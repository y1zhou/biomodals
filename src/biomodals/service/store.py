"""SQLite persistence for the local Biomodals API service."""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

import orjson

from biomodals.service.runtime_config import (
    JobAdmissionConfiguration,
    ModalConfigurationSnapshot,
)


class UserAlreadyExistsError(ValueError):
    """Raised when an administrator tries to reuse an email address."""


class UserNotFoundError(LookupError):
    """Raised when an administrator names an unknown user."""


class FirstUserMustBeAdminError(ValueError):
    """Raised when bootstrap would leave the service without an administrator."""


class LastActiveAdminError(RuntimeError):
    """Raised when a change would leave no active administrator."""


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different request."""


class JobLimitExceededError(RuntimeError):
    """Raised when a user has reached a workload's active-job limit."""


class JobNotFoundError(LookupError):
    """Raised when an owner-scoped job lookup fails."""


class JobNotCancellableError(RuntimeError):
    """Raised when cancellation is requested for a terminal job."""


class JobSubmissionConflictError(RuntimeError):
    """Raised when a stale submitter tries to attach a provider call."""


class JobState(StrEnum):
    """Durable provider-neutral job states."""

    QUEUED = "queued"
    RUNNING = "running"
    FINALIZING = "finalizing"
    CANCEL_REQUESTED = "cancel_requested"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_JOB_STATES = (
    JobState.QUEUED,
    JobState.RUNNING,
    JobState.FINALIZING,
    JobState.CANCEL_REQUESTED,
)
TERMINAL_JOB_STATES = (
    JobState.SUCCEEDED,
    JobState.PARTIAL,
    JobState.FAILED,
    JobState.CANCELLED,
)
RECONCILABLE_JOB_STATES = (*ACTIVE_JOB_STATES, JobState.BLOCKED)


class UserStatus(StrEnum):
    """Explicit account lifecycle independent of Administrator role."""

    PENDING_SETUP = "pending_setup"
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class UserRecord:
    """One administrator-provisioned service user."""

    user_id: UUID
    email: str
    display_name: str
    password_hash: str | None
    status: UserStatus
    is_admin: bool
    active_job_limit: int
    created_at: int
    updated_at: int

    @property
    def active(self) -> bool:
        """Compatibility predicate for code that needs an enabled account."""
        return self.status == UserStatus.ENABLED


@dataclass(frozen=True, slots=True)
class StoredSession:
    """An authenticated session loaded without exposing its bearer token."""

    user: UserRecord
    csrf_digest: bytes
    created_at: int
    last_seen_at: int
    absolute_expires_at: int


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One private asynchronous job."""

    job_id: UUID
    owner_user_id: UUID
    workload: str
    display_name: str
    idempotency_key: str
    request_hash: str
    parameters_json: str
    state: JobState
    modal_environment: str
    modal_app_name: str
    modal_app_version: int
    modal_call_id: str | None
    provider_operation: str | None
    run_name: str | None
    stage_history_json: str
    submission_token: str | None
    submission_lease_until: int | None
    result_volume_name: str | None
    result_volume_path: str | None
    result_filename: str | None
    result_size_bytes: int | None
    result_sha256: str | None
    result_archive_schema_version: int | None
    warnings_json: str | None
    error_code: str | None
    error_message: str | None
    created_at: int
    updated_at: int
    completed_at: int | None
    cancel_requested_at: int | None
    finalization_started_at: int | None
    finalization_retry_started_at: int | None
    finalization_retry_count: int
    blocked_at: int | None
    next_retry_at: int | None
    blocking_category: str | None
    result_previous_state: JobState | None
    result_cached: bool
    intermediates_cleaned_at: int | None

    @property
    def warnings(self) -> list[str]:
        """Decode the small internal warning list for public job views."""
        if self.warnings_json is None:
            return []
        value = orjson.loads(self.warnings_json)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("warnings_json must contain a JSON string list")
        return value

    @property
    def stage_history(self) -> list[JobStageRecord]:
        """Decode the ordered stage transitions retained for this job."""
        return _stage_history_from_json(self.stage_history_json)

    @property
    def modal_configuration(self) -> ModalConfigurationSnapshot:
        """Return the provider identity captured when this Job was admitted."""
        return ModalConfigurationSnapshot(
            environment=self.modal_environment,
            app_name=self.modal_app_name,
            app_version=self.modal_app_version,
        )


@dataclass(frozen=True, slots=True)
class JobAdmission:
    """Result of an atomic idempotency and active-limit check."""

    job: JobRecord
    created: bool


@dataclass(frozen=True, slots=True)
class JobStageRecord:
    """One durable workload operation and its observed timing."""

    provider_operation: str
    started_at: int
    completed_at: int | None
    outcome: str | None


@dataclass(frozen=True, slots=True)
class WorkloadConfigurationRecord:
    """Optional database overrides for one fixed API workload."""

    workload: str
    modal_app_name: str | None
    modal_app_version: int | None
    active_job_limit: int | None


@dataclass(frozen=True, slots=True)
class PublishedResultUsage:
    """Durable Result accounting independent of rebuildable cache files."""

    entries: int
    bytes: int


@dataclass(frozen=True, slots=True)
class BlockedJobSummary:
    """Safe aggregate that exposes no owner or Job identifier."""

    category: str
    count: int
    oldest_blocked_at: int


def _stage_history_from_json(value: str) -> list[JobStageRecord]:
    document = orjson.loads(value)
    if not isinstance(document, list):
        raise ValueError("stage_history_json must contain a JSON list")
    history: list[JobStageRecord] = []
    for item in document:
        if not isinstance(item, dict) or set(item) != {
            "provider_operation",
            "started_at",
            "completed_at",
            "outcome",
        }:
            raise ValueError("stage_history_json contains an invalid stage")
        provider_operation = item["provider_operation"]
        started_at = item["started_at"]
        completed_at = item["completed_at"]
        outcome = item["outcome"]
        if (
            not isinstance(provider_operation, str)
            or not provider_operation
            or not isinstance(started_at, int)
            or isinstance(started_at, bool)
            or (
                completed_at is not None
                and (
                    not isinstance(completed_at, int)
                    or isinstance(completed_at, bool)
                    or completed_at < started_at
                )
            )
            or outcome not in {None, "completed", "failed", "cancelled"}
            or (completed_at is None) != (outcome is None)
        ):
            raise ValueError("stage_history_json contains an invalid stage")
        history.append(
            JobStageRecord(
                provider_operation=provider_operation,
                started_at=started_at,
                completed_at=completed_at,
                outcome=outcome,
            )
        )
    return history


def _transition_stage_history_json(
    value: str,
    *,
    now: int,
    complete_operation: str | None = None,
    complete_outcome: str = "completed",
    start_operation: str | None = None,
) -> str:
    history = _stage_history_from_json(value)
    if complete_operation is not None:
        for index in range(len(history) - 1, -1, -1):
            stage = history[index]
            if (
                stage.provider_operation == complete_operation
                and stage.completed_at is None
            ):
                history[index] = JobStageRecord(
                    provider_operation=stage.provider_operation,
                    started_at=stage.started_at,
                    completed_at=max(now, stage.started_at),
                    outcome=complete_outcome,
                )
                break
    if start_operation is not None and not (
        history
        and history[-1].provider_operation == start_operation
        and history[-1].completed_at is None
    ):
        history.append(
            JobStageRecord(
                provider_operation=start_operation,
                started_at=now,
                completed_at=None,
                outcome=None,
            )
        )
    return orjson.dumps([
        {
            "provider_operation": stage.provider_operation,
            "started_at": stage.started_at,
            "completed_at": stage.completed_at,
            "outcome": stage.outcome,
        }
        for stage in history
    ]).decode()


class ServiceStore:
    """Small synchronous repository backed by one local SQLite database."""

    def __init__(self, path: str | Path) -> None:
        """Remember the database path without opening a long-lived connection."""
        self.path = Path(path)

    def initialize(self) -> None:
        """Create the database and its first schema if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.parent.is_symlink():
            raise RuntimeError("Service state directory must not be a symbolic link")
        self.path.parent.chmod(0o700)
        created = False
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
            created = True
        if self.path.is_symlink():
            raise RuntimeError("Service database must not be a symbolic link")
        with self._connection() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if created:
                conn.executescript(
                    """
                    BEGIN IMMEDIATE;

                    CREATE TABLE users (
                        user_id TEXT PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        password_hash TEXT,
                        status TEXT NOT NULL CHECK (
                            status IN ('pending_setup', 'enabled', 'disabled')
                        ),
                        is_admin INTEGER NOT NULL CHECK (is_admin IN (0, 1)),
                        active_job_limit INTEGER NOT NULL
                            CHECK (active_job_limit >= 0),
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL
                    );

                    CREATE TABLE password_tokens (
                        token_digest BLOB PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id)
                            ON DELETE CASCADE,
                        expires_at INTEGER NOT NULL
                    );
                    CREATE INDEX password_tokens_user
                        ON password_tokens(user_id);

                    CREATE TABLE sessions (
                        token_digest BLOB PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id)
                            ON DELETE CASCADE,
                        csrf_digest BLOB NOT NULL,
                        created_at INTEGER NOT NULL,
                        last_seen_at INTEGER NOT NULL,
                        absolute_expires_at INTEGER NOT NULL
                    );
                    CREATE INDEX sessions_user ON sessions(user_id);

                    CREATE TABLE jobs (
                        job_id TEXT PRIMARY KEY,
                        owner_user_id TEXT NOT NULL REFERENCES users(user_id),
                        workload TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        request_hash TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        state TEXT NOT NULL,
                        modal_environment TEXT NOT NULL,
                        modal_app_name TEXT NOT NULL,
                        modal_app_version INTEGER NOT NULL
                            CHECK (modal_app_version >= 1),
                        modal_call_id TEXT,
                        provider_operation TEXT,
                        run_name TEXT,
                        stage_history_json TEXT NOT NULL DEFAULT '[]',
                        submission_token TEXT,
                        submission_lease_until INTEGER,
                        result_volume_name TEXT,
                        result_volume_path TEXT,
                        result_filename TEXT,
                        result_size_bytes INTEGER,
                        result_sha256 TEXT,
                        result_archive_schema_version INTEGER
                            CHECK (
                                result_archive_schema_version IS NULL
                                OR result_archive_schema_version >= 1
                            ),
                        warnings_json TEXT,
                        error_code TEXT,
                        error_message TEXT,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        completed_at INTEGER,
                        cancel_requested_at INTEGER,
                        finalization_started_at INTEGER,
                        finalization_retry_started_at INTEGER,
                        finalization_retry_count INTEGER NOT NULL DEFAULT 0,
                        blocked_at INTEGER,
                        next_retry_at INTEGER,
                        blocking_category TEXT,
                        result_previous_state TEXT,
                        result_cached INTEGER NOT NULL DEFAULT 0
                            CHECK (result_cached IN (0, 1)),
                        intermediates_cleaned_at INTEGER,
                        UNIQUE (owner_user_id, workload, idempotency_key)
                    );
                    CREATE INDEX jobs_owner_created
                        ON jobs(owner_user_id, created_at DESC);
                    CREATE INDEX jobs_active
                        ON jobs(state, owner_user_id, workload);

                    CREATE TABLE service_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE workload_settings (
                        workload TEXT PRIMARY KEY,
                        modal_app_name TEXT,
                        modal_app_version INTEGER
                            CHECK (
                                modal_app_version IS NULL
                                OR modal_app_version >= 1
                            ),
                        active_job_limit INTEGER
                            CHECK (active_job_limit IS NULL OR active_job_limit >= 0)
                    );

                    PRAGMA user_version = 9;
                    COMMIT;
                    """
                )
            elif version != 9:
                raise RuntimeError(
                    "Unsupported pre-release service database version "
                    f"{version} at {self.path}; stop the service and initialize "
                    "fresh state explicitly"
                )
            conn.execute("PRAGMA journal_mode = WAL")
        self.path.chmod(0o600)
        for path in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            if path.exists():
                path.chmod(0o600)

    def check_ready(self) -> None:
        """Verify the configured database and required schema without creating it."""
        if not self.path.is_file() or self.path.is_symlink():
            raise RuntimeError("SQLite database is unavailable")
        database_uri = f"{self.path.resolve().as_uri()}?mode=rw"
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                database_uri,
                uri=True,
                timeout=5,
                isolation_level=None,
            )
            if int(conn.execute("PRAGMA user_version").fetchone()[0]) != 9:
                raise RuntimeError("SQLite schema is unavailable")
            conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("SQLite readiness check failed") from exc
        finally:
            if conn is not None:
                conn.close()

    def published_result_usage(self) -> PublishedResultUsage:
        """Sum every published immutable Result recorded in SQLite."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(result_size_bytes), 0)
                FROM jobs WHERE result_size_bytes IS NOT NULL
                """
            ).fetchone()
        return PublishedResultUsage(entries=int(row[0]), bytes=int(row[1]))

    def blocked_job_summaries(self) -> list[BlockedJobSummary]:
        """Aggregate blocked Jobs by safe service-defined category."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT blocking_category, COUNT(*) AS count,
                       MIN(blocked_at) AS oldest_blocked_at
                FROM jobs
                WHERE state = ? AND blocking_category IS NOT NULL
                      AND blocked_at IS NOT NULL
                GROUP BY blocking_category
                ORDER BY blocking_category
                """,
                (JobState.BLOCKED.value,),
            ).fetchall()
        return [
            BlockedJobSummary(
                category=str(row["blocking_category"]),
                count=int(row["count"]),
                oldest_blocked_at=int(row["oldest_blocked_at"]),
            )
            for row in rows
        ]

    def set_result_cached(self, job_id: UUID, *, cached: bool) -> None:
        """Persist whether the rebuildable local archive is currently present."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE jobs SET result_cached = ? WHERE job_id = ?",
                (int(cached), str(job_id)),
            )

    def mark_result_cache_cleared(self, job_ids: tuple[str, ...]) -> None:
        """Exclude explicitly removed archives from future cached-size views."""
        if not job_ids:
            return
        with self._transaction() as conn:
            conn.executemany(
                "UPDATE jobs SET result_cached = 0 WHERE job_id = ?",
                ((job_id,) for job_id in job_ids),
            )

    def reconcile_result_cache(self, cached_job_ids: set[str]) -> None:
        """Reconcile durable cache-presence markers with files at startup."""
        with self._transaction() as conn:
            conn.execute("UPDATE jobs SET result_cached = 0")
            conn.executemany(
                """
                UPDATE jobs SET result_cached = 1
                WHERE job_id = ? AND result_size_bytes IS NOT NULL
                """,
                ((job_id,) for job_id in cached_job_ids),
            )

    def create_user(
        self,
        *,
        email: str,
        display_name: str,
        token_digest: bytes,
        token_expires_at: int,
        now: int,
        is_admin: bool = False,
        active_job_limit: int = 2,
    ) -> UserRecord:
        """Atomically create a pending-setup user and its one-time token."""
        if active_job_limit < 0:
            raise ValueError("active_job_limit must be non-negative")
        user_id = uuid4()
        try:
            with self._transaction() as conn:
                if not is_admin:
                    first_user = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
                    if first_user is None:
                        raise FirstUserMustBeAdminError(
                            "The first User must be provisioned as an administrator"
                        )
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, email, display_name, password_hash, status,
                        is_admin, active_job_limit, created_at, updated_at
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(user_id),
                        email,
                        display_name,
                        UserStatus.PENDING_SETUP.value,
                        int(is_admin),
                        active_job_limit,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO password_tokens (token_digest, user_id, expires_at)
                    VALUES (?, ?, ?)
                    """,
                    (token_digest, str(user_id), token_expires_at),
                )
        except sqlite3.IntegrityError as exc:
            raise UserAlreadyExistsError(f"User already exists: {email}") from exc
        user = self.get_user_by_email(email)
        if user is None:  # pragma: no cover - committed insert guarantees this
            raise RuntimeError("Created user could not be loaded")
        return user

    def get_user_by_email(self, email: str) -> UserRecord | None:
        """Load a user by normalized email."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        return _user_from_row(row) if row is not None else None

    def get_user(self, user_id: UUID) -> UserRecord | None:
        """Load one user by stable identifier."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return _user_from_row(row) if row is not None else None

    def list_users(self) -> list[UserRecord]:
        """List every user in deterministic email order."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY email, user_id"
            ).fetchall()
        return [_user_from_row(row) for row in rows]

    def issue_password_token(
        self,
        user_id: UUID,
        *,
        token_digest: bytes,
        expires_at: int,
    ) -> None:
        """Replace a user's outstanding setup/reset links with one token."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT status FROM users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
            if row is None or row["status"] == UserStatus.DISABLED.value:
                raise UserNotFoundError("Enabled or pending-setup user not found")
            conn.execute(
                "DELETE FROM password_tokens WHERE user_id = ?",
                (str(user_id),),
            )
            conn.execute(
                """
                INSERT INTO password_tokens (token_digest, user_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (token_digest, str(user_id), expires_at),
            )

    def set_password_from_token(
        self,
        token_digest: bytes,
        *,
        password_hash: str,
        session_token_digest: bytes,
        csrf_digest: bytes,
        now: int,
        absolute_expires_at: int,
    ) -> UserRecord | None:
        """Replace credentials and establish one fresh session atomically."""
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT u.*
                FROM password_tokens AS t
                JOIN users AS u ON u.user_id = t.user_id
                WHERE t.token_digest = ? AND t.expires_at > ?
                  AND u.status != ?
                """,
                (token_digest, now, UserStatus.DISABLED.value),
            ).fetchone()
            if row is None:
                conn.execute(
                    "DELETE FROM password_tokens WHERE expires_at <= ?",
                    (now,),
                )
                return None
            user_id = str(row["user_id"])
            conn.execute(
                """
                UPDATE users SET password_hash = ?, status = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (password_hash, UserStatus.ENABLED.value, now, user_id),
            )
            conn.execute("DELETE FROM password_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute(
                """
                INSERT INTO sessions (
                    token_digest, user_id, csrf_digest, created_at,
                    last_seen_at, absolute_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_token_digest,
                    user_id,
                    csrf_digest,
                    now,
                    now,
                    absolute_expires_at,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _user_from_row(updated)

    def password_token_is_valid(self, token_digest: bytes, *, now: int) -> bool:
        """Cheaply reject invalid links before performing Argon2 work."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM password_tokens AS t
                JOIN users AS u ON u.user_id = t.user_id
                WHERE t.token_digest = ? AND t.expires_at > ?
                  AND u.status != ?
                """,
                (token_digest, now, UserStatus.DISABLED.value),
            ).fetchone()
        return row is not None

    def create_session_if_password_matches(
        self,
        user_id: UUID,
        *,
        expected_password_hash: str,
        replacement_password_hash: str | None,
        token_digest: bytes,
        csrf_digest: bytes,
        now: int,
        absolute_expires_at: int,
    ) -> bool:
        """Create a session only if login state did not change during hashing."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT status, password_hash FROM users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
            if (
                row is None
                or row["status"] != UserStatus.ENABLED.value
                or row["password_hash"] != expected_password_hash
            ):
                return False
            if replacement_password_hash is not None:
                conn.execute(
                    """
                    UPDATE users SET password_hash = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (replacement_password_hash, now, str(user_id)),
                )
            conn.execute(
                """
                INSERT INTO sessions (
                    token_digest, user_id, csrf_digest, created_at,
                    last_seen_at, absolute_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    token_digest,
                    str(user_id),
                    csrf_digest,
                    now,
                    now,
                    absolute_expires_at,
                ),
            )
        return True

    def authenticate_session(
        self,
        token_digest: bytes,
        *,
        now: int,
        idle_timeout_seconds: int,
    ) -> StoredSession | None:
        """Load and touch a live session, deleting it when expired or disabled."""
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT
                    s.csrf_digest,
                    s.created_at AS session_created_at,
                    s.last_seen_at,
                    s.absolute_expires_at,
                    u.*
                FROM sessions AS s
                JOIN users AS u ON u.user_id = s.user_id
                WHERE s.token_digest = ?
                """,
                (token_digest,),
            ).fetchone()
            if row is None:
                return None
            expired = (
                row["status"] != UserStatus.ENABLED.value
                or now >= int(row["absolute_expires_at"])
                or now >= int(row["last_seen_at"]) + idle_timeout_seconds
            )
            if expired:
                conn.execute(
                    "DELETE FROM sessions WHERE token_digest = ?",
                    (token_digest,),
                )
                return None
            conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE token_digest = ?",
                (now, token_digest),
            )
        return StoredSession(
            user=_user_from_row(row),
            csrf_digest=bytes(row["csrf_digest"]),
            created_at=int(row["session_created_at"]),
            last_seen_at=now,
            absolute_expires_at=int(row["absolute_expires_at"]),
        )

    def revoke_session(self, token_digest: bytes) -> None:
        """Delete one browser session."""
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE token_digest = ?",
                (token_digest,),
            )

    def disable_user(self, email: str, *, now: int) -> UserRecord:
        """Disable a user and revoke all sessions and password links."""
        user = self.get_user_by_email(email)
        if user is None:
            raise UserNotFoundError(f"User not found: {email}")
        return self.update_user(user.user_id, active=False, now=now)

    def enable_user(self, email: str, *, now: int) -> UserRecord:
        """Enable a previously disabled user without changing credentials."""
        user = self.get_user_by_email(email)
        if user is None:
            raise UserNotFoundError(f"User not found: {email}")
        return self.update_user(user.user_id, active=True, now=now)

    def set_user_admin(self, email: str, *, is_admin: bool, now: int) -> UserRecord:
        """Promote or demote a user while preserving an active administrator."""
        user = self.get_user_by_email(email)
        if user is None:
            raise UserNotFoundError(f"User not found: {email}")
        return self.update_user(user.user_id, is_admin=is_admin, now=now)

    def update_user(
        self,
        user_id: UUID,
        *,
        active: bool | None = None,
        is_admin: bool | None = None,
        active_job_limit: int | None = None,
        now: int,
    ) -> UserRecord:
        """Update one user atomically and never remove the final active admin."""
        if active_job_limit is not None and active_job_limit < 0:
            raise ValueError("active_job_limit must be non-negative")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
            if row is None:
                raise UserNotFoundError(f"User not found: {user_id}")
            current_status = UserStatus(row["status"])
            target_status = current_status
            if active is False:
                target_status = UserStatus.DISABLED
            elif active is True:
                target_status = (
                    UserStatus.ENABLED
                    if row["password_hash"] is not None
                    else UserStatus.PENDING_SETUP
                )
            target_admin = bool(row["is_admin"]) if is_admin is None else is_admin
            if (
                current_status == UserStatus.ENABLED
                and bool(row["is_admin"])
                and not (target_status == UserStatus.ENABLED and target_admin)
            ):
                active_admins = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM users WHERE status = ? AND is_admin = 1",
                        (UserStatus.ENABLED.value,),
                    ).fetchone()[0]
                )
                if active_admins <= 1:
                    raise LastActiveAdminError(
                        "The last active administrator cannot be disabled or demoted"
                    )
            target_limit = (
                int(row["active_job_limit"])
                if active_job_limit is None
                else active_job_limit
            )
            conn.execute(
                """
                UPDATE users
                SET status = ?, is_admin = ?, active_job_limit = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    target_status.value,
                    int(target_admin),
                    target_limit,
                    now,
                    str(user_id),
                ),
            )
            if target_status == UserStatus.DISABLED:
                conn.execute(
                    "DELETE FROM sessions WHERE user_id = ?",
                    (str(user_id),),
                )
                conn.execute(
                    "DELETE FROM password_tokens WHERE user_id = ?",
                    (str(user_id),),
                )
            updated = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return _user_from_row(updated)

    def get_service_setting(self, key: str) -> str | None:
        """Load one optional database Admin setting."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM service_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_service_settings(self, settings: dict[str, str | None]) -> None:
        """Create, replace, or remove non-secret settings atomically."""
        if any(not key or value == "" for key, value in settings.items()):
            raise ValueError("Service setting keys and values must not be empty")
        with self._transaction() as conn:
            conn.executemany(
                "DELETE FROM service_settings WHERE key = ?",
                ((key,) for key, value in settings.items() if value is None),
            )
            conn.executemany(
                """
                INSERT INTO service_settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ((key, value) for key, value in settings.items() if value is not None),
            )

    def get_workload_configuration(
        self,
        workload: str,
    ) -> WorkloadConfigurationRecord | None:
        """Load optional database overrides for one workload."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM workload_settings WHERE workload = ?",
                (workload,),
            ).fetchone()
        if row is None:
            return None
        return WorkloadConfigurationRecord(
            workload=str(row["workload"]),
            modal_app_name=row["modal_app_name"],
            modal_app_version=row["modal_app_version"],
            active_job_limit=row["active_job_limit"],
        )

    def set_workload_configuration(
        self,
        workload: str,
        settings: dict[str, str | int | None],
    ) -> None:
        """Create, update, or remove supplied workload overrides atomically."""
        if not workload:
            raise ValueError("workload must not be empty")
        unknown = settings.keys() - {
            "modal_app_name",
            "modal_app_version",
            "active_job_limit",
        }
        if unknown:
            raise ValueError(f"Unknown workload settings: {', '.join(sorted(unknown))}")
        modal_app_name = settings.get("modal_app_name")
        modal_app_version = settings.get("modal_app_version")
        active_job_limit = settings.get("active_job_limit")
        if modal_app_name is not None and (
            not isinstance(modal_app_name, str) or not modal_app_name
        ):
            raise ValueError("modal_app_name must not be empty")
        if modal_app_version is not None and (
            type(modal_app_version) is not int or modal_app_version < 1
        ):
            raise ValueError("modal_app_version must be positive")
        if active_job_limit is not None and (
            type(active_job_limit) is not int or active_job_limit < 0
        ):
            raise ValueError("active_job_limit must be non-negative")
        if not settings:
            return
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM workload_settings WHERE workload = ?",
                (workload,),
            ).fetchone()
            next_modal_app_name = (
                row["modal_app_name"]
                if row is not None and "modal_app_name" not in settings
                else modal_app_name
            )
            next_modal_app_version = (
                row["modal_app_version"]
                if row is not None and "modal_app_version" not in settings
                else modal_app_version
            )
            next_active_job_limit = (
                row["active_job_limit"]
                if row is not None and "active_job_limit" not in settings
                else active_job_limit
            )
            if (
                next_modal_app_name is None
                and next_modal_app_version is None
                and next_active_job_limit is None
            ):
                conn.execute(
                    "DELETE FROM workload_settings WHERE workload = ?",
                    (workload,),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO workload_settings (
                        workload, modal_app_name, modal_app_version,
                        active_job_limit
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(workload) DO UPDATE SET
                        modal_app_name = excluded.modal_app_name,
                        modal_app_version = excluded.modal_app_version,
                        active_job_limit = excluded.active_job_limit
                    """,
                    (
                        workload,
                        next_modal_app_name,
                        next_modal_app_version,
                        next_active_job_limit,
                    ),
                )

    def admit_job(
        self,
        *,
        owner_user_id: UUID,
        display_name: str,
        idempotency_key: str,
        request_hash: str,
        parameters_json: str,
        configuration: JobAdmissionConfiguration,
        now: int,
    ) -> JobAdmission:
        """Atomically apply idempotency and every active Job admission limit."""
        workload = configuration.workload
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM jobs
                WHERE owner_user_id = ? AND workload = ? AND idempotency_key = ?
                """,
                (str(owner_user_id), workload, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for another request"
                    )
                return JobAdmission(job=_job_from_row(existing), created=False)

            user = conn.execute(
                "SELECT status, active_job_limit FROM users WHERE user_id = ?",
                (str(owner_user_id),),
            ).fetchone()
            if user is None:
                raise UserNotFoundError(f"User not found: {owner_user_id}")
            if user["status"] != UserStatus.ENABLED.value:
                raise UserNotFoundError(f"Enabled User not found: {owner_user_id}")
            user_active_job_limit = int(user["active_job_limit"])

            service_rows = conn.execute(
                """
                SELECT key, value FROM service_settings
                WHERE key IN ('modal_environment', 'global_active_job_limit')
                """
            ).fetchall()
            service_settings = {
                str(row["key"]): str(row["value"]) for row in service_rows
            }
            workload_row = conn.execute(
                "SELECT * FROM workload_settings WHERE workload = ?",
                (workload,),
            ).fetchone()
            stored_environment = service_settings.get("modal_environment")
            stored_global_limit = service_settings.get("global_active_job_limit")
            modal_environment = configuration.modal_environment.resolve(
                stored_environment
            )
            global_active_job_limit = configuration.global_active_job_limit.resolve(
                int(stored_global_limit) if stored_global_limit is not None else None
            )
            modal_app_name = configuration.modal_app_name.resolve(
                workload_row["modal_app_name"] if workload_row is not None else None
            )
            modal_app_version = configuration.modal_app_version.resolve(
                int(workload_row["modal_app_version"])
                if workload_row is not None
                and workload_row["modal_app_version"] is not None
                else None
            )
            workload_active_job_limit = configuration.workload_active_job_limit.resolve(
                int(workload_row["active_job_limit"])
                if workload_row is not None
                and workload_row["active_job_limit"] is not None
                else None
            )
            limits = (
                user_active_job_limit,
                workload_active_job_limit,
                global_active_job_limit,
            )
            if any(limit < 0 for limit in limits):
                raise ValueError("active job limits must be non-negative")
            if not modal_environment.strip() or not modal_app_name.strip():
                raise ValueError("Modal Job configuration must not be empty")
            if type(modal_app_version) is not int or modal_app_version < 1:
                raise ValueError("Modal App version must be positive")

            placeholders = ", ".join("?" for _ in ACTIVE_JOB_STATES)
            states = tuple(state.value for state in ACTIVE_JOB_STATES)
            user_active_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs
                    WHERE owner_user_id = ? AND state IN ({placeholders})
                    """,  # noqa: S608 - placeholders are generated, not user input
                    (str(owner_user_id), *states),
                ).fetchone()[0]
            )
            if user_active_count >= user_active_job_limit:
                raise JobLimitExceededError(
                    f"User active Job limit ({user_active_job_limit}) reached"
                )
            workload_active_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs
                    WHERE workload = ? AND state IN ({placeholders})
                    """,  # noqa: S608 - placeholders are generated, not user input
                    (workload, *states),
                ).fetchone()[0]
            )
            if workload_active_count >= workload_active_job_limit:
                raise JobLimitExceededError(
                    f"{workload} Tool active Job limit "
                    f"({workload_active_job_limit}) reached"
                )
            global_active_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs WHERE state IN ({placeholders})
                    """,  # noqa: S608 - placeholders are generated, not user input
                    states,
                ).fetchone()[0]
            )
            if global_active_count >= global_active_job_limit:
                raise JobLimitExceededError(
                    f"Global active Job limit ({global_active_job_limit}) reached"
                )

            job_id = uuid4()
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, owner_user_id, workload, display_name,
                    idempotency_key, request_hash, parameters_json, state,
                    modal_environment, modal_app_name, modal_app_version,
                    modal_call_id,
                    provider_operation, run_name, submission_token,
                    submission_lease_until, result_volume_name,
                    result_volume_path, result_filename, result_size_bytes,
                    result_sha256, warnings_json, error_code, error_message,
                    created_at, updated_at, completed_at, cancel_requested_at,
                    finalization_started_at, finalization_retry_started_at,
                    finalization_retry_count, blocked_at, next_retry_at, blocking_category,
                    result_previous_state, result_cached,
                    intermediates_cleaned_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?,
                    ?, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, NULL, 0, NULL
                )
                """,
                (
                    str(job_id),
                    str(owner_user_id),
                    workload,
                    display_name,
                    idempotency_key,
                    request_hash,
                    parameters_json,
                    JobState.QUEUED.value,
                    modal_environment.strip(),
                    modal_app_name.strip(),
                    modal_app_version,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return JobAdmission(job=_job_from_row(row), created=True)

    def get_job(self, owner_user_id: UUID, job_id: UUID) -> JobRecord | None:
        """Load a job only when it belongs to the requesting owner."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs WHERE job_id = ? AND owner_user_id = ?
                """,
                (str(job_id), str(owner_user_id)),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def list_jobs(self, owner_user_id: UUID) -> list[JobRecord]:
        """List only one owner's jobs, newest first."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs WHERE owner_user_id = ?
                ORDER BY created_at DESC, job_id DESC
                """,
                (str(owner_user_id),),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def count_active_jobs(self, workload: str | None = None) -> int:
        """Count Jobs that consume admission capacity."""
        placeholders = ", ".join("?" for _ in ACTIVE_JOB_STATES)
        states = tuple(state.value for state in ACTIVE_JOB_STATES)
        with self._connection() as conn:
            if workload is None:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE state IN ({placeholders})",  # noqa: S608
                    states,
                ).fetchone()
            else:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs
                    WHERE state IN ({placeholders}) AND workload = ?
                    """,  # noqa: S608
                    (*states, workload),
                ).fetchone()
            return int(row[0])

    def count_running_jobs(self, workload: str | None = None) -> int:
        """Backward-compatible alias for the admission-capacity count."""
        return self.count_active_jobs(workload)

    def list_reconcilable_jobs(
        self,
        workload: str | None = None,
    ) -> list[JobRecord]:
        """List non-terminal jobs, optionally restricted to one workload."""
        placeholders = ", ".join("?" for _ in RECONCILABLE_JOB_STATES)
        workload_clause = "" if workload is None else " AND workload = ?"
        parameters: tuple[str, ...] = (
            *(state.value for state in RECONCILABLE_JOB_STATES),
            *((workload,) if workload is not None else ()),
        )
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE state IN ({placeholders}){workload_clause}
                ORDER BY created_at, job_id
                """,  # noqa: S608 - placeholders are generated, not user input
                parameters,
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def list_intermediate_cleanup_candidates(
        self,
        workload: str,
        *,
        completed_before: int,
    ) -> list[JobRecord]:
        """List terminal runs whose non-final files have passed retention."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE workload = ?
                  AND state IN (?, ?)
                  AND completed_at <= ?
                  AND run_name IS NOT NULL
                  AND intermediates_cleaned_at IS NULL
                ORDER BY completed_at, job_id
                """,
                (
                    workload,
                    JobState.SUCCEEDED.value,
                    JobState.PARTIAL.value,
                    completed_before,
                ),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def mark_intermediates_cleaned(self, job_id: UUID, *, now: int) -> JobRecord:
        """Record successful removal without changing final artifact state."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET intermediates_cleaned_at = ?, updated_at = ?
                WHERE job_id = ? AND state IN (?, ?)
                """,
                (
                    now,
                    now,
                    str(job_id),
                    JobState.SUCCEEDED.value,
                    JobState.PARTIAL.value,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def claim_submission(
        self,
        job_id: UUID,
        *,
        run_name: str,
        submission_token: str,
        now: int,
        lease_seconds: int = 120,
    ) -> JobRecord | None:
        """Lease one queued provider submission to a single request handler."""
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET run_name = ?, submission_token = ?,
                    submission_lease_until = ?, updated_at = ?
                WHERE job_id = ?
                  AND state = ?
                  AND modal_call_id IS NULL
                  AND (run_name IS NULL OR run_name = ?)
                  AND submission_token IS NULL
                  AND submission_lease_until IS NULL
                """,
                (
                    run_name,
                    submission_token,
                    now + lease_seconds,
                    now,
                    str(job_id),
                    JobState.QUEUED.value,
                    run_name,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(row)

    def release_submission(
        self,
        job_id: UUID,
        *,
        submission_token: str,
        now: int,
    ) -> JobRecord:
        """Release a failed submission attempt without losing the queued job."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET submission_token = NULL, submission_lease_until = NULL,
                    updated_at = ?
                WHERE job_id = ? AND submission_token = ?
                  AND modal_call_id IS NULL
                """,
                (now, str(job_id), submission_token),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def mark_submitted(
        self,
        job_id: UUID,
        *,
        modal_call_id: str,
        provider_operation: str,
        run_name: str,
        submission_token: str | None = None,
        now: int | None = None,
    ) -> JobRecord:
        """Attach provider identifiers after a successful asynchronous spawn."""
        provider_operation = provider_operation.strip()
        if not provider_operation:
            raise ValueError("Provider operation must not be empty")
        updated_at = int(time.time()) if now is None else now
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET modal_call_id = ?, provider_operation = ?, run_name = ?,
                    submission_token = NULL, submission_lease_until = NULL,
                    updated_at = ?
                WHERE job_id = ?
                  AND state IN (?, ?, ?, ?)
                  AND modal_call_id IS NULL
                  AND (run_name IS NULL OR run_name = ?)
                  AND (? IS NULL OR submission_token = ?)
                """,
                (
                    modal_call_id,
                    provider_operation,
                    run_name,
                    updated_at,
                    str(job_id),
                    *(state.value for state in ACTIVE_JOB_STATES),
                    run_name,
                    submission_token,
                    submission_token,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if cursor.rowcount != 1 and (
                row["modal_call_id"] != modal_call_id
                or row["provider_operation"] != provider_operation
                or row["run_name"] != run_name
            ):
                raise JobSubmissionConflictError(
                    f"Submission lease is no longer valid for job {job_id}"
                )
            if cursor.rowcount == 1:
                history_json = _transition_stage_history_json(
                    str(row["stage_history_json"]),
                    now=updated_at,
                    start_operation=provider_operation,
                )
                conn.execute(
                    "UPDATE jobs SET stage_history_json = ? WHERE job_id = ?",
                    (history_json, str(job_id)),
                )
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (str(job_id),),
                ).fetchone()
        return _job_from_row(row)

    def claim_provider_advance(
        self,
        job_id: UUID,
        *,
        expected_modal_call_id: str,
        submission_token: str,
        now: int,
        lease_seconds: int = 120,
    ) -> JobRecord | None:
        """Lease one transition away from a completed provider call."""
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET submission_token = ?, submission_lease_until = ?, updated_at = ?
                WHERE job_id = ?
                  AND modal_call_id = ?
                  AND state IN (?, ?)
                  AND submission_token IS NULL
                  AND submission_lease_until IS NULL
                """,
                (
                    submission_token,
                    now + lease_seconds,
                    now,
                    str(job_id),
                    expected_modal_call_id,
                    JobState.QUEUED.value,
                    JobState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            history_json = _transition_stage_history_json(
                str(row["stage_history_json"]),
                now=now,
                complete_operation=row["provider_operation"],
            )
            conn.execute(
                "UPDATE jobs SET stage_history_json = ? WHERE job_id = ?",
                (history_json, str(job_id)),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(row)

    def mark_provider_operation_completed(
        self,
        job_id: UUID,
        *,
        expected_modal_call_id: str,
        now: int,
    ) -> JobRecord | None:
        """Record an observed completion without claiming the next operation."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if (
                row["modal_call_id"] != expected_modal_call_id
                or JobState(row["state"]) in TERMINAL_JOB_STATES
            ):
                return None
            history_json = _transition_stage_history_json(
                str(row["stage_history_json"]),
                now=now,
                complete_operation=row["provider_operation"],
            )
            cursor = conn.execute(
                """
                UPDATE jobs
                SET stage_history_json = ?, updated_at = ?
                WHERE job_id = ? AND modal_call_id = ?
                  AND state IN (?, ?, ?, ?)
                """,
                (
                    history_json,
                    now,
                    str(job_id),
                    expected_modal_call_id,
                    *(state.value for state in ACTIVE_JOB_STATES),
                ),
            )
            if cursor.rowcount != 1:
                return None
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(updated)

    def release_provider_advance(
        self,
        job_id: UUID,
        *,
        expected_modal_call_id: str,
        submission_token: str,
        now: int,
    ) -> JobRecord:
        """Release a transition that failed before remote submission began."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET submission_token = NULL, submission_lease_until = NULL,
                    updated_at = ?
                WHERE job_id = ?
                  AND modal_call_id = ?
                  AND submission_token = ?
                """,
                (now, str(job_id), expected_modal_call_id, submission_token),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def replace_provider_call(
        self,
        job_id: UUID,
        *,
        expected_modal_call_id: str,
        modal_call_id: str,
        provider_operation: str,
        submission_token: str,
        now: int,
    ) -> JobRecord:
        """Atomically move an active Job to its next provider operation."""
        provider_operation = provider_operation.strip()
        if not provider_operation:
            raise ValueError("Provider operation must not be empty")
        with self._transaction() as conn:
            previous = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if previous is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            cursor = conn.execute(
                """
                UPDATE jobs
                SET modal_call_id = ?, provider_operation = ?,
                    submission_token = NULL, submission_lease_until = NULL,
                    updated_at = ?
                WHERE job_id = ?
                  AND modal_call_id = ?
                  AND submission_token = ?
                  AND state IN (?, ?, ?, ?)
                """,
                (
                    modal_call_id,
                    provider_operation,
                    now,
                    str(job_id),
                    expected_modal_call_id,
                    submission_token,
                    *(state.value for state in ACTIVE_JOB_STATES),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if cursor.rowcount != 1:
                raise JobSubmissionConflictError(
                    f"Provider operation changed concurrently for job {job_id}"
                )
            history_json = _transition_stage_history_json(
                str(previous["stage_history_json"]),
                now=now,
                complete_operation=previous["provider_operation"],
                start_operation=provider_operation,
            )
            conn.execute(
                "UPDATE jobs SET stage_history_json = ? WHERE job_id = ?",
                (history_json, str(job_id)),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(row)

    def request_cancel(
        self,
        owner_user_id: UUID,
        job_id: UUID,
        *,
        now: int,
    ) -> JobRecord:
        """Preserve a private job while recording an idempotent cancel request."""
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs WHERE job_id = ? AND owner_user_id = ?
                """,
                (str(job_id), str(owner_user_id)),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            state = JobState(row["state"])
            if state == JobState.CANCEL_REQUESTED:
                return _job_from_row(row)
            if state not in (JobState.QUEUED, JobState.RUNNING):
                raise JobNotCancellableError(f"Job is already {state.value}")
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, cancel_requested_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.CANCEL_REQUESTED.value, now, now, str(job_id)),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(updated)

    def set_job_state(
        self,
        job_id: UUID,
        state: JobState,
        *,
        now: int,
    ) -> JobRecord:
        """Set the state observed by the background reconciler."""
        if state in (JobState.SUCCEEDED, JobState.PARTIAL):
            raise ValueError("Use complete_job to record successful output")
        if state == JobState.FAILED:
            raise ValueError("Use fail_job to record a safe failure")
        if state == JobState.BLOCKED:
            raise ValueError("Use block_job to record a recoverable failure")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if JobState(row["state"]) in TERMINAL_JOB_STATES:
                return _job_from_row(row)
            if (
                JobState(row["state"]) == JobState.CANCEL_REQUESTED
                and state not in TERMINAL_JOB_STATES
            ):
                return _job_from_row(row)
            history_json = str(row["stage_history_json"])
            finalization_started_at = row["finalization_started_at"]
            if state == JobState.FINALIZING:
                history_json = _transition_stage_history_json(
                    history_json,
                    now=now,
                    complete_operation=row["provider_operation"],
                    start_operation="result_packaging",
                )
                finalization_started_at = finalization_started_at or now
            elif state == JobState.CANCELLED:
                history_json = _transition_stage_history_json(
                    history_json,
                    now=now,
                    complete_operation=row["provider_operation"],
                    complete_outcome="cancelled",
                )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, stage_history_json = ?, updated_at = ?,
                    completed_at = ?, finalization_started_at = ?
                WHERE job_id = ?
                """,
                (
                    state.value,
                    history_json,
                    now,
                    now if state in TERMINAL_JOB_STATES else None,
                    finalization_started_at,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(row)

    def schedule_finalization_retry(
        self,
        job_id: UUID,
        *,
        now: int,
        next_retry_at: int,
    ) -> JobRecord:
        """Persist a bounded retry schedule without losing compute outputs."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, finalization_started_at = COALESCE(
                        finalization_started_at, ?
                    ),
                    finalization_retry_started_at = COALESCE(
                        finalization_retry_started_at, ?
                    ),
                    finalization_retry_count = finalization_retry_count + 1,
                    next_retry_at = ?, updated_at = ?
                WHERE job_id = ? AND state IN (?, ?)
                """,
                (
                    JobState.FINALIZING.value,
                    now,
                    now,
                    next_retry_at,
                    now,
                    str(job_id),
                    JobState.FINALIZING.value,
                    JobState.BLOCKED.value,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def block_job(
        self,
        job_id: UUID,
        *,
        category: str,
        now: int,
        next_retry_at: int,
        previous_state: JobState | None = None,
    ) -> JobRecord:
        """Preserve outputs while recording a safe recoverable category."""
        if not category.strip():
            raise ValueError("Blocking category must not be empty")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            current_state = JobState(row["state"])
            if current_state in {JobState.FAILED, JobState.CANCELLED}:
                return _job_from_row(row)
            if current_state in {JobState.SUCCEEDED, JobState.PARTIAL}:
                if previous_state != current_state:
                    raise ValueError(
                        "A completed Result can only block with its current state"
                    )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, blocked_at = COALESCE(blocked_at, ?),
                    next_retry_at = ?, blocking_category = ?,
                    result_previous_state = COALESCE(result_previous_state, ?),
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JobState.BLOCKED.value,
                    now,
                    next_retry_at,
                    category,
                    previous_state.value if previous_state is not None else None,
                    now,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def complete_job(
        self,
        job_id: UUID,
        *,
        state: JobState,
        result_volume_name: str,
        result_volume_path: str,
        result_filename: str,
        result_size_bytes: int,
        result_sha256: str,
        result_archive_schema_version: int,
        warnings_json: str = "[]",
        result_cached: bool = False,
        now: int,
    ) -> JobRecord:
        """Record a verified immutable archive and its terminal job state."""
        if state not in (JobState.SUCCEEDED, JobState.PARTIAL):
            raise ValueError("Completed jobs must be succeeded or partial")
        if (
            type(result_archive_schema_version) is not int
            or result_archive_schema_version < 1
        ):
            raise ValueError("Result archive schema version must be positive")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if JobState(row["state"]) in (JobState.SUCCEEDED, JobState.PARTIAL):
                if result_cached and not bool(row["result_cached"]):
                    conn.execute(
                        "UPDATE jobs SET result_cached = 1 WHERE job_id = ?",
                        (str(job_id),),
                    )
                    row = conn.execute(
                        "SELECT * FROM jobs WHERE job_id = ?",
                        (str(job_id),),
                    ).fetchone()
                return _job_from_row(row)
            history_json = _transition_stage_history_json(
                str(row["stage_history_json"]),
                now=now,
                complete_operation=row["provider_operation"],
                start_operation="result_packaging",
            )
            history_json = _transition_stage_history_json(
                history_json,
                now=now,
                complete_operation="result_packaging",
            )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, result_volume_name = ?, result_volume_path = ?,
                    result_filename = ?, result_size_bytes = ?,
                    result_sha256 = ?, result_archive_schema_version = ?,
                    warnings_json = ?, error_code = NULL,
                    error_message = NULL, stage_history_json = ?, updated_at = ?,
                    completed_at = COALESCE(completed_at, ?), blocked_at = NULL,
                    next_retry_at = NULL,
                    blocking_category = NULL, result_previous_state = NULL,
                    result_cached = ?
                WHERE job_id = ?
                """,
                (
                    state.value,
                    result_volume_name,
                    result_volume_path,
                    result_filename,
                    result_size_bytes,
                    result_sha256,
                    result_archive_schema_version,
                    warnings_json,
                    history_json,
                    now,
                    now,
                    int(result_cached),
                    str(job_id),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(updated)

    def fail_job(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
        now: int,
    ) -> JobRecord:
        """Record a caller-sanitized terminal failure without provider details."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if JobState(row["state"]) in TERMINAL_JOB_STATES:
                return _job_from_row(row)
            history_json = _transition_stage_history_json(
                str(row["stage_history_json"]),
                now=now,
                complete_operation=(
                    "result_packaging"
                    if JobState(row["state"]) in {JobState.FINALIZING, JobState.BLOCKED}
                    else row["provider_operation"]
                ),
                complete_outcome="failed",
            )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, error_code = ?, error_message = ?,
                    submission_token = NULL, submission_lease_until = NULL,
                    stage_history_json = ?, updated_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (
                    JobState.FAILED.value,
                    error_code,
                    error_message,
                    history_json,
                    now,
                    now,
                    str(job_id),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return _job_from_row(updated)

    def database_bytes_for_test(self) -> bytes:
        """Return database files for tests that assert secrets are not persisted."""
        paths = (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm"))
        return b"".join(path.read_bytes() for path in paths if path.exists())

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()


def _user_from_row(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        user_id=UUID(row["user_id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        password_hash=row["password_hash"],
        status=UserStatus(row["status"]),
        is_admin=bool(row["is_admin"]),
        active_job_limit=int(row["active_job_limit"]),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=UUID(row["job_id"]),
        owner_user_id=UUID(row["owner_user_id"]),
        workload=str(row["workload"]),
        display_name=str(row["display_name"]),
        idempotency_key=str(row["idempotency_key"]),
        request_hash=str(row["request_hash"]),
        parameters_json=str(row["parameters_json"]),
        state=JobState(row["state"]),
        modal_environment=str(row["modal_environment"]),
        modal_app_name=str(row["modal_app_name"]),
        modal_app_version=int(row["modal_app_version"]),
        modal_call_id=row["modal_call_id"],
        provider_operation=row["provider_operation"],
        run_name=row["run_name"],
        stage_history_json=str(row["stage_history_json"]),
        submission_token=row["submission_token"],
        submission_lease_until=row["submission_lease_until"],
        result_volume_name=row["result_volume_name"],
        result_volume_path=row["result_volume_path"],
        result_filename=row["result_filename"],
        result_size_bytes=row["result_size_bytes"],
        result_sha256=row["result_sha256"],
        result_archive_schema_version=row["result_archive_schema_version"],
        warnings_json=row["warnings_json"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        completed_at=row["completed_at"],
        cancel_requested_at=row["cancel_requested_at"],
        finalization_started_at=row["finalization_started_at"],
        finalization_retry_started_at=row["finalization_retry_started_at"],
        finalization_retry_count=int(row["finalization_retry_count"]),
        blocked_at=row["blocked_at"],
        next_retry_at=row["next_retry_at"],
        blocking_category=row["blocking_category"],
        result_previous_state=(
            JobState(row["result_previous_state"])
            if row["result_previous_state"] is not None
            else None
        ),
        result_cached=bool(row["result_cached"]),
        intermediates_cleaned_at=row["intermediates_cleaned_at"],
    )
