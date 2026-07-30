"""SQLite persistence for the local Biomodals API service."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import orjson

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionPlan,
    SqliteExecutionRepository,
)
from biomodals.service.runtime_config import (
    JobAdmissionConfiguration,
    ModalConfigurationSnapshot,
)


class UserAlreadyExistsError(ValueError):
    """Raised when an administrator tries to reuse an email address."""


class UserNotFoundError(LookupError):
    """Raised when an administrator names an unknown user."""


class UserCursorError(ValueError):
    """Raised when an Administrator history cursor is unknown."""


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


class JobCursorError(ValueError):
    """Raised when a history cursor does not name one owner's Job."""


class JobNotCancellableError(RuntimeError):
    """Raised when cancellation is requested for a terminal job."""


class JobSubmissionConflictError(RuntimeError):
    """Raised when a stale submitter tries to attach a provider call."""


class JobStateResolutionError(RuntimeError):
    """Raised when an Administrator resolves a Job in another state."""


class JobState(StrEnum):
    """Durable provider-neutral job states."""

    QUEUED = "queued"
    RUNNING = "running"
    FINALIZING = "finalizing"
    CANCEL_REQUESTED = "cancel_requested"
    STATE_UNKNOWN = "state_unknown"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStateUnknownReason(StrEnum):
    """Safe reason that remote execution can no longer be confirmed."""

    SUBMISSION_OUTCOME_UNKNOWN = "submission_outcome_unknown"
    CANCELLATION_OUTCOME_UNKNOWN = "cancellation_outcome_unknown"


class JobOperationState(StrEnum):
    """Durable state of one operation used to advance a Job."""

    SUBMITTING = "submitting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STATE_UNKNOWN = "state_unknown"


class JobOperationExecutor(StrEnum):
    """Execution boundary responsible for one durable Job operation."""

    MODAL = "modal"
    LOCAL = "local"


PROVIDER_TRACKED_JOB_STATES = (
    JobState.QUEUED,
    JobState.RUNNING,
    JobState.FINALIZING,
    JobState.CANCEL_REQUESTED,
)
ACTIVE_JOB_STATES = (*PROVIDER_TRACKED_JOB_STATES, JobState.STATE_UNKNOWN)
TERMINAL_JOB_STATES = (
    JobState.SUCCEEDED,
    JobState.PARTIAL,
    JobState.FAILED,
    JobState.CANCELLED,
)
RECONCILABLE_JOB_STATES = (*PROVIDER_TRACKED_JOB_STATES, JobState.BLOCKED)
_SESSION_TOUCH_INTERVAL_SECONDS = 5 * 60
_SERVICE_SCHEMA_VERSION = 4
_RESULT_PACKAGING_OPERATION = "result_packaging"
_JOB_OPERATIONS_TABLE_SQL = """
CREATE TABLE job_operations (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    operation TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    executor TEXT NOT NULL CHECK (executor IN ('modal', 'local')),
    modal_call_id TEXT UNIQUE,
    state TEXT NOT NULL CHECK (
        state IN (
            'submitting', 'running', 'completed', 'failed',
            'cancelled', 'state_unknown'
        )
    ),
    submission_token TEXT,
    submission_lease_until INTEGER,
    started_at INTEGER,
    completed_at INTEGER,
    PRIMARY KEY (job_id, operation),
    UNIQUE (job_id, ordinal)
)
"""
_JOB_OPERATIONS_ACTIVE_INDEX_SQL = """
CREATE INDEX job_operations_active ON job_operations(job_id, state)
"""


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
class UserPageRecord:
    """One bounded Administrator User page and its continuation cursor."""

    users: list[UserRecord]
    next_cursor: UUID | None


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
    artifact_request_sha256: str | None
    state: JobState
    modal_environment: str
    modal_app_name: str
    modal_app_version: int
    run_name: str | None
    operations: tuple[JobOperationRecord, ...]
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
    state_unknown_at: int | None
    state_unknown_reason: JobStateUnknownReason | None
    finalization_started_at: int | None
    finalization_retry_started_at: int | None
    finalization_retry_count: int
    blocked_at: int | None
    next_retry_at: int | None
    blocking_category: str | None
    result_previous_state: JobState | None
    result_cached: bool
    intermediates_cleaned_at: int | None
    execution_run_id: UUID | None = None

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
        """Project durable operations into the public timeline shape."""
        history: list[JobStageRecord] = []
        terminal_states = {
            JobOperationState.COMPLETED,
            JobOperationState.FAILED,
            JobOperationState.CANCELLED,
        }
        for operation in self.operations:
            if operation.started_at is None:
                continue
            terminal = operation.state in terminal_states
            history.append(
                JobStageRecord(
                    operation=operation.operation,
                    started_at=operation.started_at,
                    completed_at=operation.completed_at if terminal else None,
                    outcome=operation.state.value if terminal else None,
                )
            )
        return history

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
class JobPageRecord:
    """One bounded owner-scoped history page and its continuation cursor."""

    jobs: list[JobRecord]
    next_cursor: UUID | None


@dataclass(frozen=True, slots=True)
class JobStageRecord:
    """One durable workload operation and its observed timing."""

    operation: str
    started_at: int
    completed_at: int | None
    outcome: str | None


@dataclass(frozen=True, slots=True)
class JobOperationRecord:
    """One durable remote or local operation in a Job graph."""

    job_id: UUID
    operation: str
    ordinal: int
    executor: JobOperationExecutor
    modal_call_id: str | None
    state: JobOperationState
    submission_token: str | None
    submission_lease_until: int | None
    started_at: int | None
    completed_at: int | None


@dataclass(frozen=True, slots=True)
class InitialModalOperation:
    """First paid operation durably leased in the Job admission transaction."""

    operation: str
    run_name: str
    submission_token: str
    lease_seconds: int = 120


@dataclass(frozen=True, slots=True)
class WorkloadConfigurationRecord:
    """Optional database overrides for one fixed API workload."""

    workload: str
    modal_app_name: str | None
    modal_app_version: int | None
    active_job_limit: int | None
    job_logs_visible_to_owner: bool | None


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
                    f"""
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
                        execution_run_id TEXT UNIQUE,
                        workload TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        request_hash TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        artifact_request_sha256 TEXT,
                        state TEXT NOT NULL,
                        modal_environment TEXT NOT NULL,
                        modal_app_name TEXT NOT NULL,
                        modal_app_version INTEGER NOT NULL
                            CHECK (modal_app_version >= 1),
                        run_name TEXT,
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
                        state_unknown_at INTEGER,
                        state_unknown_reason TEXT CHECK (
                            state_unknown_reason IS NULL OR state_unknown_reason IN (
                                'submission_outcome_unknown',
                                'cancellation_outcome_unknown'
                            )
                        ),
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

                    {_JOB_OPERATIONS_TABLE_SQL};
                    {_JOB_OPERATIONS_ACTIVE_INDEX_SQL};

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
                            CHECK (active_job_limit IS NULL OR active_job_limit >= 0),
                        job_logs_visible_to_owner INTEGER
                            CHECK (
                                job_logs_visible_to_owner IS NULL
                                OR job_logs_visible_to_owner IN (0, 1)
                            )
                    );

                    """
                )
                try:
                    SqliteExecutionRepository(conn).initialize_schema()
                    conn.execute(f"PRAGMA user_version = {_SERVICE_SCHEMA_VERSION}")
                except BaseException:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
            elif version != _SERVICE_SCHEMA_VERSION:
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
            if (
                int(conn.execute("PRAGMA user_version").fetchone()[0])
                != _SERVICE_SCHEMA_VERSION
            ):
                raise RuntimeError("SQLite schema is unavailable")
            conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
            conn.execute(
                "SELECT version FROM execution_schema WHERE singleton = 1"
            ).fetchone()
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

    def list_state_unknown_jobs(self) -> list[JobRecord]:
        """List Jobs that require explicit Administrator review."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE state = ?
                ORDER BY state_unknown_at, job_id
                """,
                (JobState.STATE_UNKNOWN.value,),
            ).fetchall()
            return _jobs_from_rows(conn, rows)

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

    def list_users_page(
        self,
        *,
        limit: int,
        cursor: UUID | None = None,
    ) -> UserPageRecord:
        """List a stable bounded User page after an optional cursor."""
        if type(limit) is not int or limit < 1:
            raise ValueError("User page limit must be positive")
        with self._connection() as conn:
            parameters: tuple[object, ...] = ()
            cursor_clause = ""
            if cursor is not None:
                anchor = conn.execute(
                    "SELECT email, user_id FROM users WHERE user_id = ?",
                    (str(cursor),),
                ).fetchone()
                if anchor is None:
                    raise UserCursorError("User cursor is invalid")
                cursor_clause = " WHERE email > ? OR (email = ? AND user_id > ?)"
                parameters = (
                    str(anchor["email"]),
                    str(anchor["email"]),
                    str(anchor["user_id"]),
                )
            rows = conn.execute(
                f"""
                SELECT * FROM users{cursor_clause}
                ORDER BY email, user_id
                LIMIT ?
                """,  # noqa: S608 - cursor clause is fixed service text
                (*parameters, limit + 1),
            ).fetchall()
        page_rows = rows[:limit]
        return UserPageRecord(
            users=[_user_from_row(row) for row in page_rows],
            next_cursor=(
                UUID(page_rows[-1]["user_id"])
                if len(rows) > limit and page_rows
                else None
            ),
        )

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
            last_seen_at = int(row["last_seen_at"])
            if now - last_seen_at >= _SESSION_TOUCH_INTERVAL_SECONDS:
                conn.execute(
                    "UPDATE sessions SET last_seen_at = ? WHERE token_digest = ?",
                    (now, token_digest),
                )
                last_seen_at = now
        return StoredSession(
            user=_user_from_row(row),
            csrf_digest=bytes(row["csrf_digest"]),
            created_at=int(row["session_created_at"]),
            last_seen_at=last_seen_at,
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
        display_name: str | None = None,
        active: bool | None = None,
        is_admin: bool | None = None,
        active_job_limit: int | None = None,
        now: int,
    ) -> UserRecord:
        """Update one user atomically and never remove the final active admin."""
        normalized_display_name = (
            display_name.strip() if display_name is not None else None
        )
        if normalized_display_name is not None and not normalized_display_name:
            raise ValueError("Display name is required")
        if normalized_display_name is not None and len(normalized_display_name) > 120:
            raise ValueError("Display name must not exceed 120 characters")
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
            target_display_name = (
                str(row["display_name"])
                if normalized_display_name is None
                else normalized_display_name
            )
            conn.execute(
                """
                UPDATE users
                SET display_name = ?, status = ?, is_admin = ?,
                    active_job_limit = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    target_display_name,
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
            job_logs_visible_to_owner=(
                bool(row["job_logs_visible_to_owner"])
                if row["job_logs_visible_to_owner"] is not None
                else None
            ),
        )

    def set_workload_configuration(
        self,
        workload: str,
        settings: dict[str, str | int | bool | None],
    ) -> None:
        """Create, update, or remove supplied workload overrides atomically."""
        if not workload:
            raise ValueError("workload must not be empty")
        unknown = settings.keys() - {
            "modal_app_name",
            "modal_app_version",
            "active_job_limit",
            "job_logs_visible_to_owner",
        }
        if unknown:
            raise ValueError(f"Unknown workload settings: {', '.join(sorted(unknown))}")
        modal_app_name = settings.get("modal_app_name")
        modal_app_version = settings.get("modal_app_version")
        active_job_limit = settings.get("active_job_limit")
        job_logs_visible_to_owner = settings.get("job_logs_visible_to_owner")
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
        if (
            job_logs_visible_to_owner is not None
            and type(job_logs_visible_to_owner) is not bool
        ):
            raise ValueError("job_logs_visible_to_owner must be boolean")
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
            next_job_logs_visible_to_owner = (
                bool(row["job_logs_visible_to_owner"])
                if row is not None
                and "job_logs_visible_to_owner" not in settings
                and row["job_logs_visible_to_owner"] is not None
                else job_logs_visible_to_owner
            )
            if (
                next_modal_app_name is None
                and next_modal_app_version is None
                and next_active_job_limit is None
                and next_job_logs_visible_to_owner is None
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
                        active_job_limit, job_logs_visible_to_owner
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(workload) DO UPDATE SET
                        modal_app_name = excluded.modal_app_name,
                        modal_app_version = excluded.modal_app_version,
                        active_job_limit = excluded.active_job_limit,
                        job_logs_visible_to_owner =
                            excluded.job_logs_visible_to_owner
                    """,
                    (
                        workload,
                        next_modal_app_name,
                        next_modal_app_version,
                        next_active_job_limit,
                        next_job_logs_visible_to_owner,
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
        artifact_request_sha256: str | None = None,
        configuration: JobAdmissionConfiguration,
        now: int,
        new_job_id: UUID | None = None,
        initial_operation: InitialModalOperation | None = None,
        execution_plan: ExecutionPlan | None = None,
        execution_run_id: UUID | None = None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> JobAdmission:
        """Atomically apply idempotency and every active Job admission limit."""
        workload = configuration.workload
        if initial_operation is not None and execution_plan is not None:
            raise ValueError(
                "Initial legacy operation and Execution Plan are mutually exclusive"
            )
        execution_parameters = (
            execution_run_id,
            max_active_provider_calls,
            max_active_gpu_provider_calls,
        )
        if execution_plan is None:
            if any(value is not None for value in execution_parameters):
                raise ValueError("Execution Run parameters require an Execution Plan")
        elif (
            execution_run_id is None
            or max_active_provider_calls is None
            or max_active_gpu_provider_calls is None
        ):
            raise ValueError(
                "Execution Plan admission requires complete Run parameters"
            )
        elif execution_plan.workload_name != workload:
            raise ValueError("Execution Plan workload does not match Job workload")
        if artifact_request_sha256 is not None and (
            len(artifact_request_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in artifact_request_sha256
            )
        ):
            raise ValueError("Artifact request SHA-256 must be lowercase hexadecimal")
        if initial_operation is not None:
            operation = initial_operation.operation.strip()
            run_name = initial_operation.run_name.strip()
            submission_token = initial_operation.submission_token.strip()
            if not operation:
                raise ValueError("Initial Job operation must not be empty")
            if not run_name:
                raise ValueError("Initial Job run name must not be empty")
            if not submission_token:
                raise ValueError("Initial Job submission token must not be empty")
            if initial_operation.lease_seconds < 1:
                raise ValueError("Initial Job lease must be positive")
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

            user = conn.execute(
                "SELECT status, active_job_limit FROM users WHERE user_id = ?",
                (str(owner_user_id),),
            ).fetchone()
            if user is None:
                raise UserNotFoundError(f"User not found: {owner_user_id}")
            if user["status"] != UserStatus.ENABLED.value:
                raise UserNotFoundError(f"Enabled User not found: {owner_user_id}")
            if existing is not None:
                return JobAdmission(
                    job=_job_from_row_with_operations(conn, existing),
                    created=False,
                )
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

            job_id = new_job_id or uuid4()
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, owner_user_id, execution_run_id, workload, display_name,
                    idempotency_key, request_hash, parameters_json,
                    artifact_request_sha256, state,
                    modal_environment, modal_app_name, modal_app_version,
                    run_name, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    str(job_id),
                    str(owner_user_id),
                    (None if execution_run_id is None else str(execution_run_id)),
                    workload,
                    display_name,
                    idempotency_key,
                    request_hash,
                    parameters_json,
                    artifact_request_sha256,
                    JobState.QUEUED.value,
                    modal_environment.strip(),
                    modal_app_name.strip(),
                    modal_app_version,
                    run_name if initial_operation is not None else None,
                    now,
                    now,
                ),
            )
            if execution_plan is not None:
                SqliteExecutionRepository(conn).create_run(
                    execution_run_id=cast(UUID, execution_run_id),
                    plan=execution_plan,
                    deployment=DeploymentIdentity(
                        modal_environment.strip(),
                        modal_app_name.strip(),
                        modal_app_version,
                    ),
                    max_active_provider_calls=cast(
                        int,
                        max_active_provider_calls,
                    ),
                    max_active_gpu_provider_calls=cast(
                        int,
                        max_active_gpu_provider_calls,
                    ),
                    now=now,
                )
            if initial_operation is not None:
                conn.execute(
                    """
                    INSERT INTO job_operations (
                        job_id, operation, ordinal, executor, modal_call_id, state,
                        submission_token, submission_lease_until,
                        started_at, completed_at
                    ) VALUES (?, ?, 0, ?, NULL, ?, ?, ?, NULL, NULL)
                    """,
                    (
                        str(job_id),
                        operation,
                        JobOperationExecutor.MODAL.value,
                        JobOperationState.SUBMITTING.value,
                        submission_token,
                        now + initial_operation.lease_seconds,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            job = _job_from_row_with_operations(conn, row)
        return JobAdmission(job=job, created=True)

    def get_job(self, owner_user_id: UUID, job_id: UUID) -> JobRecord | None:
        """Load a job only when it belongs to the requesting owner."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs WHERE job_id = ? AND owner_user_id = ?
                """,
                (str(job_id), str(owner_user_id)),
            ).fetchone()
            return _job_from_row_with_operations(conn, row) if row is not None else None

    def get_job_by_id(self, job_id: UUID) -> JobRecord | None:
        """Load one Job for an internal or already-authorized operation."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            return _job_from_row_with_operations(conn, row) if row is not None else None

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
            return _jobs_from_rows(conn, rows)

    def list_jobs_page(
        self,
        owner_user_id: UUID,
        *,
        limit: int,
        cursor: UUID | None = None,
    ) -> JobPageRecord:
        """List a stable bounded page after an optional owner-scoped cursor."""
        if type(limit) is not int or limit < 1:
            raise ValueError("Job page limit must be positive")
        with self._connection() as conn:
            parameters: tuple[object, ...] = (str(owner_user_id),)
            cursor_clause = ""
            if cursor is not None:
                anchor = conn.execute(
                    """
                    SELECT created_at, job_id FROM jobs
                    WHERE owner_user_id = ? AND job_id = ?
                    """,
                    (str(owner_user_id), str(cursor)),
                ).fetchone()
                if anchor is None:
                    raise JobCursorError("Job history cursor is invalid")
                cursor_clause = (
                    " AND (created_at < ? OR (created_at = ? AND job_id < ?))"
                )
                parameters = (
                    str(owner_user_id),
                    int(anchor["created_at"]),
                    int(anchor["created_at"]),
                    str(anchor["job_id"]),
                )
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE owner_user_id = ?{cursor_clause}
                ORDER BY created_at DESC, job_id DESC
                LIMIT ?
                """,  # noqa: S608 - cursor clause is fixed service text
                (*parameters, limit + 1),
            ).fetchall()
            page_rows = rows[:limit]
            return JobPageRecord(
                jobs=_jobs_from_rows(conn, page_rows),
                next_cursor=(
                    UUID(page_rows[-1]["job_id"])
                    if len(rows) > limit and page_rows
                    else None
                ),
            )

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
            return _jobs_from_rows(conn, rows)

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
            return _jobs_from_rows(conn, rows)

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
            return _job_from_row_with_operations(conn, row)

    @contextmanager
    def execution_repository(self) -> Iterator[SqliteExecutionRepository]:
        """Open one atomic kernel-state transaction in the service database."""
        with self._transaction() as conn:
            yield SqliteExecutionRepository(conn)

    def list_operations(self, job_id: UUID) -> list[JobOperationRecord]:
        """List every durable remote or local operation for one Job."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_operations
                WHERE job_id = ?
                ORDER BY ordinal
                """,
                (str(job_id),),
            ).fetchall()
            return [_operation_from_row(row) for row in rows]

    def claim_modal_operation(
        self,
        job_id: UUID,
        *,
        operation: str,
        submission_token: str,
        now: int,
        run_name: str | None = None,
        lease_seconds: int = 120,
        require_enabled_owner: bool = False,
    ) -> JobOperationRecord | None:
        """Lease one not-yet-started Modal operation exactly once."""
        operation = operation.strip()
        if not operation:
            raise ValueError("Job operation must not be empty")
        if not submission_token:
            raise ValueError("Submission token must not be empty")
        if run_name is not None and not run_name.strip():
            raise ValueError("Run name must not be empty")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        with self._transaction() as conn:
            job = conn.execute(
                """
                SELECT jobs.state, jobs.run_name, users.status AS owner_status
                FROM jobs
                JOIN users ON users.user_id = jobs.owner_user_id
                WHERE jobs.job_id = ?
                """,
                (str(job_id),),
            ).fetchone()
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if JobState(job["state"]) not in {JobState.QUEUED, JobState.RUNNING}:
                return None
            if (
                require_enabled_owner
                and job["owner_status"] != UserStatus.ENABLED.value
            ):
                return None
            if run_name is not None and job["run_name"] not in {None, run_name}:
                return None
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO job_operations (
                    job_id, operation, ordinal, executor, modal_call_id, state,
                    submission_token, submission_lease_until,
                    started_at, completed_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, NULL, NULL)
                """,
                (
                    str(job_id),
                    operation,
                    _next_operation_ordinal(conn, job_id),
                    JobOperationExecutor.MODAL.value,
                    JobOperationState.SUBMITTING.value,
                    submission_token,
                    now + lease_seconds,
                ),
            )
            if cursor.rowcount != 1:
                return None
            conn.execute(
                """
                UPDATE jobs
                SET run_name = COALESCE(run_name, ?), updated_at = ?
                WHERE job_id = ?
                """,
                (run_name, now, str(job_id)),
            )
            row = conn.execute(
                """
                SELECT * FROM job_operations
                WHERE job_id = ? AND operation = ?
                """,
                (str(job_id), operation),
            ).fetchone()
            return _operation_from_row(row)

    def release_operation(
        self,
        job_id: UUID,
        *,
        operation: str,
        submission_token: str,
        now: int,
    ) -> JobRecord:
        """Release a claim known not to have started remote work."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM job_operations
                WHERE job_id = ? AND operation = ?
                  AND state = ? AND submission_token = ?
                """,
                (
                    str(job_id),
                    operation,
                    JobOperationState.SUBMITTING.value,
                    submission_token,
                ),
            )
            if cursor.rowcount == 1:
                conn.execute(
                    "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                    (now, str(job_id)),
                )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            return _job_from_row_with_operations(conn, row)

    def attach_modal_call(
        self,
        job_id: UUID,
        *,
        operation: str,
        modal_call_id: str,
        submission_token: str,
        now: int,
    ) -> JobRecord:
        """Attach a detached Modal call to its leased Job operation."""
        with self._transaction() as conn:
            try:
                cursor = conn.execute(
                    """
                    UPDATE job_operations
                    SET modal_call_id = ?, state = ?, submission_token = NULL,
                        submission_lease_until = NULL, started_at = ?
                    WHERE job_id = ? AND operation = ?
                      AND state = ? AND submission_token = ?
                    """,
                    (
                        modal_call_id,
                        JobOperationState.RUNNING.value,
                        now,
                        str(job_id),
                        operation,
                        JobOperationState.SUBMITTING.value,
                        submission_token,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise JobSubmissionConflictError(
                    f"Modal call is already attached for job {job_id}"
                ) from exc
            job = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if cursor.rowcount != 1:
                raise JobSubmissionConflictError(
                    f"Job operation changed concurrently for job {job_id}"
                )
            conn.execute(
                "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                (now, str(job_id)),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            return _job_from_row_with_operations(conn, updated)

    def record_operation_outcome(
        self,
        job_id: UUID,
        *,
        operation: str,
        expected_modal_call_id: str,
        outcome: JobOperationState,
        now: int,
    ) -> JobRecord | None:
        """Record one observed terminal operation outcome exactly once."""
        if outcome not in {
            JobOperationState.COMPLETED,
            JobOperationState.FAILED,
            JobOperationState.CANCELLED,
        }:
            raise ValueError("Job operation outcome must be terminal")
        with self._transaction() as conn:
            call = conn.execute(
                """
                SELECT * FROM job_operations
                WHERE job_id = ? AND operation = ?
                """,
                (str(job_id), operation),
            ).fetchone()
            job = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if call is None or call["modal_call_id"] != expected_modal_call_id:
                return None
            if call["state"] == outcome.value:
                return _job_from_row_with_operations(conn, job)
            if call["state"] != JobOperationState.RUNNING.value:
                return None
            cursor = conn.execute(
                """
                UPDATE job_operations
                SET state = ?, completed_at = ?
                WHERE job_id = ? AND operation = ?
                  AND modal_call_id = ? AND state = ?
                """,
                (
                    outcome.value,
                    now,
                    str(job_id),
                    operation,
                    expected_modal_call_id,
                    JobOperationState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                return None
            conn.execute(
                "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                (now, str(job_id)),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            return _job_from_row_with_operations(conn, updated)

    def record_operation_submission_failure(
        self,
        job_id: UUID,
        *,
        operation: str,
        submission_token: str,
        now: int,
    ) -> JobRecord | None:
        """Persist a rejected submission without inventing a started stage."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE job_operations
                SET state = ?, submission_token = NULL,
                    submission_lease_until = NULL, completed_at = ?
                WHERE job_id = ? AND operation = ?
                  AND state = ? AND submission_token = ?
                """,
                (
                    JobOperationState.FAILED.value,
                    now,
                    str(job_id),
                    operation,
                    JobOperationState.SUBMITTING.value,
                    submission_token,
                ),
            )
            job = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if cursor.rowcount != 1:
                return None
            conn.execute(
                "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                (now, str(job_id)),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            return _job_from_row_with_operations(conn, updated)

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
                return _job_from_row_with_operations(conn, row)
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
            return _job_from_row_with_operations(conn, updated)

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
        if state == JobState.STATE_UNKNOWN:
            raise ValueError("Use mark_state_unknown to record remote ambiguity")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            current_state = JobState(row["state"])
            if current_state in TERMINAL_JOB_STATES:
                return _job_from_row_with_operations(conn, row)
            if current_state == JobState.STATE_UNKNOWN:
                return _job_from_row_with_operations(conn, row)
            if (
                current_state == JobState.CANCEL_REQUESTED
                and state not in TERMINAL_JOB_STATES
            ):
                return _job_from_row_with_operations(conn, row)
            finalization_started_at = row["finalization_started_at"]
            if state == JobState.FINALIZING:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO job_operations (
                        job_id, operation, ordinal, executor, modal_call_id, state,
                        submission_token, submission_lease_until,
                        started_at, completed_at
                    ) VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?, NULL)
                    """,
                    (
                        str(job_id),
                        _RESULT_PACKAGING_OPERATION,
                        _next_operation_ordinal(conn, job_id),
                        JobOperationExecutor.LOCAL.value,
                        JobOperationState.RUNNING.value,
                        now,
                    ),
                )
                finalization_started_at = finalization_started_at or now
            elif state == JobState.CANCELLED:
                conn.execute(
                    """
                    UPDATE job_operations
                    SET state = ?, submission_token = NULL,
                        submission_lease_until = NULL,
                        completed_at = COALESCE(completed_at, ?)
                    WHERE job_id = ? AND state IN (?, ?)
                    """,
                    (
                        JobOperationState.CANCELLED.value,
                        now,
                        str(job_id),
                        JobOperationState.SUBMITTING.value,
                        JobOperationState.RUNNING.value,
                    ),
                )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, updated_at = ?, completed_at = ?,
                    finalization_started_at = ?
                WHERE job_id = ?
                """,
                (
                    state.value,
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
            return _job_from_row_with_operations(conn, row)

    def mark_state_unknown(
        self,
        job_id: UUID,
        *,
        reason: JobStateUnknownReason,
        now: int,
        uncertain_operations: Iterable[str] = (),
    ) -> JobRecord:
        """Stop automation when the existence of remote work is ambiguous."""
        uncertain = tuple(dict.fromkeys(uncertain_operations))
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            current_state = JobState(row["state"])
            if current_state not in {
                *PROVIDER_TRACKED_JOB_STATES,
                JobState.STATE_UNKNOWN,
            }:
                return _job_from_row_with_operations(conn, row)
            if current_state != JobState.STATE_UNKNOWN:
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = ?, state_unknown_at = ?, state_unknown_reason = ?,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        JobState.STATE_UNKNOWN.value,
                        now,
                        reason.value,
                        now,
                        str(job_id),
                    ),
                )
            conn.execute(
                """
                UPDATE job_operations
                SET state = ?, submission_token = NULL,
                    submission_lease_until = NULL
                WHERE job_id = ? AND state = ?
                """,
                (
                    JobOperationState.STATE_UNKNOWN.value,
                    str(job_id),
                    JobOperationState.SUBMITTING.value,
                ),
            )
            for operation in uncertain:
                conn.execute(
                    """
                    UPDATE job_operations
                    SET state = ?, submission_token = NULL,
                        submission_lease_until = NULL
                    WHERE job_id = ? AND operation = ? AND state = ?
                    """,
                    (
                        JobOperationState.STATE_UNKNOWN.value,
                        str(job_id),
                        operation,
                        JobOperationState.RUNNING.value,
                    ),
                )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            return _job_from_row_with_operations(conn, updated)

    def resolve_state_unknown(self, job_id: UUID, *, now: int) -> JobRecord:
        """Mark one manually reviewed state-unknown Job as failed."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            current_state = JobState(row["state"])
            if current_state != JobState.STATE_UNKNOWN:
                raise JobStateResolutionError(
                    f"Job is {current_state.value}, not state_unknown"
                )
            conn.execute(
                """
                UPDATE job_operations
                SET state = ?, submission_token = NULL,
                    submission_lease_until = NULL,
                    completed_at = COALESCE(completed_at, ?)
                WHERE job_id = ? AND state IN (?, ?, ?)
                """,
                (
                    JobOperationState.FAILED.value,
                    now,
                    str(job_id),
                    JobOperationState.SUBMITTING.value,
                    JobOperationState.RUNNING.value,
                    JobOperationState.STATE_UNKNOWN.value,
                ),
            )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, error_code = ?, error_message = ?,
                    updated_at = ?, completed_at = ?
                WHERE job_id = ? AND state = ?
                """,
                (
                    JobState.FAILED.value,
                    "compute_failed",
                    "An administrator could not confirm the remote compute state.",
                    now,
                    now,
                    str(job_id),
                    JobState.STATE_UNKNOWN.value,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            return _job_from_row_with_operations(conn, updated)

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
            return _job_from_row_with_operations(conn, row)

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
            if current_state == JobState.STATE_UNKNOWN:
                return _job_from_row_with_operations(conn, row)
            if current_state in {JobState.FAILED, JobState.CANCELLED}:
                return _job_from_row_with_operations(conn, row)
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
            return _job_from_row_with_operations(conn, row)

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
            current_state = JobState(row["state"])
            if current_state == JobState.STATE_UNKNOWN:
                return _job_from_row_with_operations(conn, row)
            if current_state in (JobState.SUCCEEDED, JobState.PARTIAL):
                if result_cached and not bool(row["result_cached"]):
                    conn.execute(
                        "UPDATE jobs SET result_cached = 1 WHERE job_id = ?",
                        (str(job_id),),
                    )
                    row = conn.execute(
                        "SELECT * FROM jobs WHERE job_id = ?",
                        (str(job_id),),
                    ).fetchone()
                return _job_from_row_with_operations(conn, row)
            conn.execute(
                """
                UPDATE job_operations
                SET state = ?, submission_token = NULL,
                    submission_lease_until = NULL,
                    started_at = COALESCE(started_at, ?),
                    completed_at = COALESCE(completed_at, ?)
                WHERE job_id = ? AND state IN (?, ?)
                """,
                (
                    JobOperationState.COMPLETED.value,
                    now,
                    now,
                    str(job_id),
                    JobOperationState.SUBMITTING.value,
                    JobOperationState.RUNNING.value,
                ),
            )
            conn.execute(
                """
                INSERT INTO job_operations (
                    job_id, operation, ordinal, executor, modal_call_id, state,
                    submission_token, submission_lease_until,
                    started_at, completed_at
                ) VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?)
                ON CONFLICT(job_id, operation) DO UPDATE SET
                    state = excluded.state,
                    submission_token = NULL,
                    submission_lease_until = NULL,
                    started_at = COALESCE(job_operations.started_at, excluded.started_at),
                    completed_at = excluded.completed_at
                """,
                (
                    str(job_id),
                    _RESULT_PACKAGING_OPERATION,
                    _next_operation_ordinal(conn, job_id),
                    JobOperationExecutor.LOCAL.value,
                    JobOperationState.COMPLETED.value,
                    row["finalization_started_at"] or now,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, result_volume_name = ?, result_volume_path = ?,
                    result_filename = ?, result_size_bytes = ?,
                    result_sha256 = ?, result_archive_schema_version = ?,
                    warnings_json = ?, error_code = NULL,
                    error_message = NULL, updated_at = ?,
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
            return _job_from_row_with_operations(conn, updated)

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
            if JobState(row["state"]) in (
                *TERMINAL_JOB_STATES,
                JobState.STATE_UNKNOWN,
            ):
                return _job_from_row_with_operations(conn, row)
            conn.execute(
                """
                UPDATE job_operations
                SET state = ?, submission_token = NULL,
                    submission_lease_until = NULL,
                    completed_at = COALESCE(completed_at, ?)
                WHERE job_id = ? AND state IN (?, ?, ?)
                """,
                (
                    JobOperationState.FAILED.value,
                    now,
                    str(job_id),
                    JobOperationState.SUBMITTING.value,
                    JobOperationState.RUNNING.value,
                    JobOperationState.STATE_UNKNOWN.value,
                ),
            )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, error_code = ?, error_message = ?,
                    updated_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (
                    JobState.FAILED.value,
                    error_code,
                    error_message,
                    now,
                    now,
                    str(job_id),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            return _job_from_row_with_operations(conn, updated)

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


def _operation_from_row(row: sqlite3.Row) -> JobOperationRecord:
    return JobOperationRecord(
        job_id=UUID(row["job_id"]),
        operation=str(row["operation"]),
        ordinal=int(row["ordinal"]),
        executor=JobOperationExecutor(row["executor"]),
        modal_call_id=row["modal_call_id"],
        state=JobOperationState(row["state"]),
        submission_token=row["submission_token"],
        submission_lease_until=row["submission_lease_until"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _next_operation_ordinal(conn: sqlite3.Connection, job_id: UUID) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM job_operations WHERE job_id = ?",
        (str(job_id),),
    ).fetchone()
    return int(row[0])


def _jobs_from_rows(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> list[JobRecord]:
    if not rows:
        return []
    job_ids = [str(row["job_id"]) for row in rows]
    placeholders = ", ".join("?" for _ in job_ids)
    operation_rows = conn.execute(
        f"""
        SELECT * FROM job_operations
        WHERE job_id IN ({placeholders})
        ORDER BY job_id, ordinal
        """,  # noqa: S608 - placeholders are generated, not user input
        job_ids,
    ).fetchall()
    operations: dict[str, list[JobOperationRecord]] = {job_id: [] for job_id in job_ids}
    for operation_row in operation_rows:
        operations[str(operation_row["job_id"])].append(
            _operation_from_row(operation_row)
        )
    return [_job_from_row(row, tuple(operations[str(row["job_id"])])) for row in rows]


def _job_from_row_with_operations(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> JobRecord:
    return _jobs_from_rows(conn, [row])[0]


def _job_from_row(
    row: sqlite3.Row,
    operations: tuple[JobOperationRecord, ...],
) -> JobRecord:
    return JobRecord(
        job_id=UUID(row["job_id"]),
        owner_user_id=UUID(row["owner_user_id"]),
        execution_run_id=(
            UUID(row["execution_run_id"])
            if row["execution_run_id"] is not None
            else None
        ),
        workload=str(row["workload"]),
        display_name=str(row["display_name"]),
        idempotency_key=str(row["idempotency_key"]),
        request_hash=str(row["request_hash"]),
        parameters_json=str(row["parameters_json"]),
        artifact_request_sha256=row["artifact_request_sha256"],
        state=JobState(row["state"]),
        modal_environment=str(row["modal_environment"]),
        modal_app_name=str(row["modal_app_name"]),
        modal_app_version=int(row["modal_app_version"]),
        run_name=row["run_name"],
        operations=operations,
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
        state_unknown_at=row["state_unknown_at"],
        state_unknown_reason=(
            JobStateUnknownReason(row["state_unknown_reason"])
            if row["state_unknown_reason"] is not None
            else None
        ),
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
