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


class UserAlreadyExistsError(ValueError):
    """Raised when an administrator tries to reuse an email address."""


class UserNotFoundError(LookupError):
    """Raised when an administrator names an unknown user."""


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different request."""


class JobLimitExceededError(RuntimeError):
    """Raised when a user has reached a workload's active-job limit."""


class JobNotFoundError(LookupError):
    """Raised when an owner-scoped job lookup fails."""


class JobNotCancellableError(RuntimeError):
    """Raised when cancellation is requested for a terminal job."""


class JobState(StrEnum):
    """Durable provider-neutral job states."""

    QUEUED = "queued"
    RUNNING = "running"
    FINALIZING = "finalizing"
    CANCEL_REQUESTED = "cancel_requested"
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


@dataclass(frozen=True, slots=True)
class UserRecord:
    """One administrator-provisioned service user."""

    user_id: UUID
    email: str
    display_name: str
    password_hash: str | None
    active: bool
    created_at: int
    updated_at: int


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
    modal_call_id: str | None
    run_name: str | None
    result_volume_name: str | None
    result_volume_path: str | None
    result_filename: str | None
    result_size_bytes: int | None
    result_sha256: str | None
    warnings_json: str | None
    error_code: str | None
    error_message: str | None
    created_at: int
    updated_at: int
    completed_at: int | None
    cancel_requested_at: int | None
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


@dataclass(frozen=True, slots=True)
class JobAdmission:
    """Result of an atomic idempotency and active-limit check."""

    job: JobRecord
    created: bool


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
        if self.path.is_symlink():
            raise RuntimeError("Service database must not be a symbolic link")
        if not self.path.exists():
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
        self.path.chmod(0o600)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                conn.executescript(
                    """
                    BEGIN IMMEDIATE;

                    CREATE TABLE users (
                        user_id TEXT PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        password_hash TEXT,
                        active INTEGER NOT NULL CHECK (active IN (0, 1)),
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
                        modal_call_id TEXT,
                        run_name TEXT,
                        result_volume_name TEXT,
                        result_volume_path TEXT,
                        result_filename TEXT,
                        result_size_bytes INTEGER,
                        result_sha256 TEXT,
                        warnings_json TEXT,
                        error_code TEXT,
                        error_message TEXT,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        completed_at INTEGER,
                        cancel_requested_at INTEGER,
                        intermediates_cleaned_at INTEGER,
                        UNIQUE (owner_user_id, workload, idempotency_key)
                    );
                    CREATE INDEX jobs_owner_created
                        ON jobs(owner_user_id, created_at DESC);
                    CREATE INDEX jobs_active
                        ON jobs(owner_user_id, workload, state);

                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
            elif version == 1:
                conn.executescript(
                    """
                    BEGIN IMMEDIATE;
                    ALTER TABLE jobs ADD COLUMN intermediates_cleaned_at INTEGER;
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
            elif version != 2:
                raise RuntimeError(f"Unsupported service database version: {version}")
        for path in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            if path.exists():
                path.chmod(0o600)

    def create_user(
        self,
        *,
        email: str,
        display_name: str,
        token_digest: bytes,
        token_expires_at: int,
        now: int,
    ) -> UserRecord:
        """Atomically create an inactive-password user and setup token."""
        user_id = uuid4()
        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, email, display_name, password_hash, active,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, NULL, 1, ?, ?)
                    """,
                    (str(user_id), email, display_name, now, now),
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
                "SELECT active FROM users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
            if row is None or not bool(row["active"]):
                raise UserNotFoundError("Active user not found")
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
        now: int,
    ) -> UserRecord | None:
        """Consume a valid token, set the hash, and revoke user credentials."""
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT u.*
                FROM password_tokens AS t
                JOIN users AS u ON u.user_id = t.user_id
                WHERE t.token_digest = ? AND t.expires_at > ? AND u.active = 1
                """,
                (token_digest, now),
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
                UPDATE users SET password_hash = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (password_hash, now, user_id),
            )
            conn.execute("DELETE FROM password_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            updated = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _user_from_row(updated)

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
                "SELECT active, password_hash FROM users WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
            if (
                row is None
                or not bool(row["active"])
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
                not bool(row["active"])
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
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT user_id FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if row is None:
                raise UserNotFoundError(f"User not found: {email}")
            user_id = str(row["user_id"])
            conn.execute(
                """
                UPDATE users SET active = 0, updated_at = ? WHERE user_id = ?
                """,
                (now, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM password_tokens WHERE user_id = ?", (user_id,))
            updated = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _user_from_row(updated)

    def admit_job(
        self,
        *,
        owner_user_id: UUID,
        workload: str,
        display_name: str,
        idempotency_key: str,
        request_hash: str,
        parameters_json: str,
        active_limit: int,
        now: int,
    ) -> JobAdmission:
        """Atomically apply idempotency and an owner/workload active limit."""
        if active_limit < 1:
            raise ValueError("active_limit must be positive")
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

            placeholders = ", ".join("?" for _ in ACTIVE_JOB_STATES)
            active_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs
                    WHERE owner_user_id = ? AND workload = ?
                      AND state IN ({placeholders})
                    """,  # noqa: S608 - placeholders are generated, not user input
                    (
                        str(owner_user_id),
                        workload,
                        *(state.value for state in ACTIVE_JOB_STATES),
                    ),
                ).fetchone()[0]
            )
            if active_count >= active_limit:
                raise JobLimitExceededError(
                    f"Active {workload} job limit ({active_limit}) reached"
                )

            job_id = uuid4()
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, owner_user_id, workload, display_name,
                    idempotency_key, request_hash, parameters_json, state,
                    modal_call_id, run_name, result_volume_name,
                    result_volume_path, result_filename, result_size_bytes,
                    result_sha256, warnings_json, error_code, error_message,
                    created_at, updated_at, completed_at, cancel_requested_at,
                    intermediates_cleaned_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, NULL
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

    def list_reconcilable_jobs(
        self,
        workload: str | None = None,
    ) -> list[JobRecord]:
        """List non-terminal jobs, optionally restricted to one workload."""
        placeholders = ", ".join("?" for _ in ACTIVE_JOB_STATES)
        workload_clause = "" if workload is None else " AND workload = ?"
        parameters: tuple[str, ...] = (
            *(state.value for state in ACTIVE_JOB_STATES),
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

    def mark_submitted(
        self,
        job_id: UUID,
        *,
        modal_call_id: str,
        run_name: str,
        now: int | None = None,
    ) -> JobRecord:
        """Attach provider identifiers after a successful asynchronous spawn."""
        updated_at = int(time.time()) if now is None else now
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET modal_call_id = ?, run_name = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (modal_call_id, run_name, updated_at, str(job_id)),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
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
            if state in TERMINAL_JOB_STATES:
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
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, updated_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (
                    state.value,
                    now,
                    now if state in TERMINAL_JOB_STATES else None,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
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
        warnings_json: str = "[]",
        now: int,
    ) -> JobRecord:
        """Record a verified immutable archive and its terminal job state."""
        if state not in (JobState.SUCCEEDED, JobState.PARTIAL):
            raise ValueError("Completed jobs must be succeeded or partial")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if JobState(row["state"]) in (JobState.SUCCEEDED, JobState.PARTIAL):
                return _job_from_row(row)
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, result_volume_name = ?, result_volume_path = ?,
                    result_filename = ?, result_size_bytes = ?,
                    result_sha256 = ?, warnings_json = ?, error_code = NULL,
                    error_message = NULL, updated_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (
                    state.value,
                    result_volume_name,
                    result_volume_path,
                    result_filename,
                    result_size_bytes,
                    result_sha256,
                    warnings_json,
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
        active=bool(row["active"]),
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
        modal_call_id=row["modal_call_id"],
        run_name=row["run_name"],
        result_volume_name=row["result_volume_name"],
        result_volume_path=row["result_volume_path"],
        result_filename=row["result_filename"],
        result_size_bytes=row["result_size_bytes"],
        result_sha256=row["result_sha256"],
        warnings_json=row["warnings_json"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        completed_at=row["completed_at"],
        cancel_requested_at=row["cancel_requested_at"],
        intermediates_cleaned_at=row["intermediates_cleaned_at"],
    )
