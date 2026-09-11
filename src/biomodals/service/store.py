"""SQLite persistence for the local Biomodals API service."""

from __future__ import annotations

import os
import sqlite3
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


class JobStateResolutionError(RuntimeError):
    """Raised when an Administrator resolves a Job in another state."""


class JobState(StrEnum):
    """Browser-facing state projected from execution and result delivery."""

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


_SESSION_TOUCH_INTERVAL_SECONDS = 5 * 60
_SERVICE_SCHEMA_VERSION = 8
_ACTIVE_JOB_STATES = (
    JobState.QUEUED.value,
    JobState.RUNNING.value,
    JobState.FINALIZING.value,
    JobState.CANCEL_REQUESTED.value,
    JobState.STATE_UNKNOWN.value,
    JobState.BLOCKED.value,
)
_JOB_TABLES_SQL = """
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES users(user_id),
    tool TEXT NOT NULL,
    display_name TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    publication_scope_digest TEXT NOT NULL,
    modal_environment TEXT NOT NULL,
    modal_app_name TEXT NOT NULL,
    modal_app_version INTEGER NOT NULL CHECK (modal_app_version >= 1),
    root_function_call_id TEXT,
    state TEXT NOT NULL CHECK (state IN (
        'queued', 'running', 'finalizing', 'cancel_requested',
        'state_unknown', 'blocked', 'succeeded', 'partial',
        'failed', 'cancelled'
    )),
    state_reason TEXT,
    state_message TEXT,
    projection_json TEXT NOT NULL DEFAULT '{}',
    projection_observed_at INTEGER,
    max_active_provider_calls INTEGER NOT NULL
        CHECK (max_active_provider_calls >= 1),
    max_active_gpu_provider_calls INTEGER NOT NULL
        CHECK (
            max_active_gpu_provider_calls >= 1
            AND max_active_gpu_provider_calls <= max_active_provider_calls
        ),
    pending_validation_id TEXT UNIQUE,
    result_state TEXT CHECK (
        result_state IS NULL OR result_state IN ('succeeded', 'partial')
    ),
    result_filename TEXT,
    result_media_type TEXT,
    result_size_bytes INTEGER,
    result_sha256 TEXT,
    result_archive_schema TEXT,
    cache_cleared_at INTEGER,
    error_code TEXT,
    error_message TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    completed_at INTEGER,
    cancel_requested_at INTEGER,
    finalization_started_at INTEGER,
    blocked_at INTEGER,
    blocking_category TEXT,
    next_retry_at INTEGER,
    UNIQUE (owner_user_id, tool, idempotency_key)
);
CREATE INDEX jobs_owner_created ON jobs(owner_user_id, created_at DESC);
CREATE INDEX jobs_tool_state ON jobs(tool, state, created_at);
CREATE INDEX jobs_owner_state ON jobs(owner_user_id, state, created_at);
CREATE INDEX jobs_polling ON jobs(state, updated_at);
CREATE INDEX jobs_publication_scope
ON jobs(tool, publication_scope_digest, created_at);
"""


def _create_job_tables(connection: sqlite3.Connection) -> None:
    """Create the fixed service Job schema inside the caller's transaction."""
    for statement in _JOB_TABLES_SQL.split(";"):
        if statement.strip():
            connection.execute(statement)


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
    tool: str
    display_name: str
    idempotency_key: str
    request_digest: str
    publication_scope_digest: str
    state: JobState
    modal_environment: str
    modal_app_name: str
    modal_app_version: int
    root_function_call_id: str | None
    projection_json: str
    projection_observed_at: int | None
    max_active_provider_calls: int
    max_active_gpu_provider_calls: int
    pending_validation_id: UUID | None
    state_reason: str | None
    state_message: str | None
    result_state: str | None
    result_filename: str | None
    result_media_type: str | None
    result_size_bytes: int | None
    result_sha256: str | None
    result_archive_schema: str | None
    error_code: str | None
    error_message: str | None
    created_at: int
    updated_at: int
    completed_at: int | None
    cancel_requested_at: int | None
    finalization_started_at: int | None
    blocked_at: int | None
    next_retry_at: int | None
    blocking_category: str | None
    cache_cleared_at: int | None

    @property
    def warnings(self) -> list[str]:
        """Decode owner-safe warnings from the bounded projection."""
        value = self.projection.get("warnings", [])
        return list(value) if isinstance(value, list) else []

    @property
    def projection(self) -> dict[str, object]:
        """Decode the bounded website projection."""
        value = orjson.loads(self.projection_json)
        if not isinstance(value, dict):
            raise ValueError("projection_json must contain a JSON object")
        return value


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
class ToolConfigurationRecord:
    """Optional database overrides for one fixed API Tool."""

    tool: str
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

                    {_JOB_TABLES_SQL}

                    CREATE TABLE service_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE tool_settings (
                        tool TEXT PRIMARY KEY,
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
                    conn.execute(f"PRAGMA user_version = {_SERVICE_SCHEMA_VERSION}")
                except BaseException:
                    conn.rollback()
                    raise
                else:
                    conn.commit()
            elif version != _SERVICE_SCHEMA_VERSION:
                raise RuntimeError(
                    "Unsupported service database version "
                    f"{version} at {self.path}; stop the service and select "
                    "an empty state directory for this pre-release build"
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
            expected = {
                "users",
                "password_tokens",
                "sessions",
                "service_settings",
                "tool_settings",
                "jobs",
            }
            actual = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
                if not str(row[0]).startswith("sqlite_")
            }
            if actual != expected:
                raise RuntimeError("SQLite service schema is unavailable")
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
                FROM jobs
                WHERE result_size_bytes IS NOT NULL AND cache_cleared_at IS NULL
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
                WHERE blocking_category IS NOT NULL AND blocked_at IS NOT NULL
                GROUP BY blocking_category
                ORDER BY blocking_category
                """
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
        with self._read_transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs WHERE state = ? ORDER BY updated_at, job_id
                """,
                (JobState.STATE_UNKNOWN.value,),
            ).fetchall()
            return [_job_from_row(row) for row in rows]

    def set_result_cached(self, job_id: UUID, *, cached: bool) -> None:
        """Persist whether the rebuildable local archive is currently present."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET cache_cleared_at = CASE WHEN ? THEN NULL ELSE unixepoch() END
                WHERE job_id = ?
                """,
                (int(cached), str(job_id)),
            )

    def restore_result_cached(self, job_id: UUID, *, now: int) -> JobRecord:
        """Restore exact cached bytes and any prior completed Job state."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET cache_cleared_at = NULL,
                    state = CASE
                        WHEN state = ? AND blocking_category = 'result_integrity'
                        THEN result_state ELSE state END,
                    blocked_at = CASE
                        WHEN blocking_category = 'result_integrity' THEN NULL
                        ELSE blocked_at END,
                    state_message = CASE
                        WHEN blocking_category = 'result_integrity' THEN NULL
                        ELSE state_message END,
                    next_retry_at = CASE
                        WHEN blocking_category = 'result_integrity' THEN NULL
                        ELSE next_retry_at END,
                    blocking_category = CASE
                        WHEN blocking_category = 'result_integrity' THEN NULL
                        ELSE blocking_category END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.BLOCKED.value, now, str(job_id)),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def mark_result_cache_cleared(self, job_ids: tuple[str, ...]) -> None:
        """Exclude explicitly removed archives from future cached-size views."""
        if not job_ids:
            return
        with self._transaction() as conn:
            conn.executemany(
                "UPDATE jobs SET cache_cleared_at = unixepoch() WHERE job_id = ?",
                ((job_id,) for job_id in job_ids),
            )

    def reconcile_result_cache(self, cached_job_ids: set[str]) -> None:
        """Reconcile durable cache-presence markers with files at startup."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs SET cache_cleared_at = COALESCE(cache_cleared_at, unixepoch())
                WHERE result_filename IS NOT NULL
                """
            )
            conn.executemany(
                """
                UPDATE jobs SET cache_cleared_at = NULL
                WHERE job_id = ? AND result_filename IS NOT NULL
                """,
                ((job_id,) for job_id in cached_job_ids),
            )

    def has_admin(self) -> bool:
        """Include disabled and pending administrators in the bootstrap guard."""
        with self._connection() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1"
                ).fetchone()
                is not None
            )

    def bootstrap_admin(
        self,
        *,
        email: str,
        password_hash: str,
        active_job_limit: int,
        now: int,
    ) -> bool:
        """Atomically create the initial enabled admin without tokens or sessions."""
        if active_job_limit < 0:
            raise ValueError("active_job_limit must be non-negative")
        try:
            with self._transaction() as conn:
                if (
                    conn.execute(
                        "SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    return False
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, email, display_name, password_hash, status,
                        is_admin, active_job_limit, created_at, updated_at
                    ) VALUES (?, ?, 'Administrator', ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        email,
                        password_hash,
                        UserStatus.ENABLED.value,
                        active_job_limit,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise UserAlreadyExistsError(f"User already exists: {email}") from exc
        return True

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

    def get_tool_configuration(
        self,
        tool: str,
    ) -> ToolConfigurationRecord | None:
        """Load optional database overrides for one Tool."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tool_settings WHERE tool = ?",
                (tool,),
            ).fetchone()
        if row is None:
            return None
        return ToolConfigurationRecord(
            tool=str(row["tool"]),
            modal_app_version=row["modal_app_version"],
            active_job_limit=row["active_job_limit"],
            job_logs_visible_to_owner=(
                bool(row["job_logs_visible_to_owner"])
                if row["job_logs_visible_to_owner"] is not None
                else None
            ),
        )

    def set_tool_configuration(
        self,
        tool: str,
        settings: dict[str, int | bool | None],
    ) -> None:
        """Create, update, or remove supplied Tool overrides atomically."""
        if not tool:
            raise ValueError("tool must not be empty")
        unknown = settings.keys() - {
            "modal_app_version",
            "active_job_limit",
            "job_logs_visible_to_owner",
        }
        if unknown:
            raise ValueError(f"Unknown Tool settings: {', '.join(sorted(unknown))}")
        modal_app_version = settings.get("modal_app_version")
        active_job_limit = settings.get("active_job_limit")
        job_logs_visible_to_owner = settings.get("job_logs_visible_to_owner")
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
                "SELECT * FROM tool_settings WHERE tool = ?",
                (tool,),
            ).fetchone()
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
                next_modal_app_version is None
                and next_active_job_limit is None
                and next_job_logs_visible_to_owner is None
            ):
                conn.execute(
                    "DELETE FROM tool_settings WHERE tool = ?",
                    (tool,),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO tool_settings (
                        tool, modal_app_version, active_job_limit,
                        job_logs_visible_to_owner
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(tool) DO UPDATE SET
                        modal_app_version = excluded.modal_app_version,
                        active_job_limit = excluded.active_job_limit,
                        job_logs_visible_to_owner =
                            excluded.job_logs_visible_to_owner
                    """,
                    (
                        tool,
                        next_modal_app_version,
                        next_active_job_limit,
                        next_job_logs_visible_to_owner,
                    ),
                )

    def admit_job(
        self,
        *,
        owner_user_id: UUID,
        tool: str,
        display_name: str,
        idempotency_key: str,
        request_digest: str,
        publication_scope_digest: str | None = None,
        modal_environment: str,
        modal_app_name: str,
        modal_app_version: int,
        tool_active_job_limit: int,
        global_active_job_limit: int,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
        now: int,
        new_job_id: UUID | None = None,
        pending_validation_id: UUID | None = None,
    ) -> JobAdmission:
        """Atomically admit one Job before any provider side effect."""
        if not tool or not display_name or not idempotency_key:
            raise ValueError("Job identity fields must not be empty")
        if len(request_digest) != 64 or any(
            character not in "0123456789abcdef" for character in request_digest
        ):
            raise ValueError("Request digest must be lowercase SHA-256 text")
        selected_scope = publication_scope_digest or request_digest
        if len(selected_scope) != 64 or any(
            character not in "0123456789abcdef" for character in selected_scope
        ):
            raise ValueError("Publication scope digest must be lowercase SHA-256 text")
        if not modal_environment.strip() or not modal_app_name.strip():
            raise ValueError("Modal deployment identity must not be empty")
        if modal_app_version < 1:
            raise ValueError("Modal App version must be positive")
        if (
            max_active_provider_calls < 1
            or not 1 <= max_active_gpu_provider_calls <= max_active_provider_calls
        ):
            raise ValueError("Provider Call limits are invalid")
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM jobs
                WHERE owner_user_id = ? AND tool = ? AND idempotency_key = ?
                """,
                (str(owner_user_id), tool, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["request_digest"] != request_digest:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for another request"
                    )
                return JobAdmission(job=_job_from_row(existing), created=False)

            user = conn.execute(
                "SELECT status, active_job_limit FROM users WHERE user_id = ?",
                (str(owner_user_id),),
            ).fetchone()
            if user is None or user["status"] != UserStatus.ENABLED.value:
                raise UserNotFoundError(f"Enabled User not found: {owner_user_id}")
            limits = (
                int(user["active_job_limit"]),
                tool_active_job_limit,
                global_active_job_limit,
            )
            if any(limit < 0 for limit in limits):
                raise ValueError("Active Job limits must be non-negative")
            placeholders = ", ".join("?" for _ in _ACTIVE_JOB_STATES)
            user_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs
                    WHERE owner_user_id = ? AND state IN ({placeholders})
                    """,  # noqa: S608 - generated placeholders
                    (str(owner_user_id), *_ACTIVE_JOB_STATES),
                ).fetchone()[0]
            )
            if user_count >= limits[0]:
                raise JobLimitExceededError(
                    f"User active Job limit ({limits[0]}) reached"
                )
            tool_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs
                    WHERE tool = ? AND state IN ({placeholders})
                    """,  # noqa: S608 - generated placeholders
                    (tool, *_ACTIVE_JOB_STATES),
                ).fetchone()[0]
            )
            if tool_count >= limits[1]:
                raise JobLimitExceededError(
                    f"{tool} Tool active Job limit ({limits[1]}) reached"
                )
            total_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs WHERE state IN ({placeholders})
                    """,  # noqa: S608 - generated placeholders
                    _ACTIVE_JOB_STATES,
                ).fetchone()[0]
            )
            if total_count >= limits[2]:
                raise JobLimitExceededError(
                    f"Global active Job limit ({limits[2]}) reached"
                )

            job_id = new_job_id or uuid4()
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, owner_user_id, tool, display_name, idempotency_key,
                    request_digest, publication_scope_digest,
                    modal_environment, modal_app_name,
                    modal_app_version, state, max_active_provider_calls,
                    max_active_gpu_provider_calls, pending_validation_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    str(owner_user_id),
                    tool,
                    display_name,
                    idempotency_key,
                    request_digest,
                    selected_scope,
                    modal_environment.strip(),
                    modal_app_name.strip(),
                    modal_app_version,
                    JobState.QUEUED.value,
                    max_active_provider_calls,
                    max_active_gpu_provider_calls,
                    (
                        str(pending_validation_id)
                        if pending_validation_id is not None
                        else None
                    ),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        return JobAdmission(job=_job_from_row(row), created=True)

    def find_idempotent_job(
        self,
        owner_user_id: UUID,
        *,
        tool: str,
        idempotency_key: str,
    ) -> JobRecord | None:
        """Find a replay before consulting a consumed validation resource."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE owner_user_id = ? AND tool = ? AND idempotency_key = ?
                """,
                (str(owner_user_id), tool, idempotency_key),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def list_preceding_jobs_for_request(
        self,
        job_id: UUID,
    ) -> list[JobRecord]:
        """List earlier Jobs with the same Tool request identity."""
        with self._connection() as conn:
            current = conn.execute(
                """
                SELECT rowid, tool, publication_scope_digest
                FROM jobs WHERE job_id = ?
                """,
                (str(job_id),),
            ).fetchone()
            if current is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE rowid < ? AND tool = ? AND publication_scope_digest = ?
                ORDER BY rowid
                """,
                (
                    current["rowid"],
                    current["tool"],
                    current["publication_scope_digest"],
                ),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def validation_is_claimed(self, validation_id: UUID) -> bool:
        """Return whether an admitted Job still depends on one validation."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE pending_validation_id = ?",
                (str(validation_id),),
            ).fetchone()
        return row is not None

    def claimed_validation_ids(self) -> set[UUID]:
        """Return retained validation references for expiry cleanup."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT pending_validation_id FROM jobs
                WHERE pending_validation_id IS NOT NULL
                """
            ).fetchall()
        return {UUID(row[0]) for row in rows}

    def unstaged_job_ids(self) -> set[UUID]:
        """Return Jobs whose local request may still be needed for staging."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT job_id FROM jobs
                WHERE tool = 'gromacs' AND state = 'queued'
                    AND root_function_call_id IS NULL
                """
            ).fetchall()
        return {UUID(row[0]) for row in rows}

    def get_job(self, owner_user_id: UUID, job_id: UUID) -> JobRecord | None:
        """Load a Job only when it belongs to the requesting owner."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND owner_user_id = ?",
                (str(job_id), str(owner_user_id)),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def get_job_by_id(self, job_id: UUID) -> JobRecord | None:
        """Load one Job for an internal or already-authorized operation."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def list_jobs(self, owner_user_id: UUID) -> list[JobRecord]:
        """List one owner's Jobs in reverse creation order."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs WHERE owner_user_id = ?
                ORDER BY created_at DESC, job_id DESC
                """,
                (str(owner_user_id),),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def list_jobs_page(
        self,
        owner_user_id: UUID,
        *,
        limit: int,
        cursor: UUID | None = None,
    ) -> JobPageRecord:
        """Return one stable owner-scoped Job history page."""
        if type(limit) is not int or limit < 1:
            raise ValueError("Job page limit must be positive")
        with self._connection() as conn:
            clause = ""
            parameters: list[object] = [str(owner_user_id)]
            if cursor is not None:
                anchor = conn.execute(
                    """
                    SELECT created_at, job_id FROM jobs
                    WHERE owner_user_id = ? AND job_id = ?
                    """,
                    (str(owner_user_id), str(cursor)),
                ).fetchone()
                if anchor is None:
                    raise JobCursorError("Job cursor is invalid")
                clause = "AND (created_at < ? OR (created_at = ? AND job_id < ?))"
                parameters.extend([
                    anchor["created_at"],
                    anchor["created_at"],
                    anchor["job_id"],
                ])
            parameters.append(limit + 1)
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE owner_user_id = ? {clause}
                ORDER BY created_at DESC, job_id DESC LIMIT ?
                """,  # noqa: S608 - closed cursor clause
                parameters,
            ).fetchall()
        selected = rows[:limit]
        return JobPageRecord(
            jobs=[_job_from_row(row) for row in selected],
            next_cursor=(
                UUID(selected[-1]["job_id"]) if len(rows) > limit and selected else None
            ),
        )

    def count_active_jobs(self, workload: str | None = None) -> int:
        """Count locally projected nonterminal Jobs, optionally for one Tool."""
        placeholders = ", ".join("?" for _ in _ACTIVE_JOB_STATES)
        with self._connection() as conn:
            if workload is None:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE state IN ({placeholders})",  # noqa: S608 - generated placeholders
                    _ACTIVE_JOB_STATES,
                ).fetchone()
            else:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM jobs
                    WHERE tool = ? AND state IN ({placeholders})
                    """,  # noqa: S608 - generated placeholders
                    (workload, *_ACTIVE_JOB_STATES),
                ).fetchone()
        return int(row[0])

    def list_reconcilable_jobs(
        self,
        *,
        now: int,
        limit: int = 100,
    ) -> list[JobRecord]:
        """List bounded Jobs whose remote state or result needs work."""
        states = (
            JobState.QUEUED.value,
            JobState.RUNNING.value,
            JobState.FINALIZING.value,
            JobState.CANCEL_REQUESTED.value,
            JobState.STATE_UNKNOWN.value,
            JobState.BLOCKED.value,
        )
        placeholders = ", ".join("?" for _ in states)
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE state IN ({placeholders})
                  AND (state != ? OR root_function_call_id IS NOT NULL)
                  AND NOT (state = ? AND blocking_category = 'result_integrity'
                           AND next_retry_at IS NULL)
                  AND (next_retry_at IS NULL OR next_retry_at <= ?)
                ORDER BY updated_at, job_id LIMIT ?
                """,  # noqa: S608 - generated placeholders
                (
                    *states,
                    JobState.STATE_UNKNOWN.value,
                    JobState.BLOCKED.value,
                    now,
                    limit,
                ),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def touch_job(self, job_id: UUID, *, now: int) -> JobRecord:
        """Move one unchanged Job behind older reconciliation candidates."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs SET updated_at = CASE
                    WHEN updated_at >= ? THEN updated_at + 1 ELSE ? END
                WHERE job_id = ?
                """,
                (now, now, str(job_id)),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def record_launch(
        self,
        job_id: UUID,
        *,
        function_call_id: str,
        now: int,
    ) -> JobRecord:
        """Attach root launch evidence exactly once."""
        if not function_call_id:
            raise ValueError("Function Call ID must not be empty")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT root_function_call_id FROM jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            existing = row["root_function_call_id"]
            if existing is not None and existing != function_call_id:
                raise IdempotencyConflictError("Job already has another root call")
            conn.execute(
                """
                UPDATE jobs
                SET root_function_call_id = ?, state = ?, updated_at = ?,
                    state_reason = NULL, state_message = NULL
                WHERE job_id = ?
                """,
                (
                    function_call_id,
                    JobState.RUNNING.value,
                    now,
                    str(job_id),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        return _job_from_row(updated)

    def record_resume(
        self,
        job_id: UUID,
        *,
        previous_function_call_id: str,
        function_call_id: str,
        now: int,
    ) -> JobRecord:
        """Replace one completed root call with explicit resume evidence."""
        if not previous_function_call_id or not function_call_id:
            raise ValueError("Function Call IDs must not be empty")
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET root_function_call_id = ?, state = ?, updated_at = ?,
                    state_reason = NULL, state_message = NULL,
                    blocked_at = NULL, blocking_category = NULL,
                    next_retry_at = NULL
                WHERE job_id = ? AND root_function_call_id = ?
                    AND state IN (?, ?)
                """,
                (
                    function_call_id,
                    JobState.RUNNING.value,
                    now,
                    str(job_id),
                    previous_function_call_id,
                    JobState.BLOCKED.value,
                    JobState.STATE_UNKNOWN.value,
                ),
            )
            if cursor.rowcount != 1:
                raise JobStateResolutionError(
                    "Job is no longer awaiting remote recovery"
                )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        return _job_from_row(updated)

    def mark_submission_in_progress(self, job_id: UUID, *, now: int) -> JobRecord:
        """Fence one launch attempt before making the ambiguous provider call."""
        return self.mark_state_unknown(
            job_id,
            reason="submission_in_progress",
            message="The remote launch outcome has not been confirmed",
            now=now,
        )

    def mark_request_staged(self, job_id: UUID, *, now: int) -> JobRecord:
        """Release a retained validation after verified immutable staging."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET pending_validation_id = NULL, updated_at = ?,
                    state_reason = NULL, state_message = NULL,
                    next_retry_at = NULL
                WHERE job_id = ?
                """,
                (now, str(job_id)),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def defer_submission(
        self,
        job_id: UUID,
        *,
        reason: str,
        message: str,
        retry_at: int,
        now: int,
    ) -> JobRecord:
        """Keep an unlaunched Job queued behind matching active work."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state_reason = ?, state_message = ?, next_retry_at = ?,
                    updated_at = ?
                WHERE job_id = ? AND state = ?
                    AND root_function_call_id IS NULL
                """,
                (
                    reason,
                    message,
                    retry_at,
                    now,
                    str(job_id),
                    JobState.QUEUED.value,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def replace_projection(
        self,
        job_id: UUID,
        *,
        state: JobState,
        projection: dict[str, object],
        observed_at: int,
    ) -> JobRecord:
        """Atomically replace one disposable remote Job projection."""
        encoded = orjson.dumps(projection, option=orjson.OPT_SORT_KEYS).decode()
        if len(encoded) > 1024 * 1024:
            raise ValueError("Job projection exceeds one MiB")
        completed_at = (
            observed_at
            if state
            not in {
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.FINALIZING,
                JobState.CANCEL_REQUESTED,
                JobState.STATE_UNKNOWN,
                JobState.BLOCKED,
            }
            else None
        )
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, projection_json = ?,
                    projection_observed_at = ?, updated_at = ?,
                    completed_at = COALESCE(completed_at, ?),
                    state_reason = NULL, state_message = NULL
                WHERE job_id = ?
                """,
                (
                    state.value,
                    encoded,
                    observed_at,
                    observed_at,
                    completed_at,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def begin_finalization(
        self,
        job_id: UUID,
        *,
        result_state: JobState,
        projection: dict[str, object],
        now: int,
    ) -> JobRecord:
        """Persist the remote terminal outcome before preparing its Result."""
        if result_state not in {JobState.SUCCEEDED, JobState.PARTIAL}:
            raise ValueError("Result state must be succeeded or partial")
        encoded = orjson.dumps(projection, option=orjson.OPT_SORT_KEYS).decode()
        if len(encoded) > 1024 * 1024:
            raise ValueError("Job projection exceeds one MiB")
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, result_state = ?, projection_json = ?,
                    projection_observed_at = ?,
                    finalization_started_at = COALESCE(finalization_started_at, ?),
                    updated_at = ?, state_reason = NULL, state_message = NULL
                WHERE job_id = ?
                """,
                (
                    JobState.FINALIZING.value,
                    result_state.value,
                    encoded,
                    now,
                    now,
                    now,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def request_cancel(self, job_id: UUID, *, now: int) -> JobRecord:
        """Persist sticky cancellation intent before contacting Modal."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            state = JobState(row["state"])
            if state not in {
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.CANCEL_REQUESTED,
            }:
                raise JobNotCancellableError(
                    f"{state.value} Job does not accept cancellation"
                )
            target = (
                JobState.CANCELLED
                if state == JobState.QUEUED and row["root_function_call_id"] is None
                else JobState.CANCEL_REQUESTED
            )
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, cancel_requested_at = COALESCE(cancel_requested_at, ?),
                    completed_at = CASE WHEN ? = 'cancelled' THEN ? ELSE completed_at END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    target.value,
                    now,
                    target.value,
                    now,
                    now,
                    str(job_id),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        return _job_from_row(updated)

    def mark_state_unknown(
        self,
        job_id: UUID,
        *,
        reason: str,
        message: str,
        now: int,
    ) -> JobRecord:
        """Preserve ownership after an ambiguous provider outcome."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, state_reason = ?, state_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    JobState.STATE_UNKNOWN.value,
                    reason,
                    message,
                    now,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def resolve_state_unknown(
        self,
        job_id: UUID,
        *,
        resolution: str,
        function_call_id: str | None,
        now: int,
    ) -> JobRecord:
        """Apply one explicit Administrator decision to an ambiguous launch."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            if row["state"] != JobState.STATE_UNKNOWN.value:
                raise JobStateResolutionError("Job is not state_unknown")
            existing_call = row["root_function_call_id"]
            if resolution == "resume":
                selected_call = function_call_id or existing_call
                if not selected_call:
                    raise JobStateResolutionError(
                        "Resuming requires the existing Function Call ID"
                    )
                target = JobState.RUNNING
            elif resolution == "requeue":
                if existing_call is not None:
                    raise JobStateResolutionError(
                        "A Job with recorded launch evidence cannot be requeued"
                    )
                selected_call = None
                target = JobState.QUEUED
            elif resolution == "cancel":
                selected_call = existing_call
                target = JobState.CANCEL_REQUESTED
            else:
                raise ValueError("Unknown state resolution")
            conn.execute(
                """
                UPDATE jobs SET state = ?, root_function_call_id = ?,
                    cancel_requested_at = CASE
                        WHEN ? = 'cancel_requested' THEN COALESCE(cancel_requested_at, ?)
                        ELSE cancel_requested_at END,
                    state_reason = NULL, state_message = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    target.value,
                    selected_call,
                    target.value,
                    now,
                    now,
                    str(job_id),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        return _job_from_row(updated)

    def block_job(
        self,
        job_id: UUID,
        *,
        category: str,
        message: str,
        retry_at: int | None,
        now: int,
    ) -> JobRecord:
        """Record a recoverable service-owned delivery failure."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, blocking_category = ?, state_message = ?,
                    blocked_at = COALESCE(blocked_at, ?), next_retry_at = ?,
                    updated_at = ? WHERE job_id = ?
                """,
                (
                    JobState.BLOCKED.value,
                    category,
                    message,
                    now,
                    retry_at,
                    now,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def complete_job(
        self,
        job_id: UUID,
        *,
        result_state: JobState,
        result_filename: str,
        result_media_type: str,
        result_size_bytes: int,
        result_sha256: str,
        result_archive_schema: str,
        now: int,
    ) -> JobRecord:
        """Publish verified browser Result metadata."""
        if result_state not in {JobState.SUCCEEDED, JobState.PARTIAL}:
            raise ValueError("Result state must be succeeded or partial")
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, result_state = ?, result_filename = ?,
                    result_media_type = ?, result_size_bytes = ?,
                    result_sha256 = ?, result_archive_schema = ?,
                    cache_cleared_at = NULL, completed_at = ?,
                    updated_at = ?, blocked_at = NULL,
                    blocking_category = NULL, next_retry_at = NULL
                WHERE job_id = ?
                """,
                (
                    result_state.value,
                    result_state.value,
                    result_filename,
                    result_media_type,
                    result_size_bytes,
                    result_sha256,
                    result_archive_schema,
                    now,
                    now,
                    str(job_id),
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def fail_job(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
        now: int,
    ) -> JobRecord:
        """Record one owner-safe terminal failure."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, error_code = ?, error_message = ?,
                    completed_at = ?, updated_at = ? WHERE job_id = ?
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
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

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
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold one coherent SQLite read snapshot across a service projection."""
        with self._connection() as conn:
            conn.execute("BEGIN")
            try:
                yield conn
            finally:
                conn.rollback()

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
        tool=str(row["tool"]),
        display_name=str(row["display_name"]),
        idempotency_key=str(row["idempotency_key"]),
        request_digest=str(row["request_digest"]),
        publication_scope_digest=str(row["publication_scope_digest"]),
        state=JobState(row["state"]),
        modal_environment=str(row["modal_environment"]),
        modal_app_name=str(row["modal_app_name"]),
        modal_app_version=int(row["modal_app_version"]),
        root_function_call_id=row["root_function_call_id"],
        projection_json=str(row["projection_json"]),
        projection_observed_at=row["projection_observed_at"],
        max_active_provider_calls=int(row["max_active_provider_calls"]),
        max_active_gpu_provider_calls=int(row["max_active_gpu_provider_calls"]),
        pending_validation_id=(
            UUID(row["pending_validation_id"])
            if row["pending_validation_id"] is not None
            else None
        ),
        state_reason=row["state_reason"],
        state_message=row["state_message"],
        result_state=row["result_state"],
        result_filename=row["result_filename"],
        result_media_type=row["result_media_type"],
        result_size_bytes=row["result_size_bytes"],
        result_sha256=row["result_sha256"],
        result_archive_schema=row["result_archive_schema"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        completed_at=row["completed_at"],
        cancel_requested_at=row["cancel_requested_at"],
        finalization_started_at=row["finalization_started_at"],
        blocked_at=row["blocked_at"],
        next_retry_at=row["next_retry_at"],
        blocking_category=row["blocking_category"],
        cache_cleared_at=row["cache_cleared_at"],
    )
