"""Physical execution state for one remotely coordinated Direct CLI App Run."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from uuid import UUID

from biomodals.execution import SqliteExecutionRepository

LEDGER_FILENAME = "ledger.sqlite3"


class AppExecutionRunStore:
    """Own one app Run's SQLite connection and reserved state path."""

    def __init__(self, volume_root: str | Path, execution_run_id: UUID) -> None:
        """Bind storage only to the host Volume root and opaque Run ID."""
        self.volume_root = Path(volume_root)
        self.execution_run_id = execution_run_id
        self._connection: sqlite3.Connection | None = None
        self._execution: SqliteExecutionRepository | None = None
        self._lock = RLock()
        self._volume_sync_active = False

    @property
    def state_root(self) -> Path:
        """Return the reserved directory containing only execution state."""
        return (
            self.volume_root
            / ".biomodals"
            / "execution"
            / "runs"
            / str(self.execution_run_id)
        )

    @property
    def ledger_path(self) -> Path:
        """Return the per-Run App Run Ledger path."""
        return self.state_root / LEDGER_FILENAME

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active caller-owned SQLite connection."""
        with self._lock:
            return self._connect()

    @property
    def execution(self) -> SqliteExecutionRepository:
        """Return the shared execution repository on the active connection."""
        with self._lock:
            self._connect()
            if self._execution is None:
                raise RuntimeError("Execution repository was not initialized")
            return self._execution

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit or roll back one caller-owned execution transaction."""
        with self._lock:
            connection = self._connect()
            if connection.in_transaction:
                raise RuntimeError("Nested app execution transactions are unsupported")
            try:
                yield
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def closed_for_volume_sync(self) -> Iterator[None]:
        """Commit and close SQLite while the host synchronizes its Volume."""
        with self._lock:
            connection = self._connection
            if connection is not None:
                connection.commit()
            self._close()
            self._volume_sync_active = True
            try:
                yield
            finally:
                self._volume_sync_active = False

    def close(self) -> None:
        """Close the active connection without inventing an implicit commit."""
        with self._lock:
            self._close()

    def commit(self) -> None:
        """Commit coordinator-local SQLite changes without syncing its Volume."""
        with self._lock:
            self._connect().commit()

    def _connect(self) -> sqlite3.Connection:
        if self._volume_sync_active:
            raise RuntimeError("App store is closed for Volume synchronization")
        if self._connection is not None:
            return self._connection

        self.state_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.ledger_path,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            execution = SqliteExecutionRepository(connection)
            execution.initialize_schema()
            connection.commit()
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._execution = execution
        return connection

    def _close(self) -> None:
        connection = self._connection
        self._connection = None
        self._execution = None
        if connection is not None:
            connection.close()
