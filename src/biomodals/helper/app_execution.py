"""Physical execution state for one remotely coordinated Direct CLI App Run."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any
from uuid import UUID

from biomodals.execution import SqliteExecutionRepository

LEDGER_FILENAME = "ledger.sqlite3"


@dataclass(frozen=True, slots=True)
class ExecutionRequestFile:
    """Store one app's bounded immutable request bytes."""

    filename: str
    max_bytes: int
    name: str

    def path(self, execution_run_id: UUID) -> PurePosixPath:
        """Return the reserved path for one Execution Run."""
        return (
            PurePosixPath(".biomodals")
            / "execution"
            / "runs"
            / str(execution_run_id)
            / self.filename
        )

    def stage(
        self,
        output_volume: Any,
        execution_run_id: UUID,
        content: bytes,
    ) -> PurePosixPath:
        """Idempotently stage bytes through the client-side Volume API."""
        self._validate(content)
        path = self.path(execution_run_id)
        existing = self._read_volume(output_volume, path)
        if existing is not None:
            if existing != content:
                raise RuntimeError(f"Existing {self.name} conflicts with this run")
            return path
        with output_volume.batch_upload(force=True) as batch:
            batch.put_file(BytesIO(content), f"/{path.as_posix()}")
        return path

    def persist(
        self,
        volume_root: str | Path,
        execution_run_id: UUID,
        content: bytes,
    ) -> PurePosixPath:
        """Atomically create bytes from inside a mounted coordinator."""
        self._validate(content)
        relative = self.path(execution_run_id)
        path = Path(volume_root).joinpath(*relative.parts)
        if path.exists():
            if self.load(volume_root, execution_run_id) != content:
                raise RuntimeError(f"{self.name} is immutable")
            return relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        return relative

    def load(self, volume_root: str | Path, execution_run_id: UUID) -> bytes:
        """Load bytes from a coordinator-mounted Volume."""
        path = Path(volume_root).joinpath(*self.path(execution_run_id).parts)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Expected regular file: {path}")
        if path.stat().st_size > self.max_bytes:
            raise ValueError(f"{self.name} exceeds its byte limit")
        content = path.read_bytes()
        self._validate(content)
        return content

    def load_from_volume(
        self,
        output_volume: Any,
        execution_run_id: UUID,
    ) -> bytes:
        """Load bytes through the client-side Volume API."""
        path = self.path(execution_run_id)
        content = self._read_volume(output_volume, path)
        if content is None:
            raise FileNotFoundError(path.as_posix())
        self._validate(content)
        return content

    def _read_volume(
        self,
        output_volume: Any,
        path: PurePosixPath,
    ) -> bytes | None:
        content = bytearray()
        try:
            for chunk in output_volume.read_file(path.as_posix()):
                if not isinstance(chunk, bytes):
                    raise TypeError(f"Volume returned non-bytes for {path}")
                if len(content) + len(chunk) > self.max_bytes:
                    raise ValueError(f"Volume file exceeds its byte limit: {path}")
                content.extend(chunk)
        except FileNotFoundError:
            return None
        return bytes(content)

    def _validate(self, content: bytes) -> None:
        if not isinstance(content, bytes):
            raise TypeError(f"{self.name} must be bytes")
        if not 0 < len(content) <= self.max_bytes:
            raise ValueError(f"{self.name} exceeds its byte limit")


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
