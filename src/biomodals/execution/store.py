"""Provider-neutral filesystem and SQLite storage for Execution Runs."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from uuid import UUID

from biomodals.execution.artifact_store import (
    EXECUTION_ARTIFACT_TABLES,
    ExecutionArtifactStore,
)
from biomodals.execution.sqlite import SqliteExecutionRepository

LEDGER_FILENAME = "ledger.sqlite3"
COORDINATOR_PLAN_FILENAME = "workflow-plan.pkl"
_LEGACY_TABLES = {
    "artifact_files",
    "artifacts",
    "attempts",
    "node_inputs",
    "node_outputs",
    "nodes",
    "remote_calls",
    "runs",
    "workflow_artifact_files",
    "workflow_artifacts",
    "workflow_node_inputs",
    "workflow_node_outputs",
    "workflow_node_results",
    "workflow_task_outputs",
    "workflow_task_results",
}


class ExecutionStorageSync(Protocol):
    """Durability barriers required by an execution host."""

    def commit(self) -> None:
        """Make local execution state durable."""
        ...

    def reload(self) -> None:
        """Refresh state published by other execution processes."""
        ...


class ExecutionRunStore:
    """Own one Run's kernel connection and reserved state path."""

    def __init__(
        self,
        volume_root: str | Path,
        execution_run_id: UUID,
        *,
        database_path: str | Path | None = None,
        lock: Any | None = None,
        volume_io_lock: Any | None = None,
    ) -> None:
        """Bind storage only to its root and opaque Run ID."""
        self.volume_root = Path(volume_root)
        self.execution_run_id = execution_run_id
        self._database_path = None if database_path is None else Path(database_path)
        self._connection: sqlite3.Connection | None = None
        self._execution: SqliteExecutionRepository | None = None
        self._lock = RLock() if lock is None else lock
        self.volume_io_lock = RLock() if volume_io_lock is None else volume_io_lock
        self._storage_sync_active = False

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
        """Return the per-Run execution ledger path."""
        return self._database_path or self.state_root / LEDGER_FILENAME

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
                raise RuntimeError("Nested execution transactions are unsupported")
            try:
                yield
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def synchronize(self) -> Iterator[None]:
        """Serialize one compound repository and storage state boundary."""
        with self._lock:
            yield

    @contextmanager
    def closed_for_storage_sync(self) -> Iterator[None]:
        """Commit and close SQLite while the host synchronizes storage."""
        with self._lock:
            connection = self._connection
            if connection is not None:
                connection.commit()
            self._close()
            self._storage_sync_active = True
            try:
                yield
            finally:
                self._storage_sync_active = False

    def close(self) -> None:
        """Close the active connection without inventing an implicit commit."""
        with self._lock:
            self._close()

    def commit(self) -> None:
        """Commit coordinator-local SQLite changes."""
        with self._lock:
            self._connect().commit()

    def reload(self) -> None:
        """Keep the current view for storage with native coherence."""

    def _connect(self) -> sqlite3.Connection:
        if self._storage_sync_active:
            raise RuntimeError("Run store is closed for storage synchronization")
        if self._connection is not None:
            return self._connection

        self.state_root.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.ledger_path,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            execution = SqliteExecutionRepository(connection)
            execution.initialize_schema()
            self._initialize_additional_schema(connection)
            connection.commit()
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._execution = execution
        return connection

    def _initialize_additional_schema(self, connection: sqlite3.Connection) -> None:
        """Allow an adapter-owned store to share this transaction boundary."""

    def _close(self) -> None:
        connection = self._connection
        self._connection = None
        self._execution = None
        if connection is not None:
            connection.close()


class UnsupportedGraphRunStoreError(RuntimeError):
    """Raised when a graph ledger predates the execution-kernel cutover."""


class GraphExecutionRunStore(ExecutionRunStore):
    """Own one graph Run's paths, connection, and artifact boundary."""

    def __init__(
        self,
        volume_root: str | Path,
        execution_run_id: UUID,
        *,
        database_path: str | Path | None = None,
        output_root: str | Path | None = None,
        lock: Any | None = None,
        volume_io_lock: Any | None = None,
    ) -> None:
        """Select paths using only the opaque Execution Run identity."""
        super().__init__(
            volume_root,
            execution_run_id,
            database_path=database_path,
            lock=lock,
            volume_io_lock=volume_io_lock,
        )
        self._output_root = None if output_root is None else Path(output_root)
        self._artifacts: ExecutionArtifactStore | None = None

    @property
    def coordinator_plan_path(self) -> Path:
        """Return the trusted internal coordinator-plan path."""
        return self.state_root / COORDINATOR_PLAN_FILENAME

    @property
    def output_root(self) -> Path:
        """Return the separate execution-owned scientific output directory."""
        if self._output_root is not None:
            return self._output_root
        # Preserve the established physical layout for downstream consumers.
        return self.volume_root / "workflow-runs" / str(self.execution_run_id)

    @property
    def artifacts(self) -> ExecutionArtifactStore:
        """Return execution artifact storage on the active connection."""
        with self._lock:
            self._connect()
            if self._artifacts is None:
                raise RuntimeError("Execution artifact store was not initialized")
            return self._artifacts

    def write_coordinator_plan(self, content: bytes) -> None:
        """Atomically create the immutable coordinator plan for this Run."""
        if not content:
            raise ValueError("coordinator plan cannot be empty")
        with self._lock:
            if self.coordinator_plan_path.exists():
                raise FileExistsError(str(self.coordinator_plan_path))
            self.state_root.mkdir(parents=True, exist_ok=True)
            temporary_path = self.coordinator_plan_path.with_suffix(".pkl.tmp")
            temporary_path.write_bytes(content)
            temporary_path.replace(self.coordinator_plan_path)

    def read_coordinator_plan(self) -> bytes:
        """Read the trusted internal coordinator plan for this Run."""
        with self._lock:
            return self.coordinator_plan_path.read_bytes()

    def _initialize_additional_schema(self, connection: sqlite3.Connection) -> None:
        """Initialize execution-owned artifact tables on the connection."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._reject_legacy_schema(connection)
        artifacts = ExecutionArtifactStore(connection, self.execution_run_id)
        artifacts.initialize_schema()
        self._artifacts = artifacts

    @staticmethod
    def _reject_legacy_schema(connection: sqlite3.Connection) -> None:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        legacy = tables & _LEGACY_TABLES
        partial_artifacts = tables & set(EXECUTION_ARTIFACT_TABLES)
        if legacy or (
            partial_artifacts and partial_artifacts != set(EXECUTION_ARTIFACT_TABLES)
        ):
            raise UnsupportedGraphRunStoreError(
                "Unsupported pre-kernel graph ledger; initialize a fresh Execution Run"
            )

    def _close(self) -> None:
        super()._close()
        self._artifacts = None
