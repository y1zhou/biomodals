"""Physical storage owned by one remotely coordinated workflow Run."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from uuid import UUID

from biomodals.execution import SqliteExecutionRepository
from biomodals.workflow.core.artifact_store import (
    WORKFLOW_ARTIFACT_TABLES,
    WorkflowArtifactStore,
)

LEDGER_FILENAME = "ledger.sqlite3"
_LEGACY_TABLES = {
    "artifact_files",
    "artifacts",
    "attempts",
    "node_inputs",
    "node_outputs",
    "nodes",
    "remote_calls",
    "runs",
}


class UnsupportedWorkflowRunStoreError(RuntimeError):
    """Raised when a workflow ledger predates the execution-kernel cutover."""


class WorkflowRunStore:
    """Own one workflow Run's paths, connection, and transaction boundary."""

    def __init__(self, volume_root: str | Path, execution_run_id: UUID) -> None:
        """Select paths using only the opaque Execution Run identity."""
        self.volume_root = Path(volume_root)
        self.execution_run_id = execution_run_id
        self._connection: sqlite3.Connection | None = None
        self._execution: SqliteExecutionRepository | None = None
        self._artifacts: WorkflowArtifactStore | None = None
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
        """Return the per-Run SQLite repository path."""
        return self.state_root / LEDGER_FILENAME

    @property
    def output_root(self) -> Path:
        """Return the separate workflow-owned scientific output directory."""
        return self.volume_root / "workflow-runs" / str(self.execution_run_id)

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

    @property
    def artifacts(self) -> WorkflowArtifactStore:
        """Return workflow artifact storage on the active connection."""
        with self._lock:
            self._connect()
            if self._artifacts is None:
                raise RuntimeError("Workflow artifact store was not initialized")
            return self._artifacts

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit or roll back execution and artifact changes together."""
        with self._lock:
            connection = self._connect()
            if connection.in_transaction:
                raise RuntimeError("Nested workflow transactions are not supported")
            try:
                yield
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def closed_for_volume_sync(self) -> Iterator[None]:
        """Commit and close SQLite while its backing Volume is synchronized."""
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

    def _connect(self) -> sqlite3.Connection:
        if self._volume_sync_active:
            raise RuntimeError("Workflow store is closed for Volume synchronization")
        if self._connection is not None:
            return self._connection

        self.state_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.ledger_path,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._reject_legacy_schema(connection)
            execution = SqliteExecutionRepository(connection)
            execution.initialize_schema()
            artifacts = WorkflowArtifactStore(connection)
            artifacts.initialize_schema()
            connection.commit()
        except BaseException:
            connection.close()
            raise

        self._connection = connection
        self._execution = execution
        self._artifacts = artifacts
        return connection

    @staticmethod
    def _reject_legacy_schema(connection: sqlite3.Connection) -> None:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        legacy = tables & _LEGACY_TABLES
        partial_artifacts = tables & set(WORKFLOW_ARTIFACT_TABLES)
        if legacy or (
            partial_artifacts and partial_artifacts != set(WORKFLOW_ARTIFACT_TABLES)
        ):
            raise UnsupportedWorkflowRunStoreError(
                "Unsupported pre-kernel workflow ledger; initialize a fresh "
                "Execution Run"
            )

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._execution = None
        self._artifacts = None
