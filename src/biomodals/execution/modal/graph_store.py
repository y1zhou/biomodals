"""Modal Volume storage for one remotely coordinated executable graph."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from biomodals.execution.artifact_store import (
    EXECUTION_ARTIFACT_TABLES,
    ExecutionArtifactStore,
)
from biomodals.execution.modal.host import ExecutionRunStore

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


class UnsupportedGraphRunStoreError(RuntimeError):
    """Raised when a graph ledger predates the execution-kernel cutover."""


class GraphExecutionRunStore(ExecutionRunStore):
    """Own one graph Run's paths, connection, and artifact boundary."""

    def __init__(
        self,
        volume_root: str | Path,
        execution_run_id: UUID,
        *,
        lock: Any | None = None,
    ) -> None:
        """Select paths using only the opaque Execution Run identity."""
        super().__init__(volume_root, execution_run_id, lock=lock)
        self._artifacts: ExecutionArtifactStore | None = None

    @property
    def coordinator_plan_path(self) -> Path:
        """Return the trusted internal coordinator-plan path."""
        return self.state_root / COORDINATOR_PLAN_FILENAME

    @property
    def output_root(self) -> Path:
        """Return the separate execution-owned scientific output directory."""
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
        artifacts = ExecutionArtifactStore(connection)
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
