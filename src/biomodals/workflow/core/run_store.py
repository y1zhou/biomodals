"""Physical storage owned by one remotely coordinated workflow Run."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from biomodals.helper.app_execution import ExecutionRunStore
from biomodals.workflow.core.artifact_store import (
    WORKFLOW_ARTIFACT_TABLES,
    WorkflowArtifactStore,
)

WORKFLOW_PLAN_FILENAME = "workflow-plan.pkl"
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


class WorkflowRunStore(ExecutionRunStore):
    """Own one workflow Run's paths, connection, and transaction boundary."""

    def __init__(
        self,
        volume_root: str | Path,
        execution_run_id: UUID,
        *,
        lock: Any | None = None,
    ) -> None:
        """Select paths using only the opaque Execution Run identity."""
        super().__init__(volume_root, execution_run_id, lock=lock)
        self._artifacts: WorkflowArtifactStore | None = None

    @property
    def workflow_plan_path(self) -> Path:
        """Return the trusted internal workflow-plan path."""
        return self.state_root / WORKFLOW_PLAN_FILENAME

    @property
    def output_root(self) -> Path:
        """Return the separate workflow-owned scientific output directory."""
        return self.volume_root / "workflow-runs" / str(self.execution_run_id)

    @property
    def artifacts(self) -> WorkflowArtifactStore:
        """Return workflow artifact storage on the active connection."""
        with self._lock:
            self._connect()
            if self._artifacts is None:
                raise RuntimeError("Workflow artifact store was not initialized")
            return self._artifacts

    def write_workflow_plan(self, content: bytes) -> None:
        """Atomically create the immutable workflow plan for this Run."""
        if not content:
            raise ValueError("workflow plan cannot be empty")
        with self._lock:
            if self.workflow_plan_path.exists():
                raise FileExistsError(str(self.workflow_plan_path))
            self.state_root.mkdir(parents=True, exist_ok=True)
            temporary_path = self.workflow_plan_path.with_suffix(".pkl.tmp")
            temporary_path.write_bytes(content)
            temporary_path.replace(self.workflow_plan_path)

    def read_workflow_plan(self) -> bytes:
        """Read the trusted internal workflow plan for this Run."""
        with self._lock:
            return self.workflow_plan_path.read_bytes()

    def _initialize_additional_schema(self, connection: sqlite3.Connection) -> None:
        """Initialize workflow-owned tables on the shared connection."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._reject_legacy_schema(connection)
        artifacts = WorkflowArtifactStore(connection)
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
        partial_artifacts = tables & set(WORKFLOW_ARTIFACT_TABLES)
        if legacy or (
            partial_artifacts and partial_artifacts != set(WORKFLOW_ARTIFACT_TABLES)
        ):
            raise UnsupportedWorkflowRunStoreError(
                "Unsupported pre-kernel workflow ledger; initialize a fresh "
                "Execution Run"
            )

    def _close(self) -> None:
        super()._close()
        self._artifacts = None
