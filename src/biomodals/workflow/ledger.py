"""Filesystem-backed durable ledger for Biomodals workflow runs."""

from __future__ import annotations

import json
from pathlib import Path

from biomodals.schema import (
    AppRunResult,
    AttemptRecord,
    NodeStatus,
    NodeStatusRecord,
    WorkflowArtifact,
    WorkflowRun,
)


class WorkflowLedger:
    """Filesystem-backed durable state for one workflow run."""

    def __init__(self, volume_root: str | Path):
        """Initialize a ledger rooted at a mounted workflow volume path."""
        self.volume_root = Path(volume_root)
        self.workflow_name: str | None = None
        self.run_id: str | None = None

    @property
    def run_root(self) -> Path:
        """Return the root directory for the active workflow run."""
        if self.workflow_name is None or self.run_id is None:
            raise RuntimeError("Workflow run has not been initialized")
        return self.volume_root / self.workflow_name / self.run_id

    def create_run(self, run: WorkflowRun) -> WorkflowRun:
        """Create a run ledger and write its run status file."""
        self.workflow_name = run.workflow_name
        self.run_id = run.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._write_json(self.run_root / "run.json", run)
        return run

    def load_run(self, workflow_name: str, run_id: str) -> WorkflowRun:
        """Load an existing run ledger."""
        self.workflow_name = workflow_name
        self.run_id = run_id
        return WorkflowRun.model_validate(self._read_json(self.run_root / "run.json"))

    def mark_node_pending(self, node_id: str) -> NodeStatusRecord:
        """Mark a node as pending."""
        return self._write_node_status(
            NodeStatusRecord(node_id=node_id, status=NodeStatus.PENDING)
        )

    def mark_node_running(self, node_id: str, attempt_id: str) -> NodeStatusRecord:
        """Mark a node as running and record its attempt id."""
        current = self._load_node_status_or_default(node_id)
        attempts = [*current.attempts]
        if attempt_id not in attempts:
            attempts.append(attempt_id)
        return self._write_node_status(
            current.model_copy(
                update={
                    "status": NodeStatus.RUNNING,
                    "attempts": attempts,
                    "error": None,
                }
            )
        )

    def mark_node_succeeded(
        self, node_id: str, artifact_ids: list[str]
    ) -> NodeStatusRecord:
        """Mark a node as succeeded with output artifact ids."""
        current = self._load_node_status_or_default(node_id)
        return self._write_node_status(
            current.model_copy(
                update={
                    "status": NodeStatus.SUCCEEDED,
                    "output_artifact_ids": artifact_ids,
                    "error": None,
                }
            )
        )

    def mark_node_failed(self, node_id: str, error: str) -> NodeStatusRecord:
        """Mark a node as failed with an error message."""
        current = self._load_node_status_or_default(node_id)
        return self._write_node_status(
            current.model_copy(
                update={
                    "status": NodeStatus.FAILED,
                    "error": error,
                }
            )
        )

    def record_attempt_started(self, node_id: str, attempt_id: str) -> AttemptRecord:
        """Record that a node attempt started."""
        attempt = AttemptRecord(node_id=node_id, attempt_id=attempt_id)
        self._write_json(
            self._attempt_dir(node_id, attempt_id) / "started.json",
            attempt,
        )
        return attempt

    def record_app_result(
        self, node_id: str, attempt_id: str, result: AppRunResult
    ) -> Path:
        """Write the raw app result for one node attempt."""
        path = self._attempt_dir(node_id, attempt_id) / "app_result.json"
        self._write_json(path, result)
        return path

    def record_artifacts(self, artifacts: list[WorkflowArtifact]) -> list[Path]:
        """Write workflow artifact manifests."""
        paths: list[Path] = []
        for artifact in artifacts:
            path = self.run_root / "artifacts" / f"{artifact.artifact_id}.json"
            self._write_json(path, artifact)
            paths.append(path)
        return paths

    def node_is_complete(self, node_id: str) -> bool:
        """Return whether a node has succeeded and all artifacts are recorded."""
        status_path = self._node_status_path(node_id)
        if not status_path.exists():
            return False
        status = NodeStatusRecord.model_validate(self._read_json(status_path))
        if status.status is not NodeStatus.SUCCEEDED:
            return False
        return all(
            self.run_root.joinpath("artifacts", f"{artifact_id}.json").exists()
            for artifact_id in status.output_artifact_ids
        )

    def _load_node_status_or_default(self, node_id: str) -> NodeStatusRecord:
        path = self._node_status_path(node_id)
        if path.exists():
            return NodeStatusRecord.model_validate(self._read_json(path))
        return NodeStatusRecord(node_id=node_id, status=NodeStatus.PENDING)

    def _write_node_status(self, status: NodeStatusRecord) -> NodeStatusRecord:
        self._write_json(self._node_status_path(status.node_id), status)
        return status

    def _node_status_path(self, node_id: str) -> Path:
        return self.run_root / "nodes" / node_id / "status.json"

    def _attempt_dir(self, node_id: str, attempt_id: str) -> Path:
        return self.run_root / "nodes" / node_id / "attempts" / attempt_id

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "model_dump"):
            data = payload.model_dump(mode="json")
        else:
            data = payload
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    @staticmethod
    def _read_json(path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))
