"""Filesystem-backed durable ledger for Biomodals workflow runs."""

from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from biomodals.schema import (
    AppRunResult,
    ArtifactSelector,
    AttemptRecord,
    NodeExecutionPolicy,
    NodePlacement,
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

    def mark_node_running(
        self,
        node_id: str,
        attempt_id: str,
        *,
        input_artifact_ids: list[str] | None = None,
        execution_policy: NodeExecutionPolicy | None = None,
        placement: NodePlacement | None = None,
    ) -> NodeStatusRecord:
        """Mark a node as running and record its attempt id."""
        current = self._load_node_status_or_default(node_id)
        attempts = [*current.attempts]
        if attempt_id not in attempts:
            attempts.append(attempt_id)
        updates: dict[str, Any] = {
            "status": NodeStatus.RUNNING,
            "attempts": attempts,
            "error": None,
        }
        if input_artifact_ids is not None:
            updates["input_artifact_ids"] = input_artifact_ids
        if execution_policy is not None:
            updates["execution_policy"] = execution_policy
        if placement is not None:
            updates["placement"] = placement
        return self._write_node_status(current.model_copy(update=updates))

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

    def load_artifact(self, artifact_id: str) -> WorkflowArtifact:
        """Load one artifact manifest by id."""
        path = self.run_root / "artifacts" / f"{artifact_id}.json"
        return WorkflowArtifact.model_validate(self._read_json(path))

    def select_artifacts(self, selector: ArtifactSelector) -> list[WorkflowArtifact]:
        """Return artifacts matching one upstream selector."""
        artifact_dir = self.run_root / "artifacts"
        if not artifact_dir.exists():
            return []
        artifacts = [
            WorkflowArtifact.model_validate(self._read_json(path))
            for path in sorted(artifact_dir.glob("*.json"))
        ]
        return [
            artifact
            for artifact in artifacts
            if self._artifact_matches_selector(artifact, selector)
        ]

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

    @staticmethod
    def _artifact_matches_selector(
        artifact: WorkflowArtifact,
        selector: ArtifactSelector,
    ) -> bool:
        if artifact.producing_node_id != selector.producing_node_id:
            return False
        if selector.kind is not None and artifact.kind != selector.kind:
            return False
        for key, expected in selector.metadata.items():
            if artifact.metadata.get(key) != expected:
                return False
        if selector.pattern is None and selector.role is None:
            return True
        return any(
            (selector.pattern is None or fnmatch(file.path, selector.pattern))
            and (selector.role is None or file.role == selector.role)
            for file in artifact.files
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
        data: Any
        if isinstance(payload, BaseModel):
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
