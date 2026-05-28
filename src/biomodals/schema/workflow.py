"""Schemas for workflow artifacts, selectors, and durable run status."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from biomodals.schema.storage import VolumePath

# < Python 3.11 guards
try:
    from datetime import UTC
    from enum import StrEnum
except ImportError:
    from datetime import timezone

    from backports.strenum import StrEnum  # noqa: UP035

    UTC = timezone.utc  # noqa: UP017


class ArtifactKind(StrEnum):
    """Common artifact categories passed between workflow nodes."""

    STRUCTURES = "structures"
    SCORES = "scores"
    REPORT = "report"
    ARCHIVE = "archive"
    DIRECTORY = "directory"
    TABLE = "table"
    LOGS = "logs"


class NodeExecutionPolicy(StrEnum):
    """Restart behavior for an incomplete workflow node."""

    RERUN = "rerun"
    RESUME = "resume"


class NodePlacement(StrEnum):
    """Execution location for a workflow node."""

    ORCHESTRATOR = "orchestrator"
    REMOTE = "remote"


class NodeStatus(StrEnum):
    """Durable lifecycle states for one workflow node."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(StrEnum):
    """Durable lifecycle states for one workflow run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ArtifactFile(BaseModel):
    """One file recorded inside a workflow artifact."""

    path: str
    role: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowArtifact(BaseModel):
    """Durable manifest for data produced by a workflow node."""

    artifact_id: str
    producing_node_id: str
    kind: ArtifactKind
    storage: VolumePath
    files: list[ArtifactFile] = Field(default_factory=list)
    source_app_output_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactSelector(BaseModel):
    """Reference to upstream workflow artifacts consumed by a node input."""

    producing_node_id: str
    kind: ArtifactKind | None = None
    pattern: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ControlEdge(BaseModel):
    """Ordering dependency between workflow nodes without artifact passage."""

    upstream_node_id: str
    downstream_node_id: str


class WorkflowRun(BaseModel):
    """Durable status record for one workflow run."""

    workflow_name: str
    run_id: str
    status: RunStatus = RunStatus.PENDING
    dag_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttemptRecord(BaseModel):
    """Durable record for one node execution attempt."""

    node_id: str
    attempt_id: str
    status: NodeStatus = NodeStatus.RUNNING
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeStatusRecord(BaseModel):
    """Durable status record for one workflow node."""

    node_id: str
    status: NodeStatus
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.ORCHESTRATOR
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list)
    error: str | None = None
