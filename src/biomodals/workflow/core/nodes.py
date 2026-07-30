"""Small workload contracts used by the workflow execution adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import UUID

from biomodals.schema import AppRunResult, WorkflowArtifact


@dataclass(frozen=True)
class NodeRunContext:
    """Workload-owned inputs and stable paths for one workflow Node Task."""

    execution_run_id: UUID
    workload_run_key: str
    node_id: str
    task_key: str
    work_dir: Path
    cache_dir: Path
    inputs: dict[str, list[WorkflowArtifact]]
    volume_root: Path | None = None
    workflow_volume_name: str | None = None

    def resolve_workflow_artifact(self, artifact: WorkflowArtifact) -> Path:
        """Resolve one workflow-owned artifact without allowing path traversal."""
        if self.volume_root is None or self.workflow_volume_name is None:
            raise RuntimeError("Workflow volume context is unavailable")
        if artifact.storage.volume_name != self.workflow_volume_name:
            raise ValueError(
                f"Artifact {artifact.artifact_id!r} is not stored in the "
                "workflow volume"
            )
        relative = PurePosixPath(artifact.storage.path)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("Workflow artifact path must be relative and contained")
        path = self.volume_root.joinpath(*relative.parts).resolve()
        path.relative_to(self.volume_root.resolve())
        return path


@dataclass(frozen=True)
class RemoteNodeCall:
    """One prepared call that the execution kernel may durably submit."""

    function_name: str
    uses_gpu: bool
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_image_key: str | None = None
    compatibility_key: str | None = None

    def __post_init__(self) -> None:
        """Reject an incomplete provider target before Task discovery."""
        if not self.function_name:
            raise ValueError("Remote workflow function name cannot be empty")


class WorkflowNode(Protocol):
    """Protocol for one semantic workflow DAG vertex."""

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute coordinator-local workflow logic."""


class WorkflowNativeNode:
    """Base class for coordinator-local workflow Nodes."""

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute workflow-native logic."""
        raise NotImplementedError


class RemoteWorkflowNode:
    """Base class for workflow Nodes executed by one tracked provider call."""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare arguments without submitting the provider call."""
        raise NotImplementedError

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        """Normalize one durable provider result for publication."""
        return AppRunResult.model_validate(result)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Prevent bypassing the kernel's durable provider-call boundary."""
        raise RuntimeError("Remote workflow Nodes must be submitted by the kernel")


class AppBackedNode(RemoteWorkflowNode):
    """Semantic name for a remote Node implemented by a Biomodals app."""
