"""Small workload contracts used by the workflow execution adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import UUID

from biomodals.execution import AvailabilityStatus
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
    max_tasks_per_call: int = 1

    def __post_init__(self) -> None:
        """Reject an incomplete provider target before Task discovery."""
        if not self.function_name:
            raise ValueError("Remote workflow function name cannot be empty")
        if self.max_tasks_per_call < 1:
            raise ValueError("max_tasks_per_call must be positive")


@dataclass(frozen=True)
class RemotePullWorkerCall:
    """One prepared pull-worker pool backed by durable Task claims."""

    function_name: str
    uses_gpu: bool
    claim_capacity: int
    max_worker_calls: int
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_image_key: str | None = None
    compatibility_key: str | None = None

    def __post_init__(self) -> None:
        """Reject an invalid worker policy before any provider preclaim."""
        if not self.function_name:
            raise ValueError("Remote pull-worker function name cannot be empty")
        if self.claim_capacity < 1:
            raise ValueError("claim_capacity must be positive")
        if self.max_worker_calls < 1:
            raise ValueError("max_worker_calls must be positive")


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


@dataclass(frozen=True)
class RemoteWorkflowTask:
    """One independently scheduled remote Task discovered by a workflow Node."""

    task_key: str
    scientific_payload: Any
    execution_payload: Any = None

    def __post_init__(self) -> None:
        """Reject missing Task identity before the discovery transaction."""
        if not self.task_key:
            raise ValueError("Remote workflow Task key cannot be empty")


class RemoteTaskWorkflowNode:
    """Base class for workflow Nodes that discover finite remote Tasks."""

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        """Return the complete deterministic Task collection for this Node."""
        raise NotImplementedError

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        """Prepare one persisted Task for provider submission."""
        raise NotImplementedError

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[RemoteWorkflowTask, ...],
    ) -> RemoteNodeCall:
        """Prepare one provider call for a compatible fixed Task batch."""
        if len(tasks) != 1:
            raise ValueError("This workflow Node does not support Task batching")
        return self.prepare_remote_task(context, tasks[0])

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        """Normalize one durable provider result for Task publication."""
        return AppRunResult.model_validate(result)

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        """Decode one fixed-call result into every owned Task outcome."""
        if len(task_keys) != 1:
            raise ValueError("This workflow Node does not support Task batching")
        task_key = task_keys[0]
        return {task_key: self.process_remote_task_result(task_key, result, metadata)}

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[WorkflowArtifact, ...],
    ) -> AvailabilityStatus | None:
        """Optionally validate a workload-owned publication beyond its artifacts."""
        del context, task, expected_fingerprint, result, artifacts
        return None

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Build the Node publication after every discovered Task is terminal."""
        raise NotImplementedError

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Prevent bypassing the kernel's Task discovery and call ownership."""
        raise RuntimeError("Remote Task workflow Nodes must use the kernel")


class RemotePullTaskWorkflowNode(RemoteTaskWorkflowNode):
    """Remote Task Node whose calls dynamically claim bounded microbatches."""

    def prepare_pull_worker(
        self,
        context: NodeRunContext,
    ) -> RemotePullWorkerCall:
        """Prepare the immutable worker-pool binding for this Node."""
        raise NotImplementedError

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        """Prevent assigning pull Tasks during provider preclaim."""
        del context, task
        raise RuntimeError("Pull-worker Tasks are assigned through coordinator claims")


class AppBackedNode(RemoteWorkflowNode):
    """Semantic name for a remote Node implemented by a Biomodals app."""
