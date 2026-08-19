"""Executable Node contracts shared by apps and workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import UUID

from biomodals.execution.model import AvailabilityStatus
from biomodals.schema import AppRunResult, ExecutionArtifact


@dataclass(frozen=True)
class NodeRunContext:
    """Workload-owned inputs and stable paths for one Execution Node Task."""

    execution_run_id: UUID
    workload_run_key: str
    node_id: str
    task_key: str
    work_dir: Path
    cache_dir: Path
    inputs: dict[str, list[ExecutionArtifact]]
    volume_root: Path | None = None
    artifact_volume_name: str | None = None

    def resolve_artifact(self, artifact: ExecutionArtifact) -> Path:
        """Resolve one execution-owned artifact without allowing traversal."""
        if self.volume_root is None or self.artifact_volume_name is None:
            raise RuntimeError("Execution artifact volume context is unavailable")
        if artifact.storage.volume_name != self.artifact_volume_name:
            raise ValueError(
                f"Artifact {artifact.artifact_id!r} is not stored in the "
                "execution artifact volume"
            )
        relative = PurePosixPath(artifact.storage.path)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("Execution artifact path must be relative and contained")
        path = self.volume_root.joinpath(*relative.parts).resolve()
        path.relative_to(self.volume_root.resolve())
        return path

    def single_input(self, name: str) -> ExecutionArtifact:
        """Return one required input artifact without repeated cardinality checks."""
        artifacts = self.inputs.get(name) or []
        if len(artifacts) != 1:
            raise ValueError(
                f"Execution Node requires exactly one {name!r} input artifact"
            )
        return artifacts[0]

    def read_input_bytes(self, name: str) -> bytes:
        """Read one execution-owned input artifact from its mounted Volume."""
        return self.resolve_artifact(self.single_input(name)).read_bytes()


@dataclass(frozen=True)
class ProviderCallSpec:
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
            raise ValueError("Provider operation name cannot be empty")
        if self.max_tasks_per_call < 1:
            raise ValueError("max_tasks_per_call must be positive")


@dataclass(frozen=True)
class PullWorkerCallSpec:
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


class ExecutionNode(Protocol):
    """Protocol for one semantic Execution Definition vertex."""

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute coordinator-local workload logic."""


class ResultNode:
    """Shared publication hooks for one-result Execution Nodes."""

    def refresh_artifact_storage_before_result(self) -> bool:
        """Return whether decoding reads provider-published artifact storage."""
        return False

    def refresh_result_storage(self) -> None:
        """Refresh workload-owned storage before decoding completed results."""

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        """Reconstruct an already-durable workload publication when present."""
        del context
        return None

    def commit_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        """Optionally commit and validate a workload-owned publication marker."""
        del context, result, artifacts
        return None

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        """Optionally validate a workload publication beyond its artifacts."""
        del context, result, artifacts
        return None


class CoordinatorNode(ResultNode):
    """Base class for coordinator-local Execution Nodes."""

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute coordinator-local workload logic."""
        raise NotImplementedError


class ProviderNode(ResultNode):
    """Base class for a Node executed by one tracked Provider Call."""

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
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
        raise RuntimeError("Provider Nodes must be submitted by the kernel")


@dataclass(frozen=True)
class TaskDefinition:
    """One independently scheduled Task discovered by an Execution Node."""

    task_key: str
    scientific_payload: Any
    execution_payload: Any = None

    def __post_init__(self) -> None:
        """Reject missing Task identity before the discovery transaction."""
        if not self.task_key:
            raise ValueError("Task key cannot be empty")


class TaskProviderNode:
    """Base class for Nodes that discover finite provider-backed Tasks."""

    def refresh_artifact_storage_before_result(self) -> bool:
        """Return whether decoding reads provider-published artifact storage."""
        return False

    def refresh_result_storage(self) -> None:
        """Refresh workload-owned storage before decoding completed results."""

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        """Return the complete deterministic Task collection for this Node."""
        raise NotImplementedError

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Prepare one persisted Task for provider submission."""
        raise NotImplementedError

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        """Prepare one provider call for a compatible fixed Task batch."""
        if len(tasks) != 1:
            raise ValueError("This Execution Node does not support Task batching")
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
            raise ValueError("This Execution Node does not support Task batching")
        task_key = task_keys[0]
        return {task_key: self.process_remote_task_result(task_key, result, metadata)}

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        """Optionally validate a workload-owned publication beyond its artifacts."""
        del context, task, expected_fingerprint, result, artifacts
        return None

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        """Reconstruct a result whose workload publication is already durable."""
        del context, task, expected_fingerprint
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
        raise RuntimeError("Task Provider Nodes must use the kernel")


class PullTaskProviderNode(TaskProviderNode):
    """Task Provider Node whose calls claim bounded microbatches."""

    def prepare_pull_worker(
        self,
        context: NodeRunContext,
    ) -> PullWorkerCallSpec:
        """Prepare the immutable worker-pool binding for this Node."""
        raise NotImplementedError

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Prevent assigning pull Tasks during provider preclaim."""
        del context, task
        raise RuntimeError("Pull-worker Tasks are assigned through coordinator claims")
