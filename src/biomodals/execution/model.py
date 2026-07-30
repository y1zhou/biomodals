"""Immutable execution plans and lifecycle values."""

import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any
from uuid import UUID

import orjson


class RunStatus(StrEnum):
    """Lifecycle status of one Execution Run."""

    PENDING = "pending"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUSPENDED = "suspended"
    STATE_UNKNOWN = "state_unknown"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return whether the Run can no longer advance."""
        return self in {
            RunStatus.SUCCEEDED,
            RunStatus.PARTIAL,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }


class RunStatusReason(StrEnum):
    """Closed machine-readable reasons for selected Run statuses."""

    COORDINATOR_ERROR = "coordinator_error"
    RESULT_VALIDATION_UNKNOWN = "result_validation_unknown"
    SUBMISSION_OUTCOME_UNKNOWN = "submission_outcome_unknown"
    PROVIDER_OUTCOME_UNKNOWN = "provider_outcome_unknown"
    CANCELLATION_OUTCOME_UNKNOWN = "cancellation_outcome_unknown"
    REQUIRED_WORK_FAILED = "required_work_failed"
    DEPLOYMENT_UNAVAILABLE = "deployment_unavailable"


class NodeStatus(StrEnum):
    """Lifecycle status of one Execution Node."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        """Return whether the Node can no longer advance."""
        return self in {
            NodeStatus.SUCCEEDED,
            NodeStatus.PARTIAL,
            NodeStatus.FAILED,
            NodeStatus.CANCELLED,
            NodeStatus.SKIPPED,
        }


class TaskStatus(StrEnum):
    """Lifecycle status of one Task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        """Return whether the Task can no longer advance."""
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.SKIPPED,
        }


class ProviderCallStatus(StrEnum):
    """Lifecycle status of one remote Provider Call."""

    SUBMITTING = "submitting"
    ATTACHED = "attached"
    RUNNING = "running"
    OUTCOME_UNKNOWN = "outcome_unknown"
    STATE_UNKNOWN = "state_unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return whether the Provider Call can no longer advance."""
        return self in {
            ProviderCallStatus.SUCCEEDED,
            ProviderCallStatus.FAILED,
            ProviderCallStatus.CANCELLED,
        }


class DispatchMode(StrEnum):
    """Supported remote Task-to-call assignment modes."""

    FIXED_BATCH = "fixed_batch"
    PULL_WORKER = "pull_worker"


@dataclass(frozen=True)
class ProviderBinding:
    """Resolved Modal function identity and operational scheduling metadata."""

    environment: str
    app_name: str
    app_version: int
    function_name: str
    uses_gpu: bool
    runtime_image_key: str | None = None

    def __post_init__(self) -> None:
        """Require an exact deployed function identity."""
        if not self.environment:
            raise ValueError("provider environment cannot be empty")
        if not self.app_name:
            raise ValueError("provider app name cannot be empty")
        if self.app_version < 1:
            raise ValueError("provider app version must be positive")
        if not self.function_name:
            raise ValueError("provider function name cannot be empty")


class AvailabilityStatus(StrEnum):
    """Authoritative observation of one workload publication."""

    AVAILABLE = "available"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ResultProvenance(StrEnum):
    """How one validated scientific result satisfied execution work."""

    CACHE = "cache"
    CURRENT_RUN = "current_run"


class WorkStatusReason(StrEnum):
    """Closed reason shared by pruned Node and Task terminal states."""

    RESULT_ALREADY_SATISFIED = "result_already_satisfied"


@dataclass(frozen=True)
class DeploymentIdentity:
    """Exact deployed coordinator location fixed for one Execution Run."""

    environment: str
    deployment_name: str
    deployment_version: int

    def __post_init__(self) -> None:
        """Reject incomplete or non-versioned deployment locations."""
        if not self.environment:
            raise ValueError("deployment environment cannot be empty")
        if not self.deployment_name:
            raise ValueError("deployment name cannot be empty")
        if self.deployment_version < 1:
            raise ValueError("deployment version must be positive")


@dataclass(frozen=True)
class NodeDependency:
    """One immutable dependency edge between Execution Nodes."""

    node_key: str
    accept_partial: bool = False


class NodeAggregationPolicy(StrEnum):
    """How one Node aggregates its terminal Task outcomes."""

    FAIL_FAST = "fail_fast"
    COLLECT_ALL = "collect_all"
    ALLOW_PARTIAL = "allow_partial"


@dataclass(frozen=True)
class NodePlan:
    """One fixed semantic stage in an Execution Plan."""

    node_key: str
    dependencies: tuple[NodeDependency, ...] = ()
    aggregation_policy: NodeAggregationPolicy = NodeAggregationPolicy.COLLECT_ALL
    allow_empty_result: bool = False


@dataclass(frozen=True)
class TaskPlan:
    """One independently cacheable and verifiable work item."""

    task_key: str
    scientific_payload: Any
    execution_payload: Any = None

    def fingerprint(
        self,
        *,
        workload_plan_fingerprint: str,
        node_key: str,
    ) -> str:
        """Return this Task's stable scientific fingerprint."""
        return _canonical_json_sha256({
            "node": node_key,
            "plan": workload_plan_fingerprint,
            "science": self.scientific_payload,
            "task": self.task_key,
        })


@dataclass(frozen=True)
class ExecutionPlan:
    """One immutable workload DAG supplied to the execution kernel."""

    workload_name: str
    nodes: tuple[NodePlan, ...]
    workload_run_key: str | None = None
    scientific_payload: Any = None
    scientific_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate caller-supplied Node identity."""
        seen: set[str] = set()
        for node in self.nodes:
            if node.node_key in seen:
                raise ValueError(f"duplicate Node key {node.node_key!r}")
            seen.add(node.node_key)
        for node in self.nodes:
            for dependency in node.dependencies:
                if dependency.node_key not in seen:
                    raise ValueError(
                        f"Node {node.node_key!r} depends on unknown Node "
                        f"{dependency.node_key!r}"
                    )
        indegree = {node.node_key: len(node.dependencies) for node in self.nodes}
        dependents = {node.node_key: [] for node in self.nodes}
        for node in self.nodes:
            for dependency in node.dependencies:
                dependents[dependency.node_key].append(node.node_key)
        ready = deque(key for key in self.node_keys if indegree[key] == 0)
        visited = 0
        while ready:
            node_key = ready.popleft()
            visited += 1
            for dependent_key in dependents[node_key]:
                indegree[dependent_key] -= 1
                if indegree[dependent_key] == 0:
                    ready.append(dependent_key)
        if visited != len(self.nodes):
            raise ValueError("execution plan contains a cycle")

    @property
    def node_keys(self) -> tuple[str, ...]:
        """Return Node keys in caller-supplied encounter order."""
        return tuple(node.node_key for node in self.nodes)

    @property
    def terminal_node_keys(self) -> tuple[str, ...]:
        """Return Nodes with no downstream dependency in encounter order."""
        dependency_keys = {
            dependency.node_key
            for node in self.nodes
            for dependency in node.dependencies
        }
        return tuple(
            node.node_key for node in self.nodes if node.node_key not in dependency_keys
        )

    @property
    def workload_plan_fingerprint(self) -> str:
        """Return the stable fingerprint of result-affecting plan content."""
        value = {
            "nodes": [
                {
                    "aggregation_policy": node.aggregation_policy,
                    "allow_empty_result": node.allow_empty_result,
                    "dependencies": [
                        {
                            "accept_partial": dependency.accept_partial,
                            "node_key": dependency.node_key,
                        }
                        for dependency in node.dependencies
                    ],
                    "node_key": node.node_key,
                }
                for node in self.nodes
            ],
            "scientific_payload": self.scientific_payload,
            "scientific_versions": self.scientific_versions,
            "workload_name": self.workload_name,
        }
        return _canonical_json_sha256(value)


@dataclass(frozen=True)
class ExecutionRunRecord:
    """Durable state and immutable identity for one Execution Run."""

    execution_run_id: UUID
    predecessor_execution_run_id: UUID | None
    plan: ExecutionPlan
    deployment: DeploymentIdentity
    status: RunStatus
    status_reason: RunStatusReason | None
    status_message: str | None
    max_active_provider_calls: int
    max_active_gpu_provider_calls: int
    created_at: int
    updated_at: int
    started_at: int | None
    completed_at: int | None


@dataclass(frozen=True)
class ExecutionNodeRecord:
    """Durable state for one fixed Node in an Execution Run."""

    execution_run_id: UUID
    node_key: str
    ordinal: int
    dependencies: tuple[NodeDependency, ...]
    aggregation_policy: NodeAggregationPolicy
    allow_empty_result: bool
    status: NodeStatus
    status_reason: WorkStatusReason | None
    discovery_complete: bool
    result_observation: AvailabilityStatus | None
    result_observed_at: int | None
    result_provenance: ResultProvenance | None
    error_message: str | None
    created_at: int
    updated_at: int
    started_at: int | None
    completed_at: int | None


@dataclass(frozen=True)
class ExecutionTaskRecord:
    """Durable identity, payload, and lifecycle for one discovered Task."""

    execution_run_id: UUID
    node_key: str
    task_key: str
    ordinal: int
    fingerprint: str
    scientific_payload: Any
    execution_payload: Any
    status: TaskStatus
    status_reason: WorkStatusReason | None
    result_observation: AvailabilityStatus | None
    result_observed_at: int | None
    result_provenance: ResultProvenance | None
    provider_call_id: UUID | None
    worker_provider_call_id: UUID | None
    local_owned: bool
    error_message: str | None
    created_at: int
    updated_at: int
    started_at: int | None
    completed_at: int | None


@dataclass(frozen=True)
class ProviderCallRecord:
    """One durable concrete Modal function invocation."""

    provider_call_id: UUID
    execution_run_id: UUID
    node_key: str
    dispatch_batch_id: UUID
    dispatch_mode: DispatchMode
    submission_token: str
    binding: ProviderBinding
    status: ProviderCallStatus
    provider_call_handle_id: str | None
    result_envelope: Any
    error_message: str | None
    task_keys: tuple[str, ...]
    created_at: int
    updated_at: int
    attached_at: int | None
    started_at: int | None
    completed_at: int | None


@dataclass(frozen=True)
class ProviderCallPreclaim:
    """A durable call plus one-time in-process permission to spawn it."""

    call: ProviderCallRecord
    spawn_authorized: bool


@dataclass(frozen=True)
class ActiveProviderCallCounts:
    """Derived total and GPU-subset nonterminal Provider Call counts."""

    total: int
    gpu: int


@dataclass(frozen=True)
class WorkerAssignmentRecord:
    """One checkpointed pull-worker ownership decision."""

    execution_run_id: UUID
    node_key: str
    task_key: str
    task_fingerprint: str
    execution_payload: Any
    provider_call_id: UUID
    request_id: str
    ordinal: int
    created_at: int


@dataclass(frozen=True)
class PullTaskClaim:
    """Idempotent response to one bounded pull-worker claim request."""

    request_id: str
    provider_call_id: UUID
    assignments: tuple[WorkerAssignmentRecord, ...]


@dataclass(frozen=True)
class ExecutionSnapshot:
    """One read-only execution view for adapters and diagnostics."""

    run: ExecutionRunRecord
    nodes: tuple[ExecutionNodeRecord, ...]
    tasks: tuple[ExecutionTaskRecord, ...]
    provider_calls: tuple[ProviderCallRecord, ...]
    active_provider_calls: ActiveProviderCallCounts


def _canonical_json_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using the specified fixed encoding."""
    _validate_json_value(value)
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode operational persistence JSON with deterministic key order."""
    _validate_json_value(value)
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Out of range float values are not JSON compliant")
        return
    if isinstance(value, list | tuple):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _validate_json_value(item)
        return
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
