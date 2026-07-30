"""Immutable execution plans and lifecycle values."""

import json
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any


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


class AvailabilityStatus(StrEnum):
    """Authoritative observation of one workload publication."""

    AVAILABLE = "available"
    MISSING = "missing"
    UNKNOWN = "unknown"


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


def _canonical_json_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using the kernel's fixed encoding."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()
