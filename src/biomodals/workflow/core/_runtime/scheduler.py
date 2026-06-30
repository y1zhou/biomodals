"""Read-only workflow scheduling and progress decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from biomodals.workflow.core.builder import WorkflowDefinition
from biomodals.workflow.core.ledger import WorkflowLedger


class SchedulerDecisionStatus(StrEnum):
    """Read-only workflow progress classification."""

    READY = "ready"
    SUCCEEDED = "succeeded"
    BLOCKED_RUNNING = "blocked_running"
    FAILED_NO_PROGRESS = "failed_no_progress"


@dataclass(frozen=True)
class SchedulerDecision:
    """Read-only scheduler decision for one workflow loop iteration."""

    status: SchedulerDecisionStatus
    completed: set[str]
    ready: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def evaluate_progress(
    definition: WorkflowDefinition,
    *,
    ledger: WorkflowLedger,
    node_is_complete: Callable[[str], bool],
) -> SchedulerDecision:
    """Evaluate workflow progress without mutating durable state."""
    terminals = _terminal_nodes(definition)
    completion_cache: dict[str, bool] = {}

    def is_complete(node_id: str) -> bool:
        if node_id not in completion_cache:
            completion_cache[node_id] = node_is_complete(node_id)
        return completion_cache[node_id]

    completed_terminals = {node_id for node_id in terminals if is_complete(node_id)}
    incomplete_terminals = terminals - completed_terminals
    if not incomplete_terminals:
        return SchedulerDecision(
            status=SchedulerDecisionStatus.SUCCEEDED,
            completed=completed_terminals,
        )

    active_nodes = _pruned_ancestor_closure(
        definition.dependencies,
        incomplete_terminals,
        is_complete,
    )
    completed = completed_terminals | {
        node_id for node_id in active_nodes if is_complete(node_id)
    }
    ready = [
        node_id
        for node_id, dependencies in definition.dependencies.items()
        if node_id in active_nodes
        and node_id not in completed
        and dependencies.issubset(completed)
        and _node_can_make_progress(node_id, ledger=ledger)
    ]
    if ready:
        return SchedulerDecision(
            status=SchedulerDecisionStatus.READY,
            completed=completed,
            ready=ready,
        )

    running = [
        node_id
        for node_id, dependencies in definition.dependencies.items()
        if node_id in active_nodes
        and node_id not in completed
        and dependencies.issubset(completed)
        and ledger.node_is_running(node_id)
    ]
    if running:
        return SchedulerDecision(
            status=SchedulerDecisionStatus.BLOCKED_RUNNING,
            completed=completed,
            running=running,
            warnings=[
                "Workflow has in-flight nodes without a recoverable remote call: "
                f"{', '.join(sorted(running))}"
            ],
        )

    blocked = [
        node_id
        for node_id, dependencies in definition.dependencies.items()
        if node_id in active_nodes
        and node_id not in completed
        and not dependencies.issubset(completed)
    ]
    return SchedulerDecision(
        status=SchedulerDecisionStatus.FAILED_NO_PROGRESS,
        completed=completed,
        blocked=blocked,
        warnings=["No runnable workflow nodes remain"],
    )


def _terminal_nodes(definition: WorkflowDefinition) -> set[str]:
    upstream_nodes = {
        dependency
        for dependencies in definition.dependencies.values()
        for dependency in dependencies
    }
    return set(definition.nodes) - upstream_nodes


def _pruned_ancestor_closure(
    dependencies_by_node: dict[str, set[str]],
    node_ids: set[str],
    is_complete: Callable[[str], bool],
) -> set[str]:
    active: set[str] = set()
    pending = list(node_ids)
    while pending:
        node_id = pending.pop()
        if node_id in active:
            continue
        active.add(node_id)
        if is_complete(node_id):
            continue
        pending.extend(dependencies_by_node.get(node_id, set()))
    return active


def _node_can_make_progress(node_id: str, *, ledger: WorkflowLedger) -> bool:
    if not ledger.node_is_running(node_id):
        return True
    return (
        ledger.latest_remote_call(
            node_id,
            statuses=("submitted", "running", "succeeded"),
        )
        is not None
    )
