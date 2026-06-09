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
    completed = {node_id for node_id in definition.nodes if node_is_complete(node_id)}
    if len(completed) == len(definition.nodes):
        return SchedulerDecision(
            status=SchedulerDecisionStatus.SUCCEEDED,
            completed=completed,
        )

    ready = [
        node_id
        for node_id, dependencies in definition.dependencies.items()
        if node_id not in completed
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
        if node_id not in completed
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
        if node_id not in completed and not dependencies.issubset(completed)
    ]
    return SchedulerDecision(
        status=SchedulerDecisionStatus.FAILED_NO_PROGRESS,
        completed=completed,
        blocked=blocked,
        warnings=["No runnable workflow nodes remain"],
    )


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
