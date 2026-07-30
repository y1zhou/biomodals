"""Pure DAG readiness and admission decisions."""

from collections.abc import Mapping

from biomodals.execution.model import (
    AvailabilityStatus,
    ExecutionPlan,
    NodeAggregationPolicy,
    NodeStatus,
    RunStatus,
    TaskStatus,
)


def ready_node_keys(
    plan: ExecutionPlan,
    node_statuses: Mapping[str, NodeStatus],
) -> tuple[str, ...]:
    """Return pending Nodes whose dependencies have accepted outcomes."""
    ready: list[str] = []
    for node in plan.nodes:
        if node_statuses[node.node_key] != NodeStatus.PENDING:
            continue
        if all(
            node_statuses[dependency.node_key] == NodeStatus.SUCCEEDED
            or (
                dependency.accept_partial
                and node_statuses[dependency.node_key] == NodeStatus.PARTIAL
            )
            for dependency in node.dependencies
        ):
            ready.append(node.node_key)
    return tuple(ready)


def propagated_skip_node_keys(
    plan: ExecutionPlan,
    node_statuses: Mapping[str, NodeStatus],
) -> tuple[str, ...]:
    """Return pending Nodes made unreachable by terminal dependency outcomes."""
    skipped: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in plan.nodes:
            if node_statuses[node.node_key] != NodeStatus.PENDING:
                continue
            for dependency in node.dependencies:
                status = (
                    NodeStatus.SKIPPED
                    if dependency.node_key in skipped
                    else node_statuses[dependency.node_key]
                )
                accepted = status == NodeStatus.SUCCEEDED or (
                    dependency.accept_partial and status == NodeStatus.PARTIAL
                )
                if status.is_terminal and not accepted:
                    if node.node_key not in skipped:
                        skipped.add(node.node_key)
                        changed = True
                    break
    return tuple(node.node_key for node in plan.nodes if node.node_key in skipped)


def required_node_keys(
    plan: ExecutionPlan,
    observations: Mapping[str, AvailabilityStatus],
) -> tuple[str, ...] | None:
    """Return missing result work, or ``None`` when validation is unknown."""
    nodes = {node.node_key: node for node in plan.nodes}
    required: set[str] = set()

    def visit(node_key: str) -> bool:
        observation = observations[node_key]
        if observation == AvailabilityStatus.UNKNOWN:
            return False
        if observation == AvailabilityStatus.AVAILABLE:
            return True
        required.add(node_key)
        return all(
            visit(dependency.node_key) for dependency in nodes[node_key].dependencies
        )

    if not all(visit(node_key) for node_key in plan.terminal_node_keys):
        return None
    return tuple(node.node_key for node in plan.nodes if node.node_key in required)


def terminal_run_outcome(
    plan: ExecutionPlan,
    node_statuses: Mapping[str, NodeStatus],
) -> RunStatus | None:
    """Return the strict terminal-boundary Run outcome when conclusive."""
    statuses = tuple(node_statuses[key] for key in plan.terminal_node_keys)
    if not all(status.is_terminal for status in statuses):
        return None
    if NodeStatus.CANCELLED in statuses:
        return RunStatus.CANCELLED
    if NodeStatus.FAILED in statuses or NodeStatus.SKIPPED in statuses:
        return RunStatus.FAILED
    if NodeStatus.PARTIAL in statuses:
        return RunStatus.PARTIAL
    return RunStatus.SUCCEEDED


def aggregate_task_outcome(
    policy: NodeAggregationPolicy,
    task_statuses: tuple[TaskStatus, ...],
) -> NodeStatus | None:
    """Return a Node outcome once every discovered Task is terminal."""
    if not task_statuses or not all(status.is_terminal for status in task_statuses):
        return None
    if TaskStatus.CANCELLED in task_statuses:
        return NodeStatus.CANCELLED
    succeeded = task_statuses.count(TaskStatus.SUCCEEDED)
    if succeeded == len(task_statuses):
        return NodeStatus.SUCCEEDED
    if policy == NodeAggregationPolicy.ALLOW_PARTIAL and succeeded:
        return NodeStatus.PARTIAL
    return NodeStatus.FAILED
