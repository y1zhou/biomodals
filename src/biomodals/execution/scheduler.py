"""Pure DAG readiness and admission decisions."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from biomodals.execution.model import (
    AvailabilityStatus,
    ExecutionPlan,
    NodeAggregationPolicy,
    NodeStatus,
    ProviderBinding,
    RunStatus,
    TaskStatus,
)


@dataclass(frozen=True)
class NodeAdmissionRank:
    """Graph-critical ordering values for one required Node."""

    depth: int
    unblocking_span: int


@dataclass(frozen=True)
class TaskDispatchDescriptor:
    """Operational fixed-batch inputs for one ready Task."""

    node_key: str
    node_ordinal: int
    task_key: str
    task_ordinal: int
    binding: ProviderBinding
    compatibility_key: str
    max_tasks_per_call: int
    depth: int
    unblocking_span: int


@dataclass(frozen=True)
class PullWorkerDispatchDescriptor:
    """Operational inputs for one Node's derived pull-worker pool."""

    node_key: str
    node_ordinal: int
    binding: ProviderBinding
    compatibility_key: str
    claim_capacity: int
    unfinished_task_count: int
    nonterminal_worker_count: int
    next_worker_ordinal: int
    depth: int
    unblocking_span: int


@dataclass(frozen=True)
class ProviderCallCandidate:
    """One side-effect-free candidate for Provider Call admission."""

    candidate_key: str
    node_key: str
    node_ordinal: int
    task_keys: tuple[str, ...]
    task_ordinal: int
    binding: ProviderBinding
    compatibility_key: str
    depth: int
    unblocking_span: int


def required_node_ranks(
    plan: ExecutionPlan,
    *,
    required_node_keys: set[str],
    unfinished_node_keys: set[str],
) -> dict[str, NodeAdmissionRank]:
    """Calculate depth and unfinished descendant span in one required closure."""
    nodes = {node.node_key: node for node in plan.nodes}
    depths: dict[str, int] = {}

    def depth(node_key: str) -> int:
        if node_key in depths:
            return depths[node_key]
        node = nodes[node_key]
        dependency_depths = [
            depth(dependency.node_key)
            for dependency in node.dependencies
            if dependency.node_key in required_node_keys
        ]
        value = max(dependency_depths) + 1 if dependency_depths else 0
        depths[node_key] = value
        return value

    for node_key in required_node_keys:
        depth(node_key)

    dependents: dict[str, list[str]] = defaultdict(list)
    for node in plan.nodes:
        if node.node_key not in required_node_keys:
            continue
        for dependency in node.dependencies:
            if dependency.node_key in required_node_keys:
                dependents[dependency.node_key].append(node.node_key)

    def unfinished_descendants(node_key: str) -> set[str]:
        found: set[str] = set()
        remaining = list(dependents[node_key])
        while remaining:
            descendant = remaining.pop()
            if descendant in found:
                continue
            found.add(descendant)
            remaining.extend(dependents[descendant])
        return found & unfinished_node_keys

    return {
        node.node_key: NodeAdmissionRank(
            depth=depths[node.node_key],
            unblocking_span=len(unfinished_descendants(node.node_key)),
        )
        for node in plan.nodes
        if node.node_key in required_node_keys
    }


def form_fixed_batches(
    tasks: tuple[TaskDispatchDescriptor, ...],
) -> tuple[ProviderCallCandidate, ...]:
    """Group compatible Tasks into immutable bounded fixed-call candidates."""
    grouped: dict[
        tuple[str, ProviderBinding, str, int],
        list[TaskDispatchDescriptor],
    ] = {}
    for task in tasks:
        if task.max_tasks_per_call <= 0:
            raise ValueError("max_tasks_per_call must be positive")
        group_key = (
            task.node_key,
            task.binding,
            task.compatibility_key,
            task.max_tasks_per_call,
        )
        grouped.setdefault(group_key, []).append(task)

    candidates: list[ProviderCallCandidate] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda task: task.task_ordinal)
        batch_size = ordered[0].max_tasks_per_call
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset : offset + batch_size]
            first = batch[0]
            candidates.append(
                ProviderCallCandidate(
                    candidate_key=(
                        f"{first.node_key}:{first.binding.function_name}:"
                        f"{first.compatibility_key}:{first.task_ordinal}"
                    ),
                    node_key=first.node_key,
                    node_ordinal=first.node_ordinal,
                    task_keys=tuple(task.task_key for task in batch),
                    task_ordinal=first.task_ordinal,
                    binding=first.binding,
                    compatibility_key=first.compatibility_key,
                    depth=first.depth,
                    unblocking_span=first.unblocking_span,
                )
            )
    return tuple(candidates)


def form_pull_worker_candidates(
    descriptors: Iterable[PullWorkerDispatchDescriptor],
) -> tuple[ProviderCallCandidate, ...]:
    """Derive every currently useful pull-worker call candidate."""
    candidates: list[ProviderCallCandidate] = []
    for descriptor in descriptors:
        if descriptor.claim_capacity <= 0:
            raise ValueError("claim_capacity must be positive")
        if descriptor.unfinished_task_count < 0:
            raise ValueError("unfinished_task_count cannot be negative")
        if descriptor.nonterminal_worker_count < 0:
            raise ValueError("nonterminal_worker_count cannot be negative")
        if descriptor.next_worker_ordinal < 0:
            raise ValueError("next_worker_ordinal cannot be negative")
        desired_workers = (
            descriptor.unfinished_task_count + descriptor.claim_capacity - 1
        ) // descriptor.claim_capacity
        candidate_count = max(
            0,
            desired_workers - descriptor.nonterminal_worker_count,
        )
        for offset in range(candidate_count):
            worker_ordinal = descriptor.next_worker_ordinal + offset
            candidates.append(
                ProviderCallCandidate(
                    candidate_key=(
                        f"{descriptor.node_key}:{descriptor.binding.function_name}:"
                        f"{descriptor.compatibility_key}:worker-{worker_ordinal}"
                    ),
                    node_key=descriptor.node_key,
                    node_ordinal=descriptor.node_ordinal,
                    task_keys=(),
                    task_ordinal=worker_ordinal,
                    binding=descriptor.binding,
                    compatibility_key=descriptor.compatibility_key,
                    depth=descriptor.depth,
                    unblocking_span=descriptor.unblocking_span,
                )
            )
    return tuple(candidates)


def select_admissible_candidates(
    candidates: Iterable[ProviderCallCandidate],
    *,
    available_total_slots: int,
    available_gpu_slots: int,
) -> tuple[ProviderCallCandidate, ...]:
    """Greedily fill feasible call slots in deterministic priority order."""
    if available_total_slots < 0 or available_gpu_slots < 0:
        raise ValueError("available Provider Call slots cannot be negative")
    ordered = _order_admission_candidates(tuple(candidates))
    selected: list[ProviderCallCandidate] = []
    gpu_slots = available_gpu_slots
    for candidate in ordered:
        if len(selected) >= available_total_slots:
            break
        if candidate.binding.uses_gpu:
            if gpu_slots == 0:
                continue
            gpu_slots -= 1
        selected.append(candidate)
    return tuple(selected)


def _order_admission_candidates(
    candidates: tuple[ProviderCallCandidate, ...],
) -> tuple[ProviderCallCandidate, ...]:
    graph_bands: dict[tuple[int, int], list[ProviderCallCandidate]] = defaultdict(list)
    for candidate in candidates:
        graph_bands[(candidate.depth, candidate.unblocking_span)].append(candidate)

    ordered: list[ProviderCallCandidate] = []
    for graph_rank in sorted(graph_bands, reverse=True):
        encounter_order = sorted(
            graph_bands[graph_rank],
            key=lambda candidate: (
                candidate.node_ordinal,
                candidate.task_ordinal,
            ),
        )
        for uses_gpu in (True, False):
            resource_class = [
                candidate
                for candidate in encounter_order
                if candidate.binding.uses_gpu == uses_gpu
            ]
            cohorts: dict[
                tuple[str, ...],
                list[ProviderCallCandidate],
            ] = {}
            for candidate in resource_class:
                cohorts.setdefault(_image_cohort_key(candidate.binding), []).append(
                    candidate
                )
            for cohort in cohorts.values():
                ordered.extend(cohort)
    return tuple(ordered)


def _image_cohort_key(binding: ProviderBinding) -> tuple[str, ...]:
    if binding.runtime_image_key is not None:
        return ("image", binding.runtime_image_key)
    return (
        "binding",
        binding.environment,
        binding.app_name,
        str(binding.app_version),
        binding.function_name,
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
