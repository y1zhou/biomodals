"""Caller-driven composition facade for durable execution mechanics."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from biomodals.execution.model import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionNodeRecord,
    ExecutionPlan,
    ExecutionRunRecord,
    ExecutionTaskRecord,
    NodeStatus,
    ProviderBinding,
    ProviderCallPreclaim,
    ProviderCallRecord,
    ProviderCallStatus,
    PullTaskClaim,
    RunStatus,
    RunStatusReason,
    TaskPlan,
    TaskStatus,
)
from biomodals.execution.provider import (
    ProviderCallObservation,
    ProviderCallObservationKind,
    ProviderDefiniteSubmissionError,
    ProviderDeploymentUnavailableError,
    ProviderDriver,
    ProviderSubmissionOutcomeUnknownError,
)
from biomodals.execution.scheduler import (
    NodeAdmissionRank,
    ProviderCallCandidate,
    PullWorkerDispatchDescriptor,
    TaskDispatchDescriptor,
    form_fixed_batches,
    ready_node_keys,
    required_node_keys,
    required_node_ranks,
    result_probe_frontier,
    select_admissible_candidates,
)
from biomodals.execution.sqlite import SqliteExecutionRepository

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderCallSubmission:
    """One already-selected call candidate and its provider invocation."""

    candidate: ProviderCallCandidate
    submission_token: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    provider_call_id_kwarg: str | None = None
    claim_capacity: int | None = None
    function: Any | None = None


def _required_node_keys_for_run(
    repository: SqliteExecutionRepository,
    execution_run_id: UUID,
) -> tuple[str, ...] | None:
    """Derive the result-driven closure from durable Node observations."""
    observations = {
        node.node_key: (
            AvailabilityStatus.AVAILABLE
            if node.status == NodeStatus.SUCCEEDED
            else node.result_observation or AvailabilityStatus.MISSING
        )
        for node in repository.list_nodes(execution_run_id)
    }
    return required_node_keys(
        repository.get_run(execution_run_id).plan,
        observations,
    )


def _record_provider_call_observation(
    repository: SqliteExecutionRepository,
    provider_call_id: UUID,
    observation: ProviderCallObservation,
    *,
    result_envelope: Any,
    result_already_satisfied: bool,
    now: int,
) -> ProviderCallRecord:
    """Apply the provider-neutral transition shared by sync and async hosts."""
    if observation.kind == ProviderCallObservationKind.RUNNING:
        return repository.mark_provider_call_running(provider_call_id, now=now)
    if observation.kind == ProviderCallObservationKind.SUCCEEDED:
        return repository.record_provider_call_result(
            provider_call_id,
            result_envelope=result_envelope,
            now=now,
        )
    if observation.kind == ProviderCallObservationKind.FAILED:
        return repository.fail_provider_call(
            provider_call_id,
            message=observation.message or "Provider operation failed",
            now=now,
        )
    if observation.kind == ProviderCallObservationKind.CANCELLED:
        if result_already_satisfied:
            return repository.cancel_pruned_provider_call(
                provider_call_id,
                now=now,
            )
        return repository.cancel_provider_call(
            provider_call_id,
            message=observation.message or "Provider operation was cancelled",
            now=now,
        )
    return repository.mark_provider_call_state_unknown(
        provider_call_id,
        message=observation.message or "Provider Call state was inconclusive",
        now=now,
    )


class ExecutionRuntime:
    """Coordinate repository checkpoints with one provider side effect."""

    def __init__(
        self,
        repository: SqliteExecutionRepository,
        *,
        provider_driver: ProviderDriver,
        checkpoint: Callable[[], SqliteExecutionRepository | None],
        transaction: Callable[[], AbstractContextManager[object]] = nullcontext,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> None:
        """Bind host-owned state, provider operations, and durability."""
        self.repository = repository
        self._driver = provider_driver
        self._checkpoint = checkpoint
        self._transaction = transaction
        self._synchronize = synchronize

    def configure_provider_driver(self, provider_driver: ProviderDriver) -> None:
        """Replace the provider adapter before further call lifecycle work."""
        self._driver = provider_driver

    def create_or_verify_run(
        self,
        *,
        execution_run_id: UUID,
        predecessor_execution_run_id: UUID | None,
        plan: ExecutionPlan,
        deployment: DeploymentIdentity,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
        now: int,
    ) -> ExecutionRunRecord:
        """Create one Run or verify that its immutable identity still matches."""
        with self._synchronize():
            try:
                run = self.repository.get_run(execution_run_id)
            except LookupError:
                with self._transaction():
                    return self.repository.create_run(
                        execution_run_id=execution_run_id,
                        predecessor_execution_run_id=predecessor_execution_run_id,
                        plan=plan,
                        deployment=deployment,
                        max_active_provider_calls=max_active_provider_calls,
                        max_active_gpu_provider_calls=max_active_gpu_provider_calls,
                        now=now,
                    )
            if (
                run.predecessor_execution_run_id != predecessor_execution_run_id
                or run.plan != plan
                or run.deployment != deployment
                or run.max_active_provider_calls != max_active_provider_calls
                or run.max_active_gpu_provider_calls != max_active_gpu_provider_calls
            ):
                raise ValueError(
                    "Execution Run initialization does not match persisted state"
                )
            return run

    def required_node_keys(self, execution_run_id: UUID) -> tuple[str, ...] | None:
        """Derive the result-driven closure from durable Node observations."""
        with self._synchronize():
            return _required_node_keys_for_run(self.repository, execution_run_id)

    def advance_once(
        self,
        execution_run_id: UUID,
        *,
        recover_publications: Callable[[], None],
        reconcile_provider_calls: Callable[[set[str]], None],
        decode_completed_calls: Callable[[], None],
        start_ready_nodes: Callable[[set[str]], None],
        admit_remote_tasks: Callable[[set[str]], None],
        after_start_ready_nodes: Callable[[], bool] | None = None,
        reconcile_results: Callable[[], None] | None = None,
        now: Callable[[], int],
    ) -> None:
        """Apply one result-driven reconciliation and admission cycle."""
        reconcile = reconcile_results or (
            lambda: self.reconcile_nodes_and_run(execution_run_id, now=now())
        )

        def result_validation_is_suspended() -> bool:
            with self._synchronize():
                current = self.repository.get_run(execution_run_id)
            return (
                current.status == RunStatus.SUSPENDED
                and current.status_reason == RunStatusReason.RESULT_VALIDATION_UNKNOWN
            )

        def recover_publications_unless_cancelled() -> None:
            with self._synchronize():
                cancellation_is_durable = self.repository.get_run(
                    execution_run_id
                ).cancellation_is_durable
            if not cancellation_is_durable:
                recover_publications()

        recover_publications_unless_cancelled()
        reconcile()
        with self._synchronize():
            run = self.repository.get_run(execution_run_id)
        if run.status == RunStatus.CANCEL_REQUESTED:
            required_nodes = set(run.plan.node_keys)
        elif run.status == RunStatus.STATE_UNKNOWN:
            required = self.required_node_keys(execution_run_id)
            required_nodes = set(run.plan.node_keys if required is None else required)
            if required is not None:
                self.prune_unrequired_nodes(
                    execution_run_id,
                    required_node_keys=required,
                    now=now(),
                )
        elif run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
            required = self.required_node_keys(execution_run_id)
            if required is None:
                return
            required_nodes = set(required)
            self.prune_unrequired_nodes(
                execution_run_id,
                required_node_keys=required,
                now=now(),
            )
        else:
            return

        reconcile_provider_calls(required_nodes)
        decode_completed_calls()
        if result_validation_is_suspended():
            return
        recover_publications_unless_cancelled()
        reconcile()
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        with self._synchronize():
            can_continue = self.repository.get_run(execution_run_id).status in {
                RunStatus.PENDING,
                RunStatus.RUNNING,
            }
        if not can_continue:
            return
        start_ready_nodes(required_nodes)
        if after_start_ready_nodes is not None:
            while after_start_ready_nodes():
                recover_publications_unless_cancelled()
                reconcile()
                with self._synchronize():
                    can_continue = self.repository.get_run(execution_run_id).status in {
                        RunStatus.PENDING,
                        RunStatus.RUNNING,
                    }
                if not can_continue:
                    return
                required = self.required_node_keys(execution_run_id)
                if required is None:
                    return
                start_ready_nodes(set(required))
        recover_publications_unless_cancelled()
        required = self.required_node_keys(execution_run_id)
        if required is not None:
            admit_remote_tasks(set(required))
        reconcile()

    def recover_publications(
        self,
        execution_run_id: UUID,
        *,
        observe_node: Callable[[str], AvailabilityStatus],
        observe_task: Callable[[str, ExecutionTaskRecord], AvailabilityStatus | None],
        now: int,
    ) -> tuple[str, ...] | None:
        """Walk backward from results and record caller-validated publications."""
        with self._synchronize():
            repository = self.repository
            run = repository.get_run(execution_run_id)
            nodes = repository.list_nodes(execution_run_id)
        observations = {
            node.node_key: (
                AvailabilityStatus.AVAILABLE
                if node.status == NodeStatus.SUCCEEDED
                else (
                    AvailabilityStatus.MISSING
                    if node.status.is_terminal
                    else node.result_observation
                )
            )
            for node in nodes
        }
        while frontier := result_probe_frontier(run.plan, observations):
            observed = tuple(
                (node_key, observe_node(node_key)) for node_key in frontier
            )
            with self._transaction():
                repository = self.repository
                for node_key, observation in observed:
                    current = repository.get_node(
                        execution_run_id,
                        node_key,
                    )
                    if current.status.is_terminal:
                        observations[node_key] = (
                            AvailabilityStatus.AVAILABLE
                            if current.status == NodeStatus.SUCCEEDED
                            else AvailabilityStatus.MISSING
                        )
                        continue
                    repository.record_node_result_observation(
                        execution_run_id,
                        node_key,
                        observation,
                        now=now,
                    )
                    observations[node_key] = observation
            if any(
                observation == AvailabilityStatus.UNKNOWN for _, observation in observed
            ):
                return None

        required = self.required_node_keys(execution_run_id)
        if required is None:
            return None
        required_set = set(required)
        with self._synchronize():
            repository = self.repository
            nodes = repository.list_nodes(execution_run_id)
            pending_tasks = tuple(
                (node.node_key, task)
                for node in nodes
                if (
                    node.node_key in required_set
                    and node.status == NodeStatus.RUNNING
                    and node.discovery_complete
                )
                for task in repository.list_tasks_requiring_publication_recovery(
                    execution_run_id,
                    node.node_key,
                )
            )
        task_observations = []
        for node_key, task in pending_tasks:
            observation = observe_task(node_key, task)
            if observation is not None:
                task_observations.append((node_key, task.task_key, observation))
        if task_observations:
            with self._transaction():
                repository = self.repository
                for node_key, task_key, observation in task_observations:
                    if repository.get_task(
                        execution_run_id,
                        node_key,
                        task_key,
                    ).status.is_terminal:
                        continue
                    repository.record_task_result_observation(
                        execution_run_id,
                        node_key,
                        task_key,
                        observation,
                        now=now,
                    )
        return required

    def prune_unrequired_nodes(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: Collection[str],
        now: int,
    ) -> tuple[UUID, ...]:
        """Prune result-irrelevant work and cancel any attached owners."""
        with self._synchronize():
            with self._transaction():
                provider_call_ids = self.repository.prune_unrequired_nodes(
                    execution_run_id,
                    required_node_keys=set(required_node_keys),
                    now=now,
                )
            if provider_call_ids:
                self._checkpoint_state()
        for provider_call_id in provider_call_ids:
            self.request_provider_call_cancellation(
                provider_call_id, now=now, result_already_satisfied=True
            )
        return provider_call_ids

    def decode_completed_calls(
        self,
        execution_run_id: UUID,
        *,
        observe_task: Callable[[str, ExecutionTaskRecord, Any], AvailabilityStatus],
        missing_message: str,
        now: int,
    ) -> None:
        """Validate durable successful-call envelopes and finish their Tasks."""
        with self._synchronize():
            repository = self.repository
            completed = tuple(
                (call, task)
                for call in repository.list_provider_calls_requiring_reconciliation(
                    execution_run_id
                )
                if call.status == ProviderCallStatus.SUCCEEDED
                for task_key in call.task_keys
                for task in (
                    repository.get_task(
                        execution_run_id,
                        call.node_key,
                        task_key,
                    ),
                )
                if task.status == TaskStatus.RUNNING
            )
        observations = [
            (
                call.node_key,
                task,
                observe_task(call.node_key, task, call.result_envelope),
            )
            for call, task in completed
        ]
        if not observations:
            return
        with self._transaction():
            repository = self.repository
            for node_key, task, observation in observations:
                if repository.get_task(
                    execution_run_id,
                    node_key,
                    task.task_key,
                ).status.is_terminal:
                    continue
                if observation == AvailabilityStatus.MISSING:
                    repository.fail_task(
                        execution_run_id,
                        node_key,
                        task.task_key,
                        message=missing_message,
                        now=now,
                    )
                else:
                    repository.record_task_result_observation(
                        execution_run_id,
                        node_key,
                        task.task_key,
                        observation,
                        now=now,
                    )

    def start_ready_nodes(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: Collection[str],
        task_plans: Callable[[str], tuple[TaskPlan, ...]],
        observe_task: Callable[[str, ExecutionTaskRecord], AvailabilityStatus],
        now: int,
    ) -> tuple[str, ...]:
        """Start ready Nodes, discover their Tasks, and validate cache state."""
        with self._synchronize():
            repository = self.repository
            statuses = {
                node.node_key: node.status
                for node in repository.list_nodes(execution_run_id)
            }
            plan = repository.get_run(execution_run_id).plan
        required = set(required_node_keys)
        started = []
        for node_key in ready_node_keys(plan, statuses):
            if node_key not in required:
                continue
            plans = task_plans(node_key)
            with self._transaction():
                repository = self.repository
                if repository.get_node(execution_run_id, node_key).status.is_terminal:
                    continue
                repository.start_node(execution_run_id, node_key, now=now)
                records = repository.discover_tasks(
                    execution_run_id,
                    node_key,
                    plans,
                    now=now,
                )
            observations = tuple(
                (record.task_key, observe_task(node_key, record))
                for record in records
                if not record.status.is_terminal
            )
            if observations:
                with self._transaction():
                    repository = self.repository
                    for task_key, observation in observations:
                        if repository.get_task(
                            execution_run_id,
                            node_key,
                            task_key,
                        ).status.is_terminal:
                            continue
                        repository.record_task_result_observation(
                            execution_run_id,
                            node_key,
                            task_key,
                            observation,
                            now=now,
                        )
            started.append(node_key)
        return tuple(started)

    def reconcile_nodes_and_run(self, execution_run_id: UUID, *, now: int) -> None:
        """Aggregate discovered Tasks, propagate skips, and finalize the Run."""
        with self._transaction():
            repository = self.repository
            for node in repository.list_nodes(execution_run_id):
                if node.status == NodeStatus.RUNNING and node.discovery_complete:
                    repository.reconcile_node_tasks(
                        execution_run_id,
                        node.node_key,
                        now=now,
                    )
            repository.skip_unreachable_nodes(execution_run_id, now=now)
            repository.finalize_run_from_results(execution_run_id, now=now)

    def fixed_call_candidates(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: set[str],
        candidate_node_keys: set[str] | None = None,
        describe_task: Callable[
            [ExecutionNodeRecord, ExecutionTaskRecord, NodeAdmissionRank],
            TaskDispatchDescriptor | None,
        ],
        available_total_slots: int,
        available_gpu_slots: int,
        suspend_unavailable_gpu_capacity: bool = True,
        now: int,
    ) -> tuple[ProviderCallCandidate, ...]:
        """Describe unplanned Tasks once, then admit a bounded ready window."""
        if available_total_slots < 0 or available_gpu_slots < 0:
            raise ValueError("available Provider Call slots cannot be negative")
        if available_total_slots == 0:
            return ()
        eligible_node_keys = (
            required_node_keys
            if candidate_node_keys is None
            else required_node_keys & candidate_node_keys
        )
        with self._synchronize():
            repository = self.repository
            run = repository.get_run(execution_run_id)
            nodes = repository.list_nodes(execution_run_id)
            ready_nodes = {
                node.node_key: node
                for node in nodes
                if (
                    node.node_key in eligible_node_keys
                    and node.status == NodeStatus.RUNNING
                    and node.discovery_complete
                )
            }
            unplanned_tasks = repository.list_unplanned_ready_tasks(
                execution_run_id,
                ready_nodes,
            )
        ranks = required_node_ranks(
            run.plan,
            required_node_keys=required_node_keys,
            unfinished_node_keys={
                node.node_key for node in nodes if not node.status.is_terminal
            },
        )
        descriptors = []
        for task in unplanned_tasks:
            node = ready_nodes[task.node_key]
            descriptor = describe_task(node, task, ranks[node.node_key])
            if descriptor is not None:
                descriptors.append(descriptor)
        self.persist_fixed_dispatch_policy(
            execution_run_id,
            tuple(descriptors),
            now=now,
        )

        node_keys_by_rank: dict[tuple[int, int], list[str]] = {}
        for node in ready_nodes.values():
            rank = ranks[node.node_key]
            node_keys_by_rank.setdefault(
                (rank.depth, rank.unblocking_span),
                [],
            ).append(node.node_key)

        selected: list[ProviderCallCandidate] = []
        remaining_gpu_slots = available_gpu_slots
        blocked_gpu_work = False
        for depth, unblocking_span in sorted(node_keys_by_rank, reverse=True):
            node_keys = node_keys_by_rank[(depth, unblocking_span)]
            for uses_gpu in (True, False):
                remaining_total_slots = available_total_slots - len(selected)
                resource_slots = (
                    min(remaining_total_slots, remaining_gpu_slots)
                    if uses_gpu
                    else remaining_total_slots
                )
                if resource_slots <= 0:
                    if uses_gpu and run.max_active_gpu_provider_calls == 0:
                        with self._synchronize():
                            blocked = (
                                self.repository.list_ready_fixed_dispatch_descriptors(
                                    execution_run_id,
                                    node_keys,
                                    uses_gpu=True,
                                    depth=depth,
                                    unblocking_span=unblocking_span,
                                    limit_per_node=1,
                                )
                            )
                        if blocked:
                            blocked_gpu_work = True
                    continue
                lookahead = max(1, resource_slots * 4)
                with self._synchronize():
                    window = self.repository.list_ready_fixed_dispatch_descriptors(
                        execution_run_id,
                        node_keys,
                        uses_gpu=uses_gpu,
                        depth=depth,
                        unblocking_span=unblocking_span,
                        limit_per_node=lookahead,
                    )
                if not window:
                    continue
                task_capacity = resource_slots * max(
                    descriptor.max_tasks_per_call for descriptor in window
                )
                if task_capacity > lookahead:
                    with self._synchronize():
                        window = self.repository.list_ready_fixed_dispatch_descriptors(
                            execution_run_id,
                            node_keys,
                            uses_gpu=uses_gpu,
                            depth=depth,
                            unblocking_span=unblocking_span,
                            limit_per_node=task_capacity,
                        )
                admitted = select_admissible_candidates(
                    form_fixed_batches(window),
                    available_total_slots=resource_slots,
                    available_gpu_slots=(resource_slots if uses_gpu else 0),
                )
                selected.extend(admitted)
                if uses_gpu:
                    remaining_gpu_slots -= len(admitted)
                if len(selected) == available_total_slots:
                    return tuple(selected)
        if not selected and blocked_gpu_work and suspend_unavailable_gpu_capacity:
            self.suspend_for_unavailable_gpu_capacity(
                execution_run_id,
                required_node_keys=required_node_keys,
                additional_gpu_work=True,
                now=now,
            )
        return tuple(selected)

    def suspend_for_unavailable_gpu_capacity(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: set[str],
        additional_gpu_work: bool = False,
        now: int,
    ) -> None:
        """Suspend only when the remaining ready frontier requires a GPU."""
        with self._synchronize():
            with self._transaction():
                run = self.repository.get_run(execution_run_id)
                if (
                    run.status not in {RunStatus.PENDING, RunStatus.RUNNING}
                    or run.max_active_gpu_provider_calls != 0
                    or self.repository.active_provider_call_counts(
                        execution_run_id
                    ).total
                ):
                    return
                blocked_gpu_work = additional_gpu_work or bool(
                    self.repository.list_ready_fixed_dispatch_descriptors(
                        execution_run_id,
                        required_node_keys,
                        uses_gpu=True,
                        depth=0,
                        unblocking_span=0,
                        limit_per_node=1,
                    )
                )
                if not blocked_gpu_work:
                    return
                self.repository.transition_run(
                    execution_run_id,
                    RunStatus.SUSPENDED,
                    reason=RunStatusReason.RESOURCE_CAPACITY_UNAVAILABLE,
                    message=(
                        "Required GPU work is missing while "
                        "max_active_gpu_provider_calls is zero"
                    ),
                    now=now,
                )
            self._checkpoint_state()

    def persist_fixed_dispatch_policy(
        self,
        execution_run_id: UUID,
        descriptors: tuple[TaskDispatchDescriptor, ...],
        *,
        now: int,
    ) -> tuple[TaskDispatchDescriptor, ...]:
        """Bind ready fixed Tasks to the Run's durable dispatch policy."""
        with self._transaction():
            repository = self.repository
            persisted, _ = repository.persist_fixed_dispatch_policy(
                execution_run_id,
                descriptors,
                now=now,
            )
        return persisted

    def persist_pull_worker_dispatch_policy(
        self,
        execution_run_id: UUID,
        descriptor: PullWorkerDispatchDescriptor,
        *,
        now: int,
    ) -> PullWorkerDispatchDescriptor:
        """Bind one pull Node to the Run's durable worker policy."""
        with self._transaction():
            repository = self.repository
            persisted, _ = repository.persist_pull_worker_dispatch_policy(
                execution_run_id,
                descriptor,
                now=now,
            )
        return persisted

    def resolve_provider_bindings(
        self,
        execution_run_id: UUID,
        bindings: Iterable[ProviderBinding],
        *,
        now: int,
    ) -> dict[ProviderBinding, Any] | None:
        """Resolve each exact binding once before workload-owned preparation."""
        resolved: dict[ProviderBinding, Any] = {}
        for binding in bindings:
            if binding in resolved:
                continue
            function = self._resolve_provider(execution_run_id, binding, now=now)
            if function is None:
                return None
            resolved[binding] = function
        return resolved

    def submit_provider_calls(
        self,
        execution_run_id: UUID,
        submissions: tuple[ProviderCallSubmission, ...],
        *,
        now: int,
    ) -> tuple[ProviderCallRecord | None, ...]:
        """Submit one admission set with one preclaim and one attachment checkpoint."""
        if not submissions:
            return ()
        for submission in submissions:
            candidate = submission.candidate
            if candidate.max_tasks_per_call is None:
                if submission.claim_capacity is None:
                    raise ValueError("pull-worker submission is missing claim_capacity")
            elif submission.claim_capacity is not None:
                raise ValueError("fixed-batch submission cannot set claim_capacity")
            identity_kwarg = submission.provider_call_id_kwarg
            if identity_kwarg is not None:
                if not identity_kwarg:
                    raise ValueError("provider_call_id_kwarg cannot be empty")
                if identity_kwarg in submission.kwargs:
                    raise ValueError(
                        f"{identity_kwarg} is supplied by the execution runtime"
                    )
        with self._synchronize():
            fixed_descriptors = tuple(
                descriptor
                for submission in submissions
                if submission.candidate.max_tasks_per_call is not None
                for descriptor in _fixed_descriptors_for_candidate(
                    self.repository,
                    execution_run_id,
                    submission.candidate,
                )
            )
        if fixed_descriptors:
            self.persist_fixed_dispatch_policy(
                execution_run_id,
                fixed_descriptors,
                now=now,
            )

        unresolved = tuple(
            submission.candidate.binding
            for submission in submissions
            if submission.function is None
        )
        functions = self.resolve_provider_bindings(
            execution_run_id,
            unresolved,
            now=now,
        )
        if functions is None:
            return tuple(None for _ in submissions)
        resolved: list[Any | None] = []
        for submission in submissions:
            binding = submission.candidate.binding
            function = submission.function
            if function is None:
                function = functions[binding]
            else:
                functions.setdefault(binding, function)
            resolved.append(function)

        preclaims: list[ProviderCallPreclaim | None] = []
        with self._synchronize():
            with self._transaction():
                pull_unfinished_by_node: dict[str, int] = {}
                for submission in submissions:
                    candidate = submission.candidate
                    if candidate.max_tasks_per_call is None:
                        claim_capacity = cast(int, submission.claim_capacity)
                        unfinished_task_count = pull_unfinished_by_node.get(
                            candidate.node_key
                        )
                        if unfinished_task_count is None:
                            unfinished_task_count = (
                                self.repository.unfinished_pull_task_count(
                                    execution_run_id,
                                    candidate.node_key,
                                )
                            )
                            pull_unfinished_by_node[candidate.node_key] = (
                                unfinished_task_count
                            )
                        preclaim = self.repository.preclaim_pull_worker(
                            execution_run_id,
                            candidate.node_key,
                            submission_token=submission.submission_token,
                            binding=candidate.binding,
                            compatibility_key=candidate.compatibility_key,
                            claim_capacity=claim_capacity,
                            unfinished_task_count=unfinished_task_count,
                            now=now,
                        )
                    else:
                        preclaim = self.repository.preclaim_fixed_batch(
                            execution_run_id,
                            candidate.node_key,
                            candidate.task_keys,
                            submission_token=submission.submission_token,
                            binding=candidate.binding,
                            compatibility_key=candidate.compatibility_key,
                            max_tasks_per_call=candidate.max_tasks_per_call,
                            now=now,
                        )
                    preclaims.append(preclaim)
            if any(
                preclaim is not None and preclaim.spawn_authorized
                for preclaim in preclaims
            ):
                self._checkpoint_state()

        spawned: dict[UUID, str] = {}
        errors: dict[UUID, Exception] = {}
        for submission, preclaim, function in zip(
            submissions,
            preclaims,
            resolved,
            strict=True,
        ):
            if preclaim is None or not preclaim.spawn_authorized:
                continue
            with self._synchronize():
                run = self.repository.get_run(execution_run_id)
                if run.cancellation_is_durable:
                    with self._transaction():
                        self.repository.cancel_unsubmitted_provider_call(
                            preclaim.call.provider_call_id,
                            message="Run cancellation stopped submission",
                            now=now,
                        )
                    continue
            invocation_kwargs = dict(submission.kwargs)
            identity_kwarg = submission.provider_call_id_kwarg
            if identity_kwarg is not None:
                invocation_kwargs[identity_kwarg] = str(preclaim.call.provider_call_id)
            try:
                spawned[preclaim.call.provider_call_id] = self._driver.spawn(
                    function,
                    args=submission.args,
                    kwargs=invocation_kwargs,
                )
            except (
                ProviderDefiniteSubmissionError,
                ProviderSubmissionOutcomeUnknownError,
            ) as error:
                errors[preclaim.call.provider_call_id] = error
            except Exception as error:
                errors[preclaim.call.provider_call_id] = error

        authorized = tuple(
            preclaim
            for preclaim in preclaims
            if preclaim is not None and preclaim.spawn_authorized
        )
        cancellation_requested = False
        if authorized:
            try:
                with self._synchronize():
                    with self._transaction():
                        for preclaim in authorized:
                            provider_call_id = preclaim.call.provider_call_id
                            current = self.repository.get_provider_call(
                                provider_call_id,
                                include_task_keys=False,
                            )
                            if current.status.is_terminal:
                                continue
                            error = errors.get(provider_call_id)
                            if isinstance(error, ProviderDefiniteSubmissionError):
                                self.repository.fail_provider_call(
                                    provider_call_id,
                                    message=str(error),
                                    now=now,
                                )
                            elif isinstance(
                                error,
                                ProviderSubmissionOutcomeUnknownError,
                            ):
                                self.repository.mark_submission_outcome_unknown(
                                    provider_call_id,
                                    message=str(error),
                                    now=now,
                                )
                            elif error is not None:
                                self.repository.mark_submission_outcome_unknown(
                                    provider_call_id,
                                    message=(
                                        f"Unexpected provider submission error: {error}"
                                    ),
                                    now=now,
                                )
                            else:
                                self.repository.attach_provider_call(
                                    provider_call_id,
                                    provider_call_handle_id=spawned[provider_call_id],
                                    now=now,
                                )
                        run = self.repository.get_run(execution_run_id)
                        cancellation_requested = run.cancellation_is_durable
                    self._checkpoint_state()
            except Exception:
                for handle_id in spawned.values():
                    try:
                        self._driver.cancel(handle_id)
                    except Exception:
                        LOGGER.warning(
                            "Could not cancel unattached Provider Call %s",
                            handle_id,
                            exc_info=True,
                        )
                with self._synchronize():
                    with self._transaction():
                        for preclaim in authorized:
                            call = self.repository.get_provider_call(
                                preclaim.call.provider_call_id,
                                include_task_keys=False,
                            )
                            if call.status == ProviderCallStatus.SUBMITTING:
                                self.repository.mark_submission_outcome_unknown(
                                    call.provider_call_id,
                                    message="Provider Call attachment was not durable",
                                    now=now,
                                )
                    self._checkpoint_state()
                raise

        if cancellation_requested:
            for provider_call_id in spawned:
                self.request_provider_call_cancellation(provider_call_id, now=now)

        with self._synchronize():
            return tuple(
                None
                if preclaim is None
                else self.repository.get_provider_call(preclaim.call.provider_call_id)
                for preclaim in preclaims
            )

    def claim_pull_tasks(
        self,
        provider_call_id: UUID,
        *,
        request_id: str,
        capacity: int,
        now: int,
    ) -> PullTaskClaim:
        """Checkpoint pull assignments before exposing their payloads."""
        with self._synchronize():
            with self._transaction():
                claim = self.repository.claim_pull_tasks(
                    provider_call_id,
                    request_id=request_id,
                    capacity=capacity,
                    now=now,
                )
            self._checkpoint_state()
        return claim

    def record_pull_task_completions_and_claim(
        self,
        provider_call_id: UUID,
        completions: Collection[tuple[str, str, AvailabilityStatus, str | None]],
        *,
        request_id: str,
        capacity: int,
        now: int,
    ) -> PullTaskClaim:
        """Checkpoint one completed microbatch and its next claim together."""
        with self._synchronize():
            with self._transaction():
                claim = self.repository.record_pull_task_completions_and_claim(
                    provider_call_id,
                    completions,
                    request_id=request_id,
                    capacity=capacity,
                    now=now,
                )
            self._checkpoint_state()
        return claim

    def reconcile_provider_calls(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: Collection[str],
        encode_result: Callable[[Any], Any],
        now: int,
        finalize_result: Callable[[Any], Any] | None = None,
        discard_result: Callable[[Any], None] | None = None,
        recover_terminal_publications: Callable[
            [tuple[ProviderCallRecord, ...]], Collection[UUID] | None
        ]
        | None = None,
    ) -> tuple[tuple[ProviderCallRecord, ProviderCallRecord], ...]:
        """Observe and prepare outside the writer, then durably finalize results."""
        with ExitStack() as prepared_cleanup:
            with self._synchronize():
                originals = (
                    self.repository.list_provider_calls_requiring_reconciliation(
                        execution_run_id
                    )
                )
            observations: dict[UUID, tuple[ProviderCallObservation, Any]] = {}
            preparation_errors: dict[UUID, Exception] = {}
            abandoned_submissions: set[UUID] = set()
            for call in originals:
                if call.status.is_terminal:
                    continue
                if call.status == ProviderCallStatus.SUBMITTING:
                    abandoned_submissions.add(call.provider_call_id)
                    continue
                if call.provider_call_handle_id is None:
                    continue
                observation = self._driver.observe(call.provider_call_handle_id)
                prepared_result = None
                if observation.kind == ProviderCallObservationKind.SUCCEEDED:
                    try:
                        prepared_result = encode_result(observation.result)
                    except Exception as error:
                        preparation_errors[call.provider_call_id] = error
                        continue
                    if discard_result is not None:
                        prepared_cleanup.callback(
                            discard_result,
                            prepared_result,
                        )
                observations[call.provider_call_id] = (
                    observation,
                    prepared_result,
                )

            if (
                not observations
                and not preparation_errors
                and not abandoned_submissions
            ):
                return tuple((call, call) for call in originals)

            terminal_calls = tuple(
                call
                for call in originals
                if (
                    (observed := observations.get(call.provider_call_id)) is not None
                    and observed[0].kind
                    in {
                        ProviderCallObservationKind.SUCCEEDED,
                        ProviderCallObservationKind.FAILED,
                        ProviderCallObservationKind.CANCELLED,
                    }
                )
            )
            reconciled = []
            checkpoint_needed = bool(abandoned_submissions or preparation_errors)
            first_error: Exception | None = None
            with self._synchronize():
                cancellation_is_durable = self.repository.get_run(
                    execution_run_id
                ).cancellation_is_durable
                deferred_terminal_calls = (
                    frozenset(recover_terminal_publications(terminal_calls) or ())
                    if (
                        recover_terminal_publications is not None
                        and terminal_calls
                        and not cancellation_is_durable
                    )
                    else frozenset()
                )
                with self._transaction():
                    for original in originals:
                        provider_call_id = original.provider_call_id
                        if provider_call_id in abandoned_submissions:
                            current = self.repository.get_provider_call(
                                provider_call_id,
                                include_task_keys=False,
                            )
                            updated = (
                                self.repository.mark_submission_outcome_unknown(
                                    provider_call_id,
                                    message=(
                                        "Recovered an abandoned submitting "
                                        "Provider Call"
                                    ),
                                    now=now,
                                )
                                if current.status == ProviderCallStatus.SUBMITTING
                                else current
                            )
                            reconciled.append((original, updated))
                            continue

                        preparation_error = preparation_errors.get(provider_call_id)
                        if preparation_error is not None:
                            updated = self._record_result_envelope_unknown(
                                provider_call_id,
                                error=preparation_error,
                                now=now,
                            )
                            if not updated.status.is_terminal:
                                first_error = first_error or preparation_error
                            reconciled.append((original, updated))
                            continue

                        prepared = observations.get(provider_call_id)
                        if prepared is None:
                            reconciled.append((original, original))
                            continue

                        if provider_call_id in deferred_terminal_calls:
                            reconciled.append((
                                original,
                                self.repository.get_provider_call(
                                    provider_call_id,
                                    include_task_keys=False,
                                ),
                            ))
                            checkpoint_needed = True
                            continue

                        observation, prepared_result = prepared
                        current = self.repository.get_provider_call(
                            provider_call_id,
                            include_task_keys=False,
                        )
                        if current.status.is_terminal:
                            updated = current
                        else:
                            try:
                                envelope = (
                                    finalize_result(prepared_result)
                                    if finalize_result is not None
                                    and observation.kind
                                    == ProviderCallObservationKind.SUCCEEDED
                                    else prepared_result
                                )
                            except Exception as error:
                                updated = self._record_result_envelope_unknown(
                                    provider_call_id,
                                    error=error,
                                    now=now,
                                )
                                first_error = first_error or error
                            else:
                                updated = _record_provider_call_observation(
                                    self.repository,
                                    provider_call_id,
                                    observation,
                                    result_envelope=envelope,
                                    result_already_satisfied=(
                                        original.node_key not in required_node_keys
                                    ),
                                    now=now,
                                )
                        checkpoint_needed = checkpoint_needed or (
                            observation.kind != ProviderCallObservationKind.RUNNING
                        )
                        reconciled.append((original, updated))

                if checkpoint_needed:
                    self._checkpoint_state()
            if first_error is not None:
                raise first_error
            return tuple(reconciled)

    def _record_result_envelope_unknown(
        self,
        provider_call_id: UUID,
        *,
        error: Exception,
        now: int,
    ) -> ProviderCallRecord:
        """Retain ownership when a provider result cannot become durable."""
        call = self.repository.get_provider_call(
            provider_call_id,
            include_task_keys=False,
        )
        if call.status.is_terminal:
            return call
        return self.repository.mark_provider_call_state_unknown(
            provider_call_id,
            message=f"Could not create a Result Envelope: {error}",
            now=now,
        )

    def request_provider_call_cancellation(
        self,
        provider_call_id: UUID,
        *,
        now: int,
        result_already_satisfied: bool = False,
    ) -> ProviderCallRecord:
        """Request cancellation without inventing a conclusive provider outcome."""
        with self._synchronize():
            call = self.repository.get_provider_call(
                provider_call_id,
                include_task_keys=False,
            )
        if call.status.is_terminal:
            return call
        if call.provider_call_handle_id is None:
            with self._synchronize():
                with self._transaction():
                    updated = self.repository.mark_provider_cancellation_unknown(
                        provider_call_id,
                        message="Provider Call has no attached cancellation handle",
                        now=now,
                    )
                self._checkpoint_state()
            return updated
        try:
            acknowledgement = self._driver.cancel(call.provider_call_handle_id)
        except Exception as error:
            with self._synchronize():
                with self._transaction():
                    updated = self.repository.mark_provider_cancellation_unknown(
                        provider_call_id,
                        message=f"Provider cancellation was inconclusive: {error}",
                        now=now,
                    )
                self._checkpoint_state()
            return updated
        with self._synchronize():
            current = self.repository.get_provider_call(
                provider_call_id,
                include_task_keys=False,
            )
            if (
                current.status.is_terminal
                or acknowledgement is None
                or acknowledgement.kind != ProviderCallObservationKind.CANCELLED
            ):
                return current
            with self._transaction():
                updated = _record_provider_call_observation(
                    self.repository,
                    provider_call_id,
                    acknowledgement,
                    result_envelope=None,
                    result_already_satisfied=result_already_satisfied,
                    now=now,
                )
            self._checkpoint_state()
            return updated

    def cancel_run(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> ExecutionRunRecord:
        """Durably request cancellation, then ask each attached provider owner."""
        with self._synchronize():
            with self._transaction():
                provider_call_ids = self.repository.request_run_cancellation(
                    execution_run_id,
                    now=now,
                )
            self._checkpoint_state()
        for provider_call_id in provider_call_ids:
            self.request_provider_call_cancellation(provider_call_id, now=now)
        with self._synchronize():
            return self.repository.get_run(execution_run_id)

    def _resolve_provider(
        self,
        execution_run_id: UUID,
        binding: ProviderBinding,
        *,
        now: int,
    ) -> Any | None:
        """Fail closed once no attached call still needs reconciliation."""
        try:
            return self._driver.resolve(binding)
        except ProviderDeploymentUnavailableError as error:
            with self._synchronize():
                run = self.repository.get_run(execution_run_id)
                if (
                    not run.cancellation_is_durable
                    and self.repository.active_provider_call_counts(
                        execution_run_id
                    ).total
                    == 0
                ):
                    with self._transaction():
                        self.repository.transition_run(
                            execution_run_id,
                            RunStatus.FAILED,
                            reason=RunStatusReason.DEPLOYMENT_UNAVAILABLE,
                            message=str(error),
                            now=now,
                        )
                    self._checkpoint_state()
            return None

    def _checkpoint_state(self) -> None:
        replacement = self._checkpoint()
        if replacement is not None:
            self.repository = replacement


def _fixed_descriptors_for_candidate(
    repository: SqliteExecutionRepository,
    execution_run_id: UUID,
    candidate: ProviderCallCandidate,
) -> tuple[TaskDispatchDescriptor, ...]:
    """Reconstruct the persisted per-Task policy represented by one batch."""
    if candidate.max_tasks_per_call is None:
        raise ValueError("fixed-batch candidate is missing max_tasks_per_call")
    node = repository.get_node(execution_run_id, candidate.node_key)
    return tuple(
        TaskDispatchDescriptor(
            node_key=candidate.node_key,
            node_ordinal=node.ordinal,
            task_key=task.task_key,
            task_ordinal=task.ordinal,
            binding=candidate.binding,
            compatibility_key=candidate.compatibility_key,
            max_tasks_per_call=candidate.max_tasks_per_call,
            depth=candidate.depth,
            unblocking_span=candidate.unblocking_span,
        )
        for task_key in candidate.task_keys
        for task in (
            repository.get_task(
                execution_run_id,
                candidate.node_key,
                task_key,
            ),
        )
    )
