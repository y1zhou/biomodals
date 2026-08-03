"""Caller-driven composition facade for durable execution mechanics."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from uuid import UUID

from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
    ModalDefiniteSubmissionError,
    ModalDeploymentUnavailableError,
    ModalSubmissionOutcomeUnknownError,
)
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


class ModalDriver(Protocol):
    """Synchronous Modal operations required by the execution runtime."""

    def resolve(self, binding: ProviderBinding) -> Any:
        """Resolve one exact deployed function."""
        ...

    def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Submit one function invocation."""
        ...

    def observe(self, provider_call_handle_id: str) -> ModalCallObservation:
        """Observe one retained provider call."""
        ...

    def cancel(self, provider_call_handle_id: str) -> None:
        """Request provider-call cancellation."""
        ...


class _AsyncModalDriver(Protocol):
    async def resolve(self, binding: ProviderBinding) -> Any: ...

    async def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str: ...

    async def observe(
        self,
        provider_call_handle_id: str,
    ) -> ModalCallObservation: ...

    async def cancel(self, provider_call_handle_id: str) -> None: ...


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
    observation: ModalCallObservation,
    *,
    result_envelope: Any,
    result_already_satisfied: bool,
    now: int,
) -> ProviderCallRecord:
    """Apply the provider-neutral transition shared by sync and async hosts."""
    if observation.kind == ModalCallObservationKind.RUNNING:
        return repository.mark_provider_call_running(provider_call_id, now=now)
    if observation.kind == ModalCallObservationKind.SUCCEEDED:
        return repository.record_provider_call_result(
            provider_call_id,
            result_envelope=result_envelope,
            now=now,
        )
    if observation.kind == ModalCallObservationKind.FAILED:
        return repository.fail_provider_call(
            provider_call_id,
            message=observation.message or "Modal function failed",
            now=now,
        )
    if observation.kind == ModalCallObservationKind.CANCELLED:
        if result_already_satisfied:
            return repository.cancel_pruned_provider_call(
                provider_call_id,
                now=now,
            )
        return repository.cancel_provider_call(
            provider_call_id,
            message=observation.message or "Modal function was cancelled",
            now=now,
        )
    return repository.mark_provider_call_state_unknown(
        provider_call_id,
        message=observation.message or "Modal call state was inconclusive",
        now=now,
    )


class ExecutionRuntime:
    """Coordinate repository checkpoints with exactly one Modal side effect."""

    def __init__(
        self,
        repository: SqliteExecutionRepository,
        *,
        modal_driver: ModalDriver,
        checkpoint: Callable[[], SqliteExecutionRepository | None],
        transaction: Callable[[], AbstractContextManager[object]] = nullcontext,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> None:
        """Bind host-owned state, Modal operations, and its durability boundary."""
        self.repository = repository
        self._modal = modal_driver
        self._checkpoint = checkpoint
        self._transaction = transaction
        self._synchronize = synchronize

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
            plan = self.repository.get_run(execution_run_id).plan
            nodes = self.repository.list_nodes(execution_run_id)
        observations = {
            node.node_key: (
                AvailabilityStatus.AVAILABLE
                if node.status == NodeStatus.SUCCEEDED
                else node.result_observation or AvailabilityStatus.MISSING
            )
            for node in nodes
        }
        return required_node_keys(plan, observations)

    def advance_once(
        self,
        execution_run_id: UUID,
        *,
        recover_publications: Callable[[], None],
        reconcile_provider_calls: Callable[[set[str]], None],
        decode_completed_calls: Callable[[], None],
        start_ready_nodes: Callable[[set[str]], None],
        admit_remote_tasks: Callable[[set[str]], None],
        after_start_ready_nodes: Callable[[], None] | None = None,
        now: Callable[[], int],
    ) -> None:
        """Apply one result-driven reconciliation and admission cycle."""
        recover_publications()
        self.reconcile_nodes_and_run(execution_run_id, now=now())
        with self._synchronize():
            run = self.repository.get_run(execution_run_id)
        if run.status == RunStatus.CANCEL_REQUESTED:
            reconcile_provider_calls(set(run.plan.node_keys))
            decode_completed_calls()
            recover_publications()
            self.reconcile_nodes_and_run(execution_run_id, now=now())
            return
        if run.status == RunStatus.STATE_UNKNOWN:
            required = self.required_node_keys(execution_run_id)
            required_nodes = set(run.plan.node_keys if required is None else required)
            if required is not None:
                self.prune_unrequired_nodes(
                    execution_run_id,
                    required_node_keys=required,
                    now=now(),
                )
            reconcile_provider_calls(required_nodes)
            decode_completed_calls()
            recover_publications()
            self.reconcile_nodes_and_run(execution_run_id, now=now())
            return
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        required = self.required_node_keys(execution_run_id)
        if required is None:
            return
        self.prune_unrequired_nodes(
            execution_run_id,
            required_node_keys=required,
            now=now(),
        )
        reconcile_provider_calls(set(required))
        decode_completed_calls()
        recover_publications()
        self.reconcile_nodes_and_run(execution_run_id, now=now())
        with self._synchronize():
            can_continue = self.repository.get_run(execution_run_id).status in {
                RunStatus.PENDING,
                RunStatus.RUNNING,
            }
        if not can_continue:
            return
        start_ready_nodes(set(required))
        if after_start_ready_nodes is not None:
            after_start_ready_nodes()
        recover_publications()
        required = self.required_node_keys(execution_run_id)
        if required is not None:
            admit_remote_tasks(set(required))
        self.reconcile_nodes_and_run(execution_run_id, now=now())

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
                for task in repository.list_tasks(execution_run_id, node.node_key)
                if (
                    not task.status.is_terminal
                    and not (
                        task.status == TaskStatus.PENDING
                        and task.result_observation == AvailabilityStatus.MISSING
                    )
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
            self.request_provider_call_cancellation(provider_call_id, now=now)
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
                    continue
                lookahead = max(1, resource_slots * 4)
                with self._synchronize():
                    window = self.repository.list_ready_fixed_dispatch_descriptors(
                        execution_run_id,
                        node_keys,
                        uses_gpu=uses_gpu,
                        depth=depth,
                        unblocking_span=unblocking_span,
                        limit=lookahead,
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
                            limit=task_capacity,
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
        return tuple(selected)

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

    def resolve_provider_binding(
        self,
        execution_run_id: UUID,
        binding: ProviderBinding,
        *,
        now: int,
    ) -> Any | None:
        """Preflight one exact binding before workload writer coordination."""
        return self._resolve_provider(execution_run_id, binding, now=now)

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

        functions: dict[ProviderBinding, Any] = {}
        resolved: list[Any | None] = []
        for submission in submissions:
            binding = submission.candidate.binding
            function = submission.function
            if function is None:
                if binding not in functions:
                    resolved_function = self._resolve_provider(
                        execution_run_id,
                        binding,
                        now=now,
                    )
                    if resolved_function is None:
                        return tuple(None for _ in submissions)
                    functions[binding] = resolved_function
                function = functions[binding]
            else:
                functions.setdefault(binding, function)
            resolved.append(function)

        preclaims: list[ProviderCallPreclaim | None] = []
        with self._synchronize():
            with self._transaction():
                for submission in submissions:
                    candidate = submission.candidate
                    if candidate.max_tasks_per_call is None:
                        claim_capacity = cast(int, submission.claim_capacity)
                        preclaim = self.repository.preclaim_pull_worker(
                            execution_run_id,
                            candidate.node_key,
                            submission_token=submission.submission_token,
                            binding=candidate.binding,
                            compatibility_key=candidate.compatibility_key,
                            claim_capacity=claim_capacity,
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
                spawned[preclaim.call.provider_call_id] = self._modal.spawn(
                    function,
                    args=submission.args,
                    kwargs=invocation_kwargs,
                )
            except (
                ModalDefiniteSubmissionError,
                ModalSubmissionOutcomeUnknownError,
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
                                provider_call_id
                            )
                            if current.status.is_terminal:
                                continue
                            error = errors.get(provider_call_id)
                            if isinstance(error, ModalDefiniteSubmissionError):
                                self.repository.fail_provider_call(
                                    provider_call_id,
                                    message=str(error),
                                    now=now,
                                )
                            elif isinstance(
                                error,
                                ModalSubmissionOutcomeUnknownError,
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
                                        f"Unexpected Modal submission error: {error}"
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
                        self._modal.cancel(handle_id)
                    except Exception:
                        LOGGER.warning(
                            "Could not cancel unattached Modal call %s",
                            handle_id,
                            exc_info=True,
                        )
                with self._synchronize():
                    with self._transaction():
                        for preclaim in authorized:
                            call = self.repository.get_provider_call(
                                preclaim.call.provider_call_id
                            )
                            if call.status == ProviderCallStatus.SUBMITTING:
                                self.repository.mark_submission_outcome_unknown(
                                    call.provider_call_id,
                                    message="Modal call attachment was not durable",
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

    def record_pull_task_completion(
        self,
        provider_call_id: UUID,
        task_key: str,
        *,
        request_id: str,
        observation: AvailabilityStatus,
        message: str | None = None,
        now: int,
    ) -> ExecutionTaskRecord:
        """Checkpoint one idempotent worker publication report."""
        with self._synchronize():
            with self._transaction():
                task = self.repository.record_pull_task_completion(
                    provider_call_id,
                    task_key,
                    request_id=request_id,
                    observation=observation,
                    message=message,
                    now=now,
                )
            self._checkpoint_state()
        return task

    def reconcile_provider_calls(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: Collection[str],
        encode_result: Callable[[Any], Any],
        now: int,
        finalize_result: Callable[[Any], Any] | None = None,
        discard_result: Callable[[Any], None] | None = None,
    ) -> tuple[tuple[ProviderCallRecord, ProviderCallRecord], ...]:
        """Observe and prepare outside the writer, then durably finalize results."""
        with ExitStack() as prepared_cleanup:
            with self._synchronize():
                originals = (
                    self.repository.list_provider_calls_requiring_reconciliation(
                        execution_run_id
                    )
                )
            observations: dict[UUID, tuple[ModalCallObservation, Any]] = {}
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
                observation = self._modal.observe(call.provider_call_handle_id)
                prepared_result = None
                if observation.kind == ModalCallObservationKind.SUCCEEDED:
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

            reconciled = []
            checkpoint_needed = bool(abandoned_submissions or preparation_errors)
            first_error: Exception | None = None
            with self._synchronize():
                with self._transaction():
                    for original in originals:
                        provider_call_id = original.provider_call_id
                        if provider_call_id in abandoned_submissions:
                            current = self.repository.get_provider_call(
                                provider_call_id
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

                        observation, prepared_result = prepared
                        current = self.repository.get_provider_call(provider_call_id)
                        if current.status.is_terminal:
                            updated = current
                        else:
                            try:
                                envelope = (
                                    finalize_result(prepared_result)
                                    if finalize_result is not None
                                    and observation.kind
                                    == ModalCallObservationKind.SUCCEEDED
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
                            observation.kind != ModalCallObservationKind.RUNNING
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
        call = self.repository.get_provider_call(provider_call_id)
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
    ) -> ProviderCallRecord:
        """Request cancellation without inventing a conclusive provider outcome."""
        with self._synchronize():
            call = self.repository.get_provider_call(provider_call_id)
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
            self._modal.cancel(call.provider_call_handle_id)
        except Exception as error:
            with self._synchronize():
                with self._transaction():
                    updated = self.repository.mark_provider_cancellation_unknown(
                        provider_call_id,
                        message=f"Modal cancellation was inconclusive: {error}",
                        now=now,
                    )
                self._checkpoint_state()
            return updated
        with self._synchronize():
            return self.repository.get_provider_call(provider_call_id)

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
            return self._modal.resolve(binding)
        except ModalDeploymentUnavailableError as error:
            with self._synchronize():
                if (
                    self.repository.active_provider_call_counts(execution_run_id).total
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


class AsyncExecutionRuntime:
    """Async-host facade over the same durable execution transitions."""

    def __init__(
        self,
        repository: SqliteExecutionRepository,
        *,
        modal_driver: _AsyncModalDriver,
        checkpoint: Callable[[], SqliteExecutionRepository | None],
        commit_local: Callable[[], None] | None = None,
    ) -> None:
        """Bind an async provider boundary to host-owned durable state."""
        self.repository = repository
        self._modal = modal_driver
        self._checkpoint = checkpoint
        self._commit_local = commit_local

    def checkpoint(self) -> None:
        """Cross the host durability boundary for caller-owned transitions."""
        self._checkpoint_state()

    def required_node_keys(self, execution_run_id: UUID) -> tuple[str, ...] | None:
        """Derive the result-driven closure from durable Node observations."""
        return _required_node_keys_for_run(self.repository, execution_run_id)

    def persist_fixed_dispatch_policy(
        self,
        execution_run_id: UUID,
        descriptors: tuple[TaskDispatchDescriptor, ...],
        *,
        now: int,
    ) -> tuple[TaskDispatchDescriptor, ...]:
        """Bind ready fixed Tasks to the Run's durable dispatch policy."""
        persisted, changed = self.repository.persist_fixed_dispatch_policy(
            execution_run_id,
            descriptors,
            now=now,
        )
        if changed:
            self._commit_local_state()
        return persisted

    async def submit_fixed_batch(
        self,
        execution_run_id: UUID,
        candidate: ProviderCallCandidate,
        *,
        submission_token: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        provider_call_id_kwarg: str | None = None,
        now: int,
    ) -> ProviderCallRecord | None:
        """Resolve, preclaim, checkpoint, spawn once, attach, and checkpoint."""
        if candidate.max_tasks_per_call is None:
            raise ValueError("fixed-batch candidate is missing max_tasks_per_call")
        self.persist_fixed_dispatch_policy(
            execution_run_id,
            _fixed_descriptors_for_candidate(
                self.repository,
                execution_run_id,
                candidate,
            ),
            now=now,
        )
        function = await self._resolve_provider(
            execution_run_id,
            candidate.binding,
            now=now,
        )
        if function is None:
            return None
        preclaim = self.repository.preclaim_fixed_batch(
            execution_run_id,
            candidate.node_key,
            candidate.task_keys,
            submission_token=submission_token,
            binding=candidate.binding,
            compatibility_key=candidate.compatibility_key,
            max_tasks_per_call=candidate.max_tasks_per_call,
            now=now,
        )
        if preclaim is None:
            return None
        if not preclaim.spawn_authorized:
            return preclaim.call
        invocation_kwargs = {} if kwargs is None else dict(kwargs)
        if provider_call_id_kwarg is not None:
            if not provider_call_id_kwarg:
                raise ValueError("provider_call_id_kwarg cannot be empty")
            if provider_call_id_kwarg in invocation_kwargs:
                raise ValueError(
                    f"{provider_call_id_kwarg} is supplied by the execution runtime"
                )
            invocation_kwargs[provider_call_id_kwarg] = str(
                preclaim.call.provider_call_id
            )
        self._checkpoint_state()
        try:
            handle_id = await self._modal.spawn(
                function,
                args=args,
                kwargs=invocation_kwargs,
            )
        except ModalDefiniteSubmissionError as error:
            self.repository.fail_provider_call(
                preclaim.call.provider_call_id,
                message=str(error),
                now=now,
            )
            self._checkpoint_state()
            raise
        except ModalSubmissionOutcomeUnknownError as error:
            self.repository.mark_submission_outcome_unknown(
                preclaim.call.provider_call_id,
                message=str(error),
                now=now,
            )
            self._checkpoint_state()
            raise
        try:
            attached = self.repository.attach_provider_call(
                preclaim.call.provider_call_id,
                provider_call_handle_id=handle_id,
                now=now,
            )
            self._checkpoint_state()
        except Exception:
            try:
                await self._modal.cancel(handle_id)
            finally:
                call = self.repository.get_provider_call(preclaim.call.provider_call_id)
                if call.status == ProviderCallStatus.SUBMITTING:
                    self.repository.mark_submission_outcome_unknown(
                        call.provider_call_id,
                        message="Modal call attachment was not durable",
                        now=now,
                    )
                self._checkpoint_state()
            raise
        return attached

    async def reconcile_provider_call(
        self,
        provider_call_id: UUID,
        *,
        encode_result: Callable[[Any], Any],
        result_already_satisfied: bool = False,
        now: int,
    ) -> ProviderCallRecord:
        """Observe or collect an existing async call without resubmission."""
        call = self.repository.get_provider_call(provider_call_id)
        if call.status.is_terminal:
            return call
        if call.status == ProviderCallStatus.SUBMITTING:
            unknown = self.repository.mark_submission_outcome_unknown(
                provider_call_id,
                message="Recovered an abandoned submitting Provider Call",
                now=now,
            )
            self._checkpoint_state()
            return unknown
        if call.provider_call_handle_id is None:
            return call

        observation = await self._modal.observe(call.provider_call_handle_id)
        envelope = None
        if observation.kind == ModalCallObservationKind.SUCCEEDED:
            try:
                envelope = encode_result(observation.result)
            except Exception as error:
                self.repository.mark_provider_call_state_unknown(
                    provider_call_id,
                    message=f"Could not create a Result Envelope: {error}",
                    now=now,
                )
                self._checkpoint_state()
                raise
        updated = _record_provider_call_observation(
            self.repository,
            provider_call_id,
            observation,
            result_envelope=envelope,
            result_already_satisfied=result_already_satisfied,
            now=now,
        )
        if observation.kind != ModalCallObservationKind.RUNNING:
            self._checkpoint_state()
        elif updated != call:
            self._commit_local_state()
        return updated

    async def reconcile_provider_calls(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: Collection[str],
        encode_result: Callable[[Any], Any],
        now: int,
    ) -> tuple[tuple[ProviderCallRecord, ProviderCallRecord], ...]:
        """Reconcile observable calls and expose unpublished successes."""
        reconciled = []
        for original in self.repository.list_provider_calls_requiring_reconciliation(
            execution_run_id
        ):
            updated = original
            if not original.status.is_terminal:
                updated = await self.reconcile_provider_call(
                    original.provider_call_id,
                    encode_result=encode_result,
                    result_already_satisfied=(
                        original.node_key not in required_node_keys
                    ),
                    now=now,
                )
            reconciled.append((original, updated))
        return tuple(reconciled)

    async def request_provider_call_cancellation(
        self,
        provider_call_id: UUID,
        *,
        now: int,
    ) -> ProviderCallRecord:
        """Request cancellation without inventing a conclusive provider outcome."""
        call = self.repository.get_provider_call(provider_call_id)
        if call.status.is_terminal:
            return call
        if call.provider_call_handle_id is None:
            updated = self.repository.mark_provider_cancellation_unknown(
                provider_call_id,
                message="Provider Call has no attached cancellation handle",
                now=now,
            )
            self._checkpoint_state()
            return updated
        try:
            await self._modal.cancel(call.provider_call_handle_id)
        except Exception as error:
            updated = self.repository.mark_provider_cancellation_unknown(
                provider_call_id,
                message=f"Modal cancellation was inconclusive: {error}",
                now=now,
            )
            self._checkpoint_state()
            return updated
        return self.repository.get_provider_call(provider_call_id)

    async def cancel_run(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> ExecutionRunRecord:
        """Durably request cancellation, then ask each async provider owner."""
        provider_call_ids = self.repository.request_run_cancellation(
            execution_run_id,
            now=now,
        )
        self._checkpoint_state()
        for provider_call_id in provider_call_ids:
            await self.request_provider_call_cancellation(provider_call_id, now=now)
        return self.repository.get_run(execution_run_id)

    async def _resolve_provider(
        self,
        execution_run_id: UUID,
        binding: ProviderBinding,
        *,
        now: int,
    ) -> Any | None:
        """Fail closed once no attached call still needs reconciliation."""
        try:
            return await self._modal.resolve(binding)
        except ModalDeploymentUnavailableError as error:
            if self.repository.active_provider_call_counts(execution_run_id).total == 0:
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

    def _commit_local_state(self) -> None:
        if self._commit_local is None:
            self._checkpoint_state()
        else:
            self._commit_local()


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
