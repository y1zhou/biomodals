"""Caller-driven composition facade for durable execution mechanics."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol
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
    ExecutionNodeRecord,
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


class _ModalDriver(Protocol):
    def resolve(self, binding: ProviderBinding) -> Any: ...

    def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str: ...

    def observe(self, provider_call_handle_id: str) -> ModalCallObservation: ...

    def cancel(self, provider_call_handle_id: str) -> None: ...


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
        modal_driver: _ModalDriver,
        checkpoint: Callable[[], SqliteExecutionRepository | None],
        commit_local: Callable[[], None] | None = None,
        transaction: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> None:
        """Bind host-owned state, Modal operations, and its durability boundary."""
        self.repository = repository
        self._modal = modal_driver
        self._checkpoint = checkpoint
        self._commit_local = commit_local
        self._transaction = transaction

    def checkpoint(self) -> None:
        """Cross the host durability boundary for caller-owned transitions."""
        self._checkpoint_state()

    def required_node_keys(self, execution_run_id: UUID) -> tuple[str, ...] | None:
        """Derive the result-driven closure from durable Node observations."""
        return _required_node_keys_for_run(self.repository, execution_run_id)

    def recover_publications(
        self,
        execution_run_id: UUID,
        *,
        observe_node: Callable[[str], AvailabilityStatus],
        observe_task: Callable[[str, ExecutionTaskRecord], AvailabilityStatus | None],
        now: int,
    ) -> tuple[str, ...] | None:
        """Walk backward from results and record caller-validated publications."""
        repository = self.repository
        run = repository.get_run(execution_run_id)
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
            for node in repository.list_nodes(execution_run_id)
        }
        while frontier := result_probe_frontier(run.plan, observations):
            observed = tuple(
                (node_key, observe_node(node_key)) for node_key in frontier
            )
            with self._transaction():
                for node_key, observation in observed:
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
        task_observations = []
        for node in repository.list_nodes(execution_run_id):
            if (
                node.node_key not in required_set
                or node.status != NodeStatus.RUNNING
                or not node.discovery_complete
            ):
                continue
            for task in repository.list_tasks(execution_run_id, node.node_key):
                if task.status.is_terminal:
                    continue
                observation = observe_task(node.node_key, task)
                if observation is not None:
                    task_observations.append((
                        node.node_key,
                        task.task_key,
                        observation,
                    ))
        if task_observations:
            with self._transaction():
                for node_key, task_key, observation in task_observations:
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
        repository = self.repository
        observations = []
        for call in repository.list_provider_calls(execution_run_id):
            if call.status != ProviderCallStatus.SUCCEEDED:
                continue
            for task_key in call.task_keys:
                task = repository.get_task(
                    execution_run_id,
                    call.node_key,
                    task_key,
                )
                if task.status == TaskStatus.RUNNING:
                    observations.append((
                        call.node_key,
                        task,
                        observe_task(call.node_key, task, call.result_envelope),
                    ))
        if not observations:
            return
        with self._transaction():
            for node_key, task, observation in observations:
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
        repository = self.repository
        statuses = {
            node.node_key: node.status
            for node in repository.list_nodes(execution_run_id)
        }
        required = set(required_node_keys)
        started = []
        for node_key in ready_node_keys(
            repository.get_run(execution_run_id).plan, statuses
        ):
            if node_key not in required:
                continue
            plans = task_plans(node_key)
            with self._transaction():
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
                    for task_key, observation in observations:
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
        repository = self.repository
        with self._transaction():
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
        describe_task: Callable[
            [ExecutionNodeRecord, ExecutionTaskRecord, NodeAdmissionRank],
            TaskDispatchDescriptor | None,
        ],
        available_total_slots: int | None,
        available_gpu_slots: int,
        now: int,
    ) -> tuple[ProviderCallCandidate, ...]:
        """Persist workload dispatch descriptions and select admissible calls."""
        repository = self.repository
        run = repository.get_run(execution_run_id)
        nodes = repository.list_nodes(execution_run_id)
        ranks = required_node_ranks(
            run.plan,
            required_node_keys=required_node_keys,
            unfinished_node_keys={
                node.node_key for node in nodes if not node.status.is_terminal
            },
        )
        descriptors = []
        for node in nodes:
            if (
                node.node_key not in required_node_keys
                or node.status != NodeStatus.RUNNING
                or not node.discovery_complete
            ):
                continue
            for task in repository.list_tasks(execution_run_id, node.node_key):
                if (
                    task.status != TaskStatus.PENDING
                    or task.result_observation != AvailabilityStatus.MISSING
                ):
                    continue
                descriptor = describe_task(node, task, ranks[node.node_key])
                if descriptor is not None:
                    descriptors.append(descriptor)
        persisted = self.persist_fixed_dispatch_policy(
            execution_run_id,
            tuple(descriptors),
            now=now,
        )
        candidates = form_fixed_batches(persisted)
        return select_admissible_candidates(
            candidates,
            available_total_slots=(
                len(candidates)
                if available_total_slots is None
                else available_total_slots
            ),
            available_gpu_slots=available_gpu_slots,
        )

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

    def persist_pull_worker_dispatch_policy(
        self,
        execution_run_id: UUID,
        descriptor: PullWorkerDispatchDescriptor,
        *,
        now: int,
    ) -> PullWorkerDispatchDescriptor:
        """Bind one pull Node to the Run's durable worker policy."""
        persisted, changed = self.repository.persist_pull_worker_dispatch_policy(
            execution_run_id,
            descriptor,
            now=now,
        )
        if changed:
            self._commit_local_state()
        return persisted

    def submit_fixed_batch(
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
        function = self._resolve_provider(
            execution_run_id,
            candidate.binding,
            now=now,
        )
        if function is None:
            return None
        return self.submit_resolved_fixed_batch(
            execution_run_id,
            candidate,
            function=function,
            submission_token=submission_token,
            args=args,
            kwargs=kwargs,
            provider_call_id_kwarg=provider_call_id_kwarg,
            now=now,
        )

    def resolve_provider_binding(
        self,
        execution_run_id: UUID,
        binding: ProviderBinding,
        *,
        now: int,
    ) -> Any | None:
        """Preflight one exact binding before workload writer coordination."""
        return self._resolve_provider(execution_run_id, binding, now=now)

    def submit_resolved_fixed_batch(
        self,
        execution_run_id: UUID,
        candidate: ProviderCallCandidate,
        *,
        function: Any,
        submission_token: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        provider_call_id_kwarg: str | None = None,
        now: int,
    ) -> ProviderCallRecord | None:
        """Preclaim and spawn once using an already hydrated exact binding."""
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
        return self._spawn_preclaimed(
            preclaim,
            function=function,
            args=args,
            kwargs=invocation_kwargs,
            now=now,
        )

    def submit_pull_worker(
        self,
        execution_run_id: UUID,
        *,
        node_key: str,
        submission_token: str,
        binding: ProviderBinding,
        compatibility_key: str,
        claim_capacity: int,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        now: int,
    ) -> ProviderCallRecord | None:
        """Submit one pull worker after its dispatch policy is durable."""
        function = self._resolve_provider(
            execution_run_id,
            binding,
            now=now,
        )
        if function is None:
            return None
        preclaim = self.repository.preclaim_pull_worker(
            execution_run_id,
            node_key,
            submission_token=submission_token,
            binding=binding,
            compatibility_key=compatibility_key,
            claim_capacity=claim_capacity,
            now=now,
        )
        if preclaim is None:
            return None
        if not preclaim.spawn_authorized:
            return preclaim.call
        invocation_kwargs = {} if kwargs is None else dict(kwargs)
        if "provider_call_id" in invocation_kwargs:
            raise ValueError("provider_call_id is supplied by the execution runtime")
        invocation_kwargs["provider_call_id"] = str(preclaim.call.provider_call_id)
        return self._spawn_preclaimed(
            preclaim,
            function=function,
            args=args,
            kwargs=invocation_kwargs,
            now=now,
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

    def _spawn_preclaimed(
        self,
        preclaim: ProviderCallPreclaim,
        *,
        function: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        now: int,
    ) -> ProviderCallRecord:
        """Cross the durable preclaim/spawn/attachment fault boundary."""
        self._checkpoint_state()
        try:
            handle_id = self._modal.spawn(
                function,
                args=args,
                kwargs=kwargs,
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
                self._modal.cancel(handle_id)
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

    def reconcile_provider_call(
        self,
        provider_call_id: UUID,
        *,
        encode_result: Callable[[Any], Any],
        result_already_satisfied: bool = False,
        now: int,
    ) -> ProviderCallRecord:
        """Observe or collect the existing call without ever resubmitting it."""
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

        observation = self._modal.observe(call.provider_call_handle_id)
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
        else:
            self._commit_local_state()
        return updated

    def reconcile_provider_calls(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: Collection[str],
        encode_result: Callable[[Any], Any],
        now: int,
    ) -> tuple[tuple[ProviderCallRecord, ProviderCallRecord], ...]:
        """Reconcile every nonterminal call and retain before/after records."""
        reconciled = []
        for original in self.repository.list_provider_calls(execution_run_id):
            updated = original
            if not original.status.is_terminal:
                updated = self.reconcile_provider_call(
                    original.provider_call_id,
                    encode_result=encode_result,
                    result_already_satisfied=(
                        original.node_key not in required_node_keys
                    ),
                    now=now,
                )
            reconciled.append((original, updated))
        return tuple(reconciled)

    def request_provider_call_cancellation(
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
            self._modal.cancel(call.provider_call_handle_id)
        except Exception as error:
            updated = self.repository.mark_provider_cancellation_unknown(
                provider_call_id,
                message=f"Modal cancellation was inconclusive: {error}",
                now=now,
            )
            self._checkpoint_state()
            return updated
        return self.repository.get_provider_call(provider_call_id)

    def cancel_run(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> ExecutionRunRecord:
        """Durably request cancellation, then ask each attached provider owner."""
        provider_call_ids = self.repository.request_run_cancellation(
            execution_run_id,
            now=now,
        )
        self._checkpoint_state()
        for provider_call_id in provider_call_ids:
            self.request_provider_call_cancellation(provider_call_id, now=now)
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
        else:
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
        """Reconcile every nonterminal call and retain before/after records."""
        reconciled = []
        for original in self.repository.list_provider_calls(execution_run_id):
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
