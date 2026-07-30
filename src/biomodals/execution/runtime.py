"""Caller-driven composition facade for durable execution mechanics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol
from uuid import UUID

from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
    ModalDefiniteSubmissionError,
    ModalSubmissionOutcomeUnknownError,
)
from biomodals.execution.model import (
    AvailabilityStatus,
    ExecutionRunRecord,
    ExecutionTaskRecord,
    ProviderBinding,
    ProviderCallPreclaim,
    ProviderCallRecord,
    ProviderCallStatus,
    PullTaskClaim,
)
from biomodals.execution.scheduler import ProviderCallCandidate
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


class ExecutionRuntime:
    """Coordinate repository checkpoints with exactly one Modal side effect."""

    def __init__(
        self,
        repository: SqliteExecutionRepository,
        *,
        modal_driver: _ModalDriver,
        checkpoint: Callable[[], SqliteExecutionRepository | None],
    ) -> None:
        """Bind host-owned state, Modal operations, and its durability boundary."""
        self.repository = repository
        self._modal = modal_driver
        self._checkpoint = checkpoint

    def checkpoint(self) -> None:
        """Cross the host durability boundary for caller-owned transitions."""
        self._checkpoint_state()

    def submit_fixed_batch(
        self,
        execution_run_id: UUID,
        candidate: ProviderCallCandidate,
        *,
        submission_token: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        now: int,
    ) -> ProviderCallRecord | None:
        """Resolve, preclaim, checkpoint, spawn once, attach, and checkpoint."""
        function = self._modal.resolve(candidate.binding)
        preclaim = self.repository.preclaim_fixed_batch(
            execution_run_id,
            candidate.node_key,
            candidate.task_keys,
            submission_token=submission_token,
            binding=candidate.binding,
            compatibility_key=candidate.compatibility_key,
            now=now,
        )
        if preclaim is None:
            return None
        if not preclaim.spawn_authorized:
            return preclaim.call
        return self._spawn_preclaimed(
            preclaim,
            function=function,
            args=args,
            kwargs={} if kwargs is None else kwargs,
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
        """Submit one pull worker with its durable owner identity."""
        function = self._modal.resolve(binding)
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
            raise
        return attached

    def reconcile_provider_call(
        self,
        provider_call_id: UUID,
        *,
        encode_result: Callable[[Any], Any],
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
        if observation.kind == ModalCallObservationKind.RUNNING:
            updated = self.repository.mark_provider_call_running(
                provider_call_id,
                now=now,
            )
        elif observation.kind == ModalCallObservationKind.SUCCEEDED:
            try:
                envelope = encode_result(observation.result)
            except Exception as error:
                updated = self.repository.mark_provider_call_state_unknown(
                    provider_call_id,
                    message=f"Could not create a Result Envelope: {error}",
                    now=now,
                )
                self._checkpoint_state()
                raise
            updated = self.repository.record_provider_call_result(
                provider_call_id,
                result_envelope=envelope,
                now=now,
            )
        elif observation.kind == ModalCallObservationKind.FAILED:
            updated = self.repository.fail_provider_call(
                provider_call_id,
                message=observation.message or "Modal function failed",
                now=now,
            )
        elif observation.kind == ModalCallObservationKind.CANCELLED:
            updated = self.repository.cancel_provider_call(
                provider_call_id,
                message=observation.message or "Modal function was cancelled",
                now=now,
            )
        else:
            updated = self.repository.mark_provider_call_state_unknown(
                provider_call_id,
                message=observation.message or "Modal call state was inconclusive",
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
        provider_call_ids = self.repository.request_run_cancellation(
            execution_run_id,
            now=now,
        )
        self._checkpoint_state()
        for provider_call_id in provider_call_ids:
            call = self.repository.get_provider_call(provider_call_id)
            if call.provider_call_handle_id is None:
                self.repository.mark_provider_cancellation_unknown(
                    provider_call_id,
                    message="Provider Call has no attached cancellation handle",
                    now=now,
                )
                self._checkpoint_state()
                continue
            try:
                self._modal.cancel(call.provider_call_handle_id)
            except Exception as error:
                self.repository.mark_provider_cancellation_unknown(
                    provider_call_id,
                    message=f"Modal cancellation was inconclusive: {error}",
                    now=now,
                )
                self._checkpoint_state()
        return self.repository.get_run(execution_run_id)

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
    ) -> None:
        """Bind an async provider boundary to host-owned durable state."""
        self.repository = repository
        self._modal = modal_driver
        self._checkpoint = checkpoint

    def checkpoint(self) -> None:
        """Cross the host durability boundary for caller-owned transitions."""
        self._checkpoint_state()

    async def submit_fixed_batch(
        self,
        execution_run_id: UUID,
        candidate: ProviderCallCandidate,
        *,
        submission_token: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        now: int,
    ) -> ProviderCallRecord | None:
        """Resolve, preclaim, checkpoint, spawn once, attach, and checkpoint."""
        function = await self._modal.resolve(candidate.binding)
        preclaim = self.repository.preclaim_fixed_batch(
            execution_run_id,
            candidate.node_key,
            candidate.task_keys,
            submission_token=submission_token,
            binding=candidate.binding,
            compatibility_key=candidate.compatibility_key,
            now=now,
        )
        if preclaim is None:
            return None
        if not preclaim.spawn_authorized:
            return preclaim.call
        self._checkpoint_state()
        try:
            handle_id = await self._modal.spawn(
                function,
                args=args,
                kwargs={} if kwargs is None else kwargs,
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
            raise
        return attached

    async def reconcile_provider_call(
        self,
        provider_call_id: UUID,
        *,
        encode_result: Callable[[Any], Any],
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
        if observation.kind == ModalCallObservationKind.RUNNING:
            updated = self.repository.mark_provider_call_running(
                provider_call_id,
                now=now,
            )
        elif observation.kind == ModalCallObservationKind.SUCCEEDED:
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
            updated = self.repository.record_provider_call_result(
                provider_call_id,
                result_envelope=envelope,
                now=now,
            )
        elif observation.kind == ModalCallObservationKind.FAILED:
            updated = self.repository.fail_provider_call(
                provider_call_id,
                message=observation.message or "Modal function failed",
                now=now,
            )
        elif observation.kind == ModalCallObservationKind.CANCELLED:
            updated = self.repository.cancel_provider_call(
                provider_call_id,
                message=observation.message or "Modal function was cancelled",
                now=now,
            )
        else:
            updated = self.repository.mark_provider_call_state_unknown(
                provider_call_id,
                message=observation.message or "Modal call state was inconclusive",
                now=now,
            )
        self._checkpoint_state()
        return updated

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
            call = self.repository.get_provider_call(provider_call_id)
            if call.provider_call_handle_id is None:
                self.repository.mark_provider_cancellation_unknown(
                    provider_call_id,
                    message="Provider Call has no attached cancellation handle",
                    now=now,
                )
                self._checkpoint_state()
                continue
            try:
                await self._modal.cancel(call.provider_call_handle_id)
            except Exception as error:
                self.repository.mark_provider_cancellation_unknown(
                    provider_call_id,
                    message=f"Modal cancellation was inconclusive: {error}",
                    now=now,
                )
                self._checkpoint_state()
        return self.repository.get_run(execution_run_id)

    def _checkpoint_state(self) -> None:
        replacement = self._checkpoint()
        if replacement is not None:
            self.repository = replacement
