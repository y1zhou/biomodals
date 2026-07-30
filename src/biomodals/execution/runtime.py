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
    ExecutionRunRecord,
    ProviderBinding,
    ProviderCallRecord,
    ProviderCallStatus,
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


class ExecutionRuntime:
    """Coordinate repository checkpoints with exactly one Modal side effect."""

    def __init__(
        self,
        repository: SqliteExecutionRepository,
        *,
        modal_driver: _ModalDriver,
        checkpoint: Callable[[], None],
    ) -> None:
        """Bind host-owned state, Modal operations, and its durability boundary."""
        self.repository = repository
        self._modal = modal_driver
        self._checkpoint = checkpoint

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
        self._checkpoint()
        try:
            handle_id = self._modal.spawn(
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
            self._checkpoint()
            raise
        except ModalSubmissionOutcomeUnknownError as error:
            self.repository.mark_submission_outcome_unknown(
                preclaim.call.provider_call_id,
                message=str(error),
                now=now,
            )
            self._checkpoint()
            raise
        try:
            attached = self.repository.attach_provider_call(
                preclaim.call.provider_call_id,
                provider_call_handle_id=handle_id,
                now=now,
            )
            self._checkpoint()
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
            self._checkpoint()
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
                self._checkpoint()
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
        self._checkpoint()
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
        self._checkpoint()
        for provider_call_id in provider_call_ids:
            call = self.repository.get_provider_call(provider_call_id)
            if call.provider_call_handle_id is None:
                self.repository.mark_provider_cancellation_unknown(
                    provider_call_id,
                    message="Provider Call has no attached cancellation handle",
                    now=now,
                )
                self._checkpoint()
                continue
            try:
                self._modal.cancel(call.provider_call_handle_id)
            except Exception as error:
                self.repository.mark_provider_cancellation_unknown(
                    provider_call_id,
                    message=f"Modal cancellation was inconclusive: {error}",
                    now=now,
                )
                self._checkpoint()
        return self.repository.get_run(execution_run_id)
