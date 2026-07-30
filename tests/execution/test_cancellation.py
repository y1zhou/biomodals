"""Explicit Run cancellation and ownership-reconciliation tests."""

# ruff: noqa: D101, D102, D103, D107, S106

from biomodals.execution import (
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    RunStatusReason,
    TaskStatus,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.execution.runtime import ExecutionRuntime

from .provider_call_helpers import GPU_BINDING, RUN_ID, create_repository


class CancelDriver:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.cancel_error: Exception | None = None
        self.observation = ModalCallObservation(ModalCallObservationKind.CANCELLED)

    def resolve(self, binding):
        raise AssertionError("cancellation never resolves a new function")

    def spawn(self, function, *, args, kwargs):
        raise AssertionError("cancellation never spawns")

    def observe(self, provider_call_handle_id):
        return self.observation

    def cancel(self, provider_call_handle_id):
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancelled.append(provider_call_handle_id)


def test_pending_run_cancels_without_provider_work() -> None:
    repository = create_repository(task_count=1)
    # The helper started its only Node, but its Task has no execution owner.

    calls = repository.request_run_cancellation(RUN_ID, now=110)
    run = repository.finalize_run_from_results(RUN_ID, now=111)

    assert calls == ()
    assert run.status == RunStatus.CANCELLED
    assert repository.get_node(RUN_ID, "inference").status == NodeStatus.CANCELLED


def test_cancel_request_waits_for_attached_call_confirmation() -> None:
    repository = create_repository(task_count=1)
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None
    repository.attach_provider_call(
        claim.call.provider_call_id,
        provider_call_handle_id="fc-123",
        now=111,
    )
    driver = CancelDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )

    requested = runtime.cancel_run(RUN_ID, now=120)

    assert requested.status == RunStatus.CANCEL_REQUESTED
    assert driver.cancelled == ["fc-123"]
    assert (
        repository.get_task(RUN_ID, "inference", "seed-0").status == TaskStatus.RUNNING
    )

    runtime.reconcile_provider_call(
        claim.call.provider_call_id,
        encode_result=lambda value: value,
        now=121,
    )
    repository.reconcile_node_tasks(RUN_ID, "inference", now=122)
    completed = repository.finalize_run_from_results(RUN_ID, now=123)

    assert completed.status == RunStatus.CANCELLED
    assert (
        repository.get_provider_call(claim.call.provider_call_id).status
        == ProviderCallStatus.CANCELLED
    )


def test_ambiguous_cancellation_preserves_task_and_call_slots() -> None:
    repository = create_repository(task_count=1)
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None
    repository.attach_provider_call(
        claim.call.provider_call_id,
        provider_call_handle_id="fc-123",
        now=111,
    )
    driver = CancelDriver()
    driver.cancel_error = RuntimeError("cancel response lost")
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )

    run = runtime.cancel_run(RUN_ID, now=120)

    call = repository.get_provider_call(claim.call.provider_call_id)
    task = repository.get_task(RUN_ID, "inference", "seed-0")
    assert run.status == RunStatus.STATE_UNKNOWN
    assert run.status_reason == RunStatusReason.CANCELLATION_OUTCOME_UNKNOWN
    assert call.status == ProviderCallStatus.STATE_UNKNOWN
    assert task.status == TaskStatus.RUNNING
    assert repository.active_provider_call_counts(RUN_ID).total == 1
