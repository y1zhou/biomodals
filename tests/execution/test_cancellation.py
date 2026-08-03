"""Explicit Run cancellation and ownership-reconciliation tests."""

# ruff: noqa: D101, D102, D103, D107, S106

from biomodals.execution import (
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    RunStatusReason,
    TaskStatus,
    WorkStatusReason,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.execution.runtime import ExecutionRuntime

from .provider_call_helpers import (
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
)


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
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
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


def test_cancellation_closes_task_after_call_success_before_publication() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
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
    repository.record_provider_call_result(
        claim.call.provider_call_id,
        result_envelope={"tasks": {"seed-0": "outputs/seed-0"}},
        now=112,
    )
    assert (
        repository.get_task(RUN_ID, "inference", "seed-0").status == TaskStatus.RUNNING
    )

    provider_call_ids = repository.request_run_cancellation(RUN_ID, now=120)
    completed = repository.finalize_run_from_results(RUN_ID, now=121)

    assert provider_call_ids == ()
    assert (
        repository.get_task(RUN_ID, "inference", "seed-0").status
        == TaskStatus.CANCELLED
    )
    assert repository.get_node(RUN_ID, "inference").status == NodeStatus.CANCELLED
    assert completed.status == RunStatus.CANCELLED


def test_call_success_after_cancellation_does_not_reopen_publication_work() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
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
    repository.request_run_cancellation(RUN_ID, now=120)

    repository.record_provider_call_result(
        claim.call.provider_call_id,
        result_envelope={"tasks": {"seed-0": "outputs/seed-0"}},
        now=121,
    )

    assert (
        repository.get_task(RUN_ID, "inference", "seed-0").status
        == TaskStatus.CANCELLED
    )


def test_ambiguous_cancellation_preserves_task_and_call_slots() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
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


def test_mixed_unknown_calls_preserve_durable_cancellation_intent() -> None:
    repository = create_repository(task_count=2)
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    claims = []
    for index in range(2):
        claim = repository.preclaim_fixed_batch(
            RUN_ID,
            "inference",
            (f"seed-{index}",),
            submission_token=f"batch-{index}",
            binding=GPU_BINDING,
            compatibility_key="gpu",
            now=110 + index,
        )
        assert claim is not None
        repository.attach_provider_call(
            claim.call.provider_call_id,
            provider_call_handle_id=f"fc-{index}",
            now=115 + index,
        )
        claims.append(claim)

    repository.request_run_cancellation(RUN_ID, now=120)
    repository.mark_provider_cancellation_unknown(
        claims[0].call.provider_call_id,
        message="cancellation response was lost",
        now=121,
    )
    repository.mark_provider_call_state_unknown(
        claims[1].call.provider_call_id,
        message="provider state was unavailable",
        now=122,
    )
    repository.cancel_provider_call(
        claims[0].call.provider_call_id,
        message="cancelled",
        now=123,
    )
    repository.cancel_provider_call(
        claims[1].call.provider_call_id,
        message="cancelled",
        now=124,
    )

    run = repository.get_run(RUN_ID)
    assert run.status == RunStatus.CANCEL_REQUESTED
    assert run.status_reason is None


def test_result_pruning_waits_for_conclusive_provider_cancellation() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
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

    requested = runtime.request_provider_call_cancellation(
        claim.call.provider_call_id,
        now=120,
    )

    assert requested.status == ProviderCallStatus.ATTACHED
    assert driver.cancelled == ["fc-123"]
    assert (
        repository.get_task(RUN_ID, "inference", "seed-0").status == TaskStatus.RUNNING
    )

    cancelled = runtime.reconcile_provider_call(
        claim.call.provider_call_id,
        encode_result=lambda value: value,
        result_already_satisfied=True,
        now=121,
    )
    task = repository.get_task(RUN_ID, "inference", "seed-0")

    assert cancelled.status == ProviderCallStatus.CANCELLED
    assert task.status == TaskStatus.CANCELLED
    assert task.status_reason == WorkStatusReason.RESULT_ALREADY_SATISFIED


def test_result_pruning_without_an_attached_handle_preserves_unknown_ownership() -> (
    None
):
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
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
    runtime = ExecutionRuntime(
        repository,
        modal_driver=CancelDriver(),
        checkpoint=lambda: None,
    )

    call = runtime.request_provider_call_cancellation(
        claim.call.provider_call_id,
        now=120,
    )
    run = repository.get_run(RUN_ID)

    assert call.status == ProviderCallStatus.OUTCOME_UNKNOWN
    assert run.status == RunStatus.STATE_UNKNOWN
    assert run.status_reason == RunStatusReason.CANCELLATION_OUTCOME_UNKNOWN
    assert (
        repository.get_task(RUN_ID, "inference", "seed-0").status == TaskStatus.RUNNING
    )
