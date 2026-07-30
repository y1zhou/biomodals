"""Caller-driven execution runtime fault-boundary tests."""

# ruff: noqa: D101, D102, D103, D107, S106

from dataclasses import dataclass

import pytest

from biomodals.execution import ProviderCallStatus
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
    ModalDefiniteSubmissionError,
    ModalSubmissionOutcomeUnknownError,
)
from biomodals.execution.runtime import ExecutionRuntime
from biomodals.execution.scheduler import ProviderCallCandidate

from .provider_call_helpers import GPU_BINDING, RUN_ID, create_repository


@dataclass
class FakeResolvedFunction:
    name: str


class FakeModalDriver:
    def __init__(self) -> None:
        self.resolve_count = 0
        self.spawn_count = 0
        self.observe_count = 0
        self.cancelled: list[str] = []
        self.spawn_error: Exception | None = None
        self.observation = ModalCallObservation(ModalCallObservationKind.RUNNING)

    def resolve(self, binding):
        self.resolve_count += 1
        return FakeResolvedFunction(binding.function_name)

    def spawn(self, function, *, args, kwargs):
        self.spawn_count += 1
        if self.spawn_error is not None:
            raise self.spawn_error
        return f"fc-{function.name}"

    def observe(self, provider_call_handle_id):
        self.observe_count += 1
        return self.observation

    def cancel(self, provider_call_handle_id):
        self.cancelled.append(provider_call_handle_id)


def _candidate() -> ProviderCallCandidate:
    return ProviderCallCandidate(
        candidate_key="inference:0",
        node_key="inference",
        node_ordinal=0,
        task_keys=("seed-0",),
        task_ordinal=0,
        binding=GPU_BINDING,
        compatibility_key="af3",
        depth=0,
        unblocking_span=0,
    )


def test_preclaim_checkpoint_precedes_spawn_and_replay_never_spawns_twice() -> None:
    repository = create_repository(task_count=1)
    driver = FakeModalDriver()
    checkpoints: list[int] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append(driver.spawn_count),
    )

    first = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=110,
    )
    duplicate = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=111,
    )

    assert first is not None
    assert first.status == ProviderCallStatus.ATTACHED
    assert duplicate == first
    assert checkpoints == [0, 1]
    assert driver.spawn_count == 1


def test_resolution_failure_happens_before_durable_preclaim() -> None:
    repository = create_repository(task_count=1)

    class BrokenResolver(FakeModalDriver):
        def resolve(self, binding):
            raise RuntimeError("deployment unavailable")

    runtime = ExecutionRuntime(
        repository,
        modal_driver=BrokenResolver(),
        checkpoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="deployment unavailable"):
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            kwargs={"seed": 0},
            now=110,
        )

    assert repository.list_provider_calls(RUN_ID) == ()


def test_failed_preclaim_checkpoint_never_reaches_spawn_and_recovers_unknown() -> None:
    repository = create_repository(task_count=1)
    driver = FakeModalDriver()

    def fail_checkpoint():
        raise RuntimeError("checkpoint failed")

    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=fail_checkpoint,
    )

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            kwargs={"seed": 0},
            now=110,
        )

    assert driver.spawn_count == 0
    recovered = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    ).reconcile_provider_call(
        repository.list_provider_calls(RUN_ID)[0].provider_call_id,
        encode_result=lambda value: value,
        now=120,
    )
    assert recovered.status == ProviderCallStatus.OUTCOME_UNKNOWN
    assert driver.spawn_count == 0


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            ModalDefiniteSubmissionError("rejected"),
            ProviderCallStatus.FAILED,
        ),
        (
            ModalSubmissionOutcomeUnknownError("response lost"),
            ProviderCallStatus.OUTCOME_UNKNOWN,
        ),
    ],
)
def test_spawn_failure_is_durably_classified_without_reauthorization(
    error: Exception,
    expected_status: ProviderCallStatus,
) -> None:
    repository = create_repository(task_count=1)
    driver = FakeModalDriver()
    driver.spawn_error = error
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )

    with pytest.raises(type(error)):
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            kwargs={"seed": 0},
            now=110,
        )

    call = repository.list_provider_calls(RUN_ID)[0]
    assert call.status == expected_status
    replay = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=120,
    )
    assert replay == call
    assert driver.spawn_count == 1


def test_recovery_collects_attached_call_once_then_replays_durable_envelope() -> None:
    repository = create_repository(task_count=1)
    driver = FakeModalDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )
    call = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=110,
    )
    assert call is not None
    driver.observation = ModalCallObservation(
        ModalCallObservationKind.SUCCEEDED,
        result={"path": "/outputs/seed-0"},
    )

    completed = runtime.reconcile_provider_call(
        call.provider_call_id,
        encode_result=lambda result: {"tasks": {"seed-0": result}},
        now=120,
    )
    replay = runtime.reconcile_provider_call(
        call.provider_call_id,
        encode_result=lambda result: {"tasks": {"seed-0": result}},
        now=121,
    )

    assert completed.status == ProviderCallStatus.SUCCEEDED
    assert replay == completed
    assert driver.observe_count == 1
