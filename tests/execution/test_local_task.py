"""Coordinator-local Task recovery tests."""

# ruff: noqa: D103

from biomodals.execution import (
    ActiveProviderCallCounts,
    AvailabilityStatus,
    RunStatus,
    RunStatusReason,
    TaskStatus,
)

from .provider_call_helpers import RUN_ID, create_repository


def test_cache_success_never_acquires_local_ownership() -> None:
    repository = create_repository(task_count=1)
    repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-0",
        AvailabilityStatus.AVAILABLE,
        now=110,
    )

    assert not repository.acquire_local_task(
        RUN_ID,
        "inference",
        "seed-0",
        now=111,
    )
    assert repository.active_provider_call_counts(RUN_ID) == ActiveProviderCallCounts(
        total=0, gpu=0
    )


def test_missing_local_task_can_reenter_after_interruption_with_same_identity() -> None:
    repository = create_repository(task_count=1)

    assert repository.acquire_local_task(
        RUN_ID,
        "inference",
        "seed-0",
        now=110,
    )
    first = repository.get_task(RUN_ID, "inference", "seed-0")
    assert first.status == TaskStatus.RUNNING
    assert first.local_owned
    assert first.provider_call_id is None

    # A replacement coordinator validates before re-entering the same operation.
    repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-0",
        AvailabilityStatus.MISSING,
        now=120,
    )
    assert repository.acquire_local_task(
        RUN_ID,
        "inference",
        "seed-0",
        now=121,
    )

    completed = repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-0",
        AvailabilityStatus.AVAILABLE,
        now=130,
    )
    assert completed.status == TaskStatus.SUCCEEDED
    assert not repository.acquire_local_task(
        RUN_ID,
        "inference",
        "seed-0",
        now=131,
    )


def test_unknown_local_publication_suspends_and_conclusive_failure_never_replays() -> (
    None
):
    repository = create_repository(task_count=1)
    repository.acquire_local_task(
        RUN_ID,
        "inference",
        "seed-0",
        now=110,
    )

    repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-0",
        AvailabilityStatus.UNKNOWN,
        now=120,
    )
    run = repository.get_run(RUN_ID)
    assert run.status == RunStatus.SUSPENDED
    assert run.status_reason == RunStatusReason.RESULT_VALIDATION_UNKNOWN

    repository.resume_run(RUN_ID, now=121)
    failed = repository.fail_task(
        RUN_ID,
        "inference",
        "seed-0",
        message="archive publication failed",
        now=130,
    )
    assert failed.status == TaskStatus.FAILED
    assert not repository.acquire_local_task(
        RUN_ID,
        "inference",
        "seed-0",
        now=131,
    )
