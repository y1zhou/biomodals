"""SQLite-backed pull-worker assignment tests."""

# ruff: noqa: D103, S106

import pytest

from biomodals.execution import (
    AvailabilityStatus,
    DispatchMode,
    ResultProvenance,
    TaskStatus,
)

from .provider_call_helpers import (
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_pull_policy,
)


def _admit_workers(repository, count: int):
    persist_pull_policy(
        repository,
        binding=GPU_BINDING,
        compatibility_key="af3-seeds",
        claim_capacity=2,
    )
    claims = []
    for index in range(count):
        claim = repository.preclaim_pull_worker(
            RUN_ID,
            "inference",
            submission_token=f"worker-{index}",
            binding=GPU_BINDING,
            compatibility_key="af3-seeds",
            claim_capacity=2,
            now=110 + index,
        )
        assert claim is not None
        repository.attach_provider_call(
            claim.call.provider_call_id,
            provider_call_handle_id=f"fc-worker-{index}",
            now=120 + index,
        )
        repository.mark_provider_call_running(
            claim.call.provider_call_id,
            now=130 + index,
        )
        claims.append(claim)
    return claims


def test_pull_worker_count_is_derived_from_unfinished_tasks_and_claim_capacity() -> (
    None
):
    repository = create_repository(
        task_count=5,
        max_active_provider_calls=5,
        max_active_gpu_provider_calls=5,
    )

    claims = _admit_workers(repository, 3)
    excess = repository.preclaim_pull_worker(
        RUN_ID,
        "inference",
        submission_token="worker-3",
        binding=GPU_BINDING,
        compatibility_key="af3-seeds",
        claim_capacity=2,
        now=140,
    )

    assert all(claim.call.dispatch_mode == DispatchMode.PULL_WORKER for claim in claims)
    assert excess is None


def test_claim_response_is_checkpointed_idempotent_and_ordered() -> None:
    repository = create_repository(
        task_count=5,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=3,
    )
    first, second, third = _admit_workers(repository, 3)

    first_claim = repository.claim_pull_tasks(
        first.call.provider_call_id,
        request_id="first-claim",
        capacity=2,
        now=140,
    )
    duplicate = repository.claim_pull_tasks(
        first.call.provider_call_id,
        request_id="first-claim",
        capacity=2,
        now=141,
    )
    second_claim = repository.claim_pull_tasks(
        second.call.provider_call_id,
        request_id="second-claim",
        capacity=2,
        now=142,
    )
    third_claim = repository.claim_pull_tasks(
        third.call.provider_call_id,
        request_id="third-claim",
        capacity=2,
        now=143,
    )
    empty_after_race = repository.claim_pull_tasks(
        second.call.provider_call_id,
        request_id="second-empty",
        capacity=2,
        now=144,
    )

    assert [assignment.task_key for assignment in first_claim.assignments] == [
        "seed-0",
        "seed-1",
    ]
    assert duplicate == first_claim
    assert [assignment.task_key for assignment in second_claim.assignments] == [
        "seed-2",
        "seed-3",
    ]
    assert [assignment.task_key for assignment in third_claim.assignments] == ["seed-4"]
    assert empty_after_race.assignments == ()

    tasks = repository.list_tasks(RUN_ID, "inference")
    assert all(task.status == TaskStatus.RUNNING for task in tasks)
    assert tasks[0].worker_provider_call_id == first.call.provider_call_id
    assert tasks[2].worker_provider_call_id == second.call.provider_call_id
    assert tasks[4].worker_provider_call_id == third.call.provider_call_id


def test_worker_can_claim_after_spawn_before_call_attachment() -> None:
    repository = create_repository(task_count=1)
    persist_pull_policy(
        repository,
        binding=GPU_BINDING,
        compatibility_key="af3-seeds",
        claim_capacity=1,
    )
    worker = repository.preclaim_pull_worker(
        RUN_ID,
        "inference",
        submission_token="worker-starting",
        binding=GPU_BINDING,
        compatibility_key="af3-seeds",
        claim_capacity=1,
        now=110,
    )
    assert worker is not None

    claim = repository.claim_pull_tasks(
        worker.call.provider_call_id,
        request_id="startup-race",
        capacity=1,
        now=111,
    )

    assert [assignment.task_key for assignment in claim.assignments] == ["seed-0"]


def test_claim_request_conflicts_and_failed_owner_never_reassigns_tasks() -> None:
    repository = create_repository(task_count=3)
    first, second = _admit_workers(repository, 2)
    repository.claim_pull_tasks(
        first.call.provider_call_id,
        request_id="claim",
        capacity=2,
        now=140,
    )

    with pytest.raises(ValueError, match="claim request ID was reused"):
        repository.claim_pull_tasks(
            second.call.provider_call_id,
            request_id="claim",
            capacity=1,
            now=141,
        )

    repository.fail_provider_call(
        first.call.provider_call_id,
        message="worker failed",
        now=150,
    )
    remaining = repository.claim_pull_tasks(
        second.call.provider_call_id,
        request_id="remaining",
        capacity=2,
        now=151,
    )

    assert [assignment.task_key for assignment in remaining.assignments] == ["seed-2"]
    assert {
        task.task_key: task.status
        for task in repository.list_tasks(RUN_ID, "inference")
    } == {
        "seed-0": TaskStatus.FAILED,
        "seed-1": TaskStatus.FAILED,
        "seed-2": TaskStatus.RUNNING,
    }


def test_worker_completion_report_is_idempotent_and_publication_driven() -> None:
    repository = create_repository(task_count=1)
    (worker,) = _admit_workers(repository, 1)
    repository.claim_pull_tasks(
        worker.call.provider_call_id,
        request_id="claim",
        capacity=1,
        now=140,
    )

    completed = repository.record_pull_task_completion(
        worker.call.provider_call_id,
        "seed-0",
        request_id="completion",
        observation=AvailabilityStatus.AVAILABLE,
        now=150,
    )
    duplicate = repository.record_pull_task_completion(
        worker.call.provider_call_id,
        "seed-0",
        request_id="completion",
        observation=AvailabilityStatus.AVAILABLE,
        now=151,
    )

    assert completed.status == TaskStatus.SUCCEEDED
    assert completed.result_provenance == ResultProvenance.CURRENT_RUN
    assert duplicate == completed

    with pytest.raises(ValueError, match="completion request ID was reused"):
        repository.record_pull_task_completion(
            worker.call.provider_call_id,
            "seed-0",
            request_id="completion",
            observation=AvailabilityStatus.MISSING,
            now=152,
        )


def test_successful_worker_fails_any_unreported_assignment() -> None:
    repository = create_repository(task_count=2)
    (worker,) = _admit_workers(repository, 1)
    repository.claim_pull_tasks(
        worker.call.provider_call_id,
        request_id="claim",
        capacity=2,
        now=140,
    )
    repository.record_pull_task_completion(
        worker.call.provider_call_id,
        "seed-0",
        request_id="complete-0",
        observation=AvailabilityStatus.AVAILABLE,
        now=145,
    )

    repository.record_provider_call_result(
        worker.call.provider_call_id,
        result_envelope={"claimed_tasks": 2},
        now=150,
    )

    assert {
        task.task_key: task.status
        for task in repository.list_tasks(RUN_ID, "inference")
    } == {
        "seed-0": TaskStatus.SUCCEEDED,
        "seed-1": TaskStatus.FAILED,
    }
