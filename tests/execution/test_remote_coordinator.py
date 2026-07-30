"""Run-scoped coordinator loop tests."""

# ruff: noqa: D103, S106

import pytest

from biomodals.execution import (
    AvailabilityStatus,
    ProviderCallStatus,
    RunStatus,
    RunStatusReason,
)
from biomodals.execution.coordinator import (
    drive_execution_run,
    resume_execution_run,
)

from .provider_call_helpers import CPU_BINDING, RUN_ID, create_repository


class CoordinatorInterrupted(BaseException):
    """Model infrastructure loss outside application exception handling."""


def test_detached_coordinator_loop_reaches_terminal_without_client_polling() -> None:
    repository = create_repository(task_count=2)
    checkpoints: list[RunStatus] = []
    advance_count = 0

    def advance_once() -> None:
        nonlocal advance_count
        advance_count += 1
        for task in repository.list_tasks(RUN_ID, "inference"):
            repository.record_task_result_observation(
                RUN_ID,
                "inference",
                task.task_key,
                AvailabilityStatus.AVAILABLE,
                now=110,
            )
        repository.reconcile_node_tasks(RUN_ID, "inference", now=111)
        repository.finalize_run_from_results(RUN_ID, now=112)

    snapshot = drive_execution_run(
        repository,
        RUN_ID,
        advance_once=advance_once,
        checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
        now=lambda: 120,
        sleep=lambda _: None,
        poll_interval_seconds=0,
    )

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert advance_count == 1
    assert checkpoints == [RunStatus.SUCCEEDED]

    reopened = drive_execution_run(
        repository,
        RUN_ID,
        advance_once=advance_once,
        checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
        now=lambda: 130,
        sleep=lambda _: None,
        poll_interval_seconds=0,
    )
    assert reopened.run.status == RunStatus.SUCCEEDED
    assert advance_count == 1


def test_application_error_suspends_without_replacing_attached_work() -> None:
    repository = create_repository(task_count=1)
    preclaim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="call-once",
        binding=CPU_BINDING,
        compatibility_key="search",
        now=110,
    )
    assert preclaim is not None
    attached = repository.attach_provider_call(
        preclaim.call.provider_call_id,
        provider_call_handle_id="fc-live",
        now=111,
    )
    checkpoints: list[RunStatus] = []

    with pytest.raises(RuntimeError, match="workflow adapter crashed"):
        drive_execution_run(
            repository,
            RUN_ID,
            advance_once=lambda: (_ for _ in ()).throw(
                RuntimeError("workflow adapter crashed")
            ),
            checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
            now=lambda: 120,
            sleep=lambda _: None,
            poll_interval_seconds=0,
        )

    run = repository.get_run(RUN_ID)
    assert run.status == RunStatus.SUSPENDED
    assert run.status_reason == RunStatusReason.COORDINATOR_ERROR
    assert run.status_message == "workflow adapter crashed"
    assert repository.get_provider_call(attached.provider_call_id).status == (
        ProviderCallStatus.ATTACHED
    )
    assert checkpoints == [RunStatus.SUSPENDED]


def test_explicit_resume_drives_the_same_run_without_retrying_implicitly() -> None:
    repository = create_repository(task_count=1)
    checkpoints: list[RunStatus] = []

    with pytest.raises(RuntimeError, match="stop once"):
        drive_execution_run(
            repository,
            RUN_ID,
            advance_once=lambda: (_ for _ in ()).throw(RuntimeError("stop once")),
            checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
            now=lambda: 120,
            sleep=lambda _: None,
            poll_interval_seconds=0,
        )

    assert repository.get_run(RUN_ID).status == RunStatus.SUSPENDED
    resumed = resume_execution_run(
        repository,
        RUN_ID,
        checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
        now=130,
    )
    assert resumed.status == RunStatus.RUNNING

    def complete() -> None:
        repository.record_task_result_observation(
            RUN_ID,
            "inference",
            "seed-0",
            AvailabilityStatus.AVAILABLE,
            now=140,
        )
        repository.reconcile_node_tasks(RUN_ID, "inference", now=141)
        repository.finalize_run_from_results(RUN_ID, now=142)

    snapshot = drive_execution_run(
        repository,
        RUN_ID,
        advance_once=complete,
        checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
        now=lambda: 150,
        sleep=lambda _: None,
        poll_interval_seconds=0,
    )

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert checkpoints == [
        RunStatus.SUSPENDED,
        RunStatus.RUNNING,
        RunStatus.SUCCEEDED,
    ]


def test_hard_coordinator_interruption_preserves_running_state() -> None:
    repository = create_repository(task_count=1)
    checkpoints: list[RunStatus] = []

    with pytest.raises(CoordinatorInterrupted):
        drive_execution_run(
            repository,
            RUN_ID,
            advance_once=lambda: (_ for _ in ()).throw(CoordinatorInterrupted()),
            checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
            now=lambda: 120,
            sleep=lambda _: None,
            poll_interval_seconds=0,
        )

    assert repository.get_run(RUN_ID).status == RunStatus.RUNNING
    assert checkpoints == []
