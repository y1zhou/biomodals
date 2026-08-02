"""Run-scoped coordinator loop tests."""

# ruff: noqa: D103, S106

import sqlite3
from contextlib import contextmanager

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

from .provider_call_helpers import (
    CPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
)


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


def test_active_poll_cycles_do_not_cross_the_durability_boundary() -> None:
    repository = create_repository(task_count=1)
    checkpoints: list[RunStatus] = []
    advance_count = 0

    def advance_once() -> None:
        nonlocal advance_count
        advance_count += 1
        if advance_count < 3:
            return
        repository.record_task_result_observation(
            RUN_ID,
            "inference",
            "seed-0",
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
        sleep=lambda _: None,
        poll_interval_seconds=0,
    )

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert advance_count == 3
    assert checkpoints == [RunStatus.SUCCEEDED]


def test_coordinator_releases_host_lock_between_scheduling_cycles() -> None:
    repository = create_repository(task_count=1)
    lock_active = False
    cycles = 0
    events: list[str] = []

    @contextmanager
    def synchronize():
        nonlocal lock_active
        assert lock_active is False
        lock_active = True
        events.append("enter")
        try:
            yield
        finally:
            lock_active = False
            events.append("exit")

    def advance_once() -> None:
        nonlocal cycles
        assert lock_active is True
        cycles += 1
        if cycles == 2:
            repository.record_task_result_observation(
                RUN_ID,
                "inference",
                "seed-0",
                AvailabilityStatus.AVAILABLE,
                now=110,
            )
            repository.reconcile_node_tasks(RUN_ID, "inference", now=111)
            repository.finalize_run_from_results(RUN_ID, now=112)

    def sleep(_seconds: float) -> None:
        assert lock_active is False
        events.append("sleep")

    snapshot = drive_execution_run(
        repository,
        RUN_ID,
        advance_once=advance_once,
        checkpoint=lambda: None,
        sleep=sleep,
        poll_interval_seconds=0,
        synchronize=synchronize,
    )

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert cycles == 2
    assert events == ["enter", "exit", "sleep", "enter", "exit", "enter", "exit"]


def test_coordinator_accepts_a_reopened_volume_repository(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(ledger_path)
    repository = create_repository(connection=connection, task_count=1)
    connection.commit()
    active_repository = repository
    active_connection = connection

    def advance_once() -> None:
        active_repository.record_task_result_observation(
            RUN_ID,
            "inference",
            "seed-0",
            AvailabilityStatus.AVAILABLE,
            now=110,
        )
        active_repository.reconcile_node_tasks(RUN_ID, "inference", now=111)
        active_repository.finalize_run_from_results(RUN_ID, now=112)

    def checkpoint():
        nonlocal active_connection, active_repository
        active_connection.commit()
        active_connection.close()
        active_connection = sqlite3.connect(ledger_path)
        active_repository = type(repository)(active_connection)
        return active_repository

    snapshot = drive_execution_run(
        repository,
        RUN_ID,
        advance_once=advance_once,
        checkpoint=checkpoint,
        sleep=lambda _: None,
        poll_interval_seconds=0,
    )

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert active_repository is not repository


def test_coordinator_refreshes_repository_after_an_interleaved_checkpoint(
    tmp_path,
) -> None:
    """A callback may replace the SQLite connection while the driver sleeps."""
    ledger_path = tmp_path / "ledger.sqlite3"
    active_connection = sqlite3.connect(ledger_path)
    active_repository = create_repository(
        connection=active_connection,
        task_count=1,
    )
    active_connection.commit()
    initial_repository = active_repository
    cycles = 0

    def advance_once() -> None:
        nonlocal cycles
        cycles += 1
        if cycles == 1:
            return
        active_repository.record_task_result_observation(
            RUN_ID,
            "inference",
            "seed-0",
            AvailabilityStatus.AVAILABLE,
            now=110,
        )
        active_repository.reconcile_node_tasks(RUN_ID, "inference", now=111)
        active_repository.finalize_run_from_results(RUN_ID, now=112)

    def replace_repository(_seconds: float) -> None:
        nonlocal active_connection, active_repository
        active_connection.close()
        active_connection = sqlite3.connect(ledger_path)
        active_repository = type(initial_repository)(active_connection)

    snapshot = drive_execution_run(
        initial_repository,
        RUN_ID,
        advance_once=advance_once,
        checkpoint=lambda: active_repository,
        current_repository=lambda: active_repository,
        sleep=replace_repository,
        poll_interval_seconds=0,
    )

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert cycles == 2
    assert active_repository is not initial_repository


def test_application_error_suspends_without_replacing_attached_work() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=CPU_BINDING,
        compatibility_key="search",
    )
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


def test_application_error_uses_repository_reopened_by_checkpoint(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(ledger_path)
    repository = create_repository(connection=connection, task_count=1)
    connection.commit()
    active_repository = repository
    active_connection = connection

    def advance_once() -> None:
        nonlocal active_connection, active_repository
        active_connection.commit()
        active_connection.close()
        active_connection = sqlite3.connect(ledger_path)
        active_repository = type(repository)(active_connection)
        raise RuntimeError("workflow adapter crashed after checkpoint")

    def checkpoint():
        nonlocal active_connection, active_repository
        active_connection.commit()
        active_connection.close()
        active_connection = sqlite3.connect(ledger_path)
        active_repository = type(repository)(active_connection)
        return active_repository

    with pytest.raises(
        RuntimeError,
        match="workflow adapter crashed after checkpoint",
    ):
        drive_execution_run(
            repository,
            RUN_ID,
            advance_once=advance_once,
            checkpoint=checkpoint,
            now=lambda: 120,
            sleep=lambda _: None,
            poll_interval_seconds=0,
        )

    run = active_repository.get_run(RUN_ID)
    assert run.status == RunStatus.SUSPENDED
    assert run.status_reason == RunStatusReason.COORDINATOR_ERROR
    assert run.status_message == "workflow adapter crashed after checkpoint"


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
        reconcile_once=lambda: None,
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


def test_explicit_resume_reconciles_unknown_provider_ownership_in_place() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=CPU_BINDING,
        compatibility_key="search",
    )
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
    repository.attach_provider_call(
        preclaim.call.provider_call_id,
        provider_call_handle_id="fc-live",
        now=111,
    )
    repository.mark_provider_call_state_unknown(
        preclaim.call.provider_call_id,
        message="Modal state lookup was inconclusive",
        now=112,
    )
    checkpoints: list[RunStatus] = []

    def reconcile_once() -> None:
        repository.mark_provider_call_running(
            preclaim.call.provider_call_id,
            now=120,
        )

    resumed = resume_execution_run(
        repository,
        RUN_ID,
        reconcile_once=reconcile_once,
        checkpoint=lambda: checkpoints.append(repository.get_run(RUN_ID).status),
        now=120,
    )

    assert resumed.status == RunStatus.RUNNING
    assert resumed.status_reason is None
    assert len(repository.list_provider_calls(RUN_ID)) == 1
    assert checkpoints == [RunStatus.RUNNING]


def test_explicit_resume_preserves_unresolved_provider_ownership() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=CPU_BINDING,
        compatibility_key="search",
    )
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
    repository.mark_submission_outcome_unknown(
        preclaim.call.provider_call_id,
        message="Modal submission response was lost",
        now=111,
    )

    resumed = resume_execution_run(
        repository,
        RUN_ID,
        reconcile_once=lambda: None,
        checkpoint=lambda: None,
        now=120,
    )

    assert resumed.status == RunStatus.STATE_UNKNOWN
    assert resumed.status_reason == RunStatusReason.SUBMISSION_OUTCOME_UNKNOWN
    assert len(repository.list_provider_calls(RUN_ID)) == 1


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
