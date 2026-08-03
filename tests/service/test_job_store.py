"""Service metadata and embedded execution-repository contracts."""

# ruff: noqa: D103,S105

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from biomodals.execution import (
    AvailabilityStatus,
    ExecutionPlan,
    NodePlan,
    RunStatus,
    RunStatusReason,
)
from biomodals.service.auth import AuthService
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
)
from biomodals.service.store import (
    IdempotencyConflictError,
    JobCursorError,
    JobLimitExceededError,
    JobNotCancellableError,
    JobState,
    ServiceStore,
    UserNotFoundError,
)


def make_store(tmp_path: Path) -> tuple[ServiceStore, UUID, UUID]:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.initialize()
    auth = AuthService(store, frontend_url="https://biomodals.internal")
    for email, admin in (
        ("alice@example.com", True),
        ("bob@example.com", False),
    ):
        link = auth.create_user(
            email,
            display_name=email.partition("@")[0].title(),
            is_admin=admin,
        )
        auth.set_password(
            link.url.partition("#token=")[2],
            "correct horse battery staple",
        )
    alice = store.get_user_by_email("alice@example.com")
    bob = store.get_user_by_email("bob@example.com")
    assert alice is not None and bob is not None
    return store, alice.user_id, bob.user_id


def configuration(
    workload: str = "gromacs",
    *,
    workload_limit: int = 10,
    global_limit: int = 20,
) -> JobAdmissionConfiguration:
    return JobAdmissionConfiguration(
        workload=workload,
        modal_environment=DatabaseOverridableSetting("production", False),
        modal_app_name=DatabaseOverridableSetting("Gromacs", False),
        modal_app_version=DatabaseOverridableSetting(7, False),
        workload_active_job_limit=DatabaseOverridableSetting(
            workload_limit,
            False,
        ),
        global_active_job_limit=DatabaseOverridableSetting(global_limit, False),
    )


def admit(
    store: ServiceStore,
    owner_user_id: UUID,
    *,
    key: str,
    workload: str = "gromacs",
    request_hash: str = "a" * 64,
    user_limit: int = 2,
    workload_limit: int = 10,
    global_limit: int = 20,
    job_id: UUID | None = None,
    run_id: UUID | None = None,
    input_content: bytes | None = None,
):
    store.update_user(
        owner_user_id,
        active_job_limit=user_limit,
        now=99,
    )
    job_id = job_id or uuid4()
    run_id = run_id or uuid4()
    plan = ExecutionPlan(
        workload_name=workload,
        workload_run_key=f"run-{job_id.hex}",
        nodes=(NodePlan(node_key="compute"),),
        scientific_payload={"request_hash": request_hash},
    )
    return store.admit_job(
        owner_user_id=owner_user_id,
        display_name="Protein simulation",
        idempotency_key=key,
        request_hash=request_hash,
        parameters_json='{"simulation_time_ns":5}',
        artifact_request_sha256="b" * 64,
        configuration=configuration(
            workload,
            workload_limit=workload_limit,
            global_limit=global_limit,
        ),
        execution_plan=plan,
        execution_run_id=run_id,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=1,
        now=100,
        new_job_id=job_id,
        input_content=input_content,
    )


def fail_run(store: ServiceStore, run_id: UUID, *, now: int = 200) -> None:
    with store.execution_repository() as repository:
        repository.transition_run(
            run_id,
            RunStatus.FAILED,
            reason=RunStatusReason.REQUIRED_WORK_FAILED,
            message="test failure",
            now=now,
        )


def succeed_run(store: ServiceStore, run_id: UUID, *, now: int = 200) -> None:
    with store.execution_repository() as repository:
        repository.record_node_result_observation(
            run_id,
            "compute",
            AvailabilityStatus.AVAILABLE,
            now=now,
        )
        repository.finalize_run_from_results(run_id, now=now)


def test_service_schema_embeds_execution_without_a_second_stage_ledger(
    tmp_path: Path,
) -> None:
    store, _alice, _bob = make_store(tmp_path)

    with sqlite3.connect(store.path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        job_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(jobs)")}

    assert "execution_runs" in tables
    assert "execution_nodes" in tables
    assert "execution_provider_calls" in tables
    assert "job_operations" not in tables
    assert "state" not in job_columns
    assert "execution_run_id" in job_columns


def test_readiness_checks_existing_schema_without_creating_state(
    tmp_path: Path,
) -> None:
    missing = ServiceStore(tmp_path / "missing" / "service.sqlite3")
    with pytest.raises(RuntimeError, match="unavailable"):
        missing.check_ready()
    assert not missing.path.exists()

    store, _alice, _bob = make_store(tmp_path)
    store.check_ready()


def test_service_rejects_a_stale_embedded_execution_schema(tmp_path: Path) -> None:
    store, _alice, _bob = make_store(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE execution_schema SET version = 3 WHERE singleton = 1"
        )

    with pytest.raises(RuntimeError, match="execution schema version 3"):
        store.initialize()
    with pytest.raises(RuntimeError, match="execution schema is unavailable"):
        store.check_ready()


def test_admission_atomically_links_job_run_and_staged_input(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)
    job_id = UUID("11111111-1111-4111-8111-111111111111")
    run_id = UUID("22222222-2222-4222-8222-222222222222")

    admitted = admit(
        store,
        alice,
        key="one",
        job_id=job_id,
        run_id=run_id,
        input_content=b"ATOM\n",
    )

    assert admitted.created is True
    assert admitted.job.execution_run_id == run_id
    assert admitted.job.state == JobState.QUEUED
    assert admitted.job.modal_environment == "production"
    assert admitted.job.modal_app_name == "Gromacs"
    assert admitted.job.modal_app_version == 7
    assert admitted.job.run_name == f"run-{job_id.hex}"
    assert store.load_job_input(job_id) == b"ATOM\n"
    with store.execution_repository() as repository:
        assert repository.get_run(run_id).plan.workload_name == "gromacs"


def test_admission_rolls_back_job_and_run_together(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)
    run_id = uuid4()
    job_id = uuid4()
    with pytest.raises(ValueError, match="workload"):
        store.admit_job(
            owner_user_id=alice,
            display_name="Mismatch",
            idempotency_key="mismatch",
            request_hash="a" * 64,
            parameters_json="{}",
            configuration=configuration("gromacs"),
            execution_plan=ExecutionPlan(
                workload_name="other",
                nodes=(NodePlan(node_key="compute"),),
            ),
            execution_run_id=run_id,
            max_active_provider_calls=1,
            max_active_gpu_provider_calls=0,
            now=100,
            new_job_id=job_id,
        )
    assert store.get_job_by_id(job_id) is None
    with store.execution_repository() as repository:
        with pytest.raises(LookupError):
            repository.get_run(run_id)


def test_idempotency_is_owner_scoped_and_never_creates_a_second_run(
    tmp_path: Path,
) -> None:
    store, alice, bob = make_store(tmp_path)
    first = admit(store, alice, key="same")
    replay = admit(store, alice, key="same")
    other_owner = admit(store, bob, key="same")

    assert replay.created is False
    assert replay.job.job_id == first.job.job_id
    assert replay.job.execution_run_id == first.job.execution_run_id
    assert other_owner.job.job_id != first.job.job_id
    with pytest.raises(IdempotencyConflictError):
        admit(store, alice, key="same", request_hash="c" * 64)


def test_disabled_owner_cannot_replay_admission(tmp_path: Path) -> None:
    store, _alice, bob = make_store(tmp_path)
    admit(store, bob, key="one")
    store.update_user(bob, active=False, now=101)

    with pytest.raises(UserNotFoundError):
        admit(store, bob, key="one")


def test_active_limits_are_derived_from_execution_runs(tmp_path: Path) -> None:
    store, alice, bob = make_store(tmp_path)
    first = admit(store, alice, key="one", user_limit=1)
    with pytest.raises(JobLimitExceededError, match="User"):
        admit(store, alice, key="two", user_limit=1)

    assert first.job.execution_run_id is not None
    fail_run(store, first.job.execution_run_id)
    second = admit(store, alice, key="two", user_limit=1)
    assert second.created is True

    with pytest.raises(JobLimitExceededError, match="Tool"):
        admit(store, bob, key="three", workload_limit=1)


def test_job_queries_and_cursor_never_cross_owner_boundaries(tmp_path: Path) -> None:
    store, alice, bob = make_store(tmp_path)
    first = admit(store, alice, key="one")
    second = admit(store, alice, key="two")
    hidden = admit(store, bob, key="three")

    assert store.get_job(alice, hidden.job.job_id) is None
    assert {job.job_id for job in store.list_jobs(alice)} == {
        first.job.job_id,
        second.job.job_id,
    }
    page = store.list_jobs_page(alice, limit=1)
    assert len(page.jobs) == 1
    assert page.next_cursor is not None
    with pytest.raises(JobCursorError):
        store.list_jobs_page(alice, limit=1, cursor=hidden.job.job_id)


def test_cancellation_audit_does_not_mirror_execution_state(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)
    admitted = admit(store, alice, key="one")
    assert admitted.job.execution_run_id is not None

    requested = store.request_cancel(alice, admitted.job.job_id, now=150)
    with store.execution_repository() as repository:
        run = repository.get_run(admitted.job.execution_run_id)

    assert requested.state == JobState.QUEUED
    assert requested.cancel_requested_at == 150
    assert run.status == RunStatus.PENDING

    fail_run(store, admitted.job.execution_run_id)
    with pytest.raises(JobNotCancellableError):
        store.request_cancel(alice, admitted.job.job_id, now=200)


def test_result_metadata_and_delivery_block_are_service_owned(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)
    admitted = admit(store, alice, key="one")
    assert admitted.job.execution_run_id is not None
    succeed_run(store, admitted.job.execution_run_id)

    finalizing = store.get_job_by_id(admitted.job.job_id)
    assert finalizing is not None and finalizing.state == JobState.FINALIZING
    completed = store.complete_job(
        admitted.job.job_id,
        state=JobState.PARTIAL,
        result_volume_name="outputs",
        result_volume_path="result.zip",
        result_filename="result.zip",
        result_size_bytes=100,
        result_sha256="d" * 64,
        result_archive_schema_version=1,
        now=201,
    )
    assert completed.state == JobState.PARTIAL
    blocked = store.block_job(
        admitted.job.job_id,
        category="result_integrity",
        previous_state=JobState.PARTIAL,
        now=202,
        next_retry_at=302,
    )
    assert blocked.state == JobState.BLOCKED
    assert blocked.result_previous_state == JobState.PARTIAL


def test_temporary_inputs_can_be_removed_after_remote_staging(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)
    admitted = admit(
        store,
        alice,
        key="one",
        input_content=b"ATOM\n",
    )

    assert store.load_job_input(admitted.job.job_id) == b"ATOM\n"
    store.clear_job_input(admitted.job.job_id)
    assert store.load_job_input(admitted.job.job_id) is None
