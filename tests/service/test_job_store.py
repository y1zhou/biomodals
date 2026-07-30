"""Private durable-job repository contracts."""

# ruff: noqa: D101,D102,D103,S105,S106

import sqlite3
import weakref
from pathlib import Path
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
)
from biomodals.service.auth import AuthService
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
)
from biomodals.service.store import (
    IdempotencyConflictError,
    InitialModalOperation,
    JobLimitExceededError,
    JobNotCancellableError,
    JobOperationExecutor,
    JobOperationState,
    JobState,
    JobStateResolutionError,
    JobStateUnknownReason,
    ServiceStore,
    UserNotFoundError,
)


def owner(auth: AuthService, email: str) -> UUID:
    link = auth.create_user(
        email,
        display_name=email.partition("@")[0],
        is_admin=not auth.store.list_users(),
    )
    token = link.url.partition("#token=")[2]
    return auth.set_password(token, "correct horse battery staple").principal.user_id


def make_store(tmp_path: Path) -> tuple[ServiceStore, UUID, UUID]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    auth = AuthService(store, frontend_url="https://biomodals.internal")
    return store, owner(auth, "alice@example.com"), owner(auth, "bob@example.com")


def test_readiness_never_recreates_or_accepts_an_empty_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    store = ServiceStore(path)
    store.initialize()
    store.check_ready()

    path.unlink()
    with pytest.raises(RuntimeError, match="database is unavailable"):
        store.check_ready()
    assert not path.exists()

    path.touch()
    with pytest.raises(RuntimeError, match="schema is unavailable"):
        store.check_ready()


def test_job_operations_are_the_only_durable_stage_ledger(tmp_path: Path) -> None:
    store, _alice, _bob = make_store(tmp_path)

    with sqlite3.connect(store.path) as conn:
        job_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(jobs)")}
        operation_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(job_operations)")
        }

    assert {
        "modal_call_id",
        "operation",
        "stage_history_json",
        "submission_token",
        "submission_lease_until",
    }.isdisjoint(job_columns)
    assert {
        "job_id",
        "operation",
        "ordinal",
        "executor",
        "modal_call_id",
        "state",
        "submission_token",
        "submission_lease_until",
        "started_at",
        "completed_at",
    } == operation_columns


def test_service_database_embeds_host_invariant_execution_tables(
    tmp_path: Path,
) -> None:
    store, _alice, _bob = make_store(tmp_path)

    with sqlite3.connect(store.path) as conn:
        job_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(jobs)")}
        execution_foreign_keys = conn.execute(
            "PRAGMA foreign_key_list(execution_runs)"
        ).fetchall()

    assert "execution_run_id" in job_columns
    assert {str(row[2]) for row in execution_foreign_keys} <= {"execution_runs"}


def test_service_execution_repository_uses_the_service_transaction(
    tmp_path: Path,
) -> None:
    store, _alice, _bob = make_store(tmp_path)
    execution_run_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")

    with store.execution_repository() as repository:
        repository.create_run(
            execution_run_id=execution_run_id,
            plan=ExecutionPlan(
                workload_name="test",
                nodes=(NodePlan(node_key="work"),),
            ),
            deployment=DeploymentIdentity("production", "test-app", 1),
            max_active_provider_calls=1,
            max_active_gpu_provider_calls=0,
            now=10,
        )

    with store.execution_repository() as repository:
        assert repository.get_run(execution_run_id).plan.workload_name == "test"


def test_job_lifecycle_lock_registry_releases_unused_locks() -> None:
    locks = JobLifecycleLocks()
    job_id = UUID("11111111-1111-4111-8111-111111111111")
    first = locks.for_job(job_id)
    reference = weakref.ref(first)

    assert locks.for_job(job_id) is first
    del first

    assert reference() is None
    assert locks.for_job(job_id) is not None


@pytest.mark.parametrize("version", [0, 9, 10, 11])
def test_initialize_never_rewrites_an_existing_unsupported_database(
    tmp_path: Path,
    version: int,
) -> None:
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE legacy_state (value TEXT)")
        conn.execute(f"PRAGMA user_version = {version}")
    before = path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match=rf"Unsupported pre-release service database version {version}",
    ):
        ServiceStore(path).initialize()

    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def admit(
    store: ServiceStore,
    owner_user_id: UUID,
    *,
    key: str,
    request_hash: str = "a" * 64,
    workload: str = "gromacs",
    user_limit: int = 2,
    workload_limit: int = 10,
    global_limit: int = 20,
    new_job_id: UUID | None = None,
    initial_operation: InitialModalOperation | None = None,
    artifact_request_sha256: str | None = None,
):
    store.update_user(
        owner_user_id,
        active_job_limit=user_limit,
        now=1_799_999_999,
    )
    configuration = JobAdmissionConfiguration(
        workload=workload,
        modal_environment=DatabaseOverridableSetting("production", False),
        modal_app_name=DatabaseOverridableSetting("Gromacs", False),
        modal_app_version=DatabaseOverridableSetting(1, False),
        workload_active_job_limit=DatabaseOverridableSetting(
            workload_limit,
            False,
        ),
        global_active_job_limit=DatabaseOverridableSetting(global_limit, False),
    )
    return store.admit_job(
        owner_user_id=owner_user_id,
        display_name="protein · 2026-07-16",
        idempotency_key=key,
        request_hash=request_hash,
        parameters_json='{"simulation_time_ns":5}',
        artifact_request_sha256=artifact_request_sha256,
        configuration=configuration,
        now=1_800_000_000,
        new_job_id=new_job_id,
        initial_operation=initial_operation,
    )


def attach_operation(
    store: ServiceStore,
    job_id: UUID,
    *,
    operation: str,
    modal_call_id: str,
    run_name: str | None = None,
    now: int = 100,
):
    token = f"claim-{operation}"
    claimed = store.claim_modal_operation(
        job_id,
        operation=operation,
        submission_token=token,
        run_name=run_name,
        now=now,
    )
    assert claimed is not None
    return store.attach_modal_call(
        job_id,
        operation=operation,
        modal_call_id=modal_call_id,
        submission_token=token,
        now=now,
    )


def test_admission_atomically_persists_the_initial_modal_lease(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job_id = UUID("11111111-1111-4111-8111-111111111111")
    run_name = f"simulation-{job_id.hex}"
    initial = InitialModalOperation(
        operation="prepare_tpr_gpu",
        run_name=run_name,
        submission_token="initial-submitter",
        lease_seconds=120,
    )

    admitted = admit(
        store,
        alice,
        key="11111111-1111-4111-8111-111111111111",
        new_job_id=job_id,
        initial_operation=initial,
        artifact_request_sha256="b" * 64,
    )

    assert admitted.created is True
    assert admitted.job.job_id == job_id
    assert admitted.job.run_name == run_name
    assert admitted.job.artifact_request_sha256 == "b" * 64
    assert len(admitted.job.operations) == 1
    operation = admitted.job.operations[0]
    assert operation.operation == "prepare_tpr_gpu"
    assert operation.state == JobOperationState.SUBMITTING
    assert operation.submission_token == "initial-submitter"
    assert operation.submission_lease_until == 1_800_000_120

    reopened = ServiceStore(store.path)
    reopened.initialize()
    persisted = reopened.get_job(alice, job_id)
    assert persisted == admitted.job


def test_idempotency_is_scoped_to_owner_and_payload(tmp_path: Path) -> None:
    store, alice, bob = make_store(tmp_path)

    first = admit(store, alice, key="11111111-1111-4111-8111-111111111111")
    replay = admit(store, alice, key="11111111-1111-4111-8111-111111111111")
    other_owner = admit(store, bob, key="11111111-1111-4111-8111-111111111111")

    assert first.created is True
    assert replay.created is False
    assert replay.job.job_id == first.job.job_id
    assert other_owner.job.job_id != first.job.job_id
    with pytest.raises(IdempotencyConflictError):
        admit(
            store,
            alice,
            key="11111111-1111-4111-8111-111111111111",
            request_hash="b" * 64,
        )


def test_disabled_user_cannot_replay_a_queued_admission(tmp_path: Path) -> None:
    store, _alice, bob = make_store(tmp_path)
    key = "11111111-1111-4111-8111-111111111111"
    first = admit(store, bob, key=key)

    store.update_user(bob, active=False, now=1_800_000_001)

    with pytest.raises(UserNotFoundError, match="Enabled User"):
        admit(store, bob, key=key)
    preserved = store.get_job(bob, first.job.job_id)
    assert preserved is not None
    assert preserved.state == JobState.QUEUED
    assert preserved.operations == ()


def test_user_active_limit_is_transactional_across_workloads(tmp_path: Path) -> None:
    store, alice, bob = make_store(tmp_path)
    admit(store, alice, key="11111111-1111-4111-8111-111111111111")
    admit(
        store,
        alice,
        key="22222222-2222-4222-8222-222222222222",
        workload="another-tool",
    )

    with pytest.raises(JobLimitExceededError):
        admit(store, alice, key="33333333-3333-4333-8333-333333333333")

    assert admit(store, bob, key="33333333-3333-4333-8333-333333333333").created


def test_tool_and_global_active_limits_span_users_and_workloads(tmp_path: Path) -> None:
    store, alice, bob = make_store(tmp_path)
    admit(
        store,
        alice,
        key="11111111-1111-4111-8111-111111111111",
        workload_limit=1,
    )

    with pytest.raises(JobLimitExceededError, match="Tool"):
        admit(
            store,
            bob,
            key="22222222-2222-4222-8222-222222222222",
            workload_limit=1,
        )

    admit(
        store,
        bob,
        key="33333333-3333-4333-8333-333333333333",
        workload="another-tool",
        global_limit=2,
    )
    with pytest.raises(JobLimitExceededError, match="Global"):
        admit(
            store,
            bob,
            key="44444444-4444-4444-8444-444444444444",
            workload="third-tool",
            user_limit=3,
            global_limit=2,
        )


def test_job_captures_modal_configuration_at_admission(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)

    job = store.admit_job(
        owner_user_id=alice,
        display_name="Simulation",
        idempotency_key="11111111-1111-4111-8111-111111111111",
        request_hash="a" * 64,
        parameters_json="{}",
        configuration=JobAdmissionConfiguration(
            workload="gromacs",
            modal_environment=DatabaseOverridableSetting("department", False),
            modal_app_name=DatabaseOverridableSetting("GromacsDeployed", False),
            modal_app_version=DatabaseOverridableSetting(17, False),
            workload_active_job_limit=DatabaseOverridableSetting(2, False),
            global_active_job_limit=DatabaseOverridableSetting(10, False),
        ),
        now=1,
    ).job

    assert job.modal_environment == "department"
    assert job.modal_app_name == "GromacsDeployed"
    assert job.modal_app_version == 17


def test_job_queries_never_cross_owner_boundaries(tmp_path: Path) -> None:
    store, alice, bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job

    assert store.get_job(alice, job.job_id) == job
    assert store.get_job(bob, job.job_id) is None
    assert [item.job_id for item in store.list_jobs(alice)] == [job.job_id]
    assert store.list_jobs(bob) == []


def test_cancellation_is_preserved_as_a_state_transition(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    attach_operation(
        store,
        job.job_id,
        modal_call_id="fc-123",
        operation="prepare_tpr_gpu",
        run_name="api-123",
    )

    requested = store.request_cancel(alice, job.job_id, now=1_800_000_001)
    repeated = store.request_cancel(alice, job.job_id, now=1_800_000_002)

    assert requested.state == JobState.CANCEL_REQUESTED
    assert repeated.state == JobState.CANCEL_REQUESTED
    store.set_job_state(job.job_id, JobState.CANCELLED, now=1_800_000_003)
    with pytest.raises(JobNotCancellableError):
        store.request_cancel(alice, job.job_id, now=1_800_000_004)


def test_submission_lease_requires_explicit_release_before_retry(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    run_name = f"api-{job.job_id.hex}"

    claimed = store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=run_name,
        submission_token="first",
        now=100,
        lease_seconds=20,
    )
    concurrent = store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=run_name,
        submission_token="second",
        now=110,
        lease_seconds=20,
    )
    expired = store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=run_name,
        submission_token="retry",
        now=120,
        lease_seconds=20,
    )

    assert claimed is not None
    current = store.get_job(alice, job.job_id)
    assert current is not None
    assert current.run_name == run_name
    assert concurrent is None
    assert expired is None

    store.release_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        submission_token="first",
        now=120,
    )
    retry = store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=run_name,
        submission_token="retry",
        now=120,
        lease_seconds=20,
    )

    assert retry is not None
    assert retry.submission_token == "retry"


def test_disabled_owner_cannot_reclaim_a_released_initial_operation(
    tmp_path: Path,
) -> None:
    store, _alice, bob = make_store(tmp_path)
    job = admit(
        store,
        bob,
        key="11111111-1111-4111-8111-111111111111",
    ).job
    store.update_user(bob, active=False, now=101)

    replay_claim = store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=f"simulation-{job.job_id.hex}",
        submission_token="replay",
        now=102,
        require_enabled_owner=True,
    )
    lifecycle_claim = store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=f"simulation-{job.job_id.hex}",
        submission_token="reconciler",
        now=103,
    )

    assert replay_claim is None
    assert lifecycle_claim is not None


def test_cancel_during_spawn_keeps_call_attached_for_reconciliation(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    run_name = f"api-{job.job_id.hex}"
    store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=run_name,
        submission_token="submitter",
        now=100,
    )

    store.request_cancel(alice, job.job_id, now=101)
    attached = store.attach_modal_call(
        job.job_id,
        modal_call_id="fc-live",
        operation="prepare_tpr_gpu",
        submission_token="submitter",
        now=102,
    )

    assert attached.state == JobState.CANCEL_REQUESTED
    assert attached.operations[0].modal_call_id == "fc-live"


def test_job_tracks_parallel_operations(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    attached = attach_operation(
        store,
        job.job_id,
        modal_call_id="fc-prepare",
        operation="prepare_tpr_cpu",
        run_name=f"simulation-{job.job_id.hex}",
        now=100,
    )
    store.record_operation_outcome(
        job.job_id,
        operation="prepare_tpr_cpu",
        expected_modal_call_id="fc-prepare",
        outcome=JobOperationState.COMPLETED,
        now=101,
    )
    operations = (
        ("collect_traj_stats:nvt_", "fc-nvt"),
        ("collect_traj_stats:npt_", "fc-npt"),
        ("production_run_cpu", "fc-production"),
    )
    for operation, modal_call_id in operations:
        claimed = store.claim_modal_operation(
            job.job_id,
            operation=operation,
            submission_token=operation,
            now=102,
        )
        assert claimed is not None
        store.attach_modal_call(
            job.job_id,
            operation=operation,
            modal_call_id=modal_call_id,
            submission_token=operation,
            now=102,
        )

    assert attached.operations[0].operation == "prepare_tpr_cpu"
    assert [call.operation for call in store.list_operations(job.job_id)] == [
        "prepare_tpr_cpu",
        "collect_traj_stats:nvt_",
        "collect_traj_stats:npt_",
        "production_run_cpu",
    ]
    running = store.get_job(alice, job.job_id)
    assert running is not None
    assert [
        (stage.operation, stage.started_at, stage.completed_at)
        for stage in running.stage_history
    ] == [
        ("prepare_tpr_cpu", 100, 101),
        ("collect_traj_stats:nvt_", 102, None),
        ("collect_traj_stats:npt_", 102, None),
        ("production_run_cpu", 102, None),
    ]

    for operation, modal_call_id in operations:
        store.record_operation_outcome(
            job.job_id,
            operation=operation,
            expected_modal_call_id=modal_call_id,
            outcome=JobOperationState.COMPLETED,
            now=103,
        )

    finalizing = store.set_job_state(
        job.job_id,
        JobState.FINALIZING,
        now=104,
    )

    assert [
        (stage.operation, stage.started_at, stage.completed_at)
        for stage in finalizing.stage_history
    ] == [
        ("prepare_tpr_cpu", 100, 101),
        ("collect_traj_stats:nvt_", 102, 103),
        ("collect_traj_stats:npt_", 102, 103),
        ("production_run_cpu", 102, 103),
        ("result_packaging", 104, None),
    ]
    assert finalizing.operations[-1].executor == JobOperationExecutor.LOCAL

    completed = store.complete_job(
        job.job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path="api-results/result.zip",
        result_filename="result.zip",
        result_size_bytes=1,
        result_sha256="a" * 64,
        result_archive_schema_version=1,
        now=105,
    )

    assert all(
        operation.state == JobOperationState.COMPLETED
        for operation in completed.operations
    )
    assert completed.stage_history[-1].completed_at == 105


def test_operation_lease_requires_explicit_release_before_retry(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    attach_operation(
        store,
        job.job_id,
        modal_call_id="fc-prepare",
        operation="prepare_tpr_cpu",
        run_name=f"simulation-{job.job_id.hex}",
        now=100,
    )

    claimed = store.claim_modal_operation(
        job.job_id,
        operation="collect_traj_stats:nvt_",
        submission_token="first",
        now=101,
        lease_seconds=20,
    )
    expired = store.claim_modal_operation(
        job.job_id,
        operation="collect_traj_stats:nvt_",
        submission_token="second",
        now=121,
        lease_seconds=20,
    )

    assert claimed is not None
    assert expired is None

    store.release_operation(
        job.job_id,
        operation="collect_traj_stats:nvt_",
        submission_token="first",
        now=121,
    )
    retried = store.claim_modal_operation(
        job.job_id,
        operation="collect_traj_stats:nvt_",
        submission_token="second",
        now=121,
        lease_seconds=20,
    )

    assert retried is not None
    assert retried.submission_token == "second"


def test_state_unknown_consumes_capacity_until_admin_resolution(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(
        store,
        alice,
        key="11111111-1111-4111-8111-111111111111",
        user_limit=1,
    ).job
    store.claim_modal_operation(
        job.job_id,
        operation="prepare_tpr_gpu",
        run_name=f"simulation-{job.job_id.hex}",
        submission_token="uncertain-submission",
        now=100,
    )

    uncertain = store.mark_state_unknown(
        job.job_id,
        reason=JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN,
        now=120,
    )

    assert uncertain.state == JobState.STATE_UNKNOWN
    assert uncertain.state_unknown_at == 120
    assert (
        uncertain.state_unknown_reason
        == JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN
    )
    assert uncertain.operations[0].submission_token is None
    assert uncertain.operations[0].submission_lease_until is None
    assert uncertain.operations[0].state == JobOperationState.STATE_UNKNOWN
    assert store.count_active_jobs() == 1
    assert store.list_reconcilable_jobs() == []
    with pytest.raises(JobLimitExceededError, match="User"):
        admit(
            store,
            alice,
            key="22222222-2222-4222-8222-222222222222",
            user_limit=1,
        )
    assert (
        store.block_job(
            job.job_id,
            category="modal_unavailable",
            now=125,
            next_retry_at=130,
        ).state
        == JobState.STATE_UNKNOWN
    )
    assert (
        store.complete_job(
            job.job_id,
            state=JobState.SUCCEEDED,
            result_volume_name="Gromacs-outputs",
            result_volume_path="result.zip",
            result_filename="result.zip",
            result_size_bytes=1,
            result_sha256="a" * 64,
            result_archive_schema_version=1,
            now=125,
        ).state
        == JobState.STATE_UNKNOWN
    )

    resolved = store.resolve_state_unknown(job.job_id, now=130)

    assert resolved.state == JobState.FAILED
    assert resolved.error_code == "compute_failed"
    assert resolved.error_message == (
        "An administrator could not confirm the remote compute state."
    )
    assert resolved.completed_at == 130
    assert resolved.state_unknown_at == 120
    assert (
        resolved.state_unknown_reason
        == JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN
    )
    assert store.count_active_jobs() == 0


def test_only_state_unknown_can_use_admin_resolution(tmp_path: Path) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job

    with pytest.raises(JobStateResolutionError, match="queued"):
        store.resolve_state_unknown(job.job_id, now=120)
