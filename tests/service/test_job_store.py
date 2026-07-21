"""Private durable-job repository contracts."""

# ruff: noqa: D101,D102,D103,S105,S106

import sqlite3
import weakref
from pathlib import Path
from uuid import UUID

import pytest

from biomodals.service.auth import AuthService
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
)
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobNotCancellableError,
    JobState,
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


def test_job_lifecycle_lock_registry_releases_unused_locks() -> None:
    locks = JobLifecycleLocks()
    job_id = UUID("11111111-1111-4111-8111-111111111111")
    first = locks.for_job(job_id)
    reference = weakref.ref(first)

    assert locks.for_job(job_id) is first
    del first

    assert reference() is None
    assert locks.for_job(job_id) is not None


@pytest.mark.parametrize("version", [0, 8])
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
        configuration=configuration,
        now=1_800_000_000,
    )


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
    assert preserved.submission_token is None


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
    store.mark_submitted(
        job.job_id,
        modal_call_id="fc-123",
        provider_operation="prepare_tpr_gpu",
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

    claimed = store.claim_submission(
        job.job_id,
        run_name=run_name,
        submission_token="first",
        now=100,
        lease_seconds=20,
    )
    concurrent = store.claim_submission(
        job.job_id,
        run_name=run_name,
        submission_token="second",
        now=110,
        lease_seconds=20,
    )
    expired = store.claim_submission(
        job.job_id,
        run_name=run_name,
        submission_token="retry",
        now=120,
        lease_seconds=20,
    )

    assert claimed is not None
    assert claimed.run_name == run_name
    assert concurrent is None
    assert expired is None

    store.release_submission(
        job.job_id,
        submission_token="first",
        now=120,
    )
    retry = store.claim_submission(
        job.job_id,
        run_name=run_name,
        submission_token="retry",
        now=120,
        lease_seconds=20,
    )

    assert retry is not None
    assert retry.submission_token == "retry"


def test_cancel_during_spawn_keeps_call_attached_for_reconciliation(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    run_name = f"api-{job.job_id.hex}"
    store.claim_submission(
        job.job_id,
        run_name=run_name,
        submission_token="submitter",
        now=100,
    )

    store.request_cancel(alice, job.job_id, now=101)
    attached = store.mark_submitted(
        job.job_id,
        modal_call_id="fc-live",
        provider_operation="prepare_tpr_gpu",
        run_name=run_name,
        submission_token="submitter",
        now=102,
    )

    assert attached.state == JobState.CANCEL_REQUESTED
    assert attached.modal_call_id == "fc-live"


def test_job_atomically_advances_one_active_provider_operation(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    attached = store.mark_submitted(
        job.job_id,
        modal_call_id="fc-prepare",
        provider_operation="prepare_tpr_cpu",
        run_name=f"simulation-{job.job_id.hex}",
        now=100,
    )
    claimed = store.claim_provider_advance(
        job.job_id,
        expected_modal_call_id="fc-prepare",
        submission_token="next-stage",
        now=101,
    )

    advanced = store.replace_provider_call(
        job.job_id,
        expected_modal_call_id="fc-prepare",
        modal_call_id="fc-production",
        provider_operation="production_run_cpu",
        submission_token="next-stage",
        now=102,
    )

    assert attached.provider_operation == "prepare_tpr_cpu"
    assert claimed is not None
    assert claimed.stage_history[-1].completed_at == 101
    assert advanced.modal_call_id == "fc-production"
    assert advanced.provider_operation == "production_run_cpu"
    assert advanced.submission_lease_until is None
    assert [
        (stage.provider_operation, stage.started_at, stage.completed_at)
        for stage in advanced.stage_history
    ] == [
        ("prepare_tpr_cpu", 100, 101),
        ("production_run_cpu", 102, None),
    ]

    finalizing = store.set_job_state(
        job.job_id,
        JobState.FINALIZING,
        now=103,
    )

    assert [
        (stage.provider_operation, stage.started_at, stage.completed_at)
        for stage in finalizing.stage_history
    ] == [
        ("prepare_tpr_cpu", 100, 101),
        ("production_run_cpu", 102, 103),
        ("result_packaging", 103, None),
    ]


def test_provider_advance_lease_requires_explicit_release_before_retry(
    tmp_path: Path,
) -> None:
    store, alice, _bob = make_store(tmp_path)
    job = admit(store, alice, key="11111111-1111-4111-8111-111111111111").job
    store.mark_submitted(
        job.job_id,
        modal_call_id="fc-prepare",
        provider_operation="prepare_tpr_cpu",
        run_name=f"simulation-{job.job_id.hex}",
        now=100,
    )

    claimed = store.claim_provider_advance(
        job.job_id,
        expected_modal_call_id="fc-prepare",
        submission_token="first",
        now=101,
        lease_seconds=20,
    )
    expired = store.claim_provider_advance(
        job.job_id,
        expected_modal_call_id="fc-prepare",
        submission_token="second",
        now=121,
        lease_seconds=20,
    )

    assert claimed is not None
    assert expired is None

    store.release_provider_advance(
        job.job_id,
        expected_modal_call_id="fc-prepare",
        submission_token="first",
        now=121,
    )
    retried = store.claim_provider_advance(
        job.job_id,
        expected_modal_call_id="fc-prepare",
        submission_token="second",
        now=121,
        lease_seconds=20,
    )

    assert retried is not None
    assert retried.submission_token == "second"
