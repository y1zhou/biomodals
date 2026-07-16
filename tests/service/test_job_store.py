"""Private durable-job repository contracts."""

# ruff: noqa: D101,D102,D103,S105,S106

from pathlib import Path
from uuid import UUID

import pytest

from biomodals.service.auth import AuthService
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobNotCancellableError,
    JobState,
    ServiceStore,
)


def owner(auth: AuthService, email: str) -> UUID:
    link = auth.create_user(email, display_name=email.partition("@")[0])
    token = link.partition("#token=")[2]
    return auth.set_password(token, "correct horse battery staple").user_id


def make_store(tmp_path: Path) -> tuple[ServiceStore, UUID, UUID]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    auth = AuthService(store, frontend_url="https://biomodals.internal")
    return store, owner(auth, "alice@example.com"), owner(auth, "bob@example.com")


def admit(
    store: ServiceStore,
    owner_user_id: UUID,
    *,
    key: str,
    request_hash: str = "a" * 64,
):
    return store.admit_job(
        owner_user_id=owner_user_id,
        workload="gromacs",
        display_name="protein · 2026-07-16",
        idempotency_key=key,
        request_hash=request_hash,
        parameters_json='{"simulation_time_ns":5}',
        active_limit=2,
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


def test_active_limit_is_transactional_per_owner_and_workload(tmp_path: Path) -> None:
    store, alice, bob = make_store(tmp_path)
    admit(store, alice, key="11111111-1111-4111-8111-111111111111")
    admit(store, alice, key="22222222-2222-4222-8222-222222222222")

    with pytest.raises(JobLimitExceededError):
        admit(store, alice, key="33333333-3333-4333-8333-333333333333")

    assert admit(store, bob, key="33333333-3333-4333-8333-333333333333").created


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
    store.mark_submitted(job.job_id, modal_call_id="fc-123", run_name="api-123")

    requested = store.request_cancel(alice, job.job_id, now=1_800_000_001)
    repeated = store.request_cancel(alice, job.job_id, now=1_800_000_002)

    assert requested.state == JobState.CANCEL_REQUESTED
    assert repeated.state == JobState.CANCEL_REQUESTED
    store.set_job_state(job.job_id, JobState.CANCELLED, now=1_800_000_003)
    with pytest.raises(JobNotCancellableError):
        store.request_cancel(alice, job.job_id, now=1_800_000_004)


def test_submission_lease_is_stable_retryable_and_single_writer(
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
    retry = store.claim_submission(
        job.job_id,
        run_name=run_name,
        submission_token="retry",
        now=120,
        lease_seconds=20,
    )

    assert claimed is not None
    assert claimed.run_name == run_name
    assert concurrent is None
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
        run_name=run_name,
        submission_token="submitter",
        now=102,
    )

    assert attached.state == JobState.CANCEL_REQUESTED
    assert attached.modal_call_id == "fc-live"
