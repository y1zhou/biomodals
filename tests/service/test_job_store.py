"""Lean Service Job persistence contracts."""

# ruff: noqa: D103

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from biomodals.service.store import (
    DeletedSubmissionError,
    IdempotencyConflictError,
    JobNotCancellableError,
    JobNotDeletableError,
    JobNotFoundError,
    JobState,
    JobStateResolutionError,
    ServiceStore,
)

JOB_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _store(tmp_path: Path) -> tuple[ServiceStore, UUID]:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.initialize()
    user = store.create_user(
        email="admin@example.com",
        display_name="Admin",
        token_digest=b"setup",
        token_expires_at=100,
        now=1,
        is_admin=True,
        active_job_limit=10,
    )
    store.set_password_from_token(
        b"setup",
        password_hash="test",  # noqa: S106
        session_token_digest=b"session",
        csrf_digest=b"csrf",
        now=2,
        absolute_expires_at=1000,
    )
    return store, user.user_id


def _admit(store: ServiceStore, owner: UUID, *, digest: str = "a" * 64):
    return store.admit_job(
        owner_user_id=owner,
        tool="alphafold3",
        display_name="prediction",
        idempotency_key="idempotency",
        request_digest=digest,
        modal_environment="main",
        modal_app_name="AlphaFold3",
        modal_app_version=7,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=1,
        now=10,
        new_job_id=JOB_ID,
    )


def test_schema_contains_only_six_service_tables(tmp_path: Path) -> None:
    store, _owner = _store(tmp_path)
    with sqlite3.connect(store.path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {
        "users",
        "password_tokens",
        "sessions",
        "service_settings",
        "tool_settings",
        "jobs",
    }


@pytest.mark.parametrize("version", [8, 9])
def test_migration_preserves_jobs_and_authentication(
    tmp_path: Path, version: int
) -> None:
    store, owner = _store(tmp_path)
    original = _admit(store, owner).job
    with sqlite3.connect(store.path) as connection:
        if version == 8:
            connection.execute("ALTER TABLE jobs DROP COLUMN operation")
            connection.execute("ALTER TABLE jobs DROP COLUMN source_job_id")
        connection.execute("ALTER TABLE jobs DROP COLUMN deleted_at")
        connection.execute("ALTER TABLE jobs DROP COLUMN cleanup_completed_at")
        connection.execute("ALTER TABLE jobs DROP COLUMN cleanup_retry_at")
        connection.execute(f"PRAGMA user_version = {version}")
        before = connection.execute("SELECT * FROM sessions").fetchall()
    store.initialize()
    assert store.get_job_by_id(original.job_id) == original
    assert _admit(store, owner).created is False
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT * FROM sessions").fetchall() == before
        assert connection.execute("PRAGMA user_version").fetchone() == (10,)


def test_admission_replays_identity_without_execution_tables(tmp_path: Path) -> None:
    store, owner = _store(tmp_path)
    created = _admit(store, owner)
    replay = _admit(store, owner)

    assert created.created is True
    assert replay.created is False
    assert replay.job.job_id == JOB_ID
    assert replay.job.tool == "alphafold3"
    assert replay.job.root_function_call_id is None

    with pytest.raises(IdempotencyConflictError):
        _admit(store, owner, digest="b" * 64)


def test_deleted_predecessor_remains_available_for_publication_repair(
    tmp_path: Path,
) -> None:
    store, owner = _store(tmp_path)
    previous = _admit(store, owner).job
    store.fail_job(previous.job_id, error_code="test", error_message="test", now=11)
    store.delete_job(owner, previous.job_id, now=12)
    current = store.admit_job(
        owner_user_id=owner,
        tool="alphafold3",
        display_name="New explicit intent",
        idempotency_key="another-intent",
        request_digest=previous.request_digest,
        modal_environment="main",
        modal_app_name="AlphaFold3",
        modal_app_version=7,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=1,
        now=13,
    ).job
    predecessors = store.list_preceding_jobs_for_request(current.job_id)
    assert [job.job_id for job in predecessors] == [previous.job_id]
    assert predecessors[0].deleted_at == 12
    assert [job.job_id for job in store.list_jobs(owner)] == [current.job_id]


def test_unstaged_jobs_retain_queued_requests_across_tools(tmp_path: Path) -> None:
    store, owner = _store(tmp_path)
    gromacs_id = uuid4()
    store.admit_job(
        owner_user_id=owner,
        tool="gromacs",
        display_name="simulation",
        idempotency_key="gromacs-request",
        request_digest="b" * 64,
        modal_environment="main",
        modal_app_name="Gromacs",
        modal_app_version=7,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=1,
        now=10,
        new_job_id=gromacs_id,
    )
    _admit(store, owner)

    assert store.unstaged_job_ids() == {gromacs_id, JOB_ID}

    store.record_launch(gromacs_id, function_call_id="fc-test", now=11)
    assert store.unstaged_job_ids() == {JOB_ID}


@pytest.mark.parametrize("state", list(JobState))
def test_owner_deletion_eligibility_and_replay(tmp_path: Path, state: JobState) -> None:
    store, owner = _store(tmp_path)
    _admit(store, owner)
    store.replace_projection(JOB_ID, state=state, projection={}, observed_at=11)
    with pytest.raises(JobNotFoundError):
        store.delete_job(uuid4(), JOB_ID, now=12)
    if state not in {
        JobState.SUCCEEDED,
        JobState.PARTIAL,
        JobState.FAILED,
        JobState.CANCELLED,
    }:
        with pytest.raises(JobNotDeletableError):
            store.delete_job(owner, JOB_ID, now=12)
        assert store.get_job(owner, JOB_ID) is not None
        return
    store.delete_job(owner, JOB_ID, now=12)
    store.delete_job(owner, JOB_ID, now=13)
    assert store.get_job(owner, JOB_ID) is None
    assert store.list_jobs(owner) == []
    assert store.list_jobs_page(owner, limit=10).jobs == []
    assert store.list_jobs_page(owner, limit=10, cursor=JOB_ID).jobs == []
    tombstone = store.get_job_by_id(JOB_ID)
    assert tombstone is not None and tombstone.deleted_at == 12
    assert tombstone.state == state
    with pytest.raises(DeletedSubmissionError):
        _admit(store, owner)
    with pytest.raises(DeletedSubmissionError):
        store.find_idempotent_job(
            owner, tool="alphafold3", idempotency_key="idempotency"
        )
    with pytest.raises(JobNotFoundError):
        store.replace_projection(
            JOB_ID, state=JobState.RUNNING, projection={}, observed_at=14
        )
    with pytest.raises(JobNotFoundError):
        store.restore_result_cached(JOB_ID, now=14)
    assert [j.job_id for j in store.list_pending_deletions(now=14)] == [JOB_ID]
    store.record_deletion_cleanup(JOB_ID, completed=False, now=14)
    store.initialize()
    assert store.list_pending_deletions(now=73) == []
    assert len(store.list_pending_deletions(now=74)) == 1
    store.record_deletion_cleanup(JOB_ID, completed=True, now=75)
    assert store.list_pending_deletions(now=100) == []
    assert store.get_job_by_id(JOB_ID).cleanup_completed_at == 75


def test_projection_and_result_metadata_replace_atomically(tmp_path: Path) -> None:
    store, owner = _store(tmp_path)
    _admit(store, owner)
    store.record_launch(JOB_ID, function_call_id="fc-test", now=11)
    projection = {
        "stages": [
            {
                "code": "predict_structures",
                "label": "Predict structures",
                "started_at": 11,
                "ended_at": 12,
                "outcome": "completed",
                "task_counts": {"succeeded": 5},
            }
        ],
        "warnings": [],
    }
    finalizing = store.begin_finalization(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        projection=projection,
        now=12,
    )
    completed = store.complete_job(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        result_filename="prediction.tar.zst",
        result_media_type="application/zstd",
        result_size_bytes=123,
        result_sha256="c" * 64,
        result_archive_schema="alphafold3-request/1",
        now=13,
    )

    assert finalizing.state == JobState.FINALIZING
    assert finalizing.projection == projection
    assert completed.state == JobState.SUCCEEDED
    assert completed.result_filename == "prediction.tar.zst"


def test_unknown_launch_requires_an_explicit_safe_resolution(tmp_path: Path) -> None:
    store, owner = _store(tmp_path)
    _admit(store, owner)
    store.mark_submission_in_progress(JOB_ID, now=11)

    with pytest.raises(JobStateResolutionError, match="Function Call ID"):
        store.resolve_state_unknown(
            JOB_ID,
            resolution="resume",
            function_call_id=None,
            now=12,
        )

    resumed = store.resolve_state_unknown(
        JOB_ID,
        resolution="resume",
        function_call_id="fc-existing",
        now=12,
    )
    assert (resumed.state, resumed.root_function_call_id) == (
        JobState.RUNNING,
        "fc-existing",
    )


def test_unknown_launch_can_only_requeue_without_launch_evidence(
    tmp_path: Path,
) -> None:
    store, owner = _store(tmp_path)
    _admit(store, owner)
    store.mark_submission_in_progress(JOB_ID, now=11)
    requeued = store.resolve_state_unknown(
        JOB_ID,
        resolution="requeue",
        function_call_id=None,
        now=12,
    )
    assert requeued.state == JobState.QUEUED

    store.mark_submission_in_progress(JOB_ID, now=13)
    store.record_launch(JOB_ID, function_call_id="fc-existing", now=14)
    store.mark_state_unknown(
        JOB_ID,
        reason="provider_outcome_unknown",
        message="unknown",
        now=15,
    )
    with pytest.raises(JobStateResolutionError, match="cannot be requeued"):
        store.resolve_state_unknown(
            JOB_ID,
            resolution="requeue",
            function_call_id=None,
            now=16,
        )
    cancelled = store.resolve_state_unknown(
        JOB_ID,
        resolution="cancel",
        function_call_id=None,
        now=17,
    )
    assert cancelled.state == JobState.CANCEL_REQUESTED


def test_explicit_resume_replaces_completed_root_call(tmp_path: Path) -> None:
    store, owner = _store(tmp_path)
    _admit(store, owner)
    store.record_launch(JOB_ID, function_call_id="fc-root", now=11)
    store.replace_projection(
        JOB_ID,
        state=JobState.BLOCKED,
        projection={"stages": [], "warnings": []},
        observed_at=12,
    )

    resumed = store.record_resume(
        JOB_ID,
        previous_function_call_id="fc-root",
        function_call_id="fc-resume",
        now=13,
    )

    assert (resumed.state, resumed.root_function_call_id) == (
        JobState.RUNNING,
        "fc-resume",
    )


@pytest.mark.parametrize(
    "state",
    [JobState.FINALIZING, JobState.STATE_UNKNOWN, JobState.BLOCKED],
)
def test_only_active_compute_states_accept_cancellation(
    tmp_path: Path,
    state: JobState,
) -> None:
    store, owner = _store(tmp_path)
    _admit(store, owner)
    if state == JobState.FINALIZING:
        store.begin_finalization(
            JOB_ID,
            result_state=JobState.SUCCEEDED,
            projection={},
            now=11,
        )
    elif state == JobState.STATE_UNKNOWN:
        store.mark_state_unknown(
            JOB_ID,
            reason="provider_outcome_unknown",
            message="unknown",
            now=11,
        )
    else:
        store.block_job(
            JOB_ID,
            category="result_integrity",
            message="blocked",
            retry_at=None,
            now=11,
        )

    with pytest.raises(JobNotCancellableError, match="does not accept cancellation"):
        store.request_cancel(JOB_ID, now=12)
