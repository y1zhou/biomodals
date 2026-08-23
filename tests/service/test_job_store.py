"""Lean Service Job persistence contracts."""

# ruff: noqa: D103

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from biomodals.service.store import (
    IdempotencyConflictError,
    JobState,
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


def test_unstaged_jobs_select_only_queued_gromacs_requests(tmp_path: Path) -> None:
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

    assert store.unstaged_job_ids() == {gromacs_id}

    store.record_launch(gromacs_id, function_call_id="fc-test", now=11)
    assert store.unstaged_job_ids() == set()


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
