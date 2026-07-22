"""Shared paid-Modal-operation submission contracts."""

# ruff: noqa: D101,D102,D103,S106

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
)
from biomodals.service.store import JobOperationState, JobState, ServiceStore
from biomodals.service.submission import (
    ModalJobSubmitter,
    SubmissionOutcomeUnknownError,
)


@dataclass(frozen=True)
class Submitted:
    modal_call_id: str
    run_name: str
    operation: str


def admitted_job(tmp_path: Path):
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    user = store.create_user(
        email="alice@example.com",
        display_name="Alice",
        token_digest=b"setup",
        token_expires_at=100,
        now=1,
        is_admin=True,
    )
    enabled = store.set_password_from_token(
        b"setup",
        password_hash="hash",
        session_token_digest=b"session",
        csrf_digest=b"csrf",
        now=2,
        absolute_expires_at=100,
    )
    assert enabled is not None
    job = store.admit_job(
        owner_user_id=user.user_id,
        display_name="Simulation",
        idempotency_key="11111111-1111-4111-8111-111111111111",
        request_hash="a" * 64,
        parameters_json="{}",
        configuration=JobAdmissionConfiguration(
            workload="gromacs",
            modal_environment=DatabaseOverridableSetting("test", False),
            modal_app_name=DatabaseOverridableSetting("Gromacs", False),
            modal_app_version=DatabaseOverridableSetting(1, False),
            workload_active_job_limit=DatabaseOverridableSetting(2, False),
            global_active_job_limit=DatabaseOverridableSetting(2, False),
        ),
        now=3,
    ).job
    return store, job


def test_replay_attaches_only_one_paid_modal_call(tmp_path: Path) -> None:
    store, job = admitted_job(tmp_path)
    submitter = ModalJobSubmitter(store, JobLifecycleLocks(), now=lambda: 10)
    spawned = 0

    async def spawn(_job):
        nonlocal spawned
        spawned += 1
        return Submitted("fc-1", "simulation-1", "prepare")

    async def scenario():
        first = await submitter.submit(
            job,
            operation="prepare",
            run_name="simulation-1",
            submission_token="first",
            spawn=spawn,
            cancel=lambda _call_id: asyncio.sleep(0),
        )
        replay = await submitter.submit(
            job,
            operation="prepare",
            run_name="simulation-1",
            submission_token="replay",
            spawn=spawn,
            cancel=lambda _call_id: asyncio.sleep(0),
        )
        return first, replay

    first, replay = asyncio.run(scenario())

    assert first.attached is True
    assert replay.attached is False
    assert spawned == 1
    assert replay.job.operations[0].modal_call_id == "fc-1"


def test_definite_rejection_releases_the_operation_for_retry(tmp_path: Path) -> None:
    store, job = admitted_job(tmp_path)
    submitter = ModalJobSubmitter(store, JobLifecycleLocks(), now=lambda: 10)

    async def reject(_job):
        raise RuntimeError("definite rejection")

    with pytest.raises(RuntimeError, match="definite rejection"):
        asyncio.run(
            submitter.submit(
                job,
                operation="prepare",
                run_name="simulation-1",
                submission_token="first",
                spawn=reject,
                cancel=lambda _call_id: asyncio.sleep(0),
            )
        )

    assert store.list_operations(job.job_id) == []


def test_unknown_spawn_outcome_stops_automatic_retries(tmp_path: Path) -> None:
    store, job = admitted_job(tmp_path)
    submitter = ModalJobSubmitter(store, JobLifecycleLocks(), now=lambda: 10)

    async def unknown(_job):
        raise SubmissionOutcomeUnknownError("unknown")

    result = asyncio.run(
        submitter.submit(
            job,
            operation="prepare",
            run_name="simulation-1",
            submission_token="first",
            spawn=unknown,
            cancel=lambda _call_id: asyncio.sleep(0),
        )
    )

    assert result.attached is False
    assert result.job.state == JobState.STATE_UNKNOWN
    assert result.job.operations[0].state == JobOperationState.STATE_UNKNOWN


def test_unattachable_call_is_cancelled_and_marked_unknown(tmp_path: Path) -> None:
    store, job = admitted_job(tmp_path)
    submitter = ModalJobSubmitter(store, JobLifecycleLocks(), now=lambda: 10)
    cancelled: list[str] = []

    async def cancel(call_id: str) -> None:
        cancelled.append(call_id)

    result = asyncio.run(
        submitter.submit(
            job,
            operation="prepare",
            run_name="simulation-1",
            submission_token="first",
            spawn=lambda _job: asyncio.sleep(
                0,
                result=Submitted("fc-wrong", "wrong-run", "prepare"),
            ),
            cancel=cancel,
        )
    )

    assert cancelled == ["fc-wrong"]
    assert result.job.state == JobState.STATE_UNKNOWN
