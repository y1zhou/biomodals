"""Service lifecycle concurrency and recovery contracts."""

# ruff: noqa: D101,D102,D103

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from biomodals.execution import ActiveProviderCallCounts, RunStatus
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.store import JobState, ServiceStore
from biomodals.service.tool_runtime import (
    JobLifecycle,
    PreparedResult,
    ToolRegistration,
)
from biomodals.service.tools import ALPHAFOLD3_TOOL

JOB_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


class Adapter:
    prepared = 0

    async def stage(self, _job):
        return None

    async def discard_pending(self, _job):
        return None

    async def prepare_result(self, _job, cache, *, completed_at):
        self.prepared += 1
        return PreparedResult("result.tar.zst", "application/zstd", 1, "a" * 64, "v1")


def _overview(status: RunStatus):
    return SimpleNamespace(
        run=SimpleNamespace(status=status, status_message=None),
        nodes=(),
        representative_provider_calls=(),
        active_provider_calls=ActiveProviderCallCounts(total=0, gpu=0),
        node_task_status_counts=(),
    )


def _store(tmp_path: Path) -> ServiceStore:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.initialize()
    user = store.create_user(
        email="user@example.com",
        display_name="User",
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
    store.admit_job(
        owner_user_id=user.user_id,
        tool="alphafold3",
        display_name="prediction",
        idempotency_key="request",
        request_digest="a" * 64,
        modal_environment="main",
        modal_app_name="AlphaFold3",
        modal_app_version=1,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=1,
        now=10,
        new_job_id=JOB_ID,
    )
    return store


def _lifecycle(tmp_path: Path, remote, adapter: Adapter | None = None):
    store = _store(tmp_path)
    selected = adapter or Adapter()
    lifecycle = JobLifecycle(
        store,
        remote,
        (ToolRegistration(ALPHAFOLD3_TOOL, selected),),
        ArtifactCache(tmp_path / "cache"),
    )
    return store, lifecycle, selected


@pytest.mark.anyio
async def test_crash_during_spawn_is_fenced_and_never_relaunched(
    tmp_path: Path,
) -> None:
    class Remote:
        launches = 0

        async def launch(self, _locator):
            self.launches += 1
            raise asyncio.CancelledError

    remote = Remote()
    store, lifecycle, _adapter = _lifecycle(tmp_path, remote)

    with pytest.raises(asyncio.CancelledError):
        await lifecycle.advance(JOB_ID)
    unknown = store.get_job_by_id(JOB_ID)
    assert unknown is not None
    assert (unknown.state, unknown.state_reason) == (
        JobState.STATE_UNKNOWN,
        "submission_in_progress",
    )

    await lifecycle.advance(JOB_ID)
    assert remote.launches == 1


@pytest.mark.anyio
async def test_cancellation_waits_for_launch_checkpoint(tmp_path: Path) -> None:
    class Remote:
        entered = asyncio.Event()
        release = asyncio.Event()
        cancellations = 0

        async def launch(self, _locator):
            self.entered.set()
            await self.release.wait()
            return "fc-root"

        async def cancel(self, _locator):
            self.cancellations += 1
            return _overview(RunStatus.CANCEL_REQUESTED)

    remote = Remote()
    store, lifecycle, _adapter = _lifecycle(tmp_path, remote)
    launch = asyncio.create_task(lifecycle.advance(JOB_ID))
    await remote.entered.wait()
    cancel = asyncio.create_task(lifecycle.cancel(JOB_ID))
    await asyncio.sleep(0)
    assert not cancel.done()

    remote.release.set()
    await launch
    cancelled = await cancel

    assert cancelled.state == JobState.CANCEL_REQUESTED
    assert cancelled.root_function_call_id == "fc-root"
    assert remote.cancellations == 1
    assert store.get_job_by_id(JOB_ID) == cancelled


@pytest.mark.anyio
async def test_terminal_observation_defers_result_work(tmp_path: Path) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def status(self, _locator):
            return _overview(RunStatus.SUCCEEDED)

    store, lifecycle, adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)

    observed = await lifecycle.advance(JOB_ID, force_refresh=True)
    assert observed.state == JobState.FINALIZING
    assert adapter.prepared == 0

    completed = await lifecycle.advance(JOB_ID, finalize=True, background=True)
    assert completed.state == JobState.SUCCEEDED
    assert adapter.prepared == 1
