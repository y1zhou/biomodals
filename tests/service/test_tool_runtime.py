"""Service lifecycle concurrency and recovery contracts."""

# ruff: noqa: D101,D102,D103

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from biomodals.execution import ActiveProviderCallCounts, RunStatus
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.remote_execution import (
    RemoteExecutionIdentityMismatchError,
    RemoteRootExecutionFailedError,
)
from biomodals.service.store import JobState, ServiceStore
from biomodals.service.tool_runtime import (
    JobLifecycle,
    PreparedResult,
    SubmissionWait,
    ToolRegistration,
)
from biomodals.service.tools import ALPHAFOLD3_TOOL

JOB_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


class Adapter:
    prepared = 0

    async def stage(self, job):
        del job
        return None

    async def discard_pending(self, job):
        del job
        return None

    async def prepare_result(self, job, cache, *, completed_at):
        del job, cache, completed_at
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
        active_job_limit=200,
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
        tool_active_job_limit=200,
        global_active_job_limit=200,
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
    assert store.list_reconcilable_jobs(now=10**10) == []
    await lifecycle.advance(JOB_ID)
    assert remote.launches == 1


@pytest.mark.anyio
async def test_submission_wait_stays_queued_without_launching(tmp_path: Path) -> None:
    class WaitingAdapter(Adapter):
        async def stage(self, job):
            del job
            return SubmissionWait(
                "waiting_for_shared_publication",
                "Waiting for matching work",
            )

    class Remote:
        async def launch(self, _locator):
            raise AssertionError("waiting Job must not launch")

    store, lifecycle, _adapter = _lifecycle(
        tmp_path,
        Remote(),
        WaitingAdapter(),
    )

    waiting = await lifecycle.advance(JOB_ID)

    assert waiting.state == JobState.QUEUED
    assert waiting.state_reason == "waiting_for_shared_publication"
    assert waiting.state_message == "Waiting for matching work"
    assert waiting.next_retry_at is not None


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

    store.reconcile_result_cache(set())
    cleared = store.get_job_by_id(JOB_ID)
    assert cleared is not None and cleared.cache_cleared_at is not None
    await lifecycle.restore_result(cleared)
    restored = store.get_job_by_id(JOB_ID)
    assert restored is not None and restored.cache_cleared_at is None


@pytest.mark.anyio
async def test_transient_result_failure_retries_later(tmp_path: Path) -> None:
    class UnavailableAdapter(Adapter):
        async def prepare_result(self, job, cache, *, completed_at):
            del job, cache, completed_at
            raise TimeoutError("temporary outage")

    store, lifecycle, _adapter = _lifecycle(
        tmp_path,
        SimpleNamespace(),
        UnavailableAdapter(),
    )
    store.begin_finalization(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        projection={"stages": [], "warnings": []},
        now=20,
    )

    blocked = await lifecycle.advance(JOB_ID, finalize=True)

    assert blocked.state == JobState.BLOCKED
    assert blocked.state_message == "Result preparation is temporarily unavailable"
    assert blocked.next_retry_at is not None


@pytest.mark.anyio
async def test_permanent_result_failure_does_not_retry(tmp_path: Path) -> None:
    class InvalidAdapter(Adapter):
        async def prepare_result(self, job, cache, *, completed_at):
            del job, cache, completed_at
            raise ValueError("invalid archive")

    store, lifecycle, _adapter = _lifecycle(
        tmp_path,
        SimpleNamespace(),
        InvalidAdapter(),
    )
    store.begin_finalization(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        projection={"stages": [], "warnings": []},
        now=20,
    )

    failed = await lifecycle.advance(JOB_ID, finalize=True)

    assert (failed.state, failed.error_code) == (
        JobState.FAILED,
        "result_preparation_failed",
    )
    assert failed.error_message == "The Result archive could not be prepared"


@pytest.mark.anyio
async def test_background_poll_does_not_wake_an_active_coordinator(
    tmp_path: Path,
) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, _function_call_id):
            return None

        async def status(self, _locator):
            raise AssertionError("active background polling must not call status")

    _store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)
    _store.replace_projection(
        JOB_ID,
        state=JobState.RUNNING,
        projection={"stages": [], "warnings": []},
        observed_at=10**10,
    )

    active = await lifecycle.advance(JOB_ID, finalize=True, background=True)
    assert active.state == JobState.RUNNING
    assert active.updated_at > 10


@pytest.mark.anyio
async def test_terminal_root_failure_is_not_treated_as_still_running(
    tmp_path: Path,
) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, _function_call_id):
            raise RemoteRootExecutionFailedError("coordinator crashed")

    _store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)

    failed = await lifecycle.advance(JOB_ID, finalize=True, background=True)
    assert (failed.state, failed.error_code) == (
        JobState.FAILED,
        "remote_coordinator_failed",
    )
    assert failed.error_message == "The remote execution coordinator failed"


@pytest.mark.anyio
async def test_completed_root_with_nonterminal_ledger_becomes_failed(
    tmp_path: Path,
) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, _function_call_id):
            return _overview(RunStatus.STATE_UNKNOWN)

    store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)
    store.mark_state_unknown(
        JOB_ID,
        reason="provider_outcome_unknown",
        message="worker outcome unknown",
        now=20,
    )
    assert [job.job_id for job in store.list_reconcilable_jobs(now=21)] == [JOB_ID]

    failed = await lifecycle.advance(JOB_ID, finalize=True, background=True)

    assert (failed.state, failed.error_code) == (
        JobState.FAILED,
        "remote_execution_incomplete",
    )
    assert failed.error_message == (
        "Remote execution stopped without a terminal outcome"
    )


@pytest.mark.anyio
async def test_active_timeouts_rotate_a_bounded_reconciliation_page(
    tmp_path: Path,
) -> None:
    same_second = 10**10

    class Remote:
        async def poll_root(self, _locator, _function_call_id):
            return None

    store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    first = store.get_job_by_id(JOB_ID)
    assert first is not None
    job_ids = {first.job_id}
    for ordinal in range(100):
        job_id = uuid4()
        job_ids.add(job_id)
        store.admit_job(
            owner_user_id=first.owner_user_id,
            tool="alphafold3",
            display_name=f"prediction-{ordinal}",
            idempotency_key=f"request-{ordinal}",
            request_digest=f"{ordinal:064x}",
            modal_environment="main",
            modal_app_name="AlphaFold3",
            modal_app_version=1,
            tool_active_job_limit=200,
            global_active_job_limit=200,
            max_active_provider_calls=4,
            max_active_gpu_provider_calls=1,
            now=same_second,
            new_job_id=job_id,
        )
    for job_id in job_ids:
        store.record_launch(
            job_id,
            function_call_id=f"fc-{job_id}",
            now=same_second,
        )

    selected = store.list_reconcilable_jobs(now=same_second)
    deferred = job_ids - {job.job_id for job in selected}
    assert len(selected) == 100
    assert len(deferred) == 1
    for job in selected:
        await lifecycle.advance(job.job_id, finalize=True, background=True)

    next_page = store.list_reconcilable_jobs(now=same_second)
    assert deferred <= {job.job_id for job in next_page}


@pytest.mark.anyio
async def test_result_block_does_not_wake_remote_on_owner_refresh(
    tmp_path: Path,
) -> None:
    class Remote:
        async def status(self, _locator):
            raise AssertionError("blocked result delivery must stay local")

    store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    store.begin_finalization(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        projection={"stages": [], "warnings": []},
        now=20,
    )
    blocked = store.block_job(
        JOB_ID,
        category="result_preparation_failed",
        message="cache unavailable",
        retry_at=80,
        now=21,
    )

    viewed = await lifecycle.advance(JOB_ID, force_refresh=True)
    assert viewed == blocked


@pytest.mark.anyio
async def test_remote_failure_details_are_not_owner_visible(tmp_path: Path) -> None:
    raw_path = "/volumes/private/scientific-output"

    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def status(self, _locator):
            overview = _overview(RunStatus.FAILED)
            overview.run.status_message = f"failed reading {raw_path}"
            overview.nodes = (
                SimpleNamespace(
                    node_key="inference",
                    status=SimpleNamespace(value="failed"),
                    status_reason=None,
                    error_message=f"traceback at {raw_path}",
                    started_at=10,
                    completed_at=20,
                ),
            )
            return overview

    store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)
    failed = await lifecycle.advance(JOB_ID, force_refresh=True)

    assert failed.error_message == "Remote execution failed"
    assert raw_path not in str(failed.projection)
    assert raw_path not in str(failed.error_message)


@pytest.mark.anyio
async def test_remote_identity_mismatch_becomes_state_unknown(tmp_path: Path) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def status(self, _locator):
            raise RemoteExecutionIdentityMismatchError("wrong execution identity")

    _store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)
    unknown = await lifecycle.advance(JOB_ID, force_refresh=True)

    assert (unknown.state, unknown.state_reason, unknown.state_message) == (
        JobState.STATE_UNKNOWN,
        "provider_outcome_unknown",
        "wrong execution identity",
    )
