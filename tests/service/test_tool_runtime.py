"""Service lifecycle concurrency and recovery contracts."""

# ruff: noqa: D101,D102,D103

from __future__ import annotations

import asyncio
import errno
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest

from biomodals.execution import (
    ActiveProviderCallCounts,
    ExecutionRunNotFoundError,
    NodeStatus,
    RunStatus,
)
from biomodals.service import tool_runtime
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.remote_execution import (
    RemoteExecutionClient,
    RemoteExecutionIdentityMismatchError,
    RemoteExecutionNotInitializedError,
)
from biomodals.service.store import JobState, ServiceStore
from biomodals.service.tool_runtime import (
    EnvironmentPreparationError,
    JobLifecycle,
    PreparedResult,
    ResultIntegrityError,
    SubmissionWait,
    ToolRegistration,
    reconciliation_loop,
)
from biomodals.service.tools import ALPHAFOLD3_TOOL, TOOLS

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


def _store(tmp_path: Path, tool=ALPHAFOLD3_TOOL) -> ServiceStore:
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
        tool=tool.key,
        display_name="prediction",
        idempotency_key="request",
        request_digest="a" * 64,
        modal_environment="main",
        modal_app_name=tool.default_modal_app_name,
        modal_app_version=1,
        tool_active_job_limit=200,
        global_active_job_limit=200,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=1,
        now=10,
        new_job_id=JOB_ID,
    )
    return store


def _lifecycle(
    tmp_path: Path, remote, adapter: Adapter | None = None, *, tool=ALPHAFOLD3_TOOL
):
    store = _store(tmp_path, tool)
    selected = adapter or Adapter()
    lifecycle = JobLifecycle(
        store,
        remote,
        (ToolRegistration(tool, selected),),
        ArtifactCache(tmp_path / "cache"),
    )
    return store, lifecycle, selected


@pytest.mark.anyio
@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.key)
@pytest.mark.parametrize("cancel_requested", [False, True])
async def test_failed_initialization_finishes_without_remote_cancellation_loop(
    tmp_path, monkeypatch, tool, cancel_requested
):
    calls = []

    def failed(*_args, **_kwargs):
        calls.append("failed")
        raise ValueError(
            "Target deployment changed scientific versions: private-detail"
        )

    def missing():
        calls.append("status")
        raise ExecutionRunNotFoundError(str(JOB_ID))

    coordinator = SimpleNamespace(
        status=SimpleNamespace(remote=missing),
        cancel=SimpleNamespace(remote=failed),
    )
    monkeypatch.setattr(
        RemoteExecutionClient, "_coordinator", staticmethod(lambda _: coordinator)
    )
    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall.from_id",
        lambda _: SimpleNamespace(get=failed),
    )
    store, lifecycle, _adapter = _lifecycle(
        tmp_path, RemoteExecutionClient(), tool=tool
    )
    store.record_launch(JOB_ID, function_call_id="fc-root", now=20)
    if cancel_requested:
        await lifecycle.cancel(JOB_ID)

    completed = await lifecycle.advance(JOB_ID, background=True)

    assert completed.state == (
        JobState.CANCELLED if cancel_requested else JobState.FAILED
    )
    assert completed.completed_at is not None
    assert completed.root_function_call_id == "fc-root"
    assert completed.error_code == (
        None if cancel_requested else "execution_initialization_failed"
    )
    assert "private-detail" not in (completed.error_message or "")
    assert store.count_active_jobs() == 0
    assert store.list_reconcilable_jobs(now=10**10) == []
    recorded_calls = list(calls)
    restarted = JobLifecycle(
        store,
        RemoteExecutionClient(),
        tuple(lifecycle.registrations.values()),
        lifecycle.cache,
    )
    assert await restarted.advance(JOB_ID, background=True) == completed
    assert await restarted.advance(JOB_ID, force_refresh=True) == completed
    assert calls == recorded_calls


@pytest.mark.anyio
async def test_missing_previously_observed_execution_remains_unknown(tmp_path):
    class Remote:
        async def poll_root(self, _locator, _call_id):
            raise RemoteExecutionNotInitializedError("missing Run")

    store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    store.record_launch(JOB_ID, function_call_id="fc-root", now=20)
    store.replace_projection(
        JOB_ID, state=JobState.RUNNING, projection={"stages": []}, observed_at=21
    )

    unknown = await lifecycle.advance(JOB_ID, background=True)

    assert unknown.state == JobState.STATE_UNKNOWN
    assert unknown.state_reason == "remote_execution_missing"
    assert unknown.completed_at is None
    assert store.count_active_jobs() == 1


@pytest.mark.anyio
async def test_confirmed_uninitialized_run_retains_unknown_job_cancellation_intent(
    tmp_path,
):
    class Remote:
        async def poll_root(self, _locator, _call_id):
            raise RemoteExecutionNotInitializedError("missing Run")

    store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    store.record_launch(JOB_ID, function_call_id="fc-root", now=20)
    await lifecycle.cancel(JOB_ID)
    store.mark_state_unknown(
        JOB_ID, reason="provider_outcome_unknown", message="uncertain", now=21
    )

    cancelled = await lifecycle.advance(JOB_ID, background=True)

    assert cancelled.state == JobState.CANCELLED
    assert cancelled.cancel_requested_at is not None
    assert cancelled.state_reason is None
    assert cancelled.state_message is None
    assert store.count_active_jobs() == 0


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
async def test_environment_preparation_failure_retains_input_and_never_launches(
    tmp_path: Path,
) -> None:
    class FailingAdapter(Adapter):
        attempts = 0

        async def stage(self, job):
            self.attempts += 1
            raise EnvironmentPreparationError("Model cache requires operator repair")

        async def discard_pending(self, job):
            raise AssertionError("Failed preparation must retain pending input")

    adapter = FailingAdapter()
    store, lifecycle, _adapter = _lifecycle(tmp_path, object(), adapter)

    failed = await lifecycle.advance(JOB_ID)

    assert failed.state == JobState.FAILED
    assert failed.error_code == "environment_preparation_failed"
    assert failed.error_message == "Model cache requires operator repair"
    assert failed.root_function_call_id is None
    assert failed.completed_at is not None
    assert store.count_active_jobs() == 0
    assert store.list_reconcilable_jobs(now=10**10) == []
    assert await lifecycle.advance(JOB_ID, background=True) == failed
    assert await lifecycle.advance(JOB_ID, force_refresh=True) == failed
    assert adapter.attempts == 1


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

        async def cancel(self, _locator, *, root_function_call_id=None):
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
    assert remote.cancellations == 0
    assert store.get_job_by_id(JOB_ID) == cancelled
    await lifecycle.advance(JOB_ID, background=True)
    assert remote.cancellations == 1


@pytest.mark.anyio
async def test_cancellation_acknowledgement_and_status_do_not_wait_for_remote(
    tmp_path: Path,
) -> None:
    class Remote:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def launch(self, _locator):
            return "fc-root"

        async def cancel(self, _locator, *, root_function_call_id=None):
            self.entered.set()
            await self.release.wait()
            return _overview(RunStatus.CANCELLED)

    remote = Remote()
    store, lifecycle, _adapter = _lifecycle(tmp_path, remote)
    await lifecycle.advance(JOB_ID)
    try:
        acknowledged = await asyncio.wait_for(lifecycle.cancel(JOB_ID), timeout=0.1)
        assert acknowledged.state == JobState.CANCEL_REQUESTED
        assert store.get_job_by_id(JOB_ID).cancel_requested_at is not None
        assert (await lifecycle.advance(JOB_ID)).state == JobState.CANCEL_REQUESTED
        delivery = asyncio.create_task(lifecycle.advance(JOB_ID, background=True))
        await remote.entered.wait()
        viewed = await asyncio.wait_for(lifecycle.advance(JOB_ID), timeout=0.1)
        assert viewed.state == JobState.CANCEL_REQUESTED
        repeated = await asyncio.wait_for(lifecycle.cancel(JOB_ID), timeout=0.1)
        assert repeated.cancel_requested_at == acknowledged.cancel_requested_at
    finally:
        remote.release.set()
    cancelled = await delivery
    assert cancelled.state == JobState.CANCELLED


@pytest.mark.anyio
async def test_terminal_observation_defers_result_work(tmp_path: Path) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, _call_id):
            return _overview(RunStatus.SUCCEEDED)

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
async def test_background_terminal_observation_finalizes_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, _call_id):
            return _overview(RunStatus.SUCCEEDED)

        async def queued_provider_call_handles(self, _call_id, _overview):
            return frozenset()

    store, lifecycle, adapter = _lifecycle(tmp_path, Remote())
    store.record_launch(JOB_ID, function_call_id="fc-root", now=20)
    clock = iter((100, 200))
    monkeypatch.setattr(tool_runtime.time, "time", lambda: next(clock))

    completed = await lifecycle.advance(JOB_ID, finalize=True, background=True)

    assert completed.state == JobState.SUCCEEDED
    assert completed.finalization_started_at == 100
    assert completed.completed_at == 200
    assert adapter.prepared == 1


@pytest.mark.anyio
async def test_result_restoration_is_shared_and_integrity_failure_blocks(
    tmp_path: Path,
) -> None:
    class MismatchAdapter(Adapter):
        entered = asyncio.Event()
        release = asyncio.Event()
        mismatch = True

        async def prepare_result(self, job, cache, *, completed_at):
            del job, cache, completed_at
            self.prepared += 1
            self.entered.set()
            await self.release.wait()
            return (
                PreparedResult(
                    "different.tar.zst",
                    "application/zstd",
                    2,
                    "b" * 64,
                    "v2",
                )
                if self.mismatch
                else PreparedResult(
                    "result.tar.zst",
                    "application/zstd",
                    1,
                    "a" * 64,
                    "v1",
                )
            )

    adapter = MismatchAdapter()
    store, lifecycle, _adapter = _lifecycle(tmp_path, SimpleNamespace(), adapter)
    store.complete_job(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        result_filename="result.tar.zst",
        result_media_type="application/zstd",
        result_size_bytes=1,
        result_sha256="a" * 64,
        result_archive_schema="v1",
        now=20,
    )
    job = store.get_job_by_id(JOB_ID)
    assert job is not None

    first = asyncio.create_task(lifecycle.restore_result(job))
    await adapter.entered.wait()
    second = asyncio.create_task(lifecycle.restore_result(job))
    adapter.release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert adapter.prepared == 1
    assert all(isinstance(result, ResultIntegrityError) for result in results)
    blocked = store.get_job_by_id(JOB_ID)
    assert blocked is not None
    assert (blocked.state, blocked.blocking_category) == (
        JobState.BLOCKED,
        "result_integrity",
    )
    assert store.list_reconcilable_jobs(now=10**10) == []

    adapter.mismatch = False
    await lifecycle.restore_result(blocked)
    restored = store.get_job_by_id(JOB_ID)
    assert restored is not None
    assert (restored.state, restored.blocking_category) == (
        JobState.SUCCEEDED,
        None,
    )


@pytest.mark.anyio
async def test_reconciliation_processes_at_most_four_jobs_concurrently() -> None:
    stop = asyncio.Event()
    active = 0
    maximum = 0
    completed = 0
    jobs = [SimpleNamespace(job_id=uuid4()) for _ in range(8)]

    class Store:
        def list_reconcilable_jobs(self, *, now):
            del now
            return jobs

    class Lifecycle:
        store = Store()

        async def advance(self, _job_id, *, finalize, background):
            nonlocal active, maximum, completed
            assert finalize and background
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            completed += 1
            if completed == len(jobs):
                stop.set()

    await reconciliation_loop(
        cast(JobLifecycle, Lifecycle()),
        interval_seconds=60,
        stop=stop,
        wake=asyncio.Event(),
    )

    assert maximum == 4


@pytest.mark.anyio
async def test_slow_reconciliation_does_not_block_new_jobs_or_duplicate_delivery() -> (
    None
):
    stop, wake = asyncio.Event(), asyncio.Event()
    entered, release, processed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    old_id, new_id = uuid4(), uuid4()
    jobs = [SimpleNamespace(job_id=old_id)]
    old_calls = 0

    class Store:
        def list_reconcilable_jobs(self, *, now):
            return tuple(jobs)

    class Lifecycle:
        store = Store()

        async def advance(self, job_id, *, finalize, background):
            nonlocal old_calls
            if job_id == old_id:
                old_calls += 1
                entered.set()
                await release.wait()
            else:
                processed.set()
                jobs.remove(next(job for job in jobs if job.job_id == new_id))

    task = asyncio.create_task(
        reconciliation_loop(
            cast(JobLifecycle, Lifecycle()),
            interval_seconds=60,
            stop=stop,
            wake=wake,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        jobs.append(SimpleNamespace(job_id=new_id))
        wake.set()
        await asyncio.wait_for(processed.wait(), timeout=0.1)
        wake.set()
        await asyncio.sleep(0)
        assert old_calls == 1
    finally:
        stop.set()
        release.set()
        wake.set()
        await task


@pytest.mark.anyio
async def test_reconciliation_does_not_hot_poll_after_snapshot_is_exhausted() -> None:
    stop, wake, processed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0
    job = SimpleNamespace(job_id=uuid4())

    class Store:
        def list_reconcilable_jobs(self, *, now):
            return (job,)

    class Lifecycle:
        store = Store()

        async def advance(self, job_id, *, finalize, background):
            nonlocal calls
            calls += 1
            processed.set()

    task = asyncio.create_task(
        reconciliation_loop(
            cast(JobLifecycle, Lifecycle()),
            interval_seconds=60,
            stop=stop,
            wake=wake,
        )
    )
    try:
        await asyncio.wait_for(processed.wait(), timeout=1)
        await asyncio.sleep(0.02)
        assert calls == 1
    finally:
        stop.set()
        wake.set()
        await task


@pytest.mark.anyio
async def test_reconciliation_wakes_for_a_newly_admitted_job() -> None:
    stop = asyncio.Event()
    wake = asyncio.Event()
    first_scan = asyncio.Event()
    processed = asyncio.Event()
    jobs: list[SimpleNamespace] = []

    class Store:
        def list_reconcilable_jobs(self, *, now):
            del now
            first_scan.set()
            return tuple(jobs)

    class Lifecycle:
        store = Store()

        async def advance(self, _job_id, *, finalize, background):
            assert finalize and background
            processed.set()
            stop.set()

    task = asyncio.create_task(
        reconciliation_loop(
            cast(JobLifecycle, Lifecycle()),
            interval_seconds=60,
            stop=stop,
            wake=wake,
        )
    )
    await asyncio.wait_for(first_scan.wait(), timeout=1)
    jobs.append(SimpleNamespace(job_id=uuid4()))
    wake.set()
    await asyncio.wait_for(processed.wait(), timeout=1)
    await task


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
async def test_storage_exhaustion_blocks_result_preparation(tmp_path: Path) -> None:
    class FullDiskAdapter(Adapter):
        async def prepare_result(self, job, cache, *, completed_at):
            del job, cache, completed_at
            raise OSError(errno.ENOSPC, "disk full")

    store, lifecycle, _adapter = _lifecycle(
        tmp_path,
        SimpleNamespace(),
        FullDiskAdapter(),
    )
    store.begin_finalization(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        projection={"stages": [], "warnings": []},
        now=20,
    )

    blocked = await lifecycle.advance(JOB_ID, finalize=True)

    assert (blocked.state, blocked.blocking_category) == (
        JobState.BLOCKED,
        "result_preparation_failed",
    )


@pytest.mark.anyio
async def test_explicit_refresh_updates_stages_while_root_is_active(
    tmp_path: Path,
) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, _function_call_id):
            return None

        async def status(self, _locator):
            overview = _overview(RunStatus.RUNNING)
            overview.nodes = (
                SimpleNamespace(
                    node_key="prepare-environment",
                    status=NodeStatus.RUNNING,
                    started_at=20,
                    completed_at=None,
                ),
            )
            return overview

    store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)
    store.replace_projection(
        JOB_ID,
        state=JobState.RUNNING,
        projection={"stages": [{"code": "prepare_environment", "started_at": None}]},
        observed_at=10**10,
    )

    refreshed = await lifecycle.advance(JOB_ID, force_refresh=True)

    assert refreshed.projection["stages"][0]["started_at"] == 20


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
async def test_completed_root_preserves_unknown_ledger_state(
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

    unknown = await lifecycle.advance(JOB_ID, finalize=True, background=True)

    assert (unknown.state, unknown.state_reason) == (
        JobState.STATE_UNKNOWN,
        "remote_execution_state_unknown",
    )


@pytest.mark.anyio
async def test_explicit_refresh_resumes_suspended_remote_execution(
    tmp_path: Path,
) -> None:
    class Remote:
        resume_count = 0

        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, call_id):
            if call_id == "fc-resume":
                return None
            return _overview(RunStatus.SUSPENDED)

        async def status(self, _locator):
            return _overview(RunStatus.SUSPENDED)

        async def resume(self, _locator):
            self.resume_count += 1
            return "fc-resume"

    remote = Remote()
    store, lifecycle, _adapter = _lifecycle(tmp_path, remote)
    await lifecycle.advance(JOB_ID)

    resumed = await lifecycle.advance(JOB_ID, force_refresh=True)

    assert (resumed.state, resumed.root_function_call_id) == (
        JobState.RUNNING,
        "fc-resume",
    )

    still_running = await lifecycle.advance(JOB_ID, force_refresh=True)

    assert still_running.root_function_call_id == "fc-resume"
    assert remote.resume_count == 1


@pytest.mark.anyio
async def test_background_observation_does_not_resume_suspended_execution(
    tmp_path: Path,
) -> None:
    class Remote:
        async def launch(self, _locator):
            return "fc-root"

        async def poll_root(self, _locator, _function_call_id):
            return _overview(RunStatus.SUSPENDED)

        async def resume(self, _locator):
            raise AssertionError("background reconciliation must not resume")

    _store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)

    blocked = await lifecycle.advance(JOB_ID, finalize=True, background=True)

    assert (blocked.state, blocked.blocking_category) == (
        JobState.BLOCKED,
        "remote_execution_suspended",
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

        async def poll_root(self, _locator, _call_id):
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

        async def poll_root(self, _locator, _call_id):
            raise RemoteExecutionIdentityMismatchError("wrong execution identity")

        async def status(self, _locator):
            raise RemoteExecutionIdentityMismatchError("wrong execution identity")

    _store, lifecycle, _adapter = _lifecycle(tmp_path, Remote())
    await lifecycle.advance(JOB_ID)
    unknown = await lifecycle.advance(JOB_ID, force_refresh=True)

    assert (unknown.state, unknown.state_reason, unknown.state_message) == (
        JobState.STATE_UNKNOWN,
        "provider_outcome_unknown",
        "The remote execution identity could not be confirmed",
    )
