"""Owner deletion, local cleanup and durable replay boundaries."""

# ruff: noqa: D103

import asyncio
import hashlib
from uuid import UUID, uuid4

import pytest
from test_api_contract import ORIGIN, _app, _humanization_session, _request, _session
from test_artifact_cache import publish
from test_tool_runtime import JOB_ID, Adapter, _lifecycle

from biomodals.service.artifacts import ArtifactCache
from biomodals.service.http_contract import require_unsafe_session
from biomodals.service.pending import PendingRequestStore
from biomodals.service.store import JobNotFoundError, JobState
from biomodals.service.tool_runtime import (
    PreparedResult,
    reconciliation_loop,
)
from biomodals.service.tools import TOOLS


def test_delete_api_owner_visibility_and_replay(tmp_path):
    app = _app(tmp_path)
    session = _humanization_session(app)
    headers = {"Origin": ORIGIN, "Idempotency-Key": str(uuid4())}
    payload = {"pairs": [{"id": "a", "vh": "ACDE", "vl": "FGHI"}]}
    admitted = _request(
        app, "POST", "/api/v1/humanization/jobs", json=payload, headers=headers
    )
    assert admitted.status_code == 202
    job_id = UUID(admitted.json()["job_id"])
    path = f"/api/v1/jobs/{job_id}"
    assert (
        _request(app, "DELETE", path, headers=headers).json()["code"]
        == "job_not_deletable"
    )
    app.state.store.fail_job(job_id, error_code="test", error_message="test", now=20)
    foreign = _session(uuid4())
    app.dependency_overrides[require_unsafe_session] = lambda: foreign
    assert _request(app, "DELETE", path, headers=headers).status_code == 404
    app.dependency_overrides.pop(require_unsafe_session)
    assert _request(app, "DELETE", path).status_code == 403
    app.dependency_overrides[require_unsafe_session] = lambda: session
    for _ in range(2):
        response = _request(app, "DELETE", path, headers=headers)
        assert response.status_code == 202 and not response.content
    for suffix in ("", "/download", "/log-targets"):
        assert _request(app, "GET", path + suffix).status_code == 404
    for suffix in (
        "/refresh",
        "/cancel",
        "/prepare-download",
        "/retry-result-preparation",
    ):
        assert _request(app, "POST", path + suffix, headers=headers).status_code == 404
    for suffix in ("inputs", "selection", "selection.csv"):
        assert (
            _request(
                app, "GET", f"/api/v1/humanization/jobs/{job_id}/{suffix}"
            ).status_code
            == 404
        )
    assert _request(app, "GET", "/api/v1/jobs").json()["jobs"] == []
    replay = _request(
        app, "POST", "/api/v1/humanization/jobs", json=payload, headers=headers
    )
    assert replay.status_code == 409 and replay.json()["code"] == "job_deleted"
    assert app.state.store.get_job_by_id(job_id).deleted_at is not None


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.key)
def test_delete_authentication_and_all_tool_inputs(tmp_path, monkeypatch, tool):
    app = _app(tmp_path)
    session = _humanization_session(app)
    store = app.state.store
    job = store.admit_job(
        owner_user_id=session.principal.user_id,
        tool=tool.key,
        display_name="Delete me",
        idempotency_key=str(uuid4()),
        request_digest="a" * 64,
        modal_environment="main",
        modal_app_name=tool.default_modal_app_name,
        modal_app_version=1,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=8,
        max_active_gpu_provider_calls=1,
        now=3,
    ).job
    store.request_cancel(job.job_id, now=4)
    path = f"/api/v1/jobs/{job.job_id}"
    app.dependency_overrides.pop(require_unsafe_session)
    headers = {"Origin": ORIGIN}
    assert _request(app, "DELETE", path, headers=headers).status_code == 401
    monkeypatch.setattr(app.state.auth, "authenticate", lambda _: session)
    headers["Cookie"] = "__Host-biomodals-session=fixture"
    denied = _request(app, "DELETE", path, headers=headers)
    assert denied.status_code == 403 and denied.json()["code"] == "csrf_invalid"
    app.dependency_overrides[require_unsafe_session] = lambda: session
    assert _request(app, "DELETE", path, headers=headers).status_code == 202
    slug = tool.key.replace("_", "-")
    tool_routes = {
        "alphafold3": ["document", "prediction"],
        "gromacs": ["continuation", "clustering"],
        "humanization": ["inputs", "selection"],
        "nanobody_humanization": ["inputs", "selection"],
    }
    for suffix in tool_routes[tool.key]:
        response = _request(app, "GET", f"/api/v1/{slug}/jobs/{job.job_id}/{suffix}")
        assert response.status_code == 404


@pytest.mark.anyio
async def test_cleanup_waits_for_download_and_preserves_other_files(tmp_path):
    store, lifecycle, _ = _lifecycle(tmp_path, object())
    cache = lifecycle.cache
    cache.check_job_access = store.require_job_access
    content = b"native result"
    lease = await publish(cache, str(JOB_ID), content)
    other_id = str(uuid4())
    other = await publish(cache, other_id, b"other result")
    other.close()
    scratch = cache.directory / f".{JOB_ID}.archive-interrupted"
    scratch.mkdir()
    (scratch / "partial").write_bytes(b"large partial download")
    part = cache.staging_path(str(JOB_ID))
    part.write_bytes(b"partial zip")
    unrelated = cache.directory / "unattributed-historical-scratch"
    unrelated.mkdir()
    shared = cache.directory.parent / "shared-reference.json"
    shared.write_bytes(b"shared")
    store.fail_job(JOB_ID, error_code="test", error_message="test", now=20)
    job = store.get_job_by_id(JOB_ID)
    store.delete_job(job.owner_user_id, JOB_ID, now=21)
    await lifecycle.cleanup_deleted_job(store.get_job_by_id(JOB_ID))
    assert store.get_job_by_id(JOB_ID).cleanup_completed_at is None
    assert lease.read() == content
    with pytest.raises(JobNotFoundError):
        await cache.acquire_async(
            str(JOB_ID),
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    lease.close()
    await lifecycle.cleanup_deleted_job(store.get_job_by_id(JOB_ID))
    assert store.get_job_by_id(JOB_ID).cleanup_completed_at is not None
    assert sorted(p.name for p in cache.directory.iterdir()) == [
        f"{other_id}.result",
        unrelated.name,
    ]
    assert shared.read_bytes() == b"shared"
    await cache.shutdown()


@pytest.mark.anyio
async def test_deletion_during_restore_cannot_republish(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    content = b"archive"
    digest = hashlib.sha256(content).hexdigest()

    class HeldAdapter(Adapter):
        async def prepare_result(self, job, cache, *, completed_at):
            path = cache.staging_path(str(job.job_id))
            path.write_bytes(content)
            started.set()
            await release.wait()
            try:
                await cache.publish_staged(
                    str(job.job_id), path, size_bytes=len(content), sha256=digest
                )
            finally:
                path.unlink(missing_ok=True)
            return PreparedResult(
                "result.zip", "application/zip", len(content), digest, "v1"
            )

    store, lifecycle, _ = _lifecycle(tmp_path, object(), HeldAdapter())
    cache = lifecycle.cache
    cache.check_job_access = store.require_job_access
    store.complete_job(
        JOB_ID,
        result_state=JobState.SUCCEEDED,
        result_filename="result.zip",
        result_media_type="application/zip",
        result_size_bytes=len(content),
        result_sha256=digest,
        result_archive_schema="v1",
        now=20,
    )
    job = store.get_job_by_id(JOB_ID)
    task = asyncio.create_task(lifecycle.restore_result(job))
    await started.wait()
    store.delete_job(job.owner_user_id, JOB_ID, now=21)
    await lifecycle.cleanup_deleted_job(store.get_job_by_id(JOB_ID))
    assert store.get_job_by_id(JOB_ID).cleanup_completed_at is None
    release.set()
    with pytest.raises(JobNotFoundError):
        await task
    await lifecycle.cleanup_deleted_job(store.get_job_by_id(JOB_ID))
    assert store.get_job_by_id(JOB_ID).state == JobState.SUCCEEDED
    assert store.get_job_by_id(JOB_ID).cleanup_completed_at is not None
    assert list(cache.directory.iterdir()) == []
    await cache.shutdown()


@pytest.mark.anyio
async def test_cleanup_retries_pending_input_failure_after_restart(
    tmp_path, monkeypatch
):
    pending = PendingRequestStore(tmp_path)
    pending.initialize()
    pending.put(JOB_ID, b"input")

    class PendingAdapter(Adapter):
        async def discard_pending(self, job):
            pending.delete(job.job_id)

    store, lifecycle, _ = _lifecycle(tmp_path, object(), PendingAdapter())
    store.fail_job(JOB_ID, error_code="test", error_message="test", now=20)
    job = store.get_job_by_id(JOB_ID)
    store.delete_job(job.owner_user_id, JOB_ID, now=21)
    with monkeypatch.context() as patch:

        def denied(*args):
            raise PermissionError("storage unavailable")

        patch.setattr(pending, "delete", denied)
        await lifecycle.cleanup_deleted_job(store.get_job_by_id(JOB_ID))
    store.initialize()
    assert store.get_job_by_id(JOB_ID).cleanup_completed_at is None
    assert pending.get(JOB_ID) == b"input"
    # Start a fresh lifecycle/cache as the API would after restarting.
    await lifecycle.cache.shutdown()
    lifecycle = type(lifecycle)(
        store,
        object(),
        tuple(lifecycle.registrations.values()),
        ArtifactCache(tmp_path / "cache"),
    )
    stop, wake = asyncio.Event(), asyncio.Event()
    # Advance beyond the durable backoff without making the test sleep a minute.
    import biomodals.service.tool_runtime as runtime

    now = runtime.time.time()
    monkeypatch.setattr(runtime.time, "time", lambda: now + 61)
    task = asyncio.create_task(
        reconciliation_loop(lifecycle, interval_seconds=0.01, stop=stop, wake=wake)
    )
    try:
        async with asyncio.timeout(2):
            while store.get_job_by_id(JOB_ID).cleanup_completed_at is None:
                await asyncio.sleep(0.01)
    finally:
        stop.set()
        wake.set()
        await task
    assert pending.get(JOB_ID) is None
    assert store.get_job_by_id(JOB_ID).cleanup_completed_at is not None
    await lifecycle.cache.shutdown()
