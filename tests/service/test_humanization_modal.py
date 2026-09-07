"""Humanization staging, result identity and archive cache integration."""

# ruff: noqa: D103

import asyncio
import hashlib
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import orjson
import pytest

from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.humanization import modal as service_modal
from biomodals.service.humanization.modal import HumanizationToolAdapter
from biomodals.service.pending import PendingRequestStore
from biomodals.service.tool_runtime import EnvironmentPreparationError, SubmissionWait
from biomodals.workflow.humanization.contracts import AntibodyPair
from biomodals.workflow.humanization.execution import HumanizationExecutionRequest


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepare_environment: bool = False,
):
    pending = PendingRequestStore(tmp_path)
    pending.initialize()
    adapter = HumanizationToolAdapter(pending, modal_download_concurrency=2)
    job: Any = SimpleNamespace(
        job_id=uuid4(),
        modal_environment="test",
        modal_app_name="HumanizationWorkflow",
        modal_app_version=1,
    )
    request = HumanizationExecutionRequest(
        run_name="test",
        pairs=(AntibodyPair(id="001", vh="ACD", vl="EFG"),),
    )

    async def hydrate():
        return None

    volume = SimpleNamespace(hydrate=SimpleNamespace(aio=hydrate))
    monkeypatch.setattr(adapter, "_volume", lambda _: volume)
    monkeypatch.setattr(
        service_modal, "load_execution_request_from_volume", lambda *_: request
    )
    if not prepare_environment:
        monkeypatch.setattr(adapter, "_prepare_environment", lambda _: None)
    return adapter, job, request


@pytest.mark.anyio
async def test_environment_preparation_waits_deduplicates_and_pins_deployment(
    tmp_path, monkeypatch
):
    adapter, job, request = _setup(tmp_path, monkeypatch, prepare_environment=True)
    adapter.pending.put(job.job_id, request.to_bytes())
    gate = asyncio.Event()
    finished = asyncio.Event()
    calls = []

    def function(app_name, function_name, **kwargs):
        async def run():
            calls.append((app_name, function_name, kwargs))
            await gate.wait()
            if function_name == "stage_pabnativ2_models":
                finished.set()

        return SimpleNamespace(remote=SimpleNamespace(aio=run))

    monkeypatch.setattr(service_modal.modal.Function, "from_name", function)
    staged = []
    monkeypatch.setattr(
        service_modal, "stage_execution_request", lambda *_: staged.append("request")
    )
    monkeypatch.setattr(
        service_modal, "stage_execution_launch", lambda *_: staged.append("launch")
    )
    assert isinstance(await adapter.stage(job), SubmissionWait)
    await asyncio.sleep(0)
    sibling = SimpleNamespace(**{**vars(job), "job_id": uuid4()})
    assert isinstance(await adapter.stage(sibling), SubmissionWait)
    assert len(calls) == 1
    assert staged == []
    assert adapter.pending.get(job.job_id) == request.to_bytes()
    gate.set()
    await asyncio.wait_for(finished.wait(), timeout=2)
    assert await asyncio.wait_for(adapter.stage(job), timeout=2) is None
    assert staged == ["request", "launch"]
    assert await asyncio.wait_for(adapter.stage(sibling), timeout=2) is None
    assert [call[1] for call in calls] == [
        "stage_hudiff_models",
        "stage_pabnativ2_models",
    ]
    assert all(call[2] == {"environment_name": "test", "version": 1} for call in calls)

    finished.clear()
    changed = SimpleNamespace(**{
        **vars(job),
        "job_id": uuid4(),
        "modal_app_version": 2,
    })
    assert isinstance(await adapter.stage(changed), SubmissionWait)
    await asyncio.wait_for(finished.wait(), timeout=2)
    assert await adapter.stage(changed) is None
    assert len(calls) == 4
    assert all(call[2]["version"] == 2 for call in calls[2:])

    finished.clear()
    restarted, restarted_job, _ = _setup(
        tmp_path, monkeypatch, prepare_environment=True
    )
    assert isinstance(await restarted.stage(restarted_job), SubmissionWait)
    await asyncio.wait_for(finished.wait(), timeout=2)
    assert await restarted.stage(restarted_job) is None
    assert len(calls) == 6


@pytest.mark.anyio
async def test_environment_failure_prevents_launch_without_automatic_staging_retries(
    tmp_path, monkeypatch
):
    adapter, job, request = _setup(tmp_path, monkeypatch, prepare_environment=True)
    adapter.pending.put(job.job_id, request.to_bytes())
    failed = asyncio.Event()
    calls = []

    def function(app_name, function_name, **kwargs):
        async def run():
            calls.append(function_name)
            failed.set()
            raise RuntimeError("asset checksum mismatch")

        return SimpleNamespace(remote=SimpleNamespace(aio=run))

    monkeypatch.setattr(service_modal.modal.Function, "from_name", function)
    monkeypatch.setattr(
        service_modal,
        "stage_execution_request",
        lambda *_: pytest.fail("Request staged before model readiness"),
    )
    assert isinstance(await adapter.stage(job), SubmissionWait)
    await failed.wait()
    for _ in range(2):
        with pytest.raises(
            EnvironmentPreparationError, match="operator intervention"
        ) as caught:
            await adapter.stage(job)
        assert str(caught.value.__cause__) == "asset checksum mismatch"
    assert calls == ["stage_hudiff_models"]
    assert adapter.pending.get(job.job_id) == request.to_bytes()


@pytest.mark.anyio
async def test_staging_preserves_input_until_request_and_launch_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, job, request = _setup(tmp_path, monkeypatch)
    adapter.pending.put(job.job_id, request.to_bytes())
    calls = []
    monkeypatch.setattr(
        service_modal, "stage_execution_request", lambda *_: calls.append("request")
    )

    def fail_launch(*_):
        raise RuntimeError("launch unavailable")

    monkeypatch.setattr(service_modal, "stage_execution_launch", fail_launch)
    with pytest.raises(RuntimeError, match="launch unavailable"):
        await adapter.stage(job)
    assert adapter.pending.get(job.job_id) == request.to_bytes()
    monkeypatch.setattr(
        service_modal, "stage_execution_launch", lambda *_: calls.append("launch")
    )
    await adapter.stage(job)
    assert calls == ["request", "request", "launch"]
    await adapter.discard_pending(job)
    assert adapter.pending.get(job.job_id) is None
    await adapter.stage(job)


def _publication(job, request):
    files = {
        "selection.csv": b"parent_id,candidate_id\n001,abc\n",
        "imgt_mutations.parquet": b"evidence",
        "native/generation.tar.zst": b"native evidence",
    }
    manifest = {
        "schema_version": 2,
        "execution_run_id": str(job.job_id),
        "parameters": request.settings.model_dump(),
        "scientific_versions": request.scientific_versions,
        "files": [
            {
                "path": name,
                "size_bytes": len(data),
                "content_sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        ],
    }
    return files, manifest


@pytest.mark.anyio
async def test_prepare_result_publishes_verified_reproducible_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, job, request = _setup(tmp_path, monkeypatch)
    files, manifest = _publication(job, request)

    async def read_manifest(*_, max_bytes):
        assert max_bytes == service_modal.MAX_MANIFEST_BYTES
        return orjson.dumps(manifest)

    downloads = []

    def download(_volume, selected, *, concurrency):
        assert concurrency == 2
        for remote, destination in selected:
            downloads.append(remote)
            destination.parent.mkdir(parents=True, exist_ok=True)
            name = remote.split("/humanization/", 1)[1]
            destination.write_bytes(files[name])

    monkeypatch.setattr(service_modal, "read_modal_volume_file", read_manifest)
    monkeypatch.setattr(service_modal, "download_modal_volume_files", download)
    cache = ArtifactCache(tmp_path / "cache")
    try:
        result = await adapter.prepare_result(job, cache, completed_at=1)
        restored = await adapter.prepare_result(job, cache, completed_at=2)
        assert restored == result
        assert result.media_type == "application/zip"
        lease = await cache.acquire_async(
            str(job.job_id), size_bytes=result.size_bytes, sha256=result.sha256
        )
        assert lease is not None
        try:
            with zipfile.ZipFile(io.BytesIO(lease.read(result.size_bytes))) as archive:
                assert {name: archive.read(name) for name in files} == files
                assert orjson.loads(archive.read("manifest.json")) == manifest
        finally:
            lease.close()
        assert len(downloads) == 2 * len(files)
    finally:
        await cache.shutdown()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "damage",
    ["execution_run_id", "parameters", "scientific_versions", "unsafe_path", "digest"],
)
async def test_prepare_result_rejects_invalid_identity_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    adapter, job, request = _setup(tmp_path, monkeypatch)
    files, manifest = _publication(job, request)
    if damage == "execution_run_id":
        manifest[damage] = str(uuid4())
    elif damage == "parameters":
        manifest[damage]["hudiff_seed"] = 123
    elif damage == "scientific_versions":
        manifest[damage] = {"workflow": "wrong"}
    elif damage == "unsafe_path":
        manifest["files"][0]["path"] = "../outside"
    else:
        manifest["files"][0]["content_sha256"] = "0" * 64

    async def read_manifest(*_, **__):
        return orjson.dumps(manifest)

    download_calls = []

    def download(_volume, selected, *, concurrency):
        download_calls.append(concurrency)
        for remote, destination in selected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            name = remote.split("/humanization/", 1)[1]
            destination.write_bytes(files[name])

    monkeypatch.setattr(service_modal, "read_modal_volume_file", read_manifest)
    monkeypatch.setattr(service_modal, "download_modal_volume_files", download)
    cache = ArtifactCache(tmp_path / "cache")
    try:
        with pytest.raises(ArtifactIntegrityError):
            await adapter.prepare_result(job, cache, completed_at=1)
        assert download_calls == ([2] if damage == "digest" else [])
    finally:
        await cache.shutdown()
