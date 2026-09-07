"""Humanization staging, result identity and archive cache integration."""

# ruff: noqa: D103

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
from biomodals.workflow.humanization.contracts import AntibodyPair
from biomodals.workflow.humanization.execution import HumanizationExecutionRequest


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pending = PendingRequestStore(tmp_path)
    pending.initialize()
    adapter = HumanizationToolAdapter(pending, modal_download_concurrency=2)
    job: Any = SimpleNamespace(job_id=uuid4(), modal_environment="test")
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
    return adapter, job, request


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
