"""CPU preparation and verified local packaging, with no real Modal operations."""

# ruff: noqa: D103

import asyncio
import io
import zipfile
from types import SimpleNamespace
from uuid import uuid4

import orjson
import pytest
from nanobody_fixture import VHH, result_directory

from biomodals.execution import DeploymentIdentity
from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.nanobody_humanization import modal as service_modal
from biomodals.service.pending import PendingRequestStore
from biomodals.service.tool_runtime import EnvironmentPreparationError, SubmissionWait
from biomodals.workflow.nanobody_humanization.execution import NanobodyExecutionRequest
from biomodals.workflow.nanobody_humanization.preparation import VHInput, prepare_vh


def setup(tmp_path, monkeypatch):
    pending = PendingRequestStore(tmp_path)
    pending.initialize()
    adapter = service_modal.NanobodyToolAdapter(pending, modal_download_concurrency=2)
    job = SimpleNamespace(
        job_id=uuid4(),
        modal_environment="fixture",
        modal_app_name="NanobodyHumanizationWorkflow",
        modal_app_version=3,
    )
    request = NanobodyExecutionRequest(
        run_name="fixture", parents=(prepare_vh(VHInput(id="one", vhh=VHH)),)
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
async def test_preflight_hydrates_actual_new_entrypoint_without_invoking_it(
    tmp_path, monkeypatch
):
    adapter, job, _ = setup(tmp_path, monkeypatch)
    calls = []

    def resolve(app, name, **kwargs):
        async def hydrate():
            calls.append((app, name, kwargs))

        return SimpleNamespace(hydrate=SimpleNamespace(aio=hydrate))

    monkeypatch.setattr(service_modal.modal.Function, "from_name", resolve)
    await adapter.preflight(
        DeploymentIdentity(
            job.modal_environment, job.modal_app_name, job.modal_app_version
        )
    )
    assert calls == [
        (
            job.modal_app_name,
            "abnativ2_vhh_generate",
            {"environment_name": "fixture", "version": 3},
        )
    ]


def test_historical_manifest_remains_byte_identical_in_download(tmp_path, monkeypatch):
    adapter, job, request = setup(tmp_path, monkeypatch)
    root = result_directory(tmp_path / "volume", job.job_id, request)
    manifest = orjson.loads((root / "manifest.json").read_bytes())
    manifest["schema_version"] = 1
    manifest.pop("abnativ2_searches")
    manifest["parameters"].pop("abnativ2_explore")
    manifest["parameters"].pop("abnativ2_candidate_budget")
    original = orjson.dumps(manifest)
    (root / "manifest.json").write_bytes(original)
    output = io.BytesIO()
    service_modal.build_nanobody_archive(root, output)
    with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
        assert archive.read("manifest.json") == original
        assert archive.read("selection.csv") == (root / "selection.csv").read_bytes()


@pytest.mark.anyio
@pytest.mark.parametrize("fail", [False, True])
async def test_preparation_is_shared_pinned_and_never_silently_retried(
    tmp_path, monkeypatch, fail
):
    adapter, job, request = setup(tmp_path, monkeypatch)
    adapter.pending.put(job.job_id, request.to_bytes())
    gate, started = asyncio.Event(), asyncio.Event()
    calls, staged = [], []

    def function(app_name, function_name, **kwargs):
        async def run():
            calls.append((app_name, function_name, kwargs))
            started.set()
            await gate.wait()
            if fail:
                raise RuntimeError("Checksum mismatch")

        return SimpleNamespace(remote=SimpleNamespace(aio=run))

    monkeypatch.setattr(service_modal.modal.Function, "from_name", function)
    monkeypatch.setattr(
        service_modal, "stage_execution_request", lambda *_: staged.append("request")
    )
    monkeypatch.setattr(
        service_modal, "stage_execution_launch", lambda *_: staged.append("launch")
    )
    assert isinstance(await adapter.stage(job), SubmissionWait)
    await started.wait()
    assert isinstance(
        await adapter.stage(SimpleNamespace(**{**vars(job), "job_id": uuid4()})),
        SubmissionWait,
    )
    assert len(calls) == 1
    assert staged == []
    gate.set()
    task = next(iter(adapter._preparation_tasks.values()))
    await asyncio.gather(task, return_exceptions=True)
    if fail:
        for _ in range(2):
            with pytest.raises(
                EnvironmentPreparationError, match="operator intervention"
            ):
                await adapter.stage(job)
        assert len(calls) == 1
        assert adapter.pending.get(job.job_id) == request.to_bytes()
    else:
        assert await adapter.stage(job) is None
        assert staged == ["request", "launch"]
        assert [call[1] for call in calls] == [
            "stage_hudiff_nb_models",
            "stage_abnativ2_vhh_models",
        ]
        assert all(
            call[0] == job.modal_app_name
            and call[2] == {"environment_name": "fixture", "version": 3}
            for call in calls
        )
        assert adapter.pending.get(job.job_id) == request.to_bytes()
        await adapter.discard_pending(job)
        assert (await adapter.input_request(job)).parents == request.parents


@pytest.mark.anyio
@pytest.mark.parametrize(
    "damage",
    [
        None,
        "preparation_digest",
        "scientific_versions",
        "execution_run_id",
        "path",
        "digest",
    ],
)
async def test_real_publication_adapter_binds_preparation_and_verifies_all_files(
    tmp_path, monkeypatch, damage
):
    adapter, job, request = setup(tmp_path, monkeypatch)
    root = result_directory(tmp_path / "volume", job.job_id, request)
    content = (root / "manifest.json").read_bytes()
    manifest = orjson.loads(content)
    if damage == "preparation_digest":
        manifest[damage] = "0" * 64
    elif damage == "scientific_versions":
        manifest[damage] = {}
    elif damage == "execution_run_id":
        manifest[damage] = str(uuid4())
    elif damage == "path":
        manifest["files"][0]["path"] = "../outside"
    elif damage == "digest":
        manifest["files"][0]["content_sha256"] = "0" * 64

    async def read(_volume, path, *, max_bytes):
        assert path == str(service_modal.result_directory(job.job_id) / "manifest.json")
        assert len(content) < max_bytes
        return orjson.dumps(manifest)

    downloads = []

    def download(_volume, paths, *, concurrency):
        assert concurrency == 2
        for path, destination in paths:
            relative = path.removeprefix(
                str(service_modal.result_directory(job.job_id)) + "/"
            )
            downloads.append(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((root / relative).read_bytes())

    monkeypatch.setattr(service_modal, "read_modal_volume_file", read)
    monkeypatch.setattr(service_modal, "download_modal_volume_files", download)
    cache = ArtifactCache(tmp_path / "cache")
    try:
        if damage:
            with pytest.raises(ArtifactIntegrityError):
                await adapter.prepare_result(job, cache, completed_at=1)
            assert (
                await cache.acquire_async(
                    str(job.job_id), size_bytes=1, sha256="0" * 64
                )
                is None
            )
            return
        first = await adapter.prepare_result(job, cache, completed_at=1)
        assert await adapter.prepare_result(job, cache, completed_at=2) == first
        lease = await cache.acquire_async(
            str(job.job_id), size_bytes=first.size_bytes, sha256=first.sha256
        )
        assert lease is not None
        try:
            with zipfile.ZipFile(io.BytesIO(lease.read())) as archive:
                assert set(archive.namelist()) == {
                    file["path"] for file in manifest["files"]
                } | {"manifest.json"}
                assert (
                    archive.read("selection.csv")
                    == (root / "selection.csv").read_bytes()
                )
        finally:
            lease.close()
        assert len(downloads) == 2 * len(manifest["files"])
    finally:
        await cache.shutdown()
