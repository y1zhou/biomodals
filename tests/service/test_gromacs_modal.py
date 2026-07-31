"""Modal resource and Result boundaries for GROMACS service jobs."""

# ruff: noqa: D101,D102,D103,D107,S106

from __future__ import annotations

import asyncio
import hashlib
import io
import struct
import time
import zipfile
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import modal
import orjson
import pytest

from biomodals.app.bioinfo.gromacs_execution import REQUIRED_FUNCTIONS, execution_plan
from biomodals.execution import ProviderBinding
from biomodals.execution.modal import ModalCallObservationKind
from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.gromacs.archive import GROMACS_ARCHIVE_SCHEMA_VERSION
from biomodals.service.gromacs.contracts import artifact_request_sha256
from biomodals.service.gromacs.modal import (
    ArchiveNotReadyError,
    GromacsResultInvalidError,
    ModalGromacsAdapter,
)
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
)
from biomodals.service.store import JobRecord, JobState, ServiceStore, UserRecord

RUN_NAME = "first-simulation-0123456789abcdef0123456789abcdef"
SHA256 = "a" * 64
PDB = b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n"
XTC = struct.pack(
    ">iiif9fi3f",
    1995,
    1,
    0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1,
    0.0,
    0.0,
    0.0,
)
_TPR_VERSION = b"VERSION 2026.1"
TPR = (struct.pack(">II", 15, len(_TPR_VERSION)) + _TPR_VERSION).ljust(1024, b"\0")
_FIXTURE_MTIME = 1_700_000_000


def _png_chunk(chunk_type: bytes, content: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(content, checksum)
    return (
        struct.pack(">I", len(content))
        + chunk_type
        + content
        + struct.pack(">I", checksum)
    )


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    + _png_chunk(b"IDAT", zlib.compress(b"\0\0\0\0\0"))
    + _png_chunk(b"IEND", b"")
)


class AsyncMethod:
    def __init__(self, function):
        self.aio = function


class FakeCall:
    def __init__(self, object_id: str, *, result: object = "/outputs/result") -> None:
        self.object_id = object_id
        self.result = result
        self.cancelled = False
        self.get = AsyncMethod(self._get)
        self.cancel = AsyncMethod(self._cancel)

    async def _get(self, *, timeout: int):
        assert timeout == 0
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def _cancel(self):
        self.cancelled = True


class FakeFunction:
    def __init__(self, call: FakeCall) -> None:
        self.call = call
        self.hydrated = False
        self.spawn_args: tuple[object, ...] | None = None
        self.spawn_kwargs: dict[str, object] | None = None
        self.hydrate = AsyncMethod(self._hydrate)
        self.spawn = AsyncMethod(self._spawn)

    async def _hydrate(self) -> None:
        self.hydrated = True

    async def _spawn(self, *args, **kwargs):
        self.spawn_args = args
        self.spawn_kwargs = kwargs
        return self.call


def _admission_configuration() -> JobAdmissionConfiguration:
    return JobAdmissionConfiguration(
        workload="gromacs",
        modal_environment=DatabaseOverridableSetting("department-dev", False),
        modal_app_name=DatabaseOverridableSetting("GromacsAPI", False),
        modal_app_version=DatabaseOverridableSetting(17, False),
        workload_active_job_limit=DatabaseOverridableSetting(10, False),
        global_active_job_limit=DatabaseOverridableSetting(10, False),
    )


def _enable_created_user(store: ServiceStore, user: UserRecord) -> UserRecord:
    enabled = store.set_password_from_token(
        b"setup-token-digest",
        password_hash="test-hash",
        session_token_digest=b"test-session",
        csrf_digest=b"test-csrf",
        now=2,
        absolute_expires_at=3_600,
    )
    assert enabled is not None
    assert enabled.user_id == user.user_id
    return enabled


def _submitted_job(tmp_path: Path) -> tuple[ServiceStore, JobRecord]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    user = _enable_created_user(
        store,
        store.create_user(
            email="alice@example.com",
            display_name="Alice",
            token_digest=b"setup-token-digest",
            token_expires_at=3_600,
            now=1,
            is_admin=True,
            active_job_limit=10,
        ),
    )
    parameters_json = "{}"
    admission = store.admit_job(
        owner_user_id=user.user_id,
        display_name="Simulation",
        idempotency_key=str(uuid4()),
        request_hash="request-digest",
        parameters_json=parameters_json,
        artifact_request_sha256=artifact_request_sha256(PDB, parameters_json),
        configuration=_admission_configuration(),
        execution_plan=execution_plan(
            cpu_only=False,
            workload_run_key=RUN_NAME,
            pdb_sha256=hashlib.sha256(PDB).hexdigest(),
            simulation_time_ns=5,
            run_pdbfixer=False,
        ),
        execution_run_id=uuid4(),
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=1,
        now=2,
    )
    return store, admission.job


def _published_job(
    job: JobRecord,
    archive_bytes: bytes,
    *,
    schema_version: int = GROMACS_ARCHIVE_SCHEMA_VERSION,
) -> JobRecord:
    return replace(
        job,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{RUN_NAME}/result.zip",
        result_filename=f"{RUN_NAME}.zip",
        result_size_bytes=len(archive_bytes),
        result_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        result_archive_schema_version=schema_version,
        finalization_started_at=10,
        completed_at=10,
    )


def _valid_archive_bytes() -> tuple[bytes, str]:
    prefix = f"production_{RUN_NAME}"
    members = {
        "input.pdb": PDB,
        "metadata/parameters.json": b"{}",
        "metadata/provenance.json": b"{}\n",
        "metadata/stages.json": b"[]\n",
        "metadata/run.log": b"completed\n",
        "outputs/production.mdp": b"integrator = md\n",
        f"outputs/{prefix}_nopbc.xtc": XTC,
        f"outputs/{prefix}.tpr": TPR,
        f"outputs/{prefix}_nopbc_centered.pdb": PDB,
        f"outputs/rmsd_{prefix}.csv": b"time_ns,rmsd\n0.0,0.1\n",
        f"outputs/rmsd_{prefix}.png": PNG,
        f"outputs/rg_{prefix}.csv": b"time_ns,rg\n0.0,1.2\n",
        f"outputs/rg_{prefix}.png": PNG,
        f"outputs/rmsf_{prefix}.csv": b"residue_index,rmsf\n1,0.2\n",
        f"outputs/rmsf_{prefix}.png": PNG,
    }
    roles = {
        "input.pdb": "input_structure",
        "metadata/parameters.json": "normalized_parameters",
        "metadata/provenance.json": "provenance",
        "metadata/stages.json": "stages",
        "metadata/run.log": "run_log",
        "outputs/production.mdp": "production_parameters",
        f"outputs/{prefix}_nopbc.xtc": "trajectory",
        f"outputs/{prefix}.tpr": "production_topology",
        f"outputs/{prefix}_nopbc_centered.pdb": "centered_structure",
        f"outputs/rmsd_{prefix}.csv": "rmsd",
        f"outputs/rmsd_{prefix}.png": "rmsd_plot",
        f"outputs/rg_{prefix}.csv": "radius_of_gyration",
        f"outputs/rg_{prefix}.png": "radius_of_gyration_plot",
        f"outputs/rmsf_{prefix}.csv": "rmsf",
        f"outputs/rmsf_{prefix}.png": "rmsf_plot",
    }
    records = [
        {
            "path": name,
            "role": roles[name],
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in members.items()
    ]
    manifest = orjson.dumps({
        "archive_schema_version": GROMACS_ARCHIVE_SCHEMA_VERSION,
        "run_name": RUN_NAME,
        "files": records,
    })
    checksums = "".join([
        *(f"{record['sha256']}  {record['path']}\n" for record in records),
        f"{hashlib.sha256(manifest).hexdigest()}  metadata/manifest.json\n",
    ]).encode()
    output = io.BytesIO()

    def archive_info(name: str, *, source: bool = False) -> zipfile.ZipInfo:
        if source:
            source_time = time.gmtime(_FIXTURE_MTIME)[:6]
            date_time = (*source_time[:5], source_time[5] // 2 * 2)
        else:
            date_time = (1980, 1, 1, 0, 0, 0)
        info = zipfile.ZipInfo(name, date_time=date_time)
        info.compress_type = zipfile.ZIP_STORED
        if source:
            info.extra = struct.pack("<HHBI", 0x5455, 5, 1, _FIXTURE_MTIME)
        return info

    with zipfile.ZipFile(output, "w") as archive:
        for name, content in members.items():
            archive.writestr(
                archive_info(
                    name,
                    source=name == "input.pdb" or name.startswith("outputs/"),
                ),
                content,
            )
        archive.writestr(archive_info("metadata/manifest.json"), manifest)
        archive.writestr(archive_info("metadata/checksums.sha256"), checksums)
    request_digest = hashlib.sha256()
    request_digest.update(len(members["input.pdb"]).to_bytes(8, "big"))
    request_digest.update(members["input.pdb"])
    request_digest.update(members["metadata/parameters.json"])
    return output.getvalue(), request_digest.hexdigest()


def _result_marker(archive_bytes: bytes, request_sha256: str) -> bytes:
    return orjson.dumps({
        "archive_schema_version": GROMACS_ARCHIVE_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "size_bytes": len(archive_bytes),
    })


def _replace_archive_member(
    archive_bytes: bytes,
    name: str,
    content: bytes,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
        members = [
            (info, content if info.filename == name else source.read(info))
            for info in source.infolist()
        ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info, member_content in members:
            archive.writestr(info, member_content)
    return output.getvalue()


def _install_volume(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    *,
    mtimes: dict[str, int] | None = None,
) -> list[str]:
    source_mtimes = (
        mtimes if mtimes is not None else {path: 1_700_000_000 for path in files}
    )
    removed: list[str] = []

    @dataclass(frozen=True)
    class FakeFileEntry:
        path: str
        mtime: int

    class FakeVolume:
        def __init__(self) -> None:
            self.read_file = AsyncMethod(self._read_file)
            self.listdir = AsyncMethod(self._listdir)
            self.batch_upload = AsyncMethod(self._batch_upload)
            self.remove_file = AsyncMethod(self._remove_file)

        async def _read_file(self, path: str):
            if path not in files:
                raise FileNotFoundError(path)
            yield files[path]

        async def _listdir(self, path: str):
            if path in files:
                return [FakeFileEntry(path, source_mtimes[path])]
            prefix = f"{path.strip('/')}/"
            return [
                FakeFileEntry(file_path, source_mtimes[file_path])
                for file_path in files
                if file_path.startswith(prefix)
                and "/" not in file_path.removeprefix(prefix)
            ]

        @asynccontextmanager
        async def _batch_upload(self, *, force: bool):
            assert force is True

            class Upload:
                @staticmethod
                def put_file(source, path: str) -> None:
                    assert isinstance(source, (str, Path))
                    files[path] = Path(source).read_bytes()

            yield Upload()

        async def _remove_file(self, path: str, *, recursive: bool) -> None:
            assert recursive is True
            removed.append(path)
            prefix = f"{path.rstrip('/')}/"
            for existing in list(files):
                if existing == path or existing.startswith(prefix):
                    files.pop(existing)

    monkeypatch.setattr(
        modal.Volume,
        "from_name",
        lambda name, *, environment_name: (
            FakeVolume()
            if (name, environment_name) == ("Gromacs-outputs", "department-dev")
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )
    return removed


def _established_output_files() -> dict[str, bytes]:
    prefix = f"production_{RUN_NAME}"
    return {
        f"{RUN_NAME}/{RUN_NAME}.pdb": PDB,
        f"{RUN_NAME}/production.mdp": b"integrator = md\n",
        f"{RUN_NAME}/{prefix}.xtc": XTC + b"full-trajectory",
        f"{RUN_NAME}/{prefix}_nopbc.xtc": XTC,
        f"{RUN_NAME}/{prefix}.tpr": TPR,
        f"{RUN_NAME}/{prefix}_nopbc_centered.pdb": PDB,
        f"{RUN_NAME}/rmsd_{prefix}.csv": b"time_ns,rmsd\n0.0,0.1\n",
        f"{RUN_NAME}/rmsd_{prefix}.png": PNG,
        f"{RUN_NAME}/rg_{prefix}.csv": b"time_ns,rg\n0.0,1.2\n",
        f"{RUN_NAME}/rg_{prefix}.png": PNG,
        f"{RUN_NAME}/rmsf_{prefix}.csv": b"residue_index,rmsf\n1,0.2\n",
        f"{RUN_NAME}/rmsf_{prefix}.png": PNG,
    }


def _adapter(
    *,
    calls: dict[str, FakeCall] | None = None,
    function_resolver=None,
    artifact_cache: ArtifactCache | None = None,
) -> ModalGromacsAdapter:
    call_map = calls or {}
    return ModalGromacsAdapter(
        artifact_cache=artifact_cache,
        call_resolver=cast(Any, call_map.__getitem__),
        function_resolver=function_resolver,
    )


def test_preflight_hydrates_volume_and_every_required_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    class Hydratable:
        def __init__(self, kind: str, name: str) -> None:
            self.hydrate = AsyncMethod(lambda: self._hydrate(kind, name))

        async def _hydrate(self, kind: str, name: str) -> None:
            events.append((kind, name))

    monkeypatch.setattr(
        modal.Volume,
        "from_name",
        lambda name, *, environment_name: (
            Hydratable("volume", name)
            if environment_name == "candidate-env"
            else (_ for _ in ()).throw(AssertionError(environment_name))
        ),
    )

    def resolve_function(
        app_name: str,
        function_name: str,
        *,
        environment_name: str,
        version: int,
    ):
        assert (app_name, environment_name, version) == (
            "CandidateApp",
            "candidate-env",
            23,
        )
        return Hydratable("function", function_name)

    asyncio.run(
        _adapter(function_resolver=resolve_function).preflight(
            "CandidateApp",
            "candidate-env",
            23,
        )
    )

    assert events == [
        ("volume", "Gromacs-outputs"),
        *(("function", name) for name in REQUIRED_FUNCTIONS),
    ]


def test_adapter_drives_one_exact_kernel_call() -> None:
    call = FakeCall("fc-1")
    function = FakeFunction(call)
    lookups: list[tuple[str, str, str, int]] = []

    def resolve_function(
        app_name: str,
        function_name: str,
        *,
        environment_name: str,
        version: int,
    ):
        lookups.append((app_name, function_name, environment_name, version))
        return function

    adapter = _adapter(
        calls={"fc-1": call},
        function_resolver=resolve_function,
    )
    binding = ProviderBinding(
        environment="department-dev",
        app_name="GromacsAPI",
        app_version=17,
        function_name="prepare_tpr_cpu",
        uses_gpu=False,
    )

    async def exercise() -> None:
        resolved = await adapter.resolve(binding)
        call_id = await adapter.spawn(
            resolved,
            args=(),
            kwargs={"pdb_content": PDB, "run_name": RUN_NAME},
        )
        observation = await adapter.observe(call_id)
        await adapter.cancel(call_id)
        assert observation.kind == ModalCallObservationKind.SUCCEEDED

    asyncio.run(exercise())

    assert lookups == [
        ("GromacsAPI", "prepare_tpr_cpu", "department-dev", 17),
    ]
    assert function.hydrated is True
    assert function.spawn_kwargs == {"pdb_content": PDB, "run_name": RUN_NAME}
    assert call.cancelled is True


def test_service_packages_only_user_facing_outputs_and_preserves_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    files = _established_output_files()
    mtimes = {path: 1_718_372_121 + index * 120 for index, path in enumerate(files)}
    _install_volume(monkeypatch, files, mtimes=mtimes)

    archive = asyncio.run(_adapter().publish_archive(job, completed_at=10))

    archive_bytes = files[f"api-results/{RUN_NAME}/result.zip"]
    marker = orjson.loads(files[f"api-results/{RUN_NAME}/result.json"])
    assert archive.state == JobState.SUCCEEDED
    assert archive.size_bytes == len(archive_bytes)
    assert marker["archive_sha256"] == archive.sha256
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as result:
        assert result.read("input.pdb") == files[f"{RUN_NAME}/{RUN_NAME}.pdb"]
        assert result.read(f"outputs/production_{RUN_NAME}_nopbc.xtc") == XTC
        assert f"outputs/production_{RUN_NAME}.xtc" not in result.namelist()
        input_mtime = mtimes[f"{RUN_NAME}/{RUN_NAME}.pdb"]
        assert result.getinfo("input.pdb").extra == struct.pack(
            "<HHBI",
            0x5455,
            5,
            1,
            input_mtime,
        )


def test_publication_rejects_volume_input_that_differs_from_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    files = _established_output_files()
    files[f"{RUN_NAME}/{RUN_NAME}.pdb"] = (
        b"ATOM      2  CA  GLY A   1       1.000   1.000   1.000\nEND\n"
    )
    _install_volume(monkeypatch, files)

    with pytest.raises(
        GromacsResultInvalidError,
        match="does not match the admitted request",
    ):
        asyncio.run(_adapter().publish_archive(job, completed_at=10))


def test_published_archive_is_promoted_into_the_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    files = _established_output_files()
    _install_volume(monkeypatch, files)
    cache = ArtifactCache(tmp_path / "cache")

    archive = asyncio.run(
        _adapter(artifact_cache=cache).publish_archive(job, completed_at=10)
    )

    cached = tmp_path / "cache" / f"{job.job_id}.zip"
    assert archive.cache_lease is not None
    assert cached.read_bytes() == files[f"api-results/{RUN_NAME}/result.zip"]
    assert cache.clear().entries == 0
    archive.cache_lease.close()
    assert cache.clear().entries == 1


def test_rebuild_rejects_an_unavailable_archive_schema(tmp_path: Path) -> None:
    _store, job = _submitted_job(tmp_path)
    unsupported = _published_job(job, b"archive", schema_version=999)

    async def rebuild() -> bytes:
        return b"".join([
            chunk async for chunk in _adapter().rebuild_artifact(unsupported)
        ])

    with pytest.raises(ValueError, match="unsupported Result archive schema"):
        asyncio.run(rebuild())


def test_rebuild_and_recovery_use_bounded_local_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    files = _established_output_files()
    _install_volume(monkeypatch, files)
    cache = ArtifactCache(tmp_path / "cache")
    adapter = _adapter(artifact_cache=cache)
    published = asyncio.run(adapter.publish_archive(job, completed_at=10))
    completed = _published_job(
        job,
        files[f"api-results/{RUN_NAME}/result.zip"],
    )
    if published.cache_lease is not None:
        published.cache_lease.close()
    operations: list[str] = []
    original_run_bounded = cache.run_bounded

    async def recording_run_bounded(operation, /, *args, **kwargs):
        operations.append(operation.__name__)
        return await original_run_bounded(operation, *args, **kwargs)

    monkeypatch.setattr(cache, "run_bounded", recording_run_bounded)

    async def exercise() -> bytes:
        try:
            rebuilt = b"".join([
                chunk async for chunk in adapter.rebuild_artifact(completed)
            ])
            assert {"seek", "read"}.issubset(operations)
            operations.clear()
            recovered = await adapter.recover_archive(completed)
            assert recovered.sha256 == published.sha256
            assert "write" in operations
            return rebuilt
        finally:
            await cache.shutdown()

    rebuilt = asyncio.run(exercise())
    assert hashlib.sha256(rebuilt).hexdigest() == published.sha256


def test_normal_cache_restore_requires_the_published_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    archive_bytes, _request_sha256 = _valid_archive_bytes()
    completed = _published_job(job, archive_bytes)
    _install_volume(
        monkeypatch,
        {f"api-results/{RUN_NAME}/result.zip": archive_bytes},
    )

    async def restore() -> bytes:
        return b"".join([chunk async for chunk in _adapter().read_artifact(completed)])

    with pytest.raises(ArtifactIntegrityError, match="marker"):
        asyncio.run(restore())


def test_recovery_rejects_corrupt_archive_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    archive_bytes, request_sha256 = _valid_archive_bytes()
    corrupt = bytearray(archive_bytes)
    corrupt[-1] ^= 1
    _install_volume(
        monkeypatch,
        {
            f"api-results/{RUN_NAME}/result.json": _result_marker(
                archive_bytes,
                request_sha256,
            ),
            f"api-results/{RUN_NAME}/result.zip": bytes(corrupt),
        },
    )

    with pytest.raises(ValueError, match="does not match its marker"):
        asyncio.run(_adapter().recover_archive(job))


def test_recovery_rejects_invalid_zip_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    archive_bytes, request_sha256 = _valid_archive_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        manifest = orjson.loads(archive.read("metadata/manifest.json"))
    manifest["files"][0]["sha256"] = "0" * 64
    invalid_archive = _replace_archive_member(
        archive_bytes,
        "metadata/manifest.json",
        orjson.dumps(manifest),
    )
    _install_volume(
        monkeypatch,
        {
            f"api-results/{RUN_NAME}/result.json": _result_marker(
                invalid_archive,
                request_sha256,
            ),
            f"api-results/{RUN_NAME}/result.zip": invalid_archive,
        },
    )

    with pytest.raises(ValueError, match="does not match manifest"):
        asyncio.run(_adapter().recover_archive(job))


def test_missing_recovery_marker_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    _install_volume(monkeypatch, {})

    with pytest.raises(ArchiveNotReadyError, match="marker is missing"):
        asyncio.run(_adapter().recover_archive(job))


def test_recovery_rejects_archive_identity_from_another_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    archive_bytes, request_sha256 = _valid_archive_bytes()
    mismatched = replace(job, artifact_request_sha256="0" * 64)
    _install_volume(
        monkeypatch,
        {
            f"api-results/{RUN_NAME}/result.json": _result_marker(
                archive_bytes,
                request_sha256,
            ),
            f"api-results/{RUN_NAME}/result.zip": archive_bytes,
        },
    )

    with pytest.raises(ValueError, match="does not match the admitted request"):
        asyncio.run(_adapter().recover_archive(mismatched))


def test_intermediate_cleanup_preserves_published_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    files = {
        f"{RUN_NAME}/production.mdp": b"rebuildable",
        f"api-results/{RUN_NAME}/result.zip": b"published",
        f"api-results/{RUN_NAME}/result.json": b"marker",
    }
    removed = _install_volume(monkeypatch, files)

    asyncio.run(_adapter().cleanup_intermediates(job))

    assert removed == [RUN_NAME]
    assert f"{RUN_NAME}/production.mdp" not in files
    assert files[f"api-results/{RUN_NAME}/result.zip"] == b"published"
    assert files[f"api-results/{RUN_NAME}/result.json"] == b"marker"
