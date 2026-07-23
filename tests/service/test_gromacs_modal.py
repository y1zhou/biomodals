"""Modal boundary and reconciliation contracts for GROMACS jobs."""

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import modal
import orjson
import pytest

from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.gromacs.archive import GROMACS_ARCHIVE_SCHEMA_VERSION
from biomodals.service.gromacs.modal import (
    ArchiveNotReadyError,
    FinalArchive,
    GromacsReconciler,
    ModalGromacsAdapter,
    PollOutcome,
)
from biomodals.service.gromacs.router import GromacsJobOptions
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
    ModalConfigurationSnapshot,
)
from biomodals.service.store import (
    JobOperationState,
    JobRecord,
    JobState,
    JobSubmissionConflictError,
    ServiceStore,
    UserRecord,
)
from biomodals.service.submission import SubmissionOutcomeUnknownError

RUN_NAME = "first-simulation-0123456789abcdef0123456789abcdef"
SHA256 = "a" * 64
_PENDING = object()
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
PDB = b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n"


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


class AsyncMethod:
    def __init__(self, function):
        self.aio = function


@dataclass
class CallNode:
    function_call_id: str
    status: modal.call_graph.InputStatus
    children: list[CallNode] = field(default_factory=list)


class FakeCall:
    def __init__(
        self,
        object_id: str,
        *,
        result: Any = _PENDING,
        graph: list[CallNode] | None = None,
        events: list[tuple[str, str, object]] | None = None,
    ) -> None:
        self.object_id = object_id
        self.result = result
        self.graph = graph or []
        self.events = events if events is not None else []
        self.get = AsyncMethod(self._get)
        self.get_call_graph = AsyncMethod(self._get_call_graph)
        self.cancel = AsyncMethod(self._cancel)

    async def _get(self, *, timeout: int):
        self.events.append(("get", self.object_id, timeout))
        if self.result is _PENDING:
            raise TimeoutError
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def _get_call_graph(self):
        self.events.append(("graph", self.object_id, None))
        return self.graph

    async def _cancel(self, *, terminate_containers: bool):
        self.events.append(("cancel", self.object_id, terminate_containers))


class FakeFunction:
    def __init__(self, call: FakeCall) -> None:
        self.call = call
        self.spawn_kwargs: dict[str, object] | None = None
        self.spawn = AsyncMethod(self._spawn)

    async def _spawn(self, **kwargs):
        self.spawn_kwargs = kwargs
        return self.call


class FailingFunction:
    def __init__(self, error: Exception) -> None:
        self.calls = 0
        self.error = error
        self.spawn = AsyncMethod(self._spawn)

    async def _spawn(self, **_kwargs):
        self.calls += 1
        raise self.error


def _valid_archive_bytes() -> tuple[bytes, str]:
    prefix = f"production_{RUN_NAME}"
    members = {
        "input.pdb": PDB,
        "metadata/parameters.json": b"{}\n",
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
            info.extra = struct.pack(
                "<HHBI",
                0x5455,
                5,
                1,
                _FIXTURE_MTIME,
            )
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
) -> None:
    source_mtimes = (
        mtimes if mtimes is not None else {path: 1_700_000_000 for path in files}
    )

    @dataclass(frozen=True)
    class FakeFileEntry:
        path: str
        mtime: int

    class FakeVolume:
        def __init__(self) -> None:
            self.read_file = AsyncMethod(self._read_file)
            self.listdir = AsyncMethod(self._listdir)
            self.batch_upload = AsyncMethod(self._batch_upload)

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

    monkeypatch.setattr(
        modal.Volume,
        "from_name",
        lambda name, *, environment_name: (
            FakeVolume()
            if (name, environment_name) == ("Gromacs-outputs", "department-dev")
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )


def _established_output_files() -> dict[str, bytes]:
    prefix = f"production_{RUN_NAME}"
    return {
        f"{RUN_NAME}/{RUN_NAME}.pdb": PDB,
        f"{RUN_NAME}/production.mdp": b"integrator = md\n",
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
    calls: dict[str, FakeCall],
    *,
    output_volume_name: str = "Gromacs-outputs",
    function_resolver=None,
) -> ModalGromacsAdapter:
    return ModalGromacsAdapter(
        output_volume_name=output_volume_name,
        call_resolver=cast(Any, calls.__getitem__),
        function_resolver=function_resolver,
    )


def _attach_modal_operation(
    store: ServiceStore,
    job_id: UUID,
    *,
    operation: str,
    modal_call_id: str,
    now: int,
    run_name: str | None = None,
) -> JobRecord:
    token = f"setup-{operation}"
    claimed = store.claim_modal_operation(
        job_id,
        operation=operation,
        submission_token=token,
        run_name=run_name,
        now=now,
    )
    assert claimed is not None
    return store.attach_modal_call(
        job_id,
        operation=operation,
        modal_call_id=modal_call_id,
        submission_token=token,
        now=now,
    )


def _submitted_job(
    tmp_path: Path,
    *,
    cancel_requested: bool = False,
    operation: str = "prepare_tpr_gpu",
) -> tuple[ServiceStore, JobRecord]:
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
    admission = store.admit_job(
        owner_user_id=user.user_id,
        display_name="Simulation",
        idempotency_key=str(uuid4()),
        request_hash="request-digest",
        parameters_json="{}",
        configuration=_admission_configuration(),
        now=2,
    )
    if operation == "prepare_tpr_gpu":
        job = _attach_modal_operation(
            store,
            admission.job.job_id,
            modal_call_id="fc-root",
            operation=operation,
            run_name=RUN_NAME,
            now=3,
        )
        now = 3
    else:
        target_operation = operation
        job = _attach_modal_operation(
            store,
            admission.job.job_id,
            modal_call_id="fc-prepare",
            operation="prepare_tpr_gpu",
            run_name=RUN_NAME,
            now=3,
        )
        store.record_operation_outcome(
            job.job_id,
            operation="prepare_tpr_gpu",
            expected_modal_call_id="fc-prepare",
            outcome=JobOperationState.COMPLETED,
            now=4,
        )
        now = 5
        operations = [
            "collect_traj_stats:nvt_",
            "collect_traj_stats:npt_",
            "production_run_gpu",
            "collect_traj_stats:production_",
        ]
        for candidate in operations:
            operation_now = 7 if candidate == "collect_traj_stats:production_" else 5
            modal_call_id = (
                "fc-root" if candidate == target_operation else f"fc-{candidate}"
            )
            job = _attach_modal_operation(
                store,
                job.job_id,
                operation=candidate,
                modal_call_id=modal_call_id,
                now=operation_now,
            )
            if candidate == target_operation:
                break
            store.record_operation_outcome(
                job.job_id,
                operation=candidate,
                expected_modal_call_id=modal_call_id,
                outcome=JobOperationState.COMPLETED,
                now=6,
            )
        now = 7
    if cancel_requested:
        job = store.request_cancel(user.user_id, job.job_id, now=now + 1)
    return store, job


def _terminal_job(
    store: ServiceStore,
    owner_user_id: UUID,
    *,
    run_name: str,
    state: JobState,
    completed_at: int,
) -> JobRecord:
    admission = store.admit_job(
        owner_user_id=owner_user_id,
        display_name=run_name,
        idempotency_key=str(uuid4()),
        request_hash=f"digest-{run_name}",
        parameters_json="{}",
        configuration=_admission_configuration(),
        now=completed_at - 2,
    )
    job = _attach_modal_operation(
        store,
        admission.job.job_id,
        modal_call_id=f"fc-{run_name}",
        operation="collect_traj_stats:production_",
        run_name=run_name,
        now=completed_at - 1,
    )
    if state == JobState.FAILED:
        return store.fail_job(
            job.job_id,
            error_code="compute_failed",
            error_message="GROMACS job failed",
            now=completed_at,
        )
    return store.complete_job(
        job.job_id,
        state=state,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{run_name}/result.zip",
        result_filename=f"{run_name}.zip",
        result_size_bytes=123,
        result_sha256=SHA256,
        result_archive_schema_version=GROMACS_ARCHIVE_SCHEMA_VERSION,
        now=completed_at,
    )


def test_submit_resolves_the_deployed_prepare_function_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = FakeFunction(FakeCall("fc-prepare"))
    lookups: list[tuple[str, str, str | None, int | None]] = []

    def from_name(
        app_name: str,
        function_name: str,
        *,
        environment_name: str | None = None,
        version: int | None = None,
    ):
        lookups.append((app_name, function_name, environment_name, version))
        return function

    monkeypatch.setattr(modal.Function, "from_name", from_name)

    adapter = ModalGromacsAdapter()

    assert lookups == []

    submitted = asyncio.run(
        adapter.submit(
            b"PDB content",
            GromacsJobOptions(
                simulation_time_ns=3,
                run_pdbfixer=True,
                cpu_only=True,
            ),
            run_name=RUN_NAME,
            modal_configuration=ModalConfigurationSnapshot(
                environment="department-dev",
                app_name="GromacsAPI",
                app_version=17,
            ),
        )
    )

    assert lookups == [
        ("GromacsAPI", "prepare_tpr_cpu", "department-dev", 17),
    ]
    assert submitted.modal_call_id == "fc-prepare"
    assert submitted.operation == "prepare_tpr_cpu"
    assert function.spawn_kwargs == {
        "pdb_content": b"PDB content",
        "run_name": RUN_NAME,
        "simulation_time_ns": 3,
        "run_pdbfixer": True,
    }


def test_preflight_hydrates_volume_and_every_required_function_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    class Hydratable:
        def __init__(self, kind: str, name: str) -> None:
            self.kind = kind
            self.name = name
            self.hydrate = AsyncMethod(self._hydrate)

        async def _hydrate(self) -> None:
            events.append((self.kind, self.name))

    volume = Hydratable("volume", "Gromacs-outputs")

    def resolve_function(
        app_name: str,
        function_name: str,
        *,
        environment_name: str,
        version: int,
    ) -> modal.Function:
        assert (app_name, environment_name, version) == (
            "CandidateApp",
            "candidate-env",
            23,
        )
        return cast(modal.Function, Hydratable("function", function_name))

    monkeypatch.setattr(
        modal.Volume,
        "from_name",
        lambda name, *, environment_name: (
            volume
            if (name, environment_name) == ("Gromacs-outputs", "candidate-env")
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )
    adapter = ModalGromacsAdapter(
        function_resolver=resolve_function,
    )

    asyncio.run(adapter.preflight("CandidateApp", "candidate-env", 23))

    assert events == [
        ("volume", "Gromacs-outputs"),
        ("function", "prepare_tpr_cpu"),
        ("function", "prepare_tpr_gpu"),
        ("function", "collect_traj_stats"),
        ("function", "production_run_cpu"),
        ("function", "production_run_gpu"),
    ]


def test_submit_spawns_detached_call_with_normalized_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = FakeFunction(FakeCall("fc-detached"))
    monkeypatch.setattr(modal.Function, "from_name", lambda *_args, **_kwargs: function)
    adapter = ModalGromacsAdapter()

    submitted = asyncio.run(
        adapter.submit(
            b"PDB content",
            GromacsJobOptions(
                simulation_time_ns=3,
                run_pdbfixer=True,
                cpu_only=True,
            ),
            run_name=RUN_NAME,
            modal_configuration=ModalConfigurationSnapshot(
                environment="department-dev",
                app_name="GromacsAPI",
                app_version=17,
            ),
        )
    )

    assert submitted.modal_call_id == "fc-detached"
    assert submitted.run_name == RUN_NAME
    assert function.spawn_kwargs == {
        "pdb_content": b"PDB content",
        "run_name": RUN_NAME,
        "simulation_time_ns": 3,
        "run_pdbfixer": True,
    }


def test_submit_marks_a_spawn_error_as_an_unknown_provider_outcome() -> None:
    adapter = ModalGromacsAdapter(
        function_resolver=cast(
            Any,
            lambda *_args, **_kwargs: FailingFunction(
                modal.exception.ConnectionError("temporary")
            ),
        ),
    )

    with pytest.raises(SubmissionOutcomeUnknownError):
        asyncio.run(
            adapter.submit(
                b"PDB content",
                GromacsJobOptions(),
                run_name=RUN_NAME,
                modal_configuration=ModalConfigurationSnapshot(
                    environment="department-dev",
                    app_name="GromacsAPI",
                    app_version=17,
                ),
            )
        )


def test_submit_preserves_a_definite_modal_rejection() -> None:
    adapter = ModalGromacsAdapter(
        function_resolver=cast(
            Any,
            lambda *_args, **_kwargs: FailingFunction(
                modal.exception.PermissionDeniedError("not allowed")
            ),
        ),
    )

    with pytest.raises(modal.exception.PermissionDeniedError):
        asyncio.run(
            adapter.submit(
                b"PDB content",
                GromacsJobOptions(),
                run_name=RUN_NAME,
                modal_configuration=ModalConfigurationSnapshot(
                    environment="department-dev",
                    app_name="GromacsAPI",
                    app_version=17,
                ),
            )
        )


def test_submit_operation_resolves_each_deployed_stage_by_name(tmp_path: Path) -> None:
    _store, job = _submitted_job(tmp_path)
    lookups: list[tuple[str, str, str | None, int | None]] = []
    functions: list[FakeFunction] = []

    def from_name(
        app_name: str,
        function_name: str,
        *,
        environment_name: str | None = None,
        version: int | None = None,
    ):
        lookups.append((app_name, function_name, environment_name, version))
        function = FakeFunction(FakeCall(f"fc-{len(functions) + 1}"))
        functions.append(function)
        return function

    adapter = ModalGromacsAdapter(
        function_resolver=from_name,
    )

    operations = (
        "collect_traj_stats:nvt_",
        "collect_traj_stats:npt_",
        "production_run_gpu",
        "collect_traj_stats:production_",
    )
    for operation in operations:
        submitted = asyncio.run(adapter.submit_operation(job, operation))
        assert submitted.operation == operation

    assert [function_name for _, function_name, _, _ in lookups] == [
        "collect_traj_stats",
        "collect_traj_stats",
        "production_run_gpu",
        "collect_traj_stats",
    ]
    assert {version for _, _, _, version in lookups} == {17}
    assert functions[0].spawn_kwargs == {
        "traj_prefix": "nvt_",
        "run_name": RUN_NAME,
    }
    assert functions[1].spawn_kwargs == {
        "traj_prefix": "npt_",
        "run_name": RUN_NAME,
    }
    assert functions[2].spawn_kwargs == {
        "run_name": RUN_NAME,
        "simulation_time_ns": 5,
    }
    assert functions[3].spawn_kwargs == {
        "traj_prefix": "production_",
        "run_name": RUN_NAME,
        "save_processed_traj": True,
    }


@pytest.mark.parametrize(
    ("graph", "expected_kind"),
    [
        (
            [CallNode("fc-root", modal.call_graph.InputStatus.PENDING)],
            "running",
        ),
        (
            [CallNode("fc-root", modal.call_graph.InputStatus.FAILURE)],
            "failed",
        ),
        (
            [CallNode("fc-root", modal.call_graph.InputStatus.INIT_FAILURE)],
            "failed",
        ),
        (
            [CallNode("fc-root", modal.call_graph.InputStatus.TIMEOUT)],
            "failed",
        ),
        (
            [CallNode("fc-root", modal.call_graph.InputStatus.SUCCESS)],
            "running",
        ),
    ],
)
def test_poll_uses_call_graph_to_classify_get_timeout(
    graph: list[CallNode],
    expected_kind: str,
) -> None:
    root = FakeCall("fc-root", graph=graph)
    adapter = _adapter({"fc-root": root})

    outcome = asyncio.run(adapter.poll("fc-root"))

    assert outcome.kind == expected_kind
    assert ("get", "fc-root", 0) in root.events


def test_poll_waits_for_cancelled_call_graph_descendants() -> None:
    active_child = CallNode("fc-child", modal.call_graph.InputStatus.PENDING)
    root_node = CallNode(
        "fc-root",
        modal.call_graph.InputStatus.TERMINATED,
        children=[active_child],
    )
    root = FakeCall(
        "fc-root",
        result=modal.exception.InputCancellation("cancelled"),
        graph=[root_node],
    )
    adapter = _adapter({"fc-root": root})

    assert asyncio.run(adapter.poll("fc-root")).kind == "running"
    assert ("graph", "fc-root", None) in root.events

    active_child.status = modal.call_graph.InputStatus.TERMINATED
    assert asyncio.run(adapter.poll("fc-root")).kind == "cancelled"


def test_cancel_stops_root_and_only_active_descendants() -> None:
    events: list[tuple[str, str, object]] = []
    active_a = CallNode("fc-active-a", modal.call_graph.InputStatus.PENDING)
    active_b = CallNode("fc-active-b", modal.call_graph.InputStatus.PENDING)
    finished = CallNode("fc-finished", modal.call_graph.InputStatus.SUCCESS)
    root_node = CallNode(
        "fc-root",
        modal.call_graph.InputStatus.PENDING,
        children=[active_b, finished, active_a],
    )
    calls = {
        call_id: FakeCall(call_id, events=events)
        for call_id in ("fc-active-a", "fc-active-b", "fc-finished")
    }
    calls["fc-root"] = FakeCall("fc-root", graph=[root_node], events=events)
    adapter = _adapter(calls)

    asyncio.run(adapter.cancel("fc-root"))

    assert [event for event in events if event[0] == "cancel"] == [
        ("cancel", "fc-active-a", False),
        ("cancel", "fc-active-b", False),
        ("cancel", "fc-root", False),
    ]


def test_service_packages_and_publishes_established_app_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    files = _established_output_files()
    mtimes = {path: 1_718_372_121 + index * 120 for index, path in enumerate(files)}
    _install_volume(monkeypatch, files, mtimes=mtimes)

    archive = asyncio.run(_adapter({}).publish_archive(job, completed_at=10))

    archive_bytes = files[f"api-results/{RUN_NAME}/result.zip"]
    marker = orjson.loads(files[f"api-results/{RUN_NAME}/result.json"])
    assert archive.state == JobState.SUCCEEDED
    assert archive.path == f"api-results/{RUN_NAME}/result.zip"
    assert archive.size_bytes == len(archive_bytes)
    assert archive.sha256 == hashlib.sha256(archive_bytes).hexdigest()
    assert marker["archive_sha256"] == archive.sha256
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as result:
        assert result.read("input.pdb") == files[f"{RUN_NAME}/{RUN_NAME}.pdb"]
        assert result.read(f"outputs/production_{RUN_NAME}_nopbc.xtc") == XTC
        input_mtime = mtimes[f"{RUN_NAME}/{RUN_NAME}.pdb"]
        assert result.getinfo("input.pdb").extra == struct.pack(
            "<HHBI",
            0x5455,
            5,
            1,
            input_mtime,
        )


def test_published_archive_is_promoted_into_the_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    files = _established_output_files()
    _install_volume(monkeypatch, files)
    cache = ArtifactCache(tmp_path / "cache")
    adapter = ModalGromacsAdapter(
        artifact_cache=cache,
        call_resolver=cast(Any, {}.__getitem__),
    )

    archive = asyncio.run(adapter.publish_archive(job, completed_at=10))

    cached = tmp_path / "cache" / f"{job.job_id}.zip"
    assert archive.cache_lease is not None
    assert cached.read_bytes() == files[f"api-results/{RUN_NAME}/result.zip"]
    assert cache.clear().entries == 0
    archive.cache_lease.close()
    assert cache.clear().entries == 1


def test_reconciler_fans_out_after_preparation(
    tmp_path: Path,
) -> None:
    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="/volumes/Gromacs-outputs/api-run")
    functions = [
        FakeFunction(FakeCall("fc-nvt")),
        FakeFunction(FakeCall("fc-npt")),
        FakeFunction(FakeCall("fc-production")),
    ]
    lookups: list[tuple[str, str, str | None, int | None]] = []

    def from_name(
        app_name: str,
        function_name: str,
        *,
        environment_name: str | None = None,
        version: int | None = None,
    ) -> FakeFunction:
        lookups.append((app_name, function_name, environment_name, version))
        return functions[len(lookups) - 1]

    reconciler = GromacsReconciler(
        store,
        _adapter({"fc-root": prepare}, function_resolver=from_name),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())

    advanced = store.get_job(job.owner_user_id, job.job_id)
    assert advanced is not None
    assert [
        (call.operation, call.modal_call_id)
        for call in store.list_operations(job.job_id)
    ] == [
        ("prepare_tpr_gpu", "fc-root"),
        ("collect_traj_stats:nvt_", "fc-nvt"),
        ("collect_traj_stats:npt_", "fc-npt"),
        ("production_run_gpu", "fc-production"),
    ]
    assert [lookup[1] for lookup in lookups] == [
        "collect_traj_stats",
        "collect_traj_stats",
        "production_run_gpu",
    ]
    assert functions[0].spawn_kwargs == {
        "traj_prefix": "nvt_",
        "run_name": RUN_NAME,
    }
    assert functions[1].spawn_kwargs == {
        "traj_prefix": "npt_",
        "run_name": RUN_NAME,
    }
    assert functions[2].spawn_kwargs == {
        "run_name": RUN_NAME,
        "simulation_time_ns": 5,
    }


def test_reconciler_joins_parallel_analyses_before_finalizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="prepared")
    nvt = FakeCall("fc-nvt")
    npt = FakeCall("fc-npt")
    production = FakeCall("fc-production", result="produced")
    production_analysis = FakeCall("fc-production-analysis")
    calls = {"fc-root": prepare}
    functions = iter((
        FakeFunction(nvt),
        FakeFunction(npt),
        FakeFunction(production),
        FakeFunction(production_analysis),
    ))
    adapter = _adapter(
        calls,
        function_resolver=lambda *_args, **_kwargs: next(functions),
    )
    reconciler = GromacsReconciler(store, adapter, now=lambda: 10)

    asyncio.run(reconciler.reconcile())
    calls.update({call.object_id: call for call in (nvt, npt, production)})
    asyncio.run(reconciler.reconcile())
    calls[production_analysis.object_id] = production_analysis

    provider_calls = {
        call.operation: call for call in store.list_operations(job.job_id)
    }
    assert provider_calls["collect_traj_stats:nvt_"].state == (
        JobOperationState.RUNNING
    )
    assert provider_calls["collect_traj_stats:npt_"].state == (
        JobOperationState.RUNNING
    )
    assert provider_calls["production_run_gpu"].state == (JobOperationState.COMPLETED)
    assert provider_calls["collect_traj_stats:production_"].state == (
        JobOperationState.RUNNING
    )

    nvt.result = "analyzed"
    npt.result = "analyzed"
    production_analysis.result = "analyzed"

    async def publish(*_args, **_kwargs) -> FinalArchive:
        return FinalArchive(
            state=JobState.SUCCEEDED,
            volume_name="Gromacs-outputs",
            path=f"api-results/{RUN_NAME}/result.zip",
            filename=f"{RUN_NAME}.zip",
            size_bytes=123,
            sha256=SHA256,
            warnings_json="[]",
        )

    monkeypatch.setattr(adapter, "publish_archive", publish)
    asyncio.run(reconciler.reconcile())

    completed = store.get_job(job.owner_user_id, job.job_id)
    assert completed is not None
    assert completed.state == JobState.SUCCEEDED
    assert all(
        call.state == JobOperationState.COMPLETED
        for call in store.list_operations(job.job_id)
    )


def test_parallel_stage_failure_cancels_siblings_before_job_fails(
    tmp_path: Path,
) -> None:
    class CancellableCall(FakeCall):
        async def _cancel(self, *, terminate_containers: bool):
            await super()._cancel(terminate_containers=terminate_containers)
            self.result = modal.exception.InputCancellation("cancelled")

    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="prepared")
    nvt = FakeCall("fc-nvt", result=RuntimeError("analysis failed"))
    npt = CancellableCall("fc-npt")
    production = CancellableCall("fc-production")
    calls = {"fc-root": prepare}
    functions = iter((FakeFunction(nvt), FakeFunction(npt), FakeFunction(production)))
    reconciler = GromacsReconciler(
        store,
        _adapter(
            calls,
            function_resolver=lambda *_args, **_kwargs: next(functions),
        ),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())
    calls.update({call.object_id: call for call in (nvt, npt, production)})
    asyncio.run(reconciler.reconcile())

    failed = store.get_job(job.owner_user_id, job.job_id)
    assert failed is not None
    assert failed.state == JobState.FAILED
    states = {call.operation: call.state for call in store.list_operations(job.job_id)}
    assert states["collect_traj_stats:nvt_"] == JobOperationState.FAILED
    assert states["collect_traj_stats:npt_"] == JobOperationState.CANCELLED
    assert states["production_run_gpu"] == JobOperationState.CANCELLED
    assert ("cancel", "fc-npt", False) in npt.events
    assert ("cancel", "fc-production", False) in production.events


def test_definite_stage_rejection_cancels_started_siblings(
    tmp_path: Path,
) -> None:
    class CancellableCall(FakeCall):
        async def _cancel(self, *, terminate_containers: bool):
            await super()._cancel(terminate_containers=terminate_containers)
            self.result = modal.exception.InputCancellation("cancelled")

    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="prepared")
    nvt = CancellableCall("fc-nvt")
    npt = CancellableCall("fc-npt")
    rejected = FailingFunction(modal.exception.NotFoundError("deployment missing"))
    calls = {"fc-root": prepare}
    functions = iter((FakeFunction(nvt), FakeFunction(npt), rejected))
    reconciler = GromacsReconciler(
        store,
        _adapter(
            calls,
            function_resolver=lambda *_args, **_kwargs: next(functions),
        ),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())
    calls.update({call.object_id: call for call in (nvt, npt)})
    asyncio.run(reconciler.reconcile())

    failed = store.get_job(job.owner_user_id, job.job_id)
    assert failed is not None
    assert failed.state == JobState.FAILED
    states = {call.operation: call.state for call in store.list_operations(job.job_id)}
    assert states["collect_traj_stats:nvt_"] == JobOperationState.CANCELLED
    assert states["collect_traj_stats:npt_"] == JobOperationState.CANCELLED
    assert states["production_run_gpu"] == JobOperationState.FAILED
    rejected_operation = next(
        operation
        for operation in failed.operations
        if operation.operation == "production_run_gpu"
    )
    assert rejected_operation.started_at is None
    assert all(
        stage.operation != "production_run_gpu" for stage in failed.stage_history
    )
    assert rejected.calls == 1


def test_parallel_stage_failure_keeps_expired_sibling_state_unknown(
    tmp_path: Path,
) -> None:
    class DisappearingCall(FakeCall):
        async def _cancel(self, *, terminate_containers: bool):
            await super()._cancel(terminate_containers=terminate_containers)
            self.result = modal.exception.OutputExpiredError("expired")

    class CancellableCall(FakeCall):
        async def _cancel(self, *, terminate_containers: bool):
            await super()._cancel(terminate_containers=terminate_containers)
            self.result = modal.exception.InputCancellation("cancelled")

    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="prepared")
    nvt = FakeCall("fc-nvt", result=RuntimeError("analysis failed"))
    npt = DisappearingCall("fc-npt")
    production = CancellableCall("fc-production")
    calls = {"fc-root": prepare}
    functions = iter((FakeFunction(nvt), FakeFunction(npt), FakeFunction(production)))
    reconciler = GromacsReconciler(
        store,
        _adapter(
            calls,
            function_resolver=lambda *_args, **_kwargs: next(functions),
        ),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())
    calls.update({call.object_id: call for call in (nvt, npt, production)})
    asyncio.run(reconciler.reconcile())

    unresolved = store.get_job(job.owner_user_id, job.job_id)
    assert unresolved is not None
    assert unresolved.state == JobState.STATE_UNKNOWN
    assert unresolved.state_unknown_reason == "cancellation_outcome_unknown"
    states = {call.operation: call.state for call in store.list_operations(job.job_id)}
    assert states["collect_traj_stats:nvt_"] == JobOperationState.FAILED
    assert states["collect_traj_stats:npt_"] == JobOperationState.RUNNING
    assert states["production_run_gpu"] == JobOperationState.CANCELLED


def test_durable_cancellation_cannot_race_a_successor_stage_spawn(
    tmp_path: Path,
) -> None:
    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="/volumes/Gromacs-outputs/api-run")
    locks = JobLifecycleLocks()

    async def scenario() -> None:
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()
        call_ids = iter(("fc-nvt", "fc-npt", "fc-production"))

        class BlockingFunction:
            def __init__(self, call_id: str) -> None:
                self.call_id = call_id
                self.spawn = AsyncMethod(self._spawn)

            async def _spawn(self, **_kwargs):
                spawn_started.set()
                await release_spawn.wait()
                return FakeCall(self.call_id)

        reconciler = GromacsReconciler(
            store,
            _adapter(
                {"fc-root": prepare},
                function_resolver=lambda *_args, **_kwargs: BlockingFunction(
                    next(call_ids)
                ),
            ),
            lifecycle_locks=locks,
            now=lambda: 10,
        )
        reconciliation = asyncio.create_task(reconciler.reconcile())
        await spawn_started.wait()

        async def request_cancellation() -> None:
            async with locks.for_job(job.job_id):
                store.request_cancel(job.owner_user_id, job.job_id, now=11)

        cancellation = asyncio.create_task(request_cancellation())
        await asyncio.sleep(0)
        before_release = store.get_job(job.owner_user_id, job.job_id)
        assert before_release is not None
        assert before_release.state != JobState.CANCEL_REQUESTED

        release_spawn.set()
        await reconciliation
        await cancellation

    asyncio.run(scenario())
    cancelling = store.get_job(job.owner_user_id, job.job_id)
    assert cancelling is not None
    assert cancelling.state == JobState.CANCEL_REQUESTED
    assert {call.modal_call_id for call in store.list_operations(job.job_id)} == {
        "fc-root",
        "fc-nvt",
    }


def test_stage_attach_conflict_marks_job_state_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="prepared")
    duplicate = FakeCall("fc-duplicate")
    adapter = _adapter(
        {"fc-root": prepare, "fc-duplicate": duplicate},
        function_resolver=lambda *_args, **_kwargs: FakeFunction(duplicate),
    )

    def reject_attach(*_args, **_kwargs):
        raise JobSubmissionConflictError("provider operation changed concurrently")

    monkeypatch.setattr(store, "attach_modal_call", reject_attach)
    reconciler = GromacsReconciler(store, adapter, now=lambda: 10)

    asyncio.run(reconciler.reconcile())

    uncertain = store.get_job(job.owner_user_id, job.job_id)
    assert uncertain is not None
    assert uncertain.state == JobState.STATE_UNKNOWN
    assert uncertain.state_unknown_reason == "submission_outcome_unknown"
    provider_calls = store.list_operations(job.job_id)
    assert provider_calls[-1].operation == "collect_traj_stats:nvt_"
    assert provider_calls[-1].state == JobOperationState.STATE_UNKNOWN
    assert provider_calls[-1].modal_call_id is None
    assert ("cancel", "fc-duplicate", False) in duplicate.events


def test_reconciliation_bounds_concurrent_provider_polls(tmp_path: Path) -> None:
    store, first = _submitted_job(tmp_path)
    jobs = [first]
    for index in range(5):
        admitted = store.admit_job(
            owner_user_id=first.owner_user_id,
            display_name=f"Simulation {index}",
            idempotency_key=str(uuid4()),
            request_hash=f"request-{index}",
            parameters_json='{"simulation_time_ns":5}',
            configuration=_admission_configuration(),
            now=3 + index,
        )
        jobs.append(
            _attach_modal_operation(
                store,
                admitted.job.job_id,
                modal_call_id=f"fc-{index}",
                operation="prepare_tpr_gpu",
                run_name=f"simulation-{index}-0123456789abcdef0123456789abcdef",
                now=3 + index,
            )
        )

    async def scenario() -> None:
        active = 0
        peak = 0
        calls = 0
        saturated = asyncio.Event()
        release = asyncio.Event()
        adapter = _adapter({})

        async def poll(*_args, **_kwargs) -> PollOutcome:
            nonlocal active, peak, calls
            active += 1
            calls += 1
            peak = max(peak, active)
            if active == 4:
                saturated.set()
            await release.wait()
            active -= 1
            return PollOutcome("running")

        cast(Any, adapter).poll = poll
        reconciler = GromacsReconciler(
            store,
            adapter,
            now=lambda: 20,
            max_concurrent_jobs=4,
        )
        task = asyncio.create_task(reconciler.reconcile())
        await asyncio.wait_for(saturated.wait(), timeout=1)
        assert peak == 4
        release.set()
        await task
        assert calls == len(jobs)

    asyncio.run(scenario())


def test_reconciler_requires_a_positive_concurrency_bound(tmp_path: Path) -> None:
    store, _job = _submitted_job(tmp_path)

    with pytest.raises(ValueError, match="max_concurrent_jobs must be positive"):
        GromacsReconciler(store, _adapter({}), max_concurrent_jobs=0)


def test_reconciler_stops_after_an_unknown_stage_submission(
    tmp_path: Path,
) -> None:
    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="/volumes/Gromacs-outputs/api-run")
    failing = FailingFunction(modal.exception.ConnectionError("temporary"))
    now = 10
    reconciler = GromacsReconciler(
        store,
        _adapter(
            {"fc-root": prepare},
            function_resolver=lambda *_args, **_kwargs: failing,
        ),
        now=lambda: now,
    )

    asyncio.run(reconciler.reconcile())
    uncertain = store.get_job(job.owner_user_id, job.job_id)
    assert uncertain is not None
    assert uncertain.state == JobState.STATE_UNKNOWN
    assert uncertain.state_unknown_at == 10
    assert uncertain.state_unknown_reason == "submission_outcome_unknown"
    assert uncertain.operations[-1].submission_lease_until is None
    assert uncertain.operations[-1].state == JobOperationState.STATE_UNKNOWN
    assert failing.calls == 1

    now = 129
    asyncio.run(reconciler.reconcile())
    assert failing.calls == 1


def test_untracked_cancellation_is_not_declared_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    admission = store.admit_job(
        owner_user_id=user.user_id,
        display_name="Simulation",
        idempotency_key=str(uuid4()),
        request_hash="request-digest",
        parameters_json="{}",
        configuration=_admission_configuration(),
        now=2,
    )
    store.claim_modal_operation(
        admission.job.job_id,
        operation="prepare_tpr_gpu",
        run_name=RUN_NAME,
        submission_token="lost-submitter",
        now=3,
    )
    store.request_cancel(user.user_id, admission.job.job_id, now=4)
    _install_volume(monkeypatch, {})
    now = 122
    reconciler = GromacsReconciler(store, _adapter({}), now=lambda: now)

    asyncio.run(reconciler.reconcile())
    cancelling = store.get_job(user.user_id, admission.job.job_id)
    assert cancelling is not None
    assert cancelling.state == JobState.CANCEL_REQUESTED

    now = 123
    asyncio.run(reconciler.reconcile())
    uncertain = store.get_job(user.user_id, admission.job.job_id)
    assert uncertain is not None
    assert uncertain.state == JobState.STATE_UNKNOWN
    assert uncertain.state_unknown_reason == "submission_outcome_unknown"


def test_completed_stage_does_not_advance_after_cancellation(tmp_path: Path) -> None:
    store, job = _submitted_job(tmp_path, cancel_requested=True)
    root_node = CallNode("fc-root", modal.call_graph.InputStatus.SUCCESS)
    prepare = FakeCall(
        "fc-root",
        result="/volumes/Gromacs-outputs/api-run",
        graph=[root_node],
    )

    def unexpected_lookup(*_args, **_kwargs):
        raise AssertionError("cancelled Job must not launch another Modal stage")

    reconciler = GromacsReconciler(
        store,
        _adapter({"fc-root": prepare}, function_resolver=unexpected_lookup),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())

    cancelled = store.get_job(job.owner_user_id, job.job_id)
    assert cancelled is not None
    assert cancelled.state == JobState.CANCELLED
    assert cancelled.operations[0].modal_call_id == "fc-root"
    assert cancelled.stage_history[-1].completed_at == 10


def test_cancel_wins_before_result_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        cancel_requested=True,
        operation="collect_traj_stats:production_",
    )
    files = _established_output_files()
    _install_volume(monkeypatch, files)
    root_node = CallNode("fc-root", modal.call_graph.InputStatus.PENDING)
    root = FakeCall(
        "fc-root",
        result=f"/volumes/Gromacs-outputs/{RUN_NAME}",
        graph=[root_node],
    )
    reconciler = GromacsReconciler(store, _adapter({"fc-root": root}), now=lambda: 10)

    asyncio.run(reconciler.reconcile())

    completed = store.get_job(job.owner_user_id, job.job_id)
    assert completed is not None
    assert completed.state == JobState.CANCELLED
    assert completed.result_volume_path is None
    assert f"api-results/{RUN_NAME}/result.zip" not in files
    assert ("cancel", "fc-root", False) in root.events


def test_cancel_wins_after_final_stage_poll_before_state_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, stale_running = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    adapter = _adapter({"fc-root": FakeCall("fc-root", result="completed")})

    async def unexpected_publication(*_args, **_kwargs):
        raise AssertionError("accepted cancellation must prevent Result publication")

    monkeypatch.setattr(adapter, "publish_archive", unexpected_publication)
    store.request_cancel(stale_running.owner_user_id, stale_running.job_id, now=9)

    asyncio.run(GromacsReconciler(store, adapter, now=lambda: 10).reconcile())

    cancelled = store.get_job(stale_running.owner_user_id, stale_running.job_id)
    assert cancelled is not None
    assert cancelled.state == JobState.CANCELLED
    assert cancelled.cancel_requested_at == 9
    assert cancelled.result_volume_path is None
    assert cancelled.stage_history[-1].outcome == "completed"


@pytest.mark.parametrize(
    "error",
    [
        modal.exception.ConnectionError("temporary"),
        modal.exception.ExecutionError("upload failed"),
    ],
)
def test_transient_archive_publication_error_remains_finalizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    adapter = _adapter({"fc-root": FakeCall("fc-root", result=str(tmp_path))})

    async def unavailable(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(adapter, "publish_archive", unavailable)

    asyncio.run(GromacsReconciler(store, adapter, now=lambda: 10).reconcile())

    finalizing = store.get_job(job.owner_user_id, job.job_id)
    assert finalizing is not None
    assert finalizing.state == JobState.FINALIZING
    assert finalizing.error_code is None
    assert finalizing.finalization_started_at == 10
    assert finalizing.finalization_retry_started_at == 10
    assert finalizing.finalization_retry_count == 1
    assert finalizing.next_retry_at == 15


@pytest.mark.parametrize(
    "failure",
    [
        OSError("No space left on device"),
        ArtifactIntegrityError("Staged artifact is unavailable"),
    ],
)
def test_local_staging_failure_retries_and_blocks_without_losing_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    adapter = _adapter({"fc-root": FakeCall("fc-root", result=str(tmp_path))})

    async def staging_failure(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(adapter, "publish_archive", staging_failure)
    now = 10
    reconciler = GromacsReconciler(store, adapter, now=lambda: now)
    asyncio.run(reconciler.reconcile())
    retrying = store.get_job(job.owner_user_id, job.job_id)
    assert retrying is not None
    assert retrying.state == JobState.FINALIZING
    assert retrying.error_code is None
    assert retrying.next_retry_at == 15

    now = 1_810
    asyncio.run(reconciler.reconcile())
    blocked = store.get_job(job.owner_user_id, job.job_id)
    assert blocked is not None
    assert blocked.state == JobState.BLOCKED
    assert blocked.blocking_category == "local_storage"
    assert blocked.error_code is None


def test_transient_finalization_blocks_after_retry_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    adapter = _adapter({"fc-root": FakeCall("fc-root", result=str(tmp_path))})

    async def unavailable(*_args, **_kwargs):
        raise modal.exception.ConnectionError("temporary")

    monkeypatch.setattr(adapter, "publish_archive", unavailable)
    now = 10
    reconciler = GromacsReconciler(store, adapter, now=lambda: now)
    asyncio.run(reconciler.reconcile())
    now = 1_810
    asyncio.run(reconciler.reconcile())

    blocked = store.get_job(job.owner_user_id, job.job_id)
    assert blocked is not None
    assert blocked.state == JobState.BLOCKED
    assert blocked.blocking_category == "modal_unavailable"
    assert blocked.blocked_at == 1_810
    assert blocked.next_retry_at == 2_710
    assert store.count_active_jobs("gromacs") == 0


def test_unexpected_finalization_error_preserves_completed_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    adapter = _adapter({"fc-root": FakeCall("fc-root", result=str(tmp_path))})

    async def programming_error(*_args, **_kwargs):
        raise RuntimeError("unexpected archive builder failure")

    monkeypatch.setattr(adapter, "publish_archive", programming_error)
    now = 10
    reconciler = GromacsReconciler(store, adapter, now=lambda: now)
    asyncio.run(reconciler.reconcile())

    retrying = store.get_job(job.owner_user_id, job.job_id)
    assert retrying is not None
    assert retrying.state == JobState.FINALIZING
    assert retrying.error_code is None
    assert retrying.next_retry_at == 15

    now = 1_810
    asyncio.run(reconciler.reconcile())
    blocked = store.get_job(job.owner_user_id, job.job_id)
    assert blocked is not None
    assert blocked.state == JobState.BLOCKED
    assert blocked.blocking_category == "internal_service"
    assert blocked.error_code is None


def test_permanent_finalization_block_recovers_without_new_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    root = FakeCall("fc-root", result=str(tmp_path))
    adapter = _adapter({"fc-root": root})

    async def forbidden(*_args, **_kwargs):
        raise modal.exception.AuthError("permission denied")

    monkeypatch.setattr(adapter, "publish_archive", forbidden)
    now = 10
    reconciler = GromacsReconciler(store, adapter, now=lambda: now)
    asyncio.run(reconciler.reconcile())
    blocked = store.get_job(job.owner_user_id, job.job_id)
    assert blocked is not None
    assert blocked.state == JobState.BLOCKED
    assert blocked.blocking_category == "modal_configuration"
    assert blocked.next_retry_at == 910

    async def restored(*_args, **_kwargs):
        return FinalArchive(
            state=JobState.SUCCEEDED,
            volume_name="Gromacs-outputs",
            path=f"api-results/{RUN_NAME}/result.zip",
            filename=f"{RUN_NAME}.zip",
            size_bytes=123,
            sha256=SHA256,
            warnings_json="[]",
        )

    monkeypatch.setattr(adapter, "publish_archive", restored)
    now = 910
    asyncio.run(reconciler.reconcile())

    recovered = store.get_job(job.owner_user_id, job.job_id)
    assert recovered is not None
    assert recovered.state == JobState.SUCCEEDED
    assert recovered.blocking_category is None
    assert [event for event in root.events if event[0] == "get"] == [
        ("get", "fc-root", 0)
    ]


def test_result_integrity_recovery_preserves_published_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, running = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    files = _established_output_files()
    _install_volume(monkeypatch, files)
    adapter = _adapter({})
    finalizing = store.set_job_state(running.job_id, JobState.FINALIZING, now=10)
    published = asyncio.run(adapter.publish_archive(finalizing, completed_at=10))
    completed = store.complete_job(
        running.job_id,
        state=published.state,
        result_volume_name=published.volume_name,
        result_volume_path=published.path,
        result_filename=published.filename,
        result_size_bytes=published.size_bytes,
        result_sha256=published.sha256,
        result_archive_schema_version=GROMACS_ARCHIVE_SCHEMA_VERSION,
        now=10,
    )
    assert completed.result_archive_schema_version == GROMACS_ARCHIVE_SCHEMA_VERSION
    store.block_job(
        running.job_id,
        category="result_integrity",
        previous_state=JobState.SUCCEEDED,
        now=20,
        next_retry_at=30,
    )
    # Recovery must prefer the exact immutable ZIP and marker. Raw scientific
    # intermediates may already have been cleaned and cannot be assumed present.
    del files[f"{RUN_NAME}/production.mdp"]

    asyncio.run(GromacsReconciler(store, adapter, now=lambda: 30).reconcile())

    recovered = store.get_job(running.owner_user_id, running.job_id)
    assert recovered is not None
    assert recovered.state == JobState.SUCCEEDED
    assert recovered.result_sha256 == completed.result_sha256
    assert recovered.result_size_bytes == completed.result_size_bytes
    assert recovered.completed_at == 10


def test_rebuild_rejects_an_unavailable_archive_builder(tmp_path: Path) -> None:
    store, running = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    store.set_job_state(running.job_id, JobState.FINALIZING, now=10)
    completed = store.complete_job(
        running.job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{RUN_NAME}/result.zip",
        result_filename=f"{RUN_NAME}.zip",
        result_size_bytes=123,
        result_sha256=SHA256,
        result_archive_schema_version=GROMACS_ARCHIVE_SCHEMA_VERSION,
        now=10,
    )
    unsupported = replace(completed, result_archive_schema_version=999)

    async def rebuild() -> bytes:
        return b"".join([
            chunk async for chunk in _adapter({}).rebuild_artifact(unsupported)
        ])

    with pytest.raises(ValueError, match="unsupported Result archive schema"):
        asyncio.run(rebuild())


def test_expired_call_output_recovers_completed_archive_from_volume_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    root = FakeCall(
        "fc-root",
        result=modal.exception.OutputExpiredError("expired"),
    )
    adapter = _adapter({"fc-root": root})
    archive_bytes, request_sha256 = _valid_archive_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    _install_volume(
        monkeypatch,
        {
            f"api-results/{RUN_NAME}/result.json": _result_marker(
                archive_bytes, request_sha256
            ),
            f"api-results/{RUN_NAME}/result.zip": archive_bytes,
        },
    )
    reconciler = GromacsReconciler(store, adapter, now=lambda: 10)

    asyncio.run(reconciler.reconcile())

    completed = store.get_job(job.owner_user_id, job.job_id)
    assert completed is not None
    assert completed.state == JobState.SUCCEEDED
    assert completed.result_volume_path == f"api-results/{RUN_NAME}/result.zip"
    assert completed.result_sha256 == archive_sha256


def test_normal_cache_restore_requires_the_published_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, running = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    archive_bytes, _request_sha256 = _valid_archive_bytes()
    completed = store.complete_job(
        running.job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{RUN_NAME}/result.zip",
        result_filename=f"{RUN_NAME}.zip",
        result_size_bytes=len(archive_bytes),
        result_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        result_archive_schema_version=GROMACS_ARCHIVE_SCHEMA_VERSION,
        now=10,
    )
    _install_volume(
        monkeypatch,
        {f"api-results/{RUN_NAME}/result.zip": archive_bytes},
    )
    adapter = _adapter({})

    async def restore() -> bytes:
        return b"".join([chunk async for chunk in adapter.read_artifact(completed)])

    with pytest.raises(ArtifactIntegrityError, match="marker"):
        asyncio.run(restore())


def test_expired_call_rejects_marker_when_archive_bytes_are_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        operation="collect_traj_stats:production_",
    )
    root = FakeCall(
        "fc-root",
        result=modal.exception.OutputExpiredError("expired"),
    )
    archive_bytes, request_sha256 = _valid_archive_bytes()
    corrupt = bytearray(archive_bytes)
    corrupt[-1] ^= 1
    _install_volume(
        monkeypatch,
        {
            f"api-results/{RUN_NAME}/result.json": _result_marker(
                archive_bytes, request_sha256
            ),
            f"api-results/{RUN_NAME}/result.zip": bytes(corrupt),
        },
    )
    reconciler = GromacsReconciler(
        store,
        _adapter({"fc-root": root}),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())

    failed = store.get_job(job.owner_user_id, job.job_id)
    assert failed is not None
    assert failed.state == JobState.FAILED
    assert failed.error_code == "result_invalid"


def test_recovery_rejects_self_consistent_marker_for_invalid_zip_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(tmp_path)
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
                invalid_archive, request_sha256
            ),
            f"api-results/{RUN_NAME}/result.zip": invalid_archive,
        },
    )

    with pytest.raises(ValueError, match="does not match manifest"):
        asyncio.run(_adapter({}).recover_archive(job))


def test_missing_recovery_marker_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    _install_volume(monkeypatch, {})

    with pytest.raises(ArchiveNotReadyError, match="marker is missing"):
        asyncio.run(_adapter({}).recover_archive(job))


def test_reconciler_marks_an_expired_untracked_submission_state_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    admission = store.admit_job(
        owner_user_id=user.user_id,
        display_name="Simulation",
        idempotency_key=str(uuid4()),
        request_hash="request-digest",
        parameters_json="{}",
        configuration=_admission_configuration(),
        now=2,
    )
    store.claim_modal_operation(
        admission.job.job_id,
        operation="prepare_tpr_gpu",
        run_name=RUN_NAME,
        submission_token="lost-submitter",
        now=3,
    )
    _install_volume(monkeypatch, {})
    now = 122
    reconciler = GromacsReconciler(store, _adapter({}), now=lambda: now)

    asyncio.run(reconciler.reconcile())
    queued = store.get_job(user.user_id, admission.job.job_id)
    assert queued is not None
    assert queued.state == JobState.QUEUED

    now = 123
    asyncio.run(reconciler.reconcile())
    uncertain = store.get_job(user.user_id, admission.job.job_id)
    assert uncertain is not None
    assert uncertain.state == JobState.STATE_UNKNOWN
    assert uncertain.state_unknown_at == 123
    assert uncertain.state_unknown_reason == "submission_outcome_unknown"


def test_cancelled_is_terminal_only_after_call_graph_is_inactive(
    tmp_path: Path,
) -> None:
    store, job = _submitted_job(tmp_path, cancel_requested=True)
    active_child = CallNode("fc-child", modal.call_graph.InputStatus.PENDING)
    root_node = CallNode(
        "fc-root",
        modal.call_graph.InputStatus.TERMINATED,
        children=[active_child],
    )
    root = FakeCall("fc-root", graph=[root_node])
    child = FakeCall("fc-child")
    reconciler = GromacsReconciler(
        store,
        _adapter({"fc-root": root, "fc-child": child}),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())

    cancelling = store.get_job(job.owner_user_id, job.job_id)
    assert cancelling is not None
    assert cancelling.state == JobState.CANCEL_REQUESTED

    active_child.status = modal.call_graph.InputStatus.TERMINATED
    asyncio.run(reconciler.reconcile())

    cancelled = store.get_job(job.owner_user_id, job.job_id)
    assert cancelled is not None
    assert cancelled.state == JobState.CANCELLED


def test_expired_provider_status_is_not_reported_as_confirmed_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(tmp_path, cancel_requested=True)
    root = FakeCall(
        "fc-root",
        result=modal.exception.OutputExpiredError("expired"),
    )
    reconciler = GromacsReconciler(
        store,
        _adapter({"fc-root": root}),
        now=lambda: 10,
    )
    files = _established_output_files()
    _install_volume(monkeypatch, files)

    asyncio.run(reconciler.reconcile())

    unresolved = store.get_job(job.owner_user_id, job.job_id)
    assert unresolved is not None
    assert unresolved.state == JobState.STATE_UNKNOWN
    assert unresolved.state_unknown_at == 10
    assert unresolved.state_unknown_reason == "cancellation_outcome_unknown"
    assert unresolved.error_code is None
    assert f"api-results/{RUN_NAME}/result.zip" not in files


def test_expired_cancellation_retries_transient_result_recovery(
    tmp_path: Path,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        cancel_requested=True,
        operation="collect_traj_stats:production_",
    )
    root = FakeCall(
        "fc-root",
        result=modal.exception.OutputExpiredError("expired"),
    )
    adapter = _adapter({"fc-root": root})

    async def unavailable(_job: JobRecord) -> None:
        raise modal.exception.ConnectionError("temporary")

    cast(Any, adapter).recover_archive = unavailable
    reconciler = GromacsReconciler(store, adapter, now=lambda: 10)

    asyncio.run(reconciler.reconcile())

    cancelling = store.get_job(job.owner_user_id, job.job_id)
    assert cancelling is not None
    assert cancelling.state == JobState.CANCEL_REQUESTED
    assert cancelling.state_unknown_at is None


def test_intermediate_cleanup_is_opt_in_and_preserves_final_archives(
    tmp_path: Path,
) -> None:
    now = 1_000_000
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
    due_succeeded = _terminal_job(
        store,
        user.user_id,
        run_name=f"first-simulation-{'1' * 32}",
        state=JobState.SUCCEEDED,
        completed_at=100,
    )
    due_partial = _terminal_job(
        store,
        user.user_id,
        run_name=f"second-simulation-{'2' * 32}",
        state=JobState.PARTIAL,
        completed_at=200,
    )
    recent_succeeded = _terminal_job(
        store,
        user.user_id,
        run_name=f"recent-simulation-{'3' * 32}",
        state=JobState.SUCCEEDED,
        completed_at=now - 60,
    )
    due_failed = _terminal_job(
        store,
        user.user_id,
        run_name=f"failed-simulation-{'4' * 32}",
        state=JobState.FAILED,
        completed_at=300,
    )

    class CleanupAdapter:
        def __init__(self) -> None:
            self.cleaned: list[UUID] = []

        async def cleanup_intermediates(self, job: JobRecord) -> None:
            self.cleaned.append(job.job_id)

    adapter = CleanupAdapter()
    disabled = GromacsReconciler(
        store,
        cast(Any, adapter),
        now=lambda: now,
    )

    asyncio.run(disabled.reconcile())

    assert adapter.cleaned == []
    assert all(
        job.intermediates_cleaned_at is None for job in store.list_jobs(user.user_id)
    )

    final_metadata = {
        job.job_id: (
            job.state,
            job.result_volume_name,
            job.result_volume_path,
            job.result_filename,
            job.result_size_bytes,
            job.result_sha256,
        )
        for job in (due_succeeded, due_partial)
    }
    enabled = GromacsReconciler(
        store,
        cast(Any, adapter),
        now=lambda: now,
        intermediate_retention_days=2,
    )

    asyncio.run(enabled.reconcile())

    assert adapter.cleaned == [due_succeeded.job_id, due_partial.job_id]
    for original in (due_succeeded, due_partial):
        cleaned = store.get_job(user.user_id, original.job_id)
        assert cleaned is not None
        assert cleaned.intermediates_cleaned_at == now
        assert (
            cleaned.state,
            cleaned.result_volume_name,
            cleaned.result_volume_path,
            cleaned.result_filename,
            cleaned.result_size_bytes,
            cleaned.result_sha256,
        ) == final_metadata[original.job_id]
    for untouched in (recent_succeeded, due_failed):
        current = store.get_job(user.user_id, untouched.job_id)
        assert current is not None
        assert current.intermediates_cleaned_at is None
