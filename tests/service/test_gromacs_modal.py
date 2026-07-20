"""Modal boundary and reconciliation contracts for GROMACS jobs."""

# ruff: noqa: D101,D102,D103,D107,S106

from __future__ import annotations

import asyncio
import hashlib
import io
import struct
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import modal
import orjson
import pytest

from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.gromacs.modal import (
    ArchiveNotReadyError,
    FinalArchive,
    GromacsReconciler,
    ModalGromacsAdapter,
    PollOutcome,
)
from biomodals.service.gromacs.router import (
    GromacsJobOptions,
    SubmissionOutcomeUnknownError,
)
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
    ModalConfigurationSnapshot,
)
from biomodals.service.store import JobRecord, JobState, ServiceStore, UserRecord

RUN_NAME = "first-simulation-0123456789abcdef0123456789abcdef"
SHA256 = "a" * 64
_PENDING = object()
XTC = struct.pack(">i", 1995) + b"\0" * 28
TPR = b"\0\0\0\x10VERSION 2026.1\0\0\0\0"
PNG = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + b"\0" * 16
PDB = b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n"


def _admission_configuration() -> JobAdmissionConfiguration:
    return JobAdmissionConfiguration(
        workload="gromacs",
        modal_environment=DatabaseOverridableSetting("department-dev", False),
        modal_app_name=DatabaseOverridableSetting("GromacsAPI", False),
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
        "archive_schema_version": 2,
        "run_name": RUN_NAME,
        "files": records,
    })
    checksums = "".join([
        *(f"{record['sha256']}  {record['path']}\n" for record in records),
        f"{hashlib.sha256(manifest).hexdigest()}  metadata/manifest.json\n",
    ]).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr("metadata/manifest.json", manifest)
        archive.writestr("metadata/checksums.sha256", checksums)
    request_digest = hashlib.sha256()
    request_digest.update(len(members["input.pdb"]).to_bytes(8, "big"))
    request_digest.update(members["input.pdb"])
    request_digest.update(members["metadata/parameters.json"])
    return output.getvalue(), request_digest.hexdigest()


def _result_marker(archive_bytes: bytes, request_sha256: str) -> bytes:
    return orjson.dumps({
        "archive_schema_version": 2,
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
            (info.filename, content if info.filename == name else source.read(info))
            for info in source.infolist()
        ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, member_content in members:
            archive.writestr(member_name, member_content)
    return output.getvalue()


def _install_volume(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
) -> None:
    class FakeVolume:
        def __init__(self) -> None:
            self.read_file = AsyncMethod(self._read_file)
            self.batch_upload = AsyncMethod(self._batch_upload)

        async def _read_file(self, path: str):
            if path not in files:
                raise FileNotFoundError(path)
            yield files[path]

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
        app_name="GromacsAPI",
        environment_name="department-dev",
        output_volume_name=output_volume_name,
        call_resolver=cast(Any, calls.__getitem__),
        function_resolver=function_resolver,
    )


def _submitted_job(
    tmp_path: Path,
    *,
    cancel_requested: bool = False,
    provider_operation: str = "prepare_tpr_gpu",
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
    job = store.mark_submitted(
        admission.job.job_id,
        modal_call_id="fc-root",
        provider_operation=provider_operation,
        run_name=RUN_NAME,
        now=3,
    )
    if cancel_requested:
        job = store.request_cancel(user.user_id, job.job_id, now=4)
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
    job = store.mark_submitted(
        admission.job.job_id,
        modal_call_id=f"fc-{run_name}",
        provider_operation="collect_traj_stats:production_",
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
        now=completed_at,
    )


def test_submit_resolves_the_deployed_prepare_function_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = FakeFunction(FakeCall("fc-prepare"))
    lookups: list[tuple[str, str, str | None]] = []

    def from_name(
        app_name: str,
        function_name: str,
        *,
        environment_name: str | None = None,
    ):
        lookups.append((app_name, function_name, environment_name))
        return function

    monkeypatch.setattr(modal.Function, "from_name", from_name)

    adapter = ModalGromacsAdapter(
        app_name="GromacsAPI",
        environment_name="department-dev",
    )

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
            ),
        )
    )

    assert lookups == [
        ("GromacsAPI", "prepare_tpr_cpu", "department-dev"),
    ]
    assert submitted.modal_call_id == "fc-prepare"
    assert submitted.provider_operation == "prepare_tpr_cpu"
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
    ) -> modal.Function:
        assert (app_name, environment_name) == ("CandidateApp", "candidate-env")
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
        app_name="GromacsAPI",
        environment_name="department-dev",
        function_resolver=resolve_function,
    )

    asyncio.run(adapter.preflight("CandidateApp", "candidate-env"))

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
    adapter = ModalGromacsAdapter(
        app_name="GromacsAPI",
        environment_name="department-dev",
    )

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
        app_name="GromacsAPI",
        environment_name="department-dev",
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
                ),
            )
        )


def test_advance_resolves_each_deployed_stage_by_name(tmp_path: Path) -> None:
    store, job = _submitted_job(tmp_path)
    lookups: list[tuple[str, str, str | None]] = []
    functions: list[FakeFunction] = []

    def from_name(
        app_name: str,
        function_name: str,
        *,
        environment_name: str | None = None,
    ):
        lookups.append((app_name, function_name, environment_name))
        function = FakeFunction(FakeCall(f"fc-{len(functions) + 1}"))
        functions.append(function)
        return function

    adapter = ModalGromacsAdapter(
        app_name="GromacsAPI",
        environment_name="department-dev",
        function_resolver=from_name,
    )

    operations = (
        "collect_traj_stats:nvt_",
        "collect_traj_stats:npt_",
        "production_run_gpu",
        "collect_traj_stats:production_",
    )
    for index, operation in enumerate(operations, start=1):
        submission_token = f"stage-{index}"
        claimed = store.claim_provider_advance(
            job.job_id,
            expected_modal_call_id=job.modal_call_id or "",
            submission_token=submission_token,
            now=3 + index,
        )
        assert claimed is not None
        submitted = asyncio.run(adapter.advance(job))
        assert submitted.provider_operation == operation
        job = store.replace_provider_call(
            job.job_id,
            expected_modal_call_id=job.modal_call_id or "",
            modal_call_id=f"fc-{index}",
            provider_operation=operation,
            submission_token=submission_token,
            now=3 + index,
        )

    assert [function_name for _, function_name, _ in lookups] == [
        "collect_traj_stats",
        "collect_traj_stats",
        "production_run_gpu",
        "collect_traj_stats",
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
        provider_operation="collect_traj_stats:production_",
    )
    files = _established_output_files()
    _install_volume(monkeypatch, files)

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


def test_published_archive_is_promoted_into_the_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(
        tmp_path,
        provider_operation="collect_traj_stats:production_",
    )
    files = _established_output_files()
    _install_volume(monkeypatch, files)
    cache = ArtifactCache(tmp_path / "cache")
    adapter = ModalGromacsAdapter(
        app_name="GromacsAPI",
        environment_name="department-dev",
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


def test_reconciler_advances_completed_stage_to_one_direct_named_call(
    tmp_path: Path,
) -> None:
    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="/volumes/Gromacs-outputs/api-run")
    analysis = FakeFunction(FakeCall("fc-analysis"))
    lookups: list[tuple[str, str, str | None]] = []

    def from_name(
        app_name: str,
        function_name: str,
        *,
        environment_name: str | None = None,
    ) -> FakeFunction:
        lookups.append((app_name, function_name, environment_name))
        return analysis

    reconciler = GromacsReconciler(
        store,
        _adapter({"fc-root": prepare}, function_resolver=from_name),
        now=lambda: 10,
    )

    asyncio.run(reconciler.reconcile())

    advanced = store.get_job(job.owner_user_id, job.job_id)
    assert advanced is not None
    assert advanced.modal_call_id == "fc-analysis"
    assert advanced.provider_operation == "collect_traj_stats:nvt_"
    assert lookups == [("GromacsAPI", "collect_traj_stats", "department-dev")]
    assert analysis.spawn_kwargs == {
        "traj_prefix": "nvt_",
        "run_name": RUN_NAME,
    }


def test_durable_cancellation_cannot_race_a_successor_stage_spawn(
    tmp_path: Path,
) -> None:
    store, job = _submitted_job(tmp_path)
    prepare = FakeCall("fc-root", result="/volumes/Gromacs-outputs/api-run")
    locks = JobLifecycleLocks()

    async def scenario() -> None:
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()

        class BlockingFunction:
            def __init__(self) -> None:
                self.spawn = AsyncMethod(self._spawn)

            async def _spawn(self, **_kwargs):
                spawn_started.set()
                await release_spawn.wait()
                return FakeCall("fc-successor")

        reconciler = GromacsReconciler(
            store,
            _adapter(
                {"fc-root": prepare},
                function_resolver=lambda *_args, **_kwargs: BlockingFunction(),
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
    assert cancelling.modal_call_id == "fc-successor"


def test_reconciler_does_not_repeat_an_unknown_stage_submission(
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
    assert uncertain.state == JobState.QUEUED
    assert uncertain.submission_lease_until == 130
    assert failing.calls == 1

    now = 129
    asyncio.run(reconciler.reconcile())
    assert failing.calls == 1

    now = 130
    asyncio.run(reconciler.reconcile())
    failed = store.get_job(job.owner_user_id, job.job_id)
    assert failed is not None
    assert failed.state == JobState.FAILED
    assert failed.error_code == "compute_failed"
    assert failed.error_message == (
        "GROMACS stage submission was interrupted before remote compute could be "
        "tracked."
    )


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
    store.claim_submission(
        admission.job.job_id,
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
    failed = store.get_job(user.user_id, admission.job.job_id)
    assert failed is not None
    assert failed.state == JobState.FAILED
    assert failed.error_code == "compute_failed"


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
    assert cancelled.modal_call_id == "fc-root"
    assert cancelled.stage_history[-1].completed_at == 10


def test_cancel_wins_before_result_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        cancel_requested=True,
        provider_operation="collect_traj_stats:production_",
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
        provider_operation="collect_traj_stats:production_",
    )
    adapter = _adapter({"fc-root": FakeCall("fc-root", result="completed")})

    async def unexpected_publication(*_args, **_kwargs):
        raise AssertionError("accepted cancellation must prevent Result publication")

    monkeypatch.setattr(adapter, "publish_archive", unexpected_publication)
    store.request_cancel(stale_running.owner_user_id, stale_running.job_id, now=9)

    asyncio.run(
        GromacsReconciler(store, adapter, now=lambda: 10)._apply(
            stale_running,
            PollOutcome("completed"),
        )
    )

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
        provider_operation="collect_traj_stats:production_",
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
        provider_operation="collect_traj_stats:production_",
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
        provider_operation="collect_traj_stats:production_",
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


def test_permanent_finalization_block_recovers_without_new_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        provider_operation="collect_traj_stats:production_",
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
        provider_operation="collect_traj_stats:production_",
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
        now=10,
    )
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


def test_expired_call_output_recovers_completed_archive_from_volume_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        provider_operation="collect_traj_stats:production_",
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


def test_expired_call_rejects_marker_when_archive_bytes_are_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(
        tmp_path,
        provider_operation="collect_traj_stats:production_",
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


def test_reconciler_fails_an_expired_untracked_submission(
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
    store.claim_submission(
        admission.job.job_id,
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
    failed = store.get_job(user.user_id, admission.job.job_id)
    assert failed is not None
    assert failed.state == JobState.FAILED
    assert failed.error_code == "compute_failed"
    assert failed.error_message == (
        "GROMACS submission was interrupted before remote compute could be tracked."
    )


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
    assert unresolved.state == JobState.CANCEL_REQUESTED
    assert unresolved.error_code is None
    assert f"api-results/{RUN_NAME}/result.zip" not in files


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
