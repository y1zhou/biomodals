"""Modal boundary and reconciliation contracts for GROMACS jobs."""

# ruff: noqa: D101,D102,D103,D107,S106

from __future__ import annotations

import asyncio
import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import modal
import orjson
import pytest

from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)
from biomodals.service.gromacs.modal import GromacsReconciler, ModalGromacsAdapter
from biomodals.service.gromacs.router import GromacsJobOptions
from biomodals.service.store import JobRecord, JobState, ServiceStore

RUN_NAME = "api-0123456789abcdef0123456789abcdef"
SHA256 = "a" * 64
_PENDING = object()


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


def _valid_archive_bytes() -> tuple[bytes, str]:
    prefix = f"production_{RUN_NAME}"
    members = {
        "input.pdb": b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n",
        "parameters.json": b"{}\n",
        "provenance.json": b"{}\n",
        "run.log": b"completed\n",
        "outputs/production.mdp": b"integrator = md\n",
        f"outputs/{prefix}.xtc": b"trajectory",
        f"outputs/{prefix}.tpr": b"topology",
        f"outputs/{prefix}_nopbc_centered.pdb": b"MODEL\nEND\n",
        f"outputs/rmsd_{prefix}.csv": b"time,rmsd\n",
        f"outputs/rg_{prefix}.csv": b"time,rg\n",
        f"outputs/rmsf_{prefix}.csv": b"residue,rmsf\n",
    }
    roles = {
        "input.pdb": "input_structure",
        "parameters.json": "normalized_parameters",
        "provenance.json": "provenance",
        "run.log": "run_log",
        "outputs/production.mdp": "production_parameters",
        f"outputs/{prefix}.xtc": "trajectory",
        f"outputs/{prefix}.tpr": "production_topology",
        f"outputs/{prefix}_nopbc_centered.pdb": "centered_structure",
        f"outputs/rmsd_{prefix}.csv": "rmsd",
        f"outputs/rg_{prefix}.csv": "radius_of_gyration",
        f"outputs/rmsf_{prefix}.csv": "rmsf",
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
        "archive_schema_version": 1,
        "run_name": RUN_NAME,
        "files": records,
    })
    checksums = "".join([
        *(f"{record['sha256']}  {record['path']}\n" for record in records),
        f"{hashlib.sha256(manifest).hexdigest()}  manifest.json\n",
    ]).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr("manifest.json", manifest)
        archive.writestr("checksums.sha256", checksums)
    request_digest = hashlib.sha256()
    request_digest.update(len(members["input.pdb"]).to_bytes(8, "big"))
    request_digest.update(members["input.pdb"])
    request_digest.update(members["parameters.json"])
    return output.getvalue(), request_digest.hexdigest()


def _result_marker(archive_bytes: bytes, request_sha256: str) -> bytes:
    return orjson.dumps({
        "archive_schema_version": 1,
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

        async def _read_file(self, path: str):
            if path not in files:
                raise modal.exception.NotFoundError(path)
            yield files[path]

    monkeypatch.setattr(
        modal.Volume,
        "from_name",
        lambda name, *, environment_name: (
            FakeVolume()
            if (name, environment_name) == ("Gromacs-outputs", "department-dev")
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )


def _adapter(
    calls: dict[str, FakeCall],
    *,
    output_volume_name: str = "Gromacs-outputs",
) -> ModalGromacsAdapter:
    return ModalGromacsAdapter(
        app_name="GromacsAPI",
        environment_name="department-dev",
        output_volume_name=output_volume_name,
        call_resolver=cast(Any, calls.__getitem__),
    )


def _archive_result(
    *,
    status: AppRunStatus = AppRunStatus.SUCCEEDED,
    volume_name: str = "Gromacs-outputs",
    path: str = f"api-results/{RUN_NAME}/result.zip",
    media_type: str | None = "application/zip",
    kind: ArtifactKind = ArtifactKind.ARCHIVE,
    filename: str = f"{RUN_NAME}.zip",
    size_bytes: int = 123,
    sha256: str = SHA256,
    warnings: list[str] | None = None,
) -> AppRunResult:
    return AppRunResult(
        status=status,
        outputs=[
            AppOutput(
                name="gromacs_run",
                kind=kind,
                storage=VolumePath(
                    volume_name=volume_name,
                    path=path,
                    media_type=media_type,
                ),
                metadata={
                    "filename": filename,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                },
            )
        ],
        warnings=warnings or [],
    )


def _submitted_job(
    tmp_path: Path,
    *,
    cancel_requested: bool = False,
) -> tuple[ServiceStore, JobRecord]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    user = store.create_user(
        email="alice@example.com",
        display_name="Alice",
        token_digest=b"setup-token-digest",
        token_expires_at=3_600,
        now=1,
    )
    admission = store.admit_job(
        owner_user_id=user.user_id,
        workload="gromacs",
        display_name="Simulation",
        idempotency_key=str(uuid4()),
        request_hash="request-digest",
        parameters_json="{}",
        active_limit=2,
        now=2,
    )
    job = store.mark_submitted(
        admission.job.job_id,
        modal_call_id="fc-root",
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
        workload="gromacs",
        display_name=run_name,
        idempotency_key=str(uuid4()),
        request_hash=f"digest-{run_name}",
        parameters_json="{}",
        active_limit=2,
        now=completed_at - 2,
    )
    job = store.mark_submitted(
        admission.job.job_id,
        modal_call_id=f"fc-{run_name}",
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


def test_adapter_looks_up_named_function_in_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = FakeFunction(FakeCall("fc-root"))
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
        function_name="run_gromacs_job",
        environment_name="department-dev",
    )

    assert adapter.function is function
    assert lookups == [
        ("GromacsAPI", "run_gromacs_job", "department-dev"),
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
        )
    )

    assert submitted.modal_call_id == "fc-detached"
    assert submitted.run_name == RUN_NAME
    assert function.spawn_kwargs == {
        "pdb_content": b"PDB content",
        "run_name": RUN_NAME,
        "simulation_time_ns": 3,
        "run_pdbfixer": True,
        "cpu_only": True,
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


def test_final_archive_accepts_only_expected_immutable_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, job = _submitted_job(tmp_path)
    adapter = _adapter({})
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
    result = _archive_result(
        status=AppRunStatus.PARTIAL,
        size_bytes=len(archive_bytes),
        sha256=archive_sha256,
        warnings=["RMSF analysis was unavailable"],
    )

    archive = asyncio.run(adapter.final_archive(job, result))

    assert archive.state == JobState.PARTIAL
    assert archive.volume_name == "Gromacs-outputs"
    assert archive.path == f"api-results/{RUN_NAME}/result.zip"
    assert archive.filename == f"{RUN_NAME}.zip"
    assert archive.size_bytes == len(archive_bytes)
    assert archive.sha256 == archive_sha256
    assert archive.warnings_json == '["RMSF analysis was unavailable"]'


@pytest.mark.parametrize(
    "result",
    [
        AppRunResult(status=AppRunStatus.FAILED),
        AppRunResult(status=AppRunStatus.SUCCEEDED),
        _archive_result(kind=ArtifactKind.DIRECTORY),
        _archive_result(volume_name="Other-volume"),
        _archive_result(path="api-results/another-run/result.zip"),
        _archive_result(media_type=None),
        _archive_result(filename="../result.zip"),
        _archive_result(size_bytes=0),
        _archive_result(size_bytes=True),
        _archive_result(sha256="A" * 64),
    ],
)
def test_final_archive_rejects_invalid_contract(
    tmp_path: Path,
    result: AppRunResult,
) -> None:
    _store, job = _submitted_job(tmp_path)

    with pytest.raises(ValueError):
        asyncio.run(_adapter({}).final_archive(job, result))


def test_completed_archive_wins_cancel_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(tmp_path, cancel_requested=True)
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
    root_node = CallNode("fc-root", modal.call_graph.InputStatus.PENDING)
    root = FakeCall(
        "fc-root",
        result=_archive_result(
            size_bytes=len(archive_bytes),
            sha256=archive_sha256,
        ),
        graph=[root_node],
    )
    reconciler = GromacsReconciler(store, _adapter({"fc-root": root}), now=lambda: 10)

    asyncio.run(reconciler.reconcile())

    completed = store.get_job(job.owner_user_id, job.job_id)
    assert completed is not None
    assert completed.state == JobState.SUCCEEDED
    assert completed.result_volume_path == f"api-results/{RUN_NAME}/result.zip"
    assert ("cancel", "fc-root", False) in root.events


def test_expired_call_output_recovers_completed_archive_from_volume_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(tmp_path)
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
    store, job = _submitted_job(tmp_path)
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
    assert failed.error_code == "result_expired"


def test_recovery_rejects_self_consistent_marker_for_invalid_zip_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job = _submitted_job(tmp_path)
    archive_bytes, request_sha256 = _valid_archive_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        manifest = orjson.loads(archive.read("manifest.json"))
    manifest["files"][0]["sha256"] = "0" * 64
    invalid_archive = _replace_archive_member(
        archive_bytes,
        "manifest.json",
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


def test_reconciler_recovers_orphaned_submission_by_stable_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    user = store.create_user(
        email="alice@example.com",
        display_name="Alice",
        token_digest=b"setup-token-digest",
        token_expires_at=3_600,
        now=1,
    )
    admission = store.admit_job(
        owner_user_id=user.user_id,
        workload="gromacs",
        display_name="Simulation",
        idempotency_key=str(uuid4()),
        request_hash="request-digest",
        parameters_json="{}",
        active_limit=2,
        now=2,
    )
    store.claim_submission(
        admission.job.job_id,
        run_name=RUN_NAME,
        submission_token="lost-submitter",
        now=3,
    )
    archive_bytes, request_sha256 = _valid_archive_bytes()
    _install_volume(
        monkeypatch,
        {
            f"api-results/{RUN_NAME}/result.json": _result_marker(
                archive_bytes, request_sha256
            ),
            f"api-results/{RUN_NAME}/result.zip": archive_bytes,
        },
    )

    asyncio.run(GromacsReconciler(store, _adapter({}), now=lambda: 10).reconcile())

    completed = store.get_job(user.user_id, admission.job.job_id)
    assert completed is not None
    assert completed.state == JobState.SUCCEEDED


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


def test_intermediate_cleanup_is_opt_in_and_preserves_final_archives(
    tmp_path: Path,
) -> None:
    now = 1_000_000
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    user = store.create_user(
        email="alice@example.com",
        display_name="Alice",
        token_digest=b"setup-token-digest",
        token_expires_at=3_600,
        now=1,
    )
    due_succeeded = _terminal_job(
        store,
        user.user_id,
        run_name=f"api-{'1' * 32}",
        state=JobState.SUCCEEDED,
        completed_at=100,
    )
    due_partial = _terminal_job(
        store,
        user.user_id,
        run_name=f"api-{'2' * 32}",
        state=JobState.PARTIAL,
        completed_at=200,
    )
    recent_succeeded = _terminal_job(
        store,
        user.user_id,
        run_name=f"api-{'3' * 32}",
        state=JobState.SUCCEEDED,
        completed_at=now - 60,
    )
    due_failed = _terminal_job(
        store,
        user.user_id,
        run_name=f"api-{'4' * 32}",
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
