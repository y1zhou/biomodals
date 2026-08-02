"""AF3Score execution-adapter tests."""

# ruff: noqa: D101,D102,D103,D107

from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from biomodals.app.score import af3score_app
from biomodals.app.score.af3score_execution import (
    BATCHES_NODE,
    COMPLETION_REQUIRED_FILES,
    COMPLETION_SAMPLE_SUBDIR,
    METRICS_FILENAME,
    POSTPROCESS_NODE,
    PREPARE_NODE,
    AF3ScoreExecutionRequest,
    AF3ScoreExecutionRuntime,
    ChunkSpec,
    TaskSpec,
)
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.helper.app_execution import ExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OTHER_RUN_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
DEPLOYMENT = DeploymentIdentity("main", "AF3Score", 7)
INPUT_CONTENT = {"a.pdb": b"ATOM A\n", "b.pdb": b"ATOM B\n"}


class FakeVolume:
    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


class FakeClaims:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def put(self, key: str, value: str, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True


class CompletingDriver:
    def __init__(self, root: Path, request: AF3ScoreExecutionRequest) -> None:
        self.root = root / request.run_name
        self.request = request
        self.calls: dict[str, tuple[Any, dict[str, object]]] = {}
        self.spawns: list[tuple[str, dict[str, object]]] = []

    def resolve(self, binding):
        return binding

    def spawn(self, function, *, args, kwargs):
        handle = f"fc-{len(self.calls) + 1}"
        copied = dict(kwargs)
        self.calls[handle] = (function, copied)
        self.spawns.append((function.function_name, copied))
        return handle

    def observe(self, provider_call_handle_id: str):
        function, kwargs = self.calls[provider_call_handle_id]
        result = self._publish(function.function_name, kwargs)
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=result,
        )

    def cancel(self, provider_call_handle_id: str) -> None:
        pass

    def _publish(self, function_name: str, kwargs: dict[str, object]):
        if function_name == "af3score_prepare":
            json_dir = self.root / "prepare" / "input_batch" / "json" / "batch_0"
            pdb_dir = self.root / "prepare" / "input_batch" / "pdb" / "batch_0"
            json_dir.mkdir(parents=True, exist_ok=True)
            pdb_dir.mkdir(parents=True, exist_ok=True)
            for name in self.request.input_names:
                (json_dir / f"{Path(name).stem}.json").write_text("{}")
            return TaskSpec(
                total=len(self.request.inputs),
                pending=len(self.request.inputs),
                skipped=0,
                input_files=list(self.request.input_names),
                chunk_specs=[ChunkSpec("batch_0", str(json_dir), str(pdb_dir))],
                output_dir=str(self.root / "outputs"),
                failed_dir=str(self.root / "outputs" / "failed_records"),
            )
        if function_name == "af3score_run":
            for path in Path(str(kwargs["batch_json_dir"])).glob("*.json"):
                sample = self.root / "outputs" / path.stem / COMPLETION_SAMPLE_SUBDIR
                sample.mkdir(parents=True, exist_ok=True)
                for required in COMPLETION_REQUIRED_FILES:
                    (sample / required).write_text("{}")
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        metrics = self.root / METRICS_FILENAME
        metrics.write_text("name,score\na,1\n")
        af3score_app._write_metrics_publication(
            self.root,
            str(kwargs["publication_key"]),
            metrics,
        )
        return {"metrics_csv_exists": 1, "metrics_rows": 1}


def _request() -> AF3ScoreExecutionRequest:
    return AF3ScoreExecutionRequest(
        run_name="scores",
        inputs=tuple(
            (name, sha256(content).hexdigest())
            for name, content in INPUT_CONTENT.items()
        ),
        staged_input_execution_run_id=str(RUN_ID),
        prepare_workers=4,
        max_batches=2,
        app_version="b0764aa",
    )


def _stage_request_inputs(root: Path, request: AF3ScoreExecutionRequest) -> None:
    directory = (
        root
        / ".biomodals"
        / "execution"
        / "runs"
        / request.staged_input_execution_run_id
        / "inputs"
    )
    directory.mkdir(parents=True)
    for name in request.input_names:
        directory.joinpath(name).write_bytes(INPUT_CONTENT[name])


def test_af3score_request_round_trip_preserves_parallel_task_plan() -> None:
    request = _request()

    decoded = AF3ScoreExecutionRequest.from_bytes(request.to_bytes())

    assert decoded == request
    assert decoded.execution_plan.node_keys == (
        PREPARE_NODE,
        BATCHES_NODE,
        POSTPROCESS_NODE,
    )
    assert decoded.execution_plan.terminal_node_keys == (POSTPROCESS_NODE,)
    assert decoded.execution_plan.scientific_payload["inputs"] == [
        {"name": name, "sha256": sha256(content).hexdigest()}
        for name, content in INPUT_CONTENT.items()
    ]


def test_af3score_operational_limits_do_not_change_scientific_identity() -> None:
    base = AF3ScoreExecutionRequest(
        run_name="scores",
        inputs=(("a.pdb", "a" * 64),),
        staged_input_execution_run_id=str(RUN_ID),
        prepare_workers=4,
        max_batches=2,
        app_version="b0764aa",
    )
    changed = AF3ScoreExecutionRequest(
        run_name="scores",
        inputs=base.inputs,
        staged_input_execution_run_id=str(OTHER_RUN_ID),
        prepare_workers=8,
        max_batches=6,
        app_version=base.app_version,
        replace_claim_owner="old-run",
    )

    assert (
        base.execution_plan.workload_plan_fingerprint
        == changed.execution_plan.workload_plan_fingerprint
    )


def test_same_run_name_inputs_are_isolated_until_output_claim(
    tmp_path: Path,
) -> None:
    """One root Run cannot overwrite another root Run before claim ownership."""
    first_content = b"ATOM FIRST\n"
    second_content = b"ATOM SECOND\n"

    def request(execution_run_id: UUID, content: bytes) -> AF3ScoreExecutionRequest:
        return AF3ScoreExecutionRequest(
            run_name="scores",
            inputs=(("target.pdb", sha256(content).hexdigest()),),
            staged_input_execution_run_id=str(execution_run_id),
            prepare_workers=1,
            max_batches=1,
            app_version="b0764aa",
        )

    def stage(execution_run_id: UUID, content: bytes) -> None:
        path = (
            tmp_path
            / ".biomodals"
            / "execution"
            / "runs"
            / str(execution_run_id)
            / "inputs"
            / "target.pdb"
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(content)

    stage(RUN_ID, first_content)
    stage(OTHER_RUN_ID, second_content)
    claims = FakeClaims()
    first = AF3ScoreExecutionRuntime(
        request=request(RUN_ID, first_content),
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=object(),
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
    )
    second = AF3ScoreExecutionRuntime(
        request=request(OTHER_RUN_ID, second_content),
        execution_run_id=OTHER_RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, OTHER_RUN_ID),
        modal_driver=object(),
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
    )
    shared_input = tmp_path / "scores" / "inputs" / "target.pdb"

    assert not shared_input.exists()
    first._ensure_output_claim()
    assert shared_input.read_bytes() == first_content
    with pytest.raises(RuntimeError, match="already claimed"):
        second._ensure_output_claim()
    assert shared_input.read_bytes() == first_content
    first.close()
    second.close()


def test_materialization_rejects_changed_staged_input(tmp_path: Path) -> None:
    request = _request()
    _stage_request_inputs(tmp_path, request)
    staged_input = (
        tmp_path
        / ".biomodals"
        / "execution"
        / "runs"
        / str(RUN_ID)
        / "inputs"
        / "a.pdb"
    )
    staged_input.write_bytes(b"CHANGED\n")
    runtime = AF3ScoreExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=object(),
        output_volume=FakeVolume(),
        output_claims=FakeClaims(),
        output_root=tmp_path,
    )

    with pytest.raises(ValueError, match="digest changed: a.pdb"):
        runtime._ensure_output_claim()

    assert not tmp_path.joinpath("scores", "inputs", "a.pdb").exists()
    runtime.close()


def test_runtime_discovers_input_tasks_and_submits_one_gpu_batch(
    tmp_path: Path,
) -> None:
    request = _request()
    _stage_request_inputs(tmp_path, request)
    driver = CompletingDriver(tmp_path, request)
    claims = FakeClaims()
    runtime = AF3ScoreExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert [name for name, _kwargs in driver.spawns] == [
        "af3score_prepare",
        "af3score_run",
        "af3score_postprocess",
    ]
    batch_call = next(
        call for call in snapshot.provider_calls if call.node_key == BATCHES_NODE
    )
    assert batch_call.task_keys == ("a", "b")
    assert str(RUN_ID) in claims.values.values()
    runtime.close()


def test_unbound_metrics_do_not_satisfy_a_new_request(tmp_path: Path) -> None:
    request = _request()
    _stage_request_inputs(tmp_path, request)
    run_root = tmp_path / request.run_name
    run_root.mkdir()
    (run_root / METRICS_FILENAME).write_text("name,score\na,1\n")
    driver = CompletingDriver(tmp_path, request)
    claims = FakeClaims()
    runtime = AF3ScoreExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert [name for name, _kwargs in driver.spawns] == [
        "af3score_prepare",
        "af3score_run",
        "af3score_postprocess",
    ]
    assert str(RUN_ID) in claims.values.values()
    runtime.close()


def test_fingerprint_bound_metrics_satisfy_the_terminal_node(tmp_path: Path) -> None:
    request = _request()
    run_root = tmp_path / request.run_name
    run_root.mkdir()
    metrics = run_root / METRICS_FILENAME
    metrics.write_text("name,score\na,1\n")
    af3score_app._write_metrics_publication(
        run_root,
        request.execution_plan.workload_plan_fingerprint,
        metrics,
    )
    driver = CompletingDriver(tmp_path, request)
    claims = FakeClaims()
    runtime = AF3ScoreExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert driver.spawns == []
    assert claims.values == {}
    runtime.close()
