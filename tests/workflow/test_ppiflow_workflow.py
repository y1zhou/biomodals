"""Tests for the PPIFlow workflow definition."""

# ruff: noqa: D103

import pickle
import tarfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import polars as pl
import pytest
import zstandard as zstd

from biomodals.app.design import ppiflow_app
from biomodals.helper.styling import strip_ansi
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow import ppiflow_workflow
from biomodals.workflow.core import (
    NodeRunContext,
    RemoteWorkflowNode,
)
from biomodals.workflow.core._runtime import hashing
from biomodals.workflow.ppiflow import manifests as ppiflow_manifests
from biomodals.workflow.ppiflow_workflow import (
    CONF,
    build_ppiflow_workflow,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


class _FakeFunctionCall:
    def __init__(self, object_id: str, result: AppRunResult | None = None) -> None:
        self.object_id = object_id
        self.result = result or AppRunResult(status=AppRunStatus.SUCCEEDED)

    def get(self, timeout=None):
        _ = timeout
        return self.result


class _FakePPIFlowFunction:
    def __init__(self) -> None:
        self.kwargs = {}

    def _result(self) -> AppRunResult:
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ppiflow_outputs",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=ppiflow_app.CONF.output_volume_name,
                        path="demo-run",
                    ),
                )
            ],
        )

    def remote(self, **kwargs):
        self.kwargs = kwargs
        return self._result()

    def spawn(self, **kwargs):
        self.kwargs = kwargs
        return _FakeFunctionCall("fc-ppiflow", self._result())


class _FakeModalFunction:
    def __init__(
        self,
        object_id: str,
        result=None,
    ) -> None:
        self.object_id = object_id
        self.result = result
        self.args = ()
        self.kwargs = {}

    def remote(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self.result

    def spawn(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return _FakeFunctionCall(self.object_id, self.result)


def _task_yaml(*, enabled_steps: str) -> bytes:
    return f"""
task:
  gentype: binder
steps:
{enabled_steps}
""".encode()


def _upstream_structure_artifact(
    *,
    kind: ArtifactKind = ArtifactKind.STRUCTURES,
    metadata: dict[str, object] | None = None,
) -> WorkflowArtifact:
    return WorkflowArtifact(
        artifact_id="upstream-structures",
        producing_node_id="upstream",
        kind=kind,
        storage=VolumePath(volume_name="source-volume", path="upstream/results"),
        metadata=metadata or {},
    )


def _manifest_ancestor_chain(definition, node_id: str) -> list[str]:
    chain = []
    current = node_id
    while "candidate_manifest" in definition.nodes[current].inputs:
        current = (
            definition.nodes[current].inputs["candidate_manifest"].producing_node_id
        )
        chain.append(current)
    return chain


def _decorator_block(source: str, function_name: str) -> str:
    return source.split(f"def {function_name}", 1)[0].rsplit("@app.function", 1)[-1]


def _local_transform_environment(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    workflow_root = tmp_path / "workflow"
    source_root.mkdir()
    workflow_root.mkdir()
    monkeypatch.setattr(
        ppiflow_workflow, "_reload_ppiflow_source_volumes", lambda: None
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_SOURCE_VOLUME_ROOTS",
        {"source-volume": str(source_root), "workflow-volume": str(workflow_root)},
    )
    monkeypatch.setattr(
        ppiflow_workflow, "WORKFLOW_OUTPUT_MOUNTPOINT", str(workflow_root)
    )
    monkeypatch.setattr(
        ppiflow_workflow, "WORKFLOW_OUTPUT_VOLUME_NAME", "workflow-volume"
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "WORKFLOW_OUTPUT_VOLUME",
        SimpleNamespace(commit=lambda: None),
    )
    return source_root, workflow_root


def _tar_zst_bytes(files: dict[str, bytes]) -> bytes:
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    return zstd.ZstdCompressor().compress(tar_buffer.getvalue())


def test_ppiflow_workflow_declares_app_dependency() -> None:
    assert CONF.depends_on_apps == (
        "ppiflow",
        "rosetta",
        "flowpacker",
        "ligandmpnn",
        "dockq",
        "af3score",
        "alphafold3",
    )
    assert CONF.tags == {"depends_on": "-".join(CONF.depends_on_apps)}


def test_ppiflow_stage_wrappers_declare_stage_specific_mounts() -> None:
    source = Path(ppiflow_workflow.__file__).read_text(encoding="utf-8")

    for function_name in (
        "run_ppiflow_ligandmpnn_stage",
        "run_ppiflow_refold_candidate",
        "run_ppiflow_partial_stage",
        "run_ppiflow_flowpacker_stage",
        "run_ppiflow_dockq_stage",
    ):
        assert "PPI_FLOW_SOURCE_VOLUME_MOUNTS" in _decorator_block(
            source,
            function_name,
        )
    assert "PPI_FLOW_SOURCE_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_af3score_stage",
    )
    assert "PPI_FLOW_SOURCE_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_rosetta_stage",
    )


def test_ppiflow_app_step_prepares_kernel_call_and_normalizes_result(
    tmp_path: Path,
) -> None:
    fake_function = _FakePPIFlowFunction()
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  PPIFlowStep: true\n"),
        steps_yaml_bytes=b"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: /inputs/demo.pdb
    binder_chain: B
""",
    )

    definition = workflow.validate()
    spec = definition.nodes["stage1-ppiflow-design"]
    call = spec.node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id=spec.node_id,
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={},
        )
    )
    result = spec.node.process_remote_result(fake_function._result(), call.metadata)

    assert result.status == AppRunStatus.SUCCEEDED
    assert call.function_name == "ppiflow_run"
    assert call.kwargs["run_name"] == "demo-run"
    assert isinstance(call.kwargs["args"], ppiflow_app.PPIFlowArgs)
    assert result.outputs[0].storage == VolumePath(
        volume_name=ppiflow_app.CONF.output_volume_name,
        path="demo-run",
    )
    assert result.outputs[0].metadata["structure_patterns"] == (
        "outputs/*.pdb",
        "outputs/**/*.pdb",
        "outputs/*.cif",
        "outputs/**/*.cif",
    )


def test_ppiflow_app_step_preparation_does_not_submit_provider_call(
    tmp_path: Path,
) -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  PPIFlowStep: true\n"),
        steps_yaml_bytes=b"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: /inputs/demo.pdb
    binder_chain: B
""",
    )

    spec = workflow.validate().nodes["stage1-ppiflow-design"]
    submission = spec.node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id=spec.node_id,
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={},
        )
    )

    assert submission.function_name == "ppiflow_run"
    assert submission.kwargs["run_name"] == "demo-run"
    assert isinstance(submission.kwargs["args"], ppiflow_app.PPIFlowArgs)
    assert isinstance(submission.kwargs["args"].args.input_pdb, str)
    assert isinstance(submission.kwargs["args"].args.config, str)


def test_ligandmpnn_step_prepares_selected_structures_for_kernel(
    tmp_path: Path,
) -> None:
    ligandmpnn_stage = _FakeModalFunction(
        "fc-ligandmpnn-stage",
        AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="LigandMPNN_outputs",
                    kind=ArtifactKind.STRUCTURES,
                    storage=InlineBytes(
                        data=_tar_zst_bytes({
                            "outputs/seqs/selected.fa": b">selected_design\nACD\n"
                        }),
                        filename="ligandmpnn.tar.zst",
                        media_type="application/zstd",
                    ),
                ),
                AppOutput(
                    name="mpnn_seqs",
                    kind=ArtifactKind.TABLE,
                    storage=InlineBytes(
                        data=b"candidate_id,sequence\nselected,ACD\n",
                        filename="mpnn_seqs.csv",
                        media_type="text/csv",
                    ),
                ),
            ],
        ),
    )
    node = ppiflow_workflow.LigandMPNNNode(
        "MPNNStep_stage1",
        {
            "run_name": "mpnn-run",
            "seeds": "1,2",
            "model_type": "protein_mpnn",
            "batch_size": 2,
            "number_of_batches": 3,
        },
    )
    context = NodeRunContext(
        execution_run_id=RUN_ID,
        workload_run_key="run-1",
        node_id="stage1-ligandmpnn",
        task_key="node",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={"structures": [_upstream_structure_artifact()]},
    )

    submission = node.prepare_remote(context)
    result = node.process_remote_result(ligandmpnn_stage.result, submission.metadata)

    assert submission.function_name == "run_ppiflow_ligandmpnn_stage"
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert submission.kwargs["run_name"] == "mpnn-run"
    assert submission.kwargs["script_mode"] == "run"
    assert submission.kwargs["cli_args"]["--model_type"] == "protein_mpnn"
    assert submission.kwargs["cli_args"]["--batch_size"] == "2"
    assert submission.kwargs["cli_args"]["--number_of_batches"] == "3"
    assert ligandmpnn_stage.kwargs == {}
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[1].kind == ArtifactKind.TABLE
    assert result.outputs[1].name == "mpnn_seqs"
    assert b"selected,ACD" in result.outputs[1].storage.data


def test_ligandmpnn_defers_multi_structure_selection_to_tracked_stage(
    tmp_path: Path,
) -> None:
    artifacts = [
        _upstream_structure_artifact(),
        _upstream_structure_artifact(metadata={"candidate_id": "second"}),
    ]
    node = ppiflow_workflow.LigandMPNNNode(
        "MPNNStep_stage1",
        {"run_name": "mpnn-run"},
    )

    submission = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage1-ligandmpnn",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={"structures": artifacts},
        )
    )

    assert submission.kwargs["artifacts"] == artifacts


def test_flowpacker_step_prepares_selected_structures_for_kernel(
    tmp_path: Path,
) -> None:
    flowpacker = _FakeModalFunction(
        "fc-flowpacker",
        AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="flowpacker_outputs",
                    kind=ArtifactKind.ARCHIVE,
                    storage=VolumePath(
                        volume_name="FlowPacker-outputs",
                        path="workflow/fp-run/outputs/fp-run.tar.zst",
                        media_type="application/zstd",
                    ),
                )
            ],
        ),
    )
    node = ppiflow_workflow.FlowPackerNode(
        "FlowpackerStep_stage1",
        {"run_name": "fp-run", "n_samples": 2, "seed": 7},
    )
    context = NodeRunContext(
        execution_run_id=RUN_ID,
        workload_run_key="run-1",
        node_id="stage1-flowpacker",
        task_key="node",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={"structures": [_upstream_structure_artifact()]},
    )

    submission = node.prepare_remote(context)
    result = node.process_remote_result(flowpacker.result, submission.metadata)

    assert submission.function_name == "run_ppiflow_flowpacker_stage"
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert submission.kwargs["run_name"] == "fp-run"
    assert submission.kwargs["config"]["n_samples"] == 2
    assert submission.kwargs["config"]["seed"] == 7
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES


def test_partial_step_defers_structure_selection_to_tracked_stage(
    tmp_path: Path,
) -> None:
    node = ppiflow_workflow.PPIFlowPartialNode(
        "PartialStep",
        {
            "run_name": "partial-run",
            "args": {
                "name": "partial",
                "specified_hotspots": "A1",
                "input_pdb": "/placeholder.pdb",
                "binder_chain": "B",
                "start_t": 0.5,
            },
        },
    )

    submission = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-partial-ppiflow",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert submission.function_name == "run_ppiflow_partial_stage"
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]


def test_af3score_step_runs_app_sequence_and_returns_metrics_artifact(
    tmp_path: Path,
) -> None:
    node = ppiflow_workflow.AF3ScoreNode(
        "AF3scoreStep_stage1",
        {"run_name": "af3-run", "num_jobs": 4, "prepare_workers": 2},
    )

    submission = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage1-af3score",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={
                "structures": [_upstream_structure_artifact()],
                "candidate_manifest": [
                    _upstream_structure_artifact(kind=ArtifactKind.TABLE)
                ],
            },
        )
    )

    assert submission.function_name == "run_ppiflow_af3score_stage"
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert submission.kwargs["candidate_manifests"] == [
        _upstream_structure_artifact(kind=ArtifactKind.TABLE)
    ]
    assert submission.kwargs["run_name"] == "af3-run"
    assert submission.kwargs["config"]["num_jobs"] == 4
    assert submission.kwargs["config"]["prepare_workers"] == 2


def test_af3score_staging_uses_candidate_key_not_full_artifact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    long_dir = (
        source_root
        / "upstream"
        / "results"
        / ("flowpacker_outputs_" + "x" * 180)
        / "backbones"
    )
    long_dir.mkdir(parents=True)
    (long_dir / "candidate_a.pdb").write_text("ATOM\n", encoding="utf-8")
    af3_root = tmp_path / "af3"
    commits = []

    monkeypatch.setattr(
        ppiflow_workflow, "_reload_ppiflow_source_volumes", lambda: None
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_SOURCE_VOLUME_ROOTS",
        {"source-volume": str(source_root)},
    )
    monkeypatch.setattr(ppiflow_workflow, "AF3SCORE_OUTPUT_MOUNTPOINT", str(af3_root))
    monkeypatch.setattr(
        ppiflow_workflow,
        "AF3SCORE_OUTPUT_VOLUME",
        SimpleNamespace(commit=lambda: commits.append(True)),
    )

    input_names = ppiflow_workflow.stage_af3score_inputs.get_raw_f()(
        artifacts=[
            WorkflowArtifact(
                artifact_id="stage1-flowpacker-flowpacker_outputs",
                producing_node_id="stage1-flowpacker",
                kind=ArtifactKind.STRUCTURES,
                storage=VolumePath(
                    volume_name="source-volume",
                    path="upstream/results",
                ),
            )
        ],
        run_name="af3-run",
    )

    assert input_names == ["candidate_a.pdb"]
    assert (af3_root / "af3-run" / "inputs" / "candidate_a.pdb").read_text(
        encoding="utf-8"
    ) == "ATOM\n"
    assert commits == [True]


def test_af3score_step_reports_partial_for_mixed_scores(tmp_path: Path) -> None:
    af3score_stage = _FakeModalFunction(
        "fc-af3score-stage",
        AppRunResult(status=AppRunStatus.PARTIAL),
    )
    node = ppiflow_workflow.AF3ScoreNode(
        "AF3scoreStep_stage1",
        {"run_name": "af3-run"},
    )

    call = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage1-af3score",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )
    result = node.process_remote_result(af3score_stage.result, call.metadata)

    assert result.status == AppRunStatus.PARTIAL


def test_rosetta_step_submits_stage_wrapper(
    tmp_path: Path,
) -> None:
    node = ppiflow_workflow.RosettaRelaxNode(
        "RosettaRelaxStep",
        {"run_name": "rosetta-run", "rosetta_binary": "relax", "max_num_pods": 1},
    )

    submission = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-rosetta-relax",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={
                "structures": [_upstream_structure_artifact()],
                "candidate_manifest": [
                    _upstream_structure_artifact(kind=ArtifactKind.TABLE)
                ],
            },
        )
    )

    assert submission.function_name == "run_ppiflow_rosetta_stage"
    assert submission.kwargs["run_name"] == "rosetta-run"
    assert submission.kwargs["config"]["rosetta_binary"] == "relax"
    assert submission.kwargs["candidate_manifests"] == [
        _upstream_structure_artifact(kind=ArtifactKind.TABLE)
    ]


def test_rosetta_stage_records_partial_candidate_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    rosetta_root = tmp_path / "rosetta"
    run_root = rosetta_root / "rosetta-run-rosetta-id"
    (run_root / "outputs" / "1").mkdir(parents=True)
    (run_root / "outputs" / "1" / "score.sc").write_text("SCORE\n", encoding="utf-8")
    (run_root / "logs").mkdir()
    (run_root / "logs" / "1.log").write_text("ok\n", encoding="utf-8")
    job_manifest = run_root / "rosetta_job_manifest.csv"
    pl.DataFrame([
        {
            "candidate_id": "candidate-a",
            "index": 1,
            "pdb": "inputs/1/a.pdb",
            "expected_output_dir": "outputs/1",
            "expected_score_file": "outputs/1/score.sc",
            "worker_log": "logs/1.log",
        },
        {
            "candidate_id": "candidate-b",
            "index": 2,
            "pdb": "inputs/2/b.pdb",
            "expected_output_dir": "outputs/2",
            "expected_score_file": "outputs/2/score.sc",
            "worker_log": "logs/2.log",
        },
    ]).write_csv(job_manifest)

    def fake_stage_rosetta(**kwargs):
        _ = kwargs
        return {
            "run_name": "rosetta-run",
            "run_id": "rosetta-id",
            "run_root": str(run_root),
            "num_jobs": 2,
            "job_manifest": str(job_manifest),
        }

    monkeypatch.setattr(
        ppiflow_workflow,
        "stage_rosetta_inputs",
        SimpleNamespace(get_raw_f=lambda: fake_stage_rosetta),
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "ROSETTA_OUTPUT_MOUNTPOINT",
        str(rosetta_root),
    )
    monkeypatch.setattr(ppiflow_workflow, "ROSETTA_OUTPUT_VOLUME_NAME", "rosetta-vol")
    monkeypatch.setattr(
        ppiflow_workflow,
        "ROSETTA_OUTPUT_VOLUME",
        SimpleNamespace(reload=lambda: None),
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "bounded_map",
        lambda items, _worker, *, max_parallel: [None for _ in items],
    )
    monkeypatch.setattr(
        ppiflow_workflow.modal,
        "Queue",
        SimpleNamespace(objects=SimpleNamespace(delete=lambda _name: None)),
    )

    result = ppiflow_workflow.run_ppiflow_rosetta_stage.get_raw_f()(
        artifacts=[_upstream_structure_artifact()],
        config={"rosetta_binary": "relax", "max_num_pods": 1},
        step_name="RosettaRelaxStep",
        run_name="rosetta-run",
        run_id="run-1",
        node_id="stage2-rosetta-relax",
    )

    assert result.status == AppRunStatus.PARTIAL
    assert [output.name for output in result.outputs] == [
        "rosetta_outputs",
        "rosetta_job_manifest",
        "candidate_manifest",
    ]
    frame = ppiflow_manifests.read_manifest(
        workflow_root / result.outputs[-1].storage.path
    )
    assert frame.select("candidate_id", "candidate_status").to_dicts() == [
        {"candidate_id": "candidate-a", "candidate_status": "succeeded"},
        {"candidate_id": "candidate-b", "candidate_status": "failed"},
    ]


def test_refold_step_derives_af3_config_and_runs_inference(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifests" / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(
        [
            ppiflow_manifests.candidate_manifest_row(
                candidate_id="candidate-a",
                stage_name="FilterStep_stage2",
                stage_role="filter",
                operation_mode="retained",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="structure",
                        content_sha256="0" * 64,
                    )
                ],
            )
        ],
        manifest_path,
    )
    manifest = WorkflowArtifact(
        artifact_id="candidate-manifest",
        producing_node_id="stage2-filter",
        kind=ArtifactKind.TABLE,
        storage=VolumePath(
            volume_name="workflow-volume",
            path=manifest_path.relative_to(tmp_path).as_posix(),
        ),
    )
    node = ppiflow_workflow.ReFoldNode(
        "ReFoldStep",
        {"run_name": "refold-run", "model_seeds": [3], "recycle": 2, "sample": 1},
    )
    context = NodeRunContext(
        execution_run_id=RUN_ID,
        workload_run_key="run-1",
        node_id="stage2-alphafold3-refold",
        task_key="node",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={
            "structures": [_upstream_structure_artifact()],
            "candidate_manifest": [manifest],
        },
        volume_root=tmp_path,
        workflow_volume_name="workflow-volume",
    )
    [task] = node.discover_remote_tasks(context)
    submission = node.prepare_remote_task(
        replace(context, task_key=task.task_key),
        task,
    )

    assert task.task_key == "candidate-a"
    assert submission.function_name == "run_ppiflow_refold_candidate"
    assert submission.kwargs["candidate_id"] == "candidate-a"
    assert submission.kwargs["config"]["recycle"] == 2
    assert submission.kwargs["config"]["sample"] == 1
    assert submission.kwargs["config"]["model_seeds"] == [3]
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]


def test_refold_discovers_manifest_candidates_in_order(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(
        [
            ppiflow_manifests.candidate_manifest_row(
                candidate_id=candidate_id,
                stage_name="FilterStep_stage2",
                stage_role="filter",
                operation_mode="retained",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="structure",
                        content_sha256="0" * 64,
                    )
                ],
            )
            for candidate_id in ("candidate-b", "candidate-a")
        ],
        manifest_path,
    )
    manifest = WorkflowArtifact(
        artifact_id="candidate-manifest",
        producing_node_id="stage2-filter",
        kind=ArtifactKind.TABLE,
        storage=VolumePath(
            volume_name="workflow-volume",
            path=manifest_path.name,
        ),
    )
    node = ppiflow_workflow.ReFoldNode(
        "ReFoldStep",
        {"run_name": "refold-run", "max_structures": 1},
    )

    tasks = node.discover_remote_tasks(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-alphafold3-refold",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={
                "structures": [_upstream_structure_artifact()],
                "candidate_manifest": [manifest],
            },
            volume_root=tmp_path,
            workflow_volume_name="workflow-volume",
        )
    )

    assert [task.task_key for task in tasks] == ["candidate-b"]


def test_dockq_step_pairs_filtered_and_refolded_structures(tmp_path: Path) -> None:
    dockq = _FakeModalFunction(
        "fc-dockq",
        AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="dockq_scores",
                    kind=ArtifactKind.SCORES,
                    storage=InlineBytes(
                        data=b"dockq",
                        filename="dockq.tar.zst",
                        media_type="application/zstd",
                    ),
                )
            ],
        ),
    )
    node = ppiflow_workflow.DockQNode(
        "DockQStep",
        {"run_name": "dockq-run", "dockq_args": "--short"},
    )
    submission = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-dockq",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={
                "structures": [_upstream_structure_artifact()],
                "models": [_upstream_structure_artifact()],
            },
        )
    )
    result = node.process_remote_result(dockq.result, submission.metadata)

    assert submission.function_name == "run_ppiflow_dockq_stage"
    assert submission.kwargs["run_name"] == "dockq-run"
    assert submission.kwargs["config"]["dockq_args"] == "--short"
    assert submission.kwargs["reference_artifacts"] == [_upstream_structure_artifact()]
    assert submission.kwargs["model_artifacts"] == [_upstream_structure_artifact()]
    assert result.outputs[0].kind == ArtifactKind.SCORES


def test_dockq_stage_rejects_unpaired_structure_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = iter([
        [("reference.pdb", b"ATOM REF\n")],
        [
            ("model-a.pdb", b"ATOM A\n"),
            ("model-b.pdb", b"ATOM B\n"),
        ],
    ])
    monkeypatch.setattr(
        ppiflow_workflow,
        "_reload_ppiflow_source_volumes",
        lambda: None,
    )
    monkeypatch.setattr(
        ppiflow_workflow.ppiflow_staging,
        "select_structure_files_from_artifacts",
        lambda *args, **kwargs: next(selected),
    )
    with pytest.raises(ValueError, match="pairing mismatch"):
        ppiflow_workflow.run_ppiflow_dockq_stage.get_raw_f()(
            reference_artifacts=[_upstream_structure_artifact()],
            model_artifacts=[_upstream_structure_artifact()],
            candidate_manifests=None,
            config={},
            run_name="dockq-run",
        )


def test_filter_step_delegates_score_filtering(
    tmp_path: Path,
) -> None:
    node = ppiflow_workflow.FilterStructuresNode(
        "FilterStep_stage1",
        {"filters": {"iptm": "> 0.7"}},
    )

    call = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage1-filter",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={
                "structures": [_upstream_structure_artifact()],
                "scores": [_upstream_structure_artifact(kind=ArtifactKind.SCORES)],
            },
        )
    )

    assert call.function_name == "filter_ppiflow_artifacts"
    assert call.kwargs["step_name"] == "FilterStep_stage1"
    assert call.kwargs["config"] == {"filters": {"iptm": "> 0.7"}}


def test_fixed_positions_delegates_residue_energy_parsing(tmp_path: Path) -> None:
    node = ppiflow_workflow.FixedPositionsNode(
        "FixedPositions",
        {"gentype": "binder", "energy_threshold": -5},
    )

    call = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-fixed-positions",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert call.function_name == "derive_ppiflow_fixed_positions"
    assert call.kwargs["config"] == {
        "gentype": "binder",
        "energy_threshold": -5,
    }


def test_rank_step_delegates_score_aware_ranking(tmp_path: Path) -> None:
    node = ppiflow_workflow.RankNode(
        "RankStep",
        {"gentype": "binder"},
    )

    call = node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-rank",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert call.function_name == "rank_ppiflow_artifacts"
    assert call.kwargs["config"] == {"gentype": "binder"}


def test_filter_transform_selects_only_passing_structures(
    tmp_path: Path, monkeypatch
) -> None:
    source_root, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    structures_dir = source_root / "structures"
    structures_dir.mkdir()
    (structures_dir / "design-1.pdb").write_text("ATOM 1\n", encoding="utf-8")
    (structures_dir / "design-2.pdb").write_text("ATOM 2\n", encoding="utf-8")
    (source_root / "metrics.csv").write_text(
        "description,iptm\ndesign-1,0.8\ndesign-2,0.6\n",
        encoding="utf-8",
    )
    structure_artifact = _upstream_structure_artifact()
    structure_artifact.storage.path = "structures"

    result = ppiflow_workflow.filter_ppiflow_artifacts.get_raw_f()(
        structures=[structure_artifact],
        scores=[
            WorkflowArtifact(
                artifact_id="scores",
                producing_node_id="scores",
                kind=ArtifactKind.SCORES,
                storage=VolumePath(volume_name="source-volume", path="metrics.csv"),
            )
        ],
        config={"filters": {"iptm": "> 0.7"}},
        run_id="run-1",
        node_id="filter",
        step_name="FilterStep_stage1",
    )

    output_dir = workflow_root / result.outputs[0].storage.path
    assert [path.name for path in output_dir.iterdir()] == [
        "upstream-structures__design-1.pdb"
    ]
    assert result.outputs[0].metadata["retained_count"] == 1
    assert [output.name for output in result.outputs] == [
        "filtered_structures",
        "filtered_scores",
        "retained_candidate_manifest",
        "filter_audit",
    ]
    retained_manifest = ppiflow_manifests.read_manifest(
        workflow_root / result.outputs[2].storage.path
    )
    assert retained_manifest.get_column("candidate_id").to_list() == ["design-1"]
    audit = pl.read_csv(workflow_root / result.outputs[3].storage.path)
    assert audit.select("candidate_id", "passed", "reason").to_dicts() == [
        {"candidate_id": "design-1", "passed": True, "reason": "passed"},
        {"candidate_id": "design-2", "passed": False, "reason": "filtered"},
    ]


def test_fixed_position_transform_parses_rosetta_residue_energies(
    tmp_path: Path, monkeypatch
) -> None:
    source_root, _ = _local_transform_environment(monkeypatch, tmp_path)
    rosetta_root = source_root / "rosetta"
    outputs_dir = rosetta_root / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "design.pdb").write_text("ATOM\n", encoding="utf-8")
    energy_dir = rosetta_root / "interface_energy_A_B"
    energy_dir.mkdir()
    (energy_dir / "residue_energy.csv").write_text(
        'pdbname,binder_energy\ndesign_1,"{10: -6.0, 11: -4.0}"\n',
        encoding="utf-8",
    )
    artifact = _upstream_structure_artifact(
        metadata={"structure_patterns": ("outputs/*.pdb",)}
    )
    artifact.storage.path = "rosetta"

    result = ppiflow_workflow.derive_ppiflow_fixed_positions.get_raw_f()(
        artifacts=[artifact],
        config={"gentype": "binder", "energy_threshold": -5},
        run_id="run-1",
        node_id="fixed",
        step_name="FixedPositions",
    )

    assert result.outputs[0].metadata["fixed_positions"] == "A10"
    assert result.outputs[0].metadata["fixed_positions_by_structure"] == {
        "design": "A10"
    }


def test_rank_transform_uses_dockq_scores(tmp_path: Path, monkeypatch) -> None:
    source_root, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    structures_dir = source_root / "structures"
    structures_dir.mkdir()
    (structures_dir / "design-1.pdb").write_text("ATOM 1\n", encoding="utf-8")
    (structures_dir / "design-2.pdb").write_text("ATOM 2\n", encoding="utf-8")
    (source_root / "dockq.csv").write_text(
        "reference,dockq\ndesign-1.pdb,0.8\ndesign-2.pdb,0.4\n",
        encoding="utf-8",
    )
    score_artifact = WorkflowArtifact(
        artifact_id="dockq",
        producing_node_id="dockq",
        kind=ArtifactKind.SCORES,
        storage=VolumePath(volume_name="source-volume", path="dockq.csv"),
    )
    structure_artifact = _upstream_structure_artifact()
    structure_artifact.storage.path = "structures"

    result = ppiflow_workflow.rank_ppiflow_artifacts.get_raw_f()(
        structures=[structure_artifact],
        score_artifacts=[score_artifact],
        config={"gentype": "binder", "dockq_threshold": 0.49},
        run_id="run-1",
        node_id="rank",
        step_name="RankStep",
    )

    output_dir = workflow_root / result.outputs[0].storage.path
    assert [path.name for path in output_dir.iterdir()] == [
        "upstream-structures__design-1.pdb"
    ]
    assert result.outputs[1].metadata["rows"] == 1


def test_rank_transform_allows_empty_ranked_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    source_root, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    structures_dir = source_root / "structures"
    structures_dir.mkdir()
    (structures_dir / "design-1.pdb").write_text("ATOM 1\n", encoding="utf-8")
    (source_root / "dockq.csv").write_text(
        "reference,dockq\ndesign-1.pdb,0.1\n",
        encoding="utf-8",
    )
    score_artifact = WorkflowArtifact(
        artifact_id="dockq",
        producing_node_id="dockq",
        kind=ArtifactKind.SCORES,
        storage=VolumePath(volume_name="source-volume", path="dockq.csv"),
    )
    structure_artifact = _upstream_structure_artifact()
    structure_artifact.storage.path = "structures"

    result = ppiflow_workflow.rank_ppiflow_artifacts.get_raw_f()(
        structures=[structure_artifact],
        score_artifacts=[score_artifact],
        config={"gentype": "binder", "dockq_threshold": 0.49},
        run_id="run-1",
        node_id="rank",
        step_name="RankStep",
    )

    structures_output = workflow_root / result.outputs[0].storage.path
    ranked_csv = workflow_root / result.outputs[1].storage.path
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[1].metadata["rows"] == 0
    assert result.warnings == [
        "RankStep found no structures with usable ranking metrics"
    ]
    assert list(structures_output.iterdir()) == []
    assert ranked_csv.read_text(encoding="utf-8") == (
        "design,filename,rank_score,dockq,iptm,interface_score\n"
    )


def test_report_node_renders_candidate_attrition(tmp_path: Path, monkeypatch) -> None:
    _, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    report_dir = workflow_root / "report-inputs"
    report_dir.mkdir()
    ranked_csv = report_dir / "ranked_designs.csv"
    ranked_csv.write_text("design,rank_score\ndesign-1,1.0\n", encoding="utf-8")
    audit_csv = report_dir / "filter_audit.csv"
    audit_csv.write_text(
        "candidate_id,stage_name,passed,reason\n"
        "design-1,FilterStep_stage2,true,passed\n"
        "design-2,FilterStep_stage2,false,filtered\n",
        encoding="utf-8",
    )
    manifest_path = report_dir / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(
        [
            ppiflow_manifests.candidate_manifest_row(
                candidate_id="design-1",
                stage_name="FilterStep_stage2",
                stage_role="filter",
                operation_mode="retained",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                files=[],
            ),
            ppiflow_manifests.candidate_manifest_row(
                candidate_id="design-2",
                stage_name="FilterStep_stage2",
                stage_role="filter",
                operation_mode="rejected",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                files=[],
            ),
        ],
        manifest_path,
    )
    node = ppiflow_workflow.ReportNode("ReportStep")

    result = node.run(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-report",
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={
                "rank": [
                    WorkflowArtifact(
                        artifact_id="rank",
                        producing_node_id="rank",
                        kind=ArtifactKind.TABLE,
                        storage=VolumePath(
                            volume_name="workflow-volume",
                            path="report-inputs/ranked_designs.csv",
                        ),
                    )
                ],
                "filter_audit": [
                    WorkflowArtifact(
                        artifact_id="audit",
                        producing_node_id="filter",
                        kind=ArtifactKind.TABLE,
                        storage=VolumePath(
                            volume_name="workflow-volume",
                            path="report-inputs/filter_audit.csv",
                        ),
                    )
                ],
                "candidate_manifest": [
                    WorkflowArtifact(
                        artifact_id="manifest",
                        producing_node_id="filter",
                        kind=ArtifactKind.TABLE,
                        storage=VolumePath(
                            volume_name="workflow-volume",
                            path="report-inputs/candidate_manifest.parquet",
                            media_type=ppiflow_manifests.MANIFEST_MEDIA_TYPE,
                        ),
                    )
                ],
            },
        )
    )

    markdown = result.outputs[0].storage.data.decode("utf-8")
    assert "## Candidate Attrition" in markdown
    assert "FilterStep_stage2" in markdown
    assert "## Ranked Designs" in markdown


def test_submit_ppiflow_workflow_dry_run_prints_dag_without_orchestrator(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    task_yaml = tmp_path / "task.yaml"
    steps_yaml = tmp_path / "steps.yaml"
    input_pdb = tmp_path / "demo.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    task_yaml.write_bytes(_task_yaml(enabled_steps="  PPIFlowStep: true\n"))
    steps_yaml.write_text(
        f"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: {input_pdb}
    binder_chain: B
""",
        encoding="utf-8",
    )

    class UnexpectedExecutionCoordinator:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("dry-run should not construct the orchestrator")

    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "ExecutionCoordinator",
        UnexpectedExecutionCoordinator,
    )

    raw_f = ppiflow_workflow.submit_ppiflow_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        task_yaml=str(task_yaml),
        steps_yaml=str(steps_yaml),
        run_id="demo",
        dry_run=True,
    )

    stdout = strip_ansi(capsys.readouterr().out)
    assert "[workflow] DAG graph: node_id [execution; class] <- dependency" in stdout
    assert (
        "[workflow]   stage1-ppiflow-design [provider; PPIFlowDesignNode] <- -"
        in stdout
    )
    assert "ppiflow_workflow.PPIFlowDesignNode" not in stdout
    assert "Submitting PPIFlow workflow" not in stdout


def test_submit_ppiflow_workflow_creates_new_execution_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_yaml = tmp_path / "task.yaml"
    steps_yaml = tmp_path / "steps.yaml"
    task_yaml.write_bytes(_task_yaml(enabled_steps="  PPIFlowStep: true\n"))
    steps_yaml.write_text(
        f"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: {ppiflow_app.CONF.output_volume_mountpoint}/inputs/demo.pdb
    binder_chain: B
""",
        encoding="utf-8",
    )
    calls = {}

    def fake_stage_ppiflow_app_inputs(**kwargs):
        calls["staging"] = kwargs
        return kwargs["steps_doc"]

    class FakeOrchestratorMethod:
        def spawn(self, **kwargs):
            calls["spawn"] = kwargs
            return _FakeFunctionCall(
                "call-1",
                AppRunResult(status=AppRunStatus.SUCCEEDED),
            )

    class FakeExecutionCoordinator:
        def __init__(self, **kwargs) -> None:
            calls["coordinator"] = kwargs
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        ppiflow_workflow,
        "_stage_ppiflow_app_inputs",
        fake_stage_ppiflow_app_inputs,
    )
    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "ExecutionCoordinator",
        FakeExecutionCoordinator,
    )

    raw_f = ppiflow_workflow.submit_ppiflow_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        task_yaml=str(task_yaml),
        steps_yaml=str(steps_yaml),
        run_id="demo",
        wait=True,
    )

    assert "force" not in calls["staging"]
    UUID(str(calls["coordinator"]["execution_run_id"]))
    assert calls["spawn"]["workload_run_key"] == "demo"
    assert calls["coordinator"]["deployment_environment"] == "development"
    assert calls["coordinator"]["deployment_name"] == ppiflow_workflow.CONF.name
    assert calls["coordinator"]["deployment_version"] == 1
    assert calls["spawn"]["max_active_provider_calls"] == 4
    assert calls["spawn"]["max_active_gpu_provider_calls"] == 4
    assert "force" not in calls["spawn"]


def test_ppiflow_full_binder_chain_uses_specific_node_classes() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(
            enabled_steps="""  PPIFlowStep: true
  MPNNStep_stage1: true
  FlowpackerStep_stage1: true
  AF3scoreStep_stage1: true
  FilterStep_stage1: true
  RosettaFixStep: true
  PartialStep: true
  MPNNStep_stage2: true
  FlowpackerStep_stage2: true
  AF3scoreStep_stage2: true
  FilterStep_stage2: true
  ReFoldStep: true
  DockQStep: true
  RosettaRelaxStep: true
  RankStep: true
  ReportStep: true
"""
        ),
        steps_yaml_bytes=b"{}\n",
    )

    definition = workflow.validate()

    assert list(definition.nodes) == [
        "stage1-ppiflow-design",
        "stage1-ligandmpnn",
        "stage1-flowpacker",
        "stage1-af3score",
        "stage1-filter",
        "stage2-rosetta-fix",
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
        "stage2-ligandmpnn",
        "stage2-flowpacker",
        "stage2-af3score",
        "stage2-filter",
        "stage2-alphafold3-refold",
        "stage2-dockq",
        "stage2-rosetta-relax",
        "stage2-rank",
        "stage2-report",
    ]
    assert [
        type(definition.nodes[node_id].node).__name__ for node_id in definition.nodes
    ] == [
        "PPIFlowDesignNode",
        "LigandMPNNNode",
        "FlowPackerNode",
        "AF3ScoreNode",
        "FilterStructuresNode",
        "RosettaFixNode",
        "FixedPositionsNode",
        "PPIFlowPartialNode",
        "LigandMPNNNode",
        "FlowPackerNode",
        "AF3ScoreNode",
        "FilterStructuresNode",
        "ReFoldNode",
        "DockQNode",
        "RosettaRelaxNode",
        "RankNode",
        "ReportNode",
    ]
    assert definition.dependencies["stage2-fixed-positions"] == {"stage2-rosetta-fix"}
    assert definition.dependencies["stage2-partial-ppiflow"] == {
        "stage2-fixed-positions"
    }
    assert definition.dependencies["stage2-dockq"] == {
        "stage2-filter",
        "stage2-alphafold3-refold",
    }
    assert (
        definition.nodes["stage2-alphafold3-refold"].aggregation_policy
        == ppiflow_workflow.NodeAggregationPolicy.ALLOW_PARTIAL
    )
    assert definition.nodes["stage2-dockq"].partial_dependencies == {
        "stage2-alphafold3-refold"
    }
    assert definition.dependencies["stage2-rosetta-relax"] == {
        "stage2-filter",
        "stage2-dockq",
    }
    assert definition.dependencies["stage2-rank"] == {
        "stage2-af3score",
        "stage2-alphafold3-refold",
        "stage2-rosetta-relax",
        "stage2-dockq",
    }
    assert definition.nodes["stage2-rank"].partial_dependencies == {
        "stage2-alphafold3-refold"
    }
    assert (
        definition.nodes["stage2-rank"].inputs["refold_metrics"].kind
        == ArtifactKind.TABLE
    )
    assert (
        definition.nodes["stage2-rank"].inputs["candidate_manifest"].role
        == ppiflow_manifests.MANIFEST_FILE_ROLE
    )
    assert (
        definition.nodes["stage2-report"].inputs["filter_tables"].kind
        == ArtifactKind.TABLE
    )
    assert definition.dependencies["stage2-report"] == {
        "stage2-rank",
        "stage1-ligandmpnn",
        "stage1-filter",
        "stage2-ligandmpnn",
        "stage2-filter",
        "stage2-alphafold3-refold",
    }
    restored = pickle.loads(pickle.dumps(workflow))  # noqa: S301
    assert restored.validate() == definition


def test_ppiflow_candidate_manifest_edges_select_manifest_role() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(
            enabled_steps="""  PPIFlowStep: true
  MPNNStep_stage1: true
  AF3scoreStep_stage1: true
  FilterStep_stage1: true
  RosettaFixStep: true
  PartialStep: true
  MPNNStep_stage2: true
  AF3scoreStep_stage2: true
  FilterStep_stage2: true
  ReFoldStep: true
  DockQStep: true
  RosettaRelaxStep: true
  RankStep: true
"""
        ),
        steps_yaml_bytes=b"{}\n",
    )

    definition = workflow.validate()

    expected_manifest_sources = {
        "stage1-ligandmpnn": "stage1-ppiflow-design",
        "stage1-af3score": "stage1-ligandmpnn",
        "stage1-filter": "stage1-ligandmpnn",
        "stage2-rosetta-fix": "stage1-filter",
        "stage2-fixed-positions": "stage2-rosetta-fix",
        "stage2-partial-ppiflow": "stage2-fixed-positions",
        "stage2-ligandmpnn": "stage2-partial-ppiflow",
        "stage2-af3score": "stage2-ligandmpnn",
        "stage2-filter": "stage2-ligandmpnn",
        "stage2-alphafold3-refold": "stage2-filter",
        "stage2-dockq": "stage2-filter",
        "stage2-rosetta-relax": "stage2-filter",
        "stage2-rank": "stage2-rosetta-relax",
    }
    for node_id, source_node_id in expected_manifest_sources.items():
        selector = definition.nodes[node_id].inputs["candidate_manifest"]
        assert selector.producing_node_id == source_node_id
        assert selector.kind == ArtifactKind.TABLE
        assert selector.role == ppiflow_manifests.MANIFEST_FILE_ROLE


def test_ppiflow_stage2_scientific_nodes_consume_only_retained_manifests() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(
            enabled_steps="""  PPIFlowStep: true
  AF3scoreStep_stage1: true
  FilterStep_stage1: true
  RosettaFixStep: true
  PartialStep: true
  MPNNStep_stage2: true
  AF3scoreStep_stage2: true
  FilterStep_stage2: true
  ReFoldStep: true
  DockQStep: true
  RosettaRelaxStep: true
  RankStep: true
"""
        ),
        steps_yaml_bytes=b"{}\n",
    )

    definition = workflow.validate()

    stage1_retained_path = [
        "stage2-rosetta-fix",
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
        "stage2-ligandmpnn",
        "stage2-af3score",
        "stage2-filter",
    ]
    for node_id in stage1_retained_path:
        assert "stage1-filter" in _manifest_ancestor_chain(definition, node_id)

    stage2_retained_path = [
        "stage2-alphafold3-refold",
        "stage2-dockq",
        "stage2-rosetta-relax",
        "stage2-rank",
    ]
    for node_id in stage2_retained_path:
        assert "stage2-filter" in _manifest_ancestor_chain(definition, node_id)


def test_ppiflow_candidate_concurrency_is_copied_to_node_configs() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=b"""
task:
  gentype: binder
  candidate_concurrency: 3
steps:
  MPNNStep_stage1: true
  AF3scoreStep_stage1: true
""",
        steps_yaml_bytes=b"""
MPNNStep_stage1:
  candidate_concurrency: 2
AF3scoreStep_stage1: {}
""",
    )

    definition = workflow.validate()

    assert (
        definition.nodes["stage1-ligandmpnn"].node.config["candidate_concurrency"] == 2
    )
    assert definition.nodes["stage1-af3score"].node.config["candidate_concurrency"] == 3


def test_ppiflow_max_child_calls_caps_stage_fanout_configs() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=b"""
task:
  gentype: binder
  candidate_concurrency: 5
steps:
  MPNNStep_stage1: true
  AF3scoreStep_stage1: true
  RosettaFixStep: true
""",
        steps_yaml_bytes=b"""
MPNNStep_stage1:
  candidate_concurrency: 4
AF3scoreStep_stage1:
  num_jobs: 6
RosettaFixStep:
  max_num_pods: 7
""",
        max_child_calls=2,
    )

    definition = workflow.validate()

    assert (
        definition.nodes["stage1-ligandmpnn"].node.config["candidate_concurrency"] == 2
    )
    assert definition.nodes["stage1-af3score"].node.config["num_jobs"] == 2
    assert definition.nodes["stage1-af3score"].node.config["max_child_calls"] == 2
    assert definition.nodes["stage2-rosetta-fix"].node.config["max_num_pods"] == 2


def test_ppiflow_stage2_only_requires_existing_input() -> None:
    try:
        build_ppiflow_workflow(
            task_yaml_bytes=_task_yaml(enabled_steps="  RosettaFixStep: true\n"),
            steps_yaml_bytes=b"{}\n",
            stage=2,
        )
    except ValueError as exc:
        assert "stage=2 PPIFlow runs require a Stage2Input" in str(exc)
    else:
        raise AssertionError("stage=2 without existing structures should fail")


def test_ppiflow_stage2_only_uses_existing_input_node() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  RosettaFixStep: true\n"),
        steps_yaml_bytes=b"""
Stage2Input:
  volume_name: source-volume
  path: existing/stage1-filtered
RosettaFixStep: {}
""",
        stage=2,
    )

    definition = workflow.validate()

    assert list(definition.nodes) == [
        "stage2-existing-input",
        "stage2-rosetta-fix",
    ]
    assert isinstance(
        definition.nodes["stage2-existing-input"].node,
        ppiflow_workflow.ExistingStructuresNode,
    )
    assert definition.dependencies["stage2-rosetta-fix"] == {"stage2-existing-input"}
    assert (
        definition.nodes["stage2-rosetta-fix"].inputs["candidate_manifest"].kind
        == ArtifactKind.TABLE
    )
    assert (
        definition.nodes["stage2-rosetta-fix"].inputs["candidate_manifest"].role
        == ppiflow_manifests.MANIFEST_FILE_ROLE
    )
    assert isinstance(
        definition.nodes["stage2-existing-input"].node,
        RemoteWorkflowNode,
    )


def test_ppiflow_stage2_only_manifest_feeds_downstream_nodes() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(
            enabled_steps="""  RosettaFixStep: true
  PartialStep: true
  MPNNStep_stage2: true
  AF3scoreStep_stage2: true
  FilterStep_stage2: true
  ReFoldStep: true
  DockQStep: true
  RosettaRelaxStep: true
  RankStep: true
"""
        ),
        steps_yaml_bytes=b"""
Stage2Input:
  volume_name: source-volume
  path: existing/stage1-filtered
  manifest_path: manifests/candidate_manifest.parquet
RosettaFixStep: {}
PartialStep: {}
MPNNStep_stage2: {}
AF3scoreStep_stage2: {}
FilterStep_stage2: {}
ReFoldStep: {}
DockQStep: {}
RosettaRelaxStep: {}
RankStep: {}
""",
        stage=2,
    )

    definition = workflow.validate()

    assert definition.nodes["stage2-existing-input"].node.config["manifest_path"] == (
        "manifests/candidate_manifest.parquet"
    )
    for node_id in [
        "stage2-rosetta-fix",
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
        "stage2-ligandmpnn",
    ]:
        assert "stage2-existing-input" in _manifest_ancestor_chain(
            definition,
            node_id,
        )
    assert (
        definition
        .nodes["stage2-alphafold3-refold"]
        .inputs["candidate_manifest"]
        .producing_node_id
        == "stage2-filter"
    )


def test_stage2_input_node_returns_structures_and_manifest(tmp_path: Path) -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="stage2_input_structures",
                kind=ArtifactKind.STRUCTURES,
                storage=VolumePath(volume_name="source-volume", path="existing"),
            ),
            AppOutput(
                name="candidate_manifest",
                kind=ArtifactKind.TABLE,
                storage=VolumePath(
                    volume_name="workflow-volume",
                    path="ppiflow/run/node/result/stage2_input/candidate_manifest.parquet",
                    media_type=ppiflow_manifests.MANIFEST_MEDIA_TYPE,
                ),
            ),
        ],
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  RosettaFixStep: true\n"),
        steps_yaml_bytes=b"""
Stage2Input:
  volume_name: source-volume
  path: existing
RosettaFixStep: {}
""",
        stage=2,
    )
    spec = workflow.validate().nodes["stage2-existing-input"]

    call = spec.node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id=spec.node_id,
            task_key="node",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={},
        )
    )
    node_result = spec.node.process_remote_result(result, call.metadata)

    assert node_result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert node_result.outputs[1].kind == ArtifactKind.TABLE
    assert call.kwargs["storage"] == VolumePath(
        volume_name="source-volume",
        path="existing",
    )
    assert call.kwargs["step_name"] == "Stage2Input"


def test_stage2_input_generated_manifest_does_not_affect_dag_hash() -> None:
    task_yaml = _task_yaml(enabled_steps="  RosettaFixStep: true\n")
    steps_yaml = b"""
Stage2Input:
  volume_name: source-volume
  path: existing
RosettaFixStep: {}
"""

    first_hash = hashing.dag_hash(
        build_ppiflow_workflow(
            task_yaml_bytes=task_yaml,
            steps_yaml_bytes=steps_yaml,
            stage=2,
        ).validate()
    )
    repeated_hash = hashing.dag_hash(
        build_ppiflow_workflow(
            task_yaml_bytes=task_yaml,
            steps_yaml_bytes=steps_yaml,
            stage=2,
        ).validate()
    )
    changed_hash = hashing.dag_hash(
        build_ppiflow_workflow(
            task_yaml_bytes=task_yaml,
            steps_yaml_bytes=b"""
Stage2Input:
  volume_name: source-volume
  path: other-existing
RosettaFixStep: {}
""",
            stage=2,
        ).validate()
    )

    assert first_hash == repeated_hash
    assert first_hash != changed_hash


def test_stage2_input_normalization_scans_path_and_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    existing_dir = source_root / "existing"
    existing_dir.mkdir()
    (existing_dir / "design-a.pdb").write_text("ATOM A\n", encoding="utf-8")
    (existing_dir / "design-b.pdb").write_text("ATOM B\n", encoding="utf-8")

    result = ppiflow_workflow.normalize_ppiflow_stage2_input.get_raw_f()(
        storage=VolumePath(volume_name="source-volume", path="existing"),
        config={"run_name": "stage2-run", "structure_patterns": "*.pdb"},
        run_id="run-1",
        node_id="stage2-existing-input",
        step_name="Stage2Input",
    )

    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].metadata["structure_count"] == 2
    assert result.outputs[1].kind == ArtifactKind.TABLE
    assert result.outputs[1].storage.media_type == ppiflow_manifests.MANIFEST_MEDIA_TYPE
    manifest_path = workflow_root / result.outputs[1].storage.path
    frame = ppiflow_manifests.read_manifest(manifest_path)
    assert frame.get_column("candidate_id").to_list() == [
        "stage2_input_000001",
        "stage2_input_000002",
    ]
    assert frame.get_column("source_path").to_list() == [
        "existing/design-a.pdb",
        "existing/design-b.pdb",
    ]


def test_stage2_input_normalization_accepts_explicit_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    existing_dir = source_root / "existing"
    existing_dir.mkdir()
    (existing_dir / "design-a.pdb").write_text("ATOM A\n", encoding="utf-8")
    explicit_manifest = workflow_root / "provided" / "candidate_manifest.parquet"
    ppiflow_manifests.write_manifest(
        [
            ppiflow_manifests.candidate_manifest_row(
                candidate_id="provided-candidate",
                stage_name="Stage2Input",
                stage_role="stage2_input",
                operation_mode="provided_manifest",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                source_artifact_id="provided",
                source_path="existing/design-a.pdb",
                derived_path="existing/design-a.pdb",
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="structure",
                        volume_name="source-volume",
                        app_volume_path="existing/design-a.pdb",
                    )
                ],
            )
        ],
        explicit_manifest,
    )

    result = ppiflow_workflow.normalize_ppiflow_stage2_input.get_raw_f()(
        storage=VolumePath(volume_name="source-volume", path="existing"),
        config={
            "manifest_volume_name": "workflow-volume",
            "manifest_path": "provided/candidate_manifest.parquet",
        },
        run_id="run-1",
        node_id="stage2-existing-input",
        step_name="Stage2Input",
    )

    frame = ppiflow_manifests.read_manifest(
        workflow_root / result.outputs[1].storage.path
    )
    assert frame.get_column("candidate_id").to_list() == ["provided-candidate"]


def test_structure_consuming_steps_fail_clearly_without_inputs(
    tmp_path: Path,
) -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  FlowpackerStep_stage1: true\n"),
        steps_yaml_bytes=b"FlowpackerStep_stage1: {}\n",
    )

    spec = workflow.validate().nodes["stage1-flowpacker"]
    try:
        spec.node.prepare_remote(
            NodeRunContext(
                execution_run_id=RUN_ID,
                workload_run_key="run-1",
                node_id=spec.node_id,
                task_key="node",
                work_dir=tmp_path / "result",
                cache_dir=tmp_path / "cache",
                inputs={},
            )
        )
    except ValueError as exc:
        assert "requires structure inputs" in str(exc)
    else:
        raise AssertionError("missing PPIFlow inputs should fail clearly")


def test_submit_ppiflow_workflow_can_enable_strict_external_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_yaml = tmp_path / "task.yaml"
    steps_yaml = tmp_path / "steps.yaml"
    task_yaml.write_bytes(_task_yaml(enabled_steps="  PPIFlowStep: true\n"))
    steps_yaml.write_text(
        f"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: {ppiflow_app.CONF.output_volume_mountpoint}/inputs/demo.pdb
    binder_chain: B
""",
        encoding="utf-8",
    )
    calls = {}

    class FakeOrchestratorMethod:
        def spawn(self, **kwargs):
            calls["spawn"] = kwargs
            return _FakeFunctionCall(
                "call-1",
                AppRunResult(status=AppRunStatus.SUCCEEDED),
            )

    class FakeExecutionCoordinator:
        def __init__(self, **kwargs) -> None:
            calls["coordinator"] = kwargs
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "ExecutionCoordinator",
        FakeExecutionCoordinator,
    )

    raw_f = ppiflow_workflow.submit_ppiflow_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        task_yaml=str(task_yaml),
        steps_yaml=str(steps_yaml),
        run_id="demo",
        wait=True,
        strict_artifact_checks=True,
    )

    assert calls["spawn"]["strict_external_artifact_checks"] is True
    assert (
        calls["spawn"]["external_artifact_checker_function_name"]
        == "check_ppiflow_external_artifact"
    )
