"""Tests for the PPIFlow workflow definition."""

# ruff: noqa: D103

import hashlib
import pickle
import tarfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import orjson
import polars as pl
import pytest
import yaml
import zstandard as zstd
from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.design import ligandmpnn_app, ppiflow_app
from biomodals.app.fold.alphafold3 import (
    inference_inputs,
    modal_adapters,
    request_results,
)
from biomodals.execution import PullTaskClaim, WorkerAssignmentRecord
from biomodals.helper import shell as shell_helper
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
    RemotePullTaskWorkflowNode,
    RemoteWorkflowNode,
    hashing,
)
from biomodals.workflow.core.execution import execution_plan
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

    def get_raw_f(self):
        return self.remote


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

    assert "ALPHAFOLD3_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_refold_candidate",
    )
    assert "PPI_FLOW_SOURCE_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_dockq_stage",
    )
    assert "PPI_FLOW_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_partial_candidate",
    )
    assert "PPI_FLOW_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_design_stage",
    )
    assert "LIGANDMPNN_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_ligandmpnn_candidate",
    )
    assert "FLOWPACKER_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_flowpacker_stage",
    )
    assert "AF3SCORE_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "prepare_ppiflow_af3score_stage",
    )
    assert "AF3SCORE_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_af3score_batch",
    )
    assert "AF3SCORE_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "postprocess_ppiflow_af3score_stage",
    )
    assert "PPI_FLOW_SOURCE_VOLUME_MOUNTS" in _decorator_block(
        source,
        "prepare_ppiflow_rosetta_stage",
    )
    assert "ROSETTA_TASK_VOLUME_MOUNTS" in _decorator_block(
        source,
        "run_ppiflow_rosetta_worker",
    )
    assert "PPI_FLOW_SOURCE_VOLUME_MOUNTS" in _decorator_block(
        source,
        "finalize_ppiflow_rosetta_stage",
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

    assert submission.function_name == "run_ppiflow_design_stage"
    assert submission.kwargs["run_name"] == "demo-run"
    assert submission.kwargs["run_id"] == "run-1"
    assert submission.kwargs["node_id"] == "stage1-ppiflow-design"
    assert isinstance(submission.kwargs["args"], ppiflow_app.PPIFlowArgs)
    assert isinstance(submission.kwargs["args"].args.input_pdb, str)
    assert isinstance(submission.kwargs["args"].args.config, str)


def test_ligandmpnn_step_prepares_selected_structures_for_kernel(
    tmp_path: Path,
) -> None:
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
        task_key="candidate-1",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={"structures": [_upstream_structure_artifact()]},
    )

    submission = node.prepare_remote_task(
        context,
        ppiflow_workflow.RemoteWorkflowTask(
            task_key="candidate-1",
            scientific_payload={"candidate_id": "candidate-1"},
            execution_payload={"candidate_id": "candidate-1"},
        ),
    )

    assert submission.function_name == "run_ppiflow_ligandmpnn_candidate"
    assert submission.uses_gpu is True
    assert submission.kwargs["candidate_id"] == "candidate-1"
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert submission.kwargs["run_name"] == "mpnn-run"
    assert submission.kwargs["script_mode"] == "run"
    assert submission.kwargs["cli_args"]["--model_type"] == "protein_mpnn"
    assert submission.kwargs["cli_args"]["--batch_size"] == "2"
    assert submission.kwargs["cli_args"]["--number_of_batches"] == "3"


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

    submission = node.prepare_remote_task(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage1-ligandmpnn",
            task_key="candidate-1",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={"structures": artifacts},
        ),
        ppiflow_workflow.RemoteWorkflowTask(
            task_key="candidate-1",
            scientific_payload={"candidate_id": "candidate-1"},
            execution_payload={"candidate_id": "candidate-1"},
        ),
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


def test_flowpacker_stage_executes_batch_in_tracked_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flowpacker = _FakeModalFunction(
        "unused",
        AppRunResult(status=AppRunStatus.SUCCEEDED),
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "_reload_ppiflow_source_volumes",
        lambda: None,
    )
    monkeypatch.setattr(
        ppiflow_workflow.ppiflow_staging,
        "select_structure_files_from_artifacts",
        lambda *args, **kwargs: [("candidate.pdb", b"ATOM\n")],
    )
    monkeypatch.setattr(
        ppiflow_workflow.flowpacker_app,
        "run_flowpacker_workflow",
        flowpacker,
    )

    ppiflow_workflow.run_ppiflow_flowpacker_stage.get_raw_f()(
        artifacts=[_upstream_structure_artifact()],
        config={"seed": 7},
        run_name="flowpacker-run",
    )

    assert flowpacker.kwargs["input_files"] == [("candidate.pdb", b"ATOM\n")]
    assert flowpacker.kwargs["run_name"] == "flowpacker-run"
    assert flowpacker.kwargs["seed"] == 7


def test_partial_step_prepares_one_kernel_candidate_task(
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

    submission = node.prepare_remote_task(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-partial-ppiflow",
            task_key="candidate-1",
            work_dir=tmp_path / "result",
            cache_dir=tmp_path / "cache",
            inputs={"structures": [_upstream_structure_artifact()]},
        ),
        ppiflow_workflow.RemoteWorkflowTask(
            task_key="candidate-1",
            scientific_payload={"candidate_id": "candidate-1"},
            execution_payload={"candidate_id": "candidate-1"},
        ),
    )

    assert submission.function_name == "run_ppiflow_partial_candidate"
    assert submission.uses_gpu is True
    assert submission.kwargs["candidate_id"] == "candidate-1"
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]


def test_design_stage_publishes_digest_bearing_candidate_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    output_root = tmp_path / "ppiflow-output"
    output_root.mkdir()
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_OUTPUT_MOUNTPOINT",
        str(output_root),
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_SOURCE_VOLUME_ROOTS",
        {
            **ppiflow_workflow.PPI_FLOW_SOURCE_VOLUME_ROOTS,
            ppiflow_app.CONF.output_volume_name: str(output_root),
        },
    )

    def run_raw(*, args, run_name):
        del args
        outputs_dir = output_root / run_name / "outputs"
        outputs_dir.mkdir(parents=True)
        (outputs_dir / "design-b.pdb").write_bytes(b"MODEL B\n")
        (outputs_dir / "design-a.cif").write_bytes(b"data_A\n")
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ppiflow_outputs",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=ppiflow_app.CONF.output_volume_name,
                        path=run_name,
                    ),
                )
            ],
        )

    monkeypatch.setattr(
        ppiflow_app,
        "ppiflow_run_workflow",
        SimpleNamespace(get_raw_f=lambda: run_raw),
    )
    result = ppiflow_workflow.run_ppiflow_design_stage.get_raw_f()(
        args=ppiflow_app.PPIFlowArgs.model_validate({
            "args": {
                "name": "design",
                "specified_hotspots": "A1",
                "input_pdb": "/input.pdb",
                "binder_chain": "B",
            }
        }),
        run_name="design-run",
        run_id="run-1",
        node_id="design",
        step_name="PPIFlowStep",
    )

    assert [output.kind for output in result.outputs] == [
        ArtifactKind.STRUCTURES,
        ArtifactKind.TABLE,
    ]
    manifest = ppiflow_manifests.read_manifest(
        workflow_root / result.outputs[1].storage.path
    )
    assert manifest.height == 2
    assert manifest.get_column("candidate_id").n_unique() == 2
    assert all(
        file_record["content_sha256"]
        for row in manifest.iter_rows(named=True)
        for file_record in row["files"]
    )


def test_ligandmpnn_candidate_runs_science_in_kernel_owned_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    structure_dir = source_root / "upstream" / "results"
    structure_dir.mkdir(parents=True)
    structure_bytes = b"ATOM\n"
    (structure_dir / "candidate.pdb").write_bytes(structure_bytes)
    manifest_path = workflow_root / "candidate_manifest.parquet"
    ppiflow_manifests.write_manifest(
        [
            ppiflow_manifests.candidate_manifest_row(
                candidate_id="candidate-1",
                stage_name="source",
                stage_role="source",
                operation_mode="source",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="structure",
                        volume_name="source-volume",
                        app_volume_path="upstream/results/candidate.pdb",
                        path="candidate.pdb",
                        content_sha256=hashlib.sha256(structure_bytes).hexdigest(),
                    )
                ],
            )
        ],
        manifest_path,
    )
    calls = []
    archive = _tar_zst_bytes({
        "outputs/backbones/designed.pdb": b"MODEL\n",
        "outputs/seqs/designed.fa": b">designed\nACD\n",
    })

    def run_raw(**kwargs):
        calls.append(kwargs)
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="LigandMPNN_outputs",
                    kind=ArtifactKind.ARCHIVE,
                    storage=InlineBytes(
                        data=archive,
                        filename="ligandmpnn.tar.zst",
                        media_type="application/zstd",
                    ),
                )
            ],
        )

    monkeypatch.setattr(
        ligandmpnn_app,
        "ligandmpnn_run",
        SimpleNamespace(get_raw_f=lambda: run_raw),
    )
    result = ppiflow_workflow.run_ppiflow_ligandmpnn_candidate.get_raw_f()(
        artifacts=[_upstream_structure_artifact()],
        candidate_manifests=[
            WorkflowArtifact(
                artifact_id="candidate-manifest",
                producing_node_id="source",
                kind=ArtifactKind.TABLE,
                storage=VolumePath(
                    volume_name="workflow-volume",
                    path=manifest_path.relative_to(workflow_root).as_posix(),
                ),
            )
        ],
        candidate_id="candidate-1",
        config={"seeds": [1]},
        step_name="MPNNStep_stage1",
        run_name="mpnn-run",
        script_mode="run",
        cli_args={"--model_type": "protein_mpnn"},
    )

    assert len(calls) == 1
    assert calls[0]["run_name"] == "mpnn-run-candidate-1"
    assert calls[0]["struct_bytes"] == structure_bytes
    assert [output.kind for output in result.outputs] == [
        ArtifactKind.STRUCTURES,
        ArtifactKind.TABLE,
    ]
    assert result.outputs[1].name == "mpnn_seqs"
    assert b"ACD" in result.outputs[1].storage.data
    [file_record] = result.outputs[0].metadata["candidate_files"]
    assert file_record["content_sha256"] == hashlib.sha256(archive).hexdigest()


def test_partial_candidate_runs_science_in_kernel_owned_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root, workflow_root = _local_transform_environment(monkeypatch, tmp_path)
    structure_dir = source_root / "upstream" / "results"
    structure_dir.mkdir(parents=True)
    structure_bytes = b"ATOM\n"
    (structure_dir / "candidate.pdb").write_bytes(structure_bytes)
    manifest_path = workflow_root / "candidate_manifest.parquet"
    ppiflow_manifests.write_manifest(
        [
            ppiflow_manifests.candidate_manifest_row(
                candidate_id="candidate-1",
                stage_name="source",
                stage_role="source",
                operation_mode="source",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="structure",
                        volume_name="source-volume",
                        app_volume_path="upstream/results/candidate.pdb",
                        path="candidate.pdb",
                        content_sha256=hashlib.sha256(structure_bytes).hexdigest(),
                    )
                ],
            )
        ],
        manifest_path,
    )
    output_root = tmp_path / "ppiflow-output"
    output_root.mkdir()
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_OUTPUT_MOUNTPOINT",
        str(output_root),
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_OUTPUT_VOLUME",
        SimpleNamespace(commit=lambda: None),
    )

    calls = []

    def run_raw(*, args, run_name):
        calls.append((args, run_name))
        run_root = output_root / run_name
        structure = run_root / "outputs" / "designed.pdb"
        structure.parent.mkdir(parents=True)
        structure.write_bytes(b"MODEL\n")
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ppiflow_outputs",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=ppiflow_app.CONF.output_volume_name,
                        path=run_name,
                    ),
                )
            ],
        )

    monkeypatch.setattr(
        ppiflow_app,
        "ppiflow_run_workflow",
        SimpleNamespace(get_raw_f=lambda: run_raw),
    )
    result = ppiflow_workflow.run_ppiflow_partial_candidate.get_raw_f()(
        artifacts=[_upstream_structure_artifact()],
        candidate_manifests=[
            WorkflowArtifact(
                artifact_id="candidate-manifest",
                producing_node_id="source",
                kind=ArtifactKind.TABLE,
                storage=VolumePath(
                    volume_name="workflow-volume",
                    path=manifest_path.relative_to(workflow_root).as_posix(),
                ),
            )
        ],
        candidate_id="candidate-1",
        config={
            "args": {
                "name": "partial",
                "specified_hotspots": "A1",
                "input_pdb": "/placeholder.pdb",
                "fixed_positions": "B1",
            }
        },
        step_name="PartialStep",
        run_name="partial-run",
    )

    assert len(calls) == 1
    assert calls[0][1] == "partial-run-candidate-1"
    assert Path(calls[0][0].args.input_pdb).read_bytes() == structure_bytes
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    [file_record] = result.outputs[0].metadata["candidate_files"]
    assert file_record["app_volume_path"] == (
        "partial-run-candidate-1/outputs/designed.pdb"
    )
    assert file_record["content_sha256"] == hashlib.sha256(b"MODEL\n").hexdigest()


def test_af3score_step_runs_app_sequence_and_returns_metrics_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_node = ppiflow_workflow.AF3ScorePrepareNode(
        "AF3scoreStep_stage1",
        {"run_name": "af3-run", "num_jobs": 4, "prepare_workers": 2},
    )

    submission = prepare_node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage1-af3score-prepare",
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

    assert submission.function_name == "prepare_ppiflow_af3score_stage"
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert submission.kwargs["candidate_manifests"] == [
        _upstream_structure_artifact(kind=ArtifactKind.TABLE)
    ]
    assert submission.kwargs["execution_run_name"] == (
        f"ppiflow-af3score-{RUN_ID}-stage1-af3score-prepare"
    )
    assert submission.kwargs["config"]["num_jobs"] == 4
    assert submission.kwargs["config"]["prepare_workers"] == 2

    plan_path = tmp_path / "af3score_task_plan.json"
    batch_json_dir = tmp_path / "af3-batch-json"
    batch_json_dir.mkdir()
    for candidate_id in ("candidate-a", "candidate-b"):
        batch_json_dir.joinpath(f"{candidate_id}.json").write_text("{}")
    plan_path.write_text(
        orjson.dumps({
            "run_name": "af3-run",
            "input_files": ["candidate-a.pdb", "candidate-b.pdb"],
            "input_digests": {
                "candidate-a": "candidate-a",
                "candidate-b": "candidate-b",
                "unrelated": "unrelated",
            },
            "publication_key": "request-key",
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "input_name": f"{candidate_id}.pdb",
                    "scientific_payload": {
                        "candidate_id": candidate_id,
                        "content_sha256": candidate_id,
                    },
                    "chunk": {
                        "batch_name": "batch-0",
                        "batch_json_dir": str(batch_json_dir),
                        "batch_pdb_dir": "/af3/batch/pdb",
                        "task_count": 2,
                    },
                }
                for candidate_id in ("candidate-a", "candidate-b")
            ],
        }).decode(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "PPI_FLOW_SOURCE_VOLUME_ROOTS",
        {"workflow-volume": str(tmp_path)},
    )
    plan_artifact = WorkflowArtifact(
        artifact_id="af3score-plan",
        producing_node_id="stage1-af3score-prepare",
        kind=ArtifactKind.TABLE,
        storage=VolumePath(
            volume_name="workflow-volume",
            path=plan_path.name,
            media_type="application/json",
        ),
    )
    batch_node = ppiflow_workflow.AF3ScoreBatchNode(
        "AF3scoreStep_stage1",
        {"run_name": "af3-run"},
    )
    batch_context = NodeRunContext(
        execution_run_id=RUN_ID,
        workload_run_key="run-1",
        node_id="stage1-af3score-batches",
        task_key="candidate-a",
        work_dir=tmp_path / "batch-result",
        cache_dir=tmp_path / "batch-cache",
        inputs={"af3score_plan": [plan_artifact]},
    )
    tasks = batch_node.discover_remote_tasks(batch_context)
    batch_call = batch_node.prepare_remote_task_batch(batch_context, tasks)
    assert [task.task_key for task in tasks] == ["candidate-a", "candidate-b"]
    assert batch_call.function_name == "run_ppiflow_af3score_batch"
    assert batch_call.max_tasks_per_call == 2
    assert batch_call.kwargs["task_keys"] == ["candidate-a", "candidate-b"]
    assert batch_call.kwargs["input_digests"] == {
        "candidate-a": "candidate-a",
        "candidate-b": "candidate-b",
    }

    postprocess_node = ppiflow_workflow.AF3ScoreNode(
        "AF3scoreStep_stage1",
        {"run_name": "af3-run"},
    )
    postprocess_call = postprocess_node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage1-af3score",
            task_key="node",
            work_dir=tmp_path / "post-result",
            cache_dir=tmp_path / "post-cache",
            inputs={"af3score_plan": [plan_artifact]},
        )
    )
    assert postprocess_call.function_name == "postprocess_ppiflow_af3score_stage"


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

    artifacts = [
        WorkflowArtifact(
            artifact_id="stage1-flowpacker-flowpacker_outputs",
            producing_node_id="stage1-flowpacker",
            kind=ArtifactKind.STRUCTURES,
            storage=VolumePath(
                volume_name="source-volume",
                path="upstream/results",
            ),
        )
    ]
    staged, physical_run_name, publication_key = (
        ppiflow_workflow._stage_af3score_candidate_inputs(
            artifacts=artifacts,
            candidate_manifests=None,
            execution_run_name="execution-a",
        )
    )

    assert [record["input_name"] for record in staged] == ["candidate_a.pdb"]
    assert physical_run_name == f"execution-a-{publication_key}"
    first_input = af3_root / physical_run_name / "inputs" / "candidate_a.pdb"
    assert first_input.read_text(encoding="utf-8") == "ATOM\n"
    assert commits == [True]

    monkeypatch.setattr(
        ppiflow_workflow,
        "DECLARED_MODEL_IDENTITY",
        "AlphaFold3/af3.bin:v2",
    )
    _staged, model_run_name, model_key = (
        ppiflow_workflow._stage_af3score_candidate_inputs(
            artifacts=artifacts,
            candidate_manifests=None,
            execution_run_name="execution-model-v2",
        )
    )
    assert model_key != publication_key
    assert model_run_name != physical_run_name

    (long_dir / "candidate_a.pdb").write_text("CHANGED\n", encoding="utf-8")
    _staged, changed_run_name, changed_key = (
        ppiflow_workflow._stage_af3score_candidate_inputs(
            artifacts=artifacts,
            candidate_manifests=None,
            execution_run_name="execution-b",
        )
    )

    assert changed_key != publication_key
    assert changed_run_name != physical_run_name
    assert first_input.read_text(encoding="utf-8") == "ATOM\n"
    assert (af3_root / changed_run_name / "inputs" / "candidate_a.pdb").read_text(
        encoding="utf-8"
    ) == "CHANGED\n"
    assert commits == [True, True, True]


def test_af3score_prepare_publishes_candidate_to_batch_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_pdb_dir = tmp_path / "batch" / "pdb"
    batch_pdb_dir.mkdir(parents=True)
    for candidate_id in ("candidate-a", "candidate-b"):
        (batch_pdb_dir / f"{candidate_id}.pdb").write_text(
            "ATOM\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        ppiflow_workflow,
        "_stage_af3score_candidate_inputs",
        lambda **_kwargs: (
            [
                {
                    "candidate_id": candidate_id,
                    "input_name": f"{candidate_id}.pdb",
                    "scientific_payload": {
                        "candidate_id": candidate_id,
                        "content_sha256": candidate_id,
                    },
                }
                for candidate_id in ("candidate-a", "candidate-b")
            ],
            "ppiflow-af3score-request-key",
            "request-key",
        ),
    )
    monkeypatch.setattr(
        ppiflow_workflow.af3score_app,
        "af3score_prepare",
        _FakeModalFunction(
            "fc-prepare",
            SimpleNamespace(
                pending=2,
                chunk_specs=[
                    SimpleNamespace(
                        batch_name="batch-0",
                        batch_json_dir=str(tmp_path / "batch" / "json"),
                        batch_pdb_dir=str(batch_pdb_dir),
                    )
                ],
            ),
        ),
    )

    result = ppiflow_workflow.prepare_ppiflow_af3score_stage.get_raw_f()(
        artifacts=[_upstream_structure_artifact()],
        candidate_manifests=[],
        config={"num_jobs": 2},
        step_name="AF3scoreStep_stage1",
        execution_run_name="af3-run",
    )

    [output] = result.outputs
    assert isinstance(output.storage, InlineBytes)
    plan = orjson.loads(output.storage.data)
    assert [candidate["candidate_id"] for candidate in plan["candidates"]] == [
        "candidate-a",
        "candidate-b",
    ]
    assert {candidate["chunk"]["batch_name"] for candidate in plan["candidates"]} == {
        "batch-0"
    }
    assert {candidate["chunk"]["task_count"] for candidate in plan["candidates"]} == {2}
    assert plan["run_name"] == "ppiflow-af3score-request-key"


def test_af3score_step_reports_partial_for_mixed_scores(tmp_path: Path) -> None:
    af3score_stage = _FakeModalFunction(
        "fc-af3score-stage",
        AppRunResult(status=AppRunStatus.PARTIAL),
    )
    node = ppiflow_workflow.AF3ScoreNode(
        "AF3scoreStep_stage1",
        {"run_name": "af3-run"},
    )

    del tmp_path
    result = node.process_remote_result(af3score_stage.result, {})

    assert result.status == AppRunStatus.PARTIAL


def test_rosetta_nodes_bind_prepare_pull_worker_and_finalizer(
    tmp_path: Path,
) -> None:
    task = ppiflow_workflow.RosettaTaskSpec(
        task_key="candidate-a",
        index=1,
        binary="relax",
        pdb="inputs/1/candidate-a.pdb",
        rosetta_script=None,
        flags_file=None,
        output_dir="outputs/1",
        worker_log="logs/1.log",
        expected_files=("outputs/1/score.sc",),
        input_sha256="a" * 64,
        candidate_id="candidate-a",
    )
    plan_path = tmp_path / "rosetta-plan.json"
    plan_path.write_text(
        orjson.dumps({
            "schema_version": 1,
            "run_name": "rosetta-run",
            "run_id": "rosetta-id",
            "run_root": str(
                Path(ppiflow_workflow.ROSETTA_OUTPUT_MOUNTPOINT)
                / "rosetta-run-rosetta-id"
            ),
            "job_manifest": str(
                Path(ppiflow_workflow.ROSETTA_OUTPUT_MOUNTPOINT)
                / "rosetta-run-rosetta-id"
                / "rosetta_job_manifest.csv"
            ),
            "num_jobs": 1,
            "worker_count": 1,
            "claim_capacity": 1,
            "max_parallel_per_worker": 1,
            "tasks": [task.to_dict()],
        }).decode(),
        encoding="utf-8",
    )
    plan_artifact = WorkflowArtifact(
        artifact_id="rosetta-plan",
        producing_node_id="stage2-rosetta-relax-prepare",
        kind=ArtifactKind.TABLE,
        storage=VolumePath(
            volume_name="workflow-volume",
            path=plan_path.name,
            media_type="application/json",
        ),
    )
    outcome_artifact = plan_artifact.model_copy(
        update={
            "artifact_id": "rosetta-outcomes",
            "producing_node_id": "stage2-rosetta-relax-workers",
        }
    )
    prepare_node = ppiflow_workflow.RosettaPrepareNode(
        "RosettaRelaxStep",
        {"run_name": "rosetta-run", "rosetta_binary": "relax", "max_num_pods": 1},
    )
    prepare_call = prepare_node.prepare_remote(
        NodeRunContext(
            execution_run_id=RUN_ID,
            workload_run_key="run-1",
            node_id="stage2-rosetta-relax-prepare",
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
    assert prepare_call.function_name == "prepare_ppiflow_rosetta_stage"
    assert prepare_call.kwargs["run_name"] == "rosetta-run"
    assert prepare_call.kwargs["config"]["rosetta_binary"] == "relax"
    assert prepare_call.kwargs["candidate_manifests"] == [
        _upstream_structure_artifact(kind=ArtifactKind.TABLE)
    ]

    worker_node = ppiflow_workflow.RosettaWorkerNode(
        "RosettaRelaxStep",
        {"run_name": "rosetta-run"},
    )
    worker_context = NodeRunContext(
        execution_run_id=RUN_ID,
        workload_run_key="run-1",
        node_id="stage2-rosetta-relax-workers",
        task_key="node",
        work_dir=tmp_path / "worker-result",
        cache_dir=tmp_path / "worker-cache",
        inputs={"rosetta_plan": [plan_artifact]},
        volume_root=tmp_path,
        workflow_volume_name="workflow-volume",
    )
    assert isinstance(worker_node, RemotePullTaskWorkflowNode)
    [discovered] = worker_node.discover_remote_tasks(worker_context)
    assert discovered.task_key == "candidate-a"
    assert discovered.scientific_payload == task.scientific_payload
    worker_call = worker_node.prepare_pull_worker(worker_context)
    assert worker_call.function_name == "run_ppiflow_rosetta_worker"
    assert worker_call.claim_capacity == 1
    assert worker_call.max_worker_calls == 1
    assert worker_call.kwargs["max_parallel"] == 1

    finalizer = ppiflow_workflow.RosettaRelaxNode(
        "RosettaRelaxStep",
        {"run_name": "rosetta-run"},
    )
    finalizer_call = finalizer.prepare_remote(
        replace(
            worker_context,
            node_id="stage2-rosetta-relax",
            inputs={
                "rosetta_plan": [plan_artifact],
                "rosetta_outcomes": [outcome_artifact],
            },
        )
    )
    assert finalizer_call.function_name == "finalize_ppiflow_rosetta_stage"
    assert finalizer_call.kwargs["plan_artifacts"] == [plan_artifact]
    assert finalizer_call.kwargs["outcome_artifacts"] == [outcome_artifact]


def test_rosetta_prepare_publishes_deterministic_task_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _local_transform_environment(monkeypatch, tmp_path)
    structures = source_root / "upstream" / "results"
    structures.mkdir(parents=True)
    (structures / "b.pdb").write_text("ATOM B\n", encoding="utf-8")
    (structures / "a.pdb").write_text("ATOM A\n", encoding="utf-8")
    rosetta_root = tmp_path / "rosetta"
    commits = []
    monkeypatch.setattr(
        ppiflow_workflow,
        "ROSETTA_OUTPUT_MOUNTPOINT",
        str(rosetta_root),
    )
    monkeypatch.setattr(ppiflow_workflow, "ROSETTA_OUTPUT_VOLUME_NAME", "rosetta-vol")
    monkeypatch.setattr(
        ppiflow_workflow,
        "ROSETTA_OUTPUT_VOLUME",
        SimpleNamespace(commit=lambda: commits.append(True)),
    )
    run_root = rosetta_root / "rosetta-run-run-1-stage2-rosetta-relax-prepare"
    preserved_output = run_root / "outputs" / "1" / "score.sc"
    preserved_output.parent.mkdir(parents=True)
    preserved_output.write_text("CACHED\n", encoding="utf-8")
    preserved_marker = run_root / ".biomodals" / "tasks" / "existing.json"
    preserved_marker.parent.mkdir(parents=True)
    preserved_marker.write_text("{}\n", encoding="utf-8")

    result = ppiflow_workflow.prepare_ppiflow_rosetta_stage.get_raw_f()(
        artifacts=[_upstream_structure_artifact()],
        candidate_manifests=[],
        config={"rosetta_binary": "relax", "max_num_pods": 2},
        step_name="RosettaRelaxStep",
        run_name="rosetta-run",
        run_id="run-1",
        node_id="stage2-rosetta-relax-prepare",
    )

    assert result.status == AppRunStatus.SUCCEEDED
    [output] = result.outputs
    assert isinstance(output.storage, InlineBytes)
    plan = orjson.loads(output.storage.data)
    assert plan["num_jobs"] == 2
    assert plan["worker_count"] == 1
    assert plan["claim_capacity"] == 2
    assert [task["candidate_id"] for task in plan["tasks"]] == ["a", "b"]
    assert [task["index"] for task in plan["tasks"]] == [1, 2]
    assert [task["input_sha256"] for task in plan["tasks"]] == [
        hashlib.sha256(b"ATOM A\n").hexdigest(),
        hashlib.sha256(b"ATOM B\n").hexdigest(),
    ]
    assert Path(plan["job_manifest"]).is_file()
    assert preserved_output.read_text(encoding="utf-8") == "CACHED\n"
    assert preserved_marker.read_text(encoding="utf-8") == "{}\n"
    assert commits == [True]


def test_rosetta_worker_claims_executes_and_checkpoints_microbatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_call_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    task = ppiflow_workflow.RosettaTaskSpec(
        task_key="candidate-a",
        index=1,
        binary="relax",
        pdb="inputs/1/candidate-a.pdb",
        rosetta_script=None,
        flags_file=None,
        output_dir="outputs/1",
        worker_log="logs/1.log",
        expected_files=("outputs/1/score.sc",),
        input_sha256="a" * 64,
        candidate_id="candidate-a",
    )
    assignment = WorkerAssignmentRecord(
        execution_run_id=RUN_ID,
        node_key="stage2-rosetta-relax-workers",
        task_key=task.task_key,
        task_fingerprint="task-fingerprint",
        execution_payload={"task": task.to_dict(), "run_root": "unused"},
        provider_call_id=provider_call_id,
        request_id="claim",
        ordinal=0,
        created_at=1,
    )
    claim_count = 0
    completions = []

    def claim(call_id, request_id, capacity):
        nonlocal claim_count
        assignments = (assignment,) if claim_count == 0 else ()
        claim_count += 1
        return PullTaskClaim(
            request_id=request_id,
            provider_call_id=provider_call_id,
            assignments=assignments,
        )

    def complete(call_id, batch):
        completions.extend(
            (call_id, task_key, request_id, result)
            for task_key, request_id, result in batch
        )

    coordinator = SimpleNamespace(
        claim_tasks=SimpleNamespace(remote=claim),
        complete_tasks=SimpleNamespace(remote=complete),
    )
    commits = []
    monkeypatch.setattr(
        ppiflow_workflow,
        "ROSETTA_OUTPUT_MOUNTPOINT",
        str(tmp_path),
    )
    monkeypatch.setattr(
        ppiflow_workflow,
        "ROSETTA_OUTPUT_VOLUME",
        SimpleNamespace(commit=lambda: commits.append(True)),
    )

    def run_command(command, *, output_mode, log_file):
        assert output_mode == "log"
        Path(command[-1], "score.sc").write_text("SCORE\n", encoding="utf-8")
        Path(log_file).write_text("worker log\n", encoding="utf-8")

    monkeypatch.setattr(shell_helper, "run_command", run_command)
    run_root = tmp_path / "rosetta-run-rosetta-id"
    input_pdb = run_root / task.pdb
    input_pdb.parent.mkdir(parents=True)
    input_pdb.write_text("ATOM\n", encoding="utf-8")

    summary = ppiflow_workflow.run_ppiflow_rosetta_worker.get_raw_f()(
        coordinator=coordinator,
        provider_call_id=str(provider_call_id),
        run_name="rosetta-run",
        run_id="rosetta-id",
        claim_capacity=1,
        max_parallel=1,
    )

    assert summary == {"claimed_tasks": 1, "claim_requests": 2}
    assert completions[0][0:3] == (
        str(provider_call_id),
        "candidate-a",
        f"{provider_call_id}:complete:task-fingerprint",
    )
    assert completions[0][3].status == AppRunStatus.SUCCEEDED
    assert commits == [True]


def test_rosetta_finalizer_preserves_usable_partial_candidate_manifest(
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

    tasks = [
        ppiflow_workflow.RosettaTaskSpec(
            task_key=candidate_id,
            index=index,
            binary="relax",
            pdb=f"inputs/{index}/{candidate_id}.pdb",
            rosetta_script=None,
            flags_file=None,
            output_dir=f"outputs/{index}",
            worker_log=f"logs/{index}.log",
            expected_files=(f"outputs/{index}/score.sc",),
            input_sha256=f"{index}" * 64,
            candidate_id=candidate_id,
        )
        for index, candidate_id in enumerate(
            ("candidate-a", "candidate-b"),
            start=1,
        )
    ]
    plan_path = workflow_root / "rosetta-plan.json"
    plan_path.write_text(
        orjson.dumps({
            "schema_version": 1,
            "run_name": "rosetta-run",
            "run_id": "rosetta-id",
            "run_root": str(run_root),
            "num_jobs": 2,
            "job_manifest": str(job_manifest),
            "worker_count": 1,
            "claim_capacity": 2,
            "max_parallel_per_worker": 2,
            "tasks": [task.to_dict() for task in tasks],
        }).decode(),
        encoding="utf-8",
    )
    outcomes_path = workflow_root / "rosetta-outcomes.json"
    outcomes_path.write_text(
        orjson.dumps({
            "schema_version": 1,
            "succeeded": ["candidate-a"],
            "errors": {"candidate-b": "worker failed"},
        }).decode(),
        encoding="utf-8",
    )
    plan_artifact = WorkflowArtifact(
        artifact_id="rosetta-plan",
        producing_node_id="stage2-rosetta-relax-prepare",
        kind=ArtifactKind.TABLE,
        storage=VolumePath(
            volume_name="workflow-volume",
            path=plan_path.name,
            media_type="application/json",
        ),
    )
    outcomes_artifact = WorkflowArtifact(
        artifact_id="rosetta-outcomes",
        producing_node_id="stage2-rosetta-relax-workers",
        kind=ArtifactKind.TABLE,
        storage=VolumePath(
            volume_name="workflow-volume",
            path=outcomes_path.name,
            media_type="application/json",
        ),
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

    result = ppiflow_workflow.finalize_ppiflow_rosetta_stage.get_raw_f()(
        plan_artifacts=[plan_artifact],
        outcome_artifacts=[outcomes_artifact],
        config={"rosetta_binary": "relax", "max_num_pods": 1},
        step_name="RosettaRelaxStep",
        run_id="run-1",
        node_id="stage2-rosetta-relax",
    )

    assert result.status == AppRunStatus.SUCCEEDED
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
    assert result.metrics == {
        "successful_candidates": 1,
        "failed_candidates": 1,
    }
    assert result.warnings == ["candidate-b: worker failed"]


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
    assert submission.uses_gpu is True
    assert submission.kwargs["candidate_id"] == "candidate-a"
    assert submission.kwargs["config"]["recycle"] == 2
    assert submission.kwargs["config"]["sample"] == 1
    assert submission.kwargs["config"]["model_seeds"] == [3]
    assert submission.kwargs["artifacts"] == [_upstream_structure_artifact()]


def test_refold_uses_alphafold3_helpers_from_their_owning_modules() -> None:
    assert ppiflow_workflow.AF3Config is AF3Config
    assert (
        ppiflow_workflow.prepare_inference_run is inference_inputs.prepare_inference_run
    )
    assert ppiflow_workflow.stage_inference_run is modal_adapters.stage_inference_run
    assert ppiflow_workflow.RequestPublication is request_results.RequestPublication
    assert (
        ppiflow_workflow.load_request_manifest is request_results.load_request_manifest
    )
    assert (
        ppiflow_workflow.request_manifest_from_result
        is request_results.request_manifest_from_result
    )
    assert (
        ppiflow_workflow.create_request_archive
        is request_results.create_request_archive
    )


def test_refold_builds_af3_config_without_app_reexports() -> None:
    config = ppiflow_workflow._af3_config_for_refold(
        structure_name="candidate.pdb",
        structure_bytes=(
            b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            b"  1.00  0.00           C\n"
        ),
        run_name="refold-run",
        config={"model_seeds": [3]},
    )

    assert config.name == "refold-run"
    assert config.modelSeeds == [3]
    assert config.sequences[0].protein.id == "A"
    assert config.sequences[0].protein.sequence == "A"


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


def test_dockq_stage_executes_batch_in_tracked_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = iter([
        [("candidate.pdb", b"ATOM REF\n")],
        [("candidate.pdb", b"ATOM MODEL\n")],
    ])
    dockq = _FakeModalFunction(
        "unused",
        AppRunResult(status=AppRunStatus.SUCCEEDED),
    )
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
    monkeypatch.setattr(
        ppiflow_workflow.dockq_app,
        "run_dockq_workflow",
        dockq,
    )

    ppiflow_workflow.run_ppiflow_dockq_stage.get_raw_f()(
        reference_artifacts=[_upstream_structure_artifact()],
        model_artifacts=[_upstream_structure_artifact()],
        candidate_manifests=None,
        config={},
        run_name="dockq-run",
    )

    assert dockq.kwargs["run_name"] == "dockq-run"
    assert len(dockq.kwargs["pairs"]) == 1
    assert dockq.kwargs["pairs"][0]["candidate_id"] == "candidate"


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
    assert result.outputs[2].metadata["rows"] == 1
    assert result.outputs[2].metadata["manifest_schema_version"] == 1


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


def test_ppiflow_stages_local_inputs_by_content_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads: list[tuple[Path, str]] = []
    remote_payloads = {
        "existing/first.pdb": b"ATOM remote\n",
        "existing/renamed.pdb": b"ATOM remote\n",
    }

    class FakeBatchUpload:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def put_file(self, local_path: Path, remote_path: str) -> None:
            uploads.append((local_path, remote_path))

    class FakeVolume:
        def batch_upload(self, *, force: bool):
            assert force is True
            return FakeBatchUpload()

        def read_file(self, path: str):
            yield remote_payloads[path]

    volume_root = "/mnt/PPIFlow-outputs"
    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint=volume_root,
            output_volume_name="PPIFlow-outputs",
            repo_commit_hash=ppiflow_app.CONF.repo_commit_hash,
            version=ppiflow_app.CONF.version,
        ),
    )
    first = tmp_path / "first.pdb"
    same = tmp_path / "renamed.pdb"
    changed = tmp_path / "changed.pdb"
    first.write_bytes(b"ATOM same\n")
    same.write_bytes(b"ATOM same\n")
    changed.write_bytes(b"ATOM changed\n")

    def stage(path: Path):
        return ppiflow_workflow._stage_ppiflow_app_inputs(
            steps_doc={
                "PPIFlowStep": {
                    "args": {
                        "name": "demo",
                        "specified_hotspots": "A1",
                        "input_pdb": str(path),
                        "binder_chain": "B",
                    }
                }
            },
            run_id="demo",
            app_steps=("PPIFlowStep",),
        )

    first_steps = stage(first)
    same_steps = stage(same)
    changed_steps = stage(changed)
    remote_first_steps = stage(Path(f"{volume_root}/existing/first.pdb"))
    remote_same_steps = stage(Path(f"{volume_root}/existing/renamed.pdb"))
    first_path = first_steps["PPIFlowStep"]["args"]["input_pdb"]
    same_path = same_steps["PPIFlowStep"]["args"]["input_pdb"]
    changed_path = changed_steps["PPIFlowStep"]["args"]["input_pdb"]

    assert first_path == same_path
    assert first_path != changed_path
    assert hashlib.sha256(first.read_bytes()).hexdigest() in str(first_path)

    task_yaml = _task_yaml(enabled_steps="  PPIFlowStep: true\n")

    def workflow_hash(steps: object) -> str:
        return hashing.dag_hash(
            build_ppiflow_workflow(
                task_yaml_bytes=task_yaml,
                steps_yaml_bytes=yaml.safe_dump(steps).encode(),
            ).validate()
        )

    assert workflow_hash(first_steps) == workflow_hash(same_steps)
    assert workflow_hash(first_steps) != workflow_hash(changed_steps)
    assert workflow_hash(remote_first_steps) == workflow_hash(remote_same_steps)
    assert len(uploads) == 3


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
    assert calls["spawn"]["max_parallel_nodes"] == 16
    assert calls["spawn"]["max_active_provider_calls"] == 4
    assert calls["spawn"]["max_active_gpu_provider_calls"] == 4
    assert "force" not in calls["spawn"]


def test_submit_ppiflow_workflow_uses_successor_operation_for_restart(
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
    predecessor = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    class UnexpectedRunMethod:
        def spawn(self, **_kwargs):
            raise AssertionError("restart must not create a root run")

    class FakePrepareMethod:
        def remote(self, **kwargs):
            calls["prepare"] = kwargs

    class FakeDriveMethod:
        def spawn(self, **kwargs):
            calls["drive"] = kwargs
            return _FakeFunctionCall("call-1")

    class FakeCoordinator:
        run = UnexpectedRunMethod()
        prepare_restart_from = FakePrepareMethod()
        drive_prepared = FakeDriveMethod()

    def fake_coordinator_handle(**kwargs):
        calls["coordinator"] = kwargs
        return FakeCoordinator()

    monkeypatch.setattr(
        ppiflow_workflow,
        "_stage_ppiflow_app_inputs",
        lambda **kwargs: kwargs["steps_doc"],
    )
    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "execution_coordinator_handle",
        fake_coordinator_handle,
    )

    raw_f = ppiflow_workflow.submit_ppiflow_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        task_yaml=str(task_yaml),
        steps_yaml=str(steps_yaml),
        run_id="demo",
        wait=False,
        use_deployed_coordinator=True,
        deployment_name="ppiflow-prod",
        deployment_version=7,
        restart_from=predecessor,
    )

    assert calls["coordinator"]["deployment"].environment == "main"
    assert calls["prepare"]["predecessor_execution_run_id"] == predecessor
    assert calls["prepare"]["workload_run_key"] == "demo"
    assert calls["prepare"]["workflow"].name == "ppiflow-v2"
    assert calls["drive"] == {}


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
        "stage1-af3score-prepare",
        "stage1-af3score-batches",
        "stage1-af3score",
        "stage1-filter",
        "stage2-rosetta-fix-prepare",
        "stage2-rosetta-fix-workers",
        "stage2-rosetta-fix",
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
        "stage2-ligandmpnn",
        "stage2-flowpacker",
        "stage2-af3score-prepare",
        "stage2-af3score-batches",
        "stage2-af3score",
        "stage2-filter",
        "stage2-alphafold3-refold",
        "stage2-dockq",
        "stage2-rosetta-relax-prepare",
        "stage2-rosetta-relax-workers",
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
        "AF3ScorePrepareNode",
        "AF3ScoreBatchNode",
        "AF3ScoreNode",
        "FilterStructuresNode",
        "RosettaPrepareNode",
        "RosettaWorkerNode",
        "RosettaFixNode",
        "FixedPositionsNode",
        "PPIFlowPartialNode",
        "LigandMPNNNode",
        "FlowPackerNode",
        "AF3ScorePrepareNode",
        "AF3ScoreBatchNode",
        "AF3ScoreNode",
        "FilterStructuresNode",
        "ReFoldNode",
        "DockQNode",
        "RosettaPrepareNode",
        "RosettaWorkerNode",
        "RosettaRelaxNode",
        "RankNode",
        "ReportNode",
    ]
    assert definition.dependencies["stage2-fixed-positions"] == {"stage2-rosetta-fix"}
    assert definition.dependencies["stage2-rosetta-fix-prepare"] == {"stage1-filter"}
    assert definition.dependencies["stage2-rosetta-fix-workers"] == {
        "stage2-rosetta-fix-prepare"
    }
    assert definition.dependencies["stage2-rosetta-fix"] == {
        "stage2-rosetta-fix-prepare",
        "stage2-rosetta-fix-workers",
    }
    assert definition.dependencies["stage1-af3score-batches"] == {
        "stage1-af3score-prepare"
    }
    assert definition.dependencies["stage1-af3score"] == {
        "stage1-af3score-prepare",
        "stage1-af3score-batches",
    }
    assert definition.dependencies["stage2-partial-ppiflow"] == {
        "stage2-fixed-positions"
    }
    assert (
        definition.nodes["stage1-ligandmpnn"].aggregation_policy
        == ppiflow_workflow.NodeAggregationPolicy.ALLOW_PARTIAL
    )
    assert definition.nodes["stage1-flowpacker"].partial_dependencies == {
        "stage1-ligandmpnn"
    }
    assert definition.dependencies["stage2-dockq"] == {
        "stage2-filter",
        "stage2-alphafold3-refold",
    }
    assert (
        definition.nodes["stage2-partial-ppiflow"].aggregation_policy
        == ppiflow_workflow.NodeAggregationPolicy.ALLOW_PARTIAL
    )
    assert (
        definition.nodes["stage2-ligandmpnn"].aggregation_policy
        == ppiflow_workflow.NodeAggregationPolicy.ALLOW_PARTIAL
    )
    assert definition.nodes["stage2-ligandmpnn"].partial_dependencies == {
        "stage2-partial-ppiflow"
    }
    assert (
        definition.nodes["stage2-alphafold3-refold"].aggregation_policy
        == ppiflow_workflow.NodeAggregationPolicy.ALLOW_PARTIAL
    )
    assert definition.nodes["stage2-dockq"].partial_dependencies == {
        "stage2-alphafold3-refold"
    }
    assert definition.dependencies["stage2-rosetta-relax-prepare"] == {
        "stage2-filter",
        "stage2-dockq",
    }
    assert definition.dependencies["stage2-rosetta-relax-workers"] == {
        "stage2-rosetta-relax-prepare"
    }
    assert definition.dependencies["stage2-rosetta-relax"] == {
        "stage2-rosetta-relax-prepare",
        "stage2-rosetta-relax-workers",
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
        "stage1-af3score-prepare": "stage1-ligandmpnn",
        "stage1-filter": "stage1-ligandmpnn",
        "stage2-rosetta-fix-prepare": "stage1-filter",
        "stage2-fixed-positions": "stage2-rosetta-fix",
        "stage2-partial-ppiflow": "stage2-fixed-positions",
        "stage2-ligandmpnn": "stage2-partial-ppiflow",
        "stage2-af3score-prepare": "stage2-ligandmpnn",
        "stage2-filter": "stage2-ligandmpnn",
        "stage2-alphafold3-refold": "stage2-filter",
        "stage2-dockq": "stage2-filter",
        "stage2-rosetta-relax-prepare": "stage2-filter",
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

    assert "stage1-filter" in _manifest_ancestor_chain(
        definition,
        "stage2-rosetta-fix-prepare",
    )
    stage1_retained_path = [
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
        "stage2-ligandmpnn",
        "stage2-af3score-prepare",
        "stage2-filter",
    ]
    for node_id in stage1_retained_path:
        assert "stage2-rosetta-fix" in _manifest_ancestor_chain(
            definition,
            node_id,
        )

    stage2_retained_path = [
        "stage2-alphafold3-refold",
        "stage2-dockq",
        "stage2-rosetta-relax-prepare",
    ]
    for node_id in stage2_retained_path:
        assert "stage2-filter" in _manifest_ancestor_chain(definition, node_id)
    assert "stage2-rosetta-relax" in _manifest_ancestor_chain(
        definition,
        "stage2-rank",
    )


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


def test_ppiflow_operational_fanout_does_not_change_scientific_dag_hash() -> None:
    task_yaml = b"""
task:
  gentype: binder
  candidate_concurrency: 8
steps:
  MPNNStep_stage1: true
  AF3scoreStep_stage1: true
  RosettaFixStep: true
"""

    def workflow_hash(max_child_calls: int, *, max_structures: int = 5) -> str:
        return hashing.dag_hash(
            build_ppiflow_workflow(
                task_yaml_bytes=task_yaml,
                steps_yaml_bytes=f"""
MPNNStep_stage1:
  candidate_concurrency: 6
  max_structures: {max_structures}
AF3scoreStep_stage1:
  num_jobs: 7
  prepare_workers: 4
RosettaFixStep:
  max_num_pods: 5
""".encode(),
                max_child_calls=max_child_calls,
            ).validate()
        )

    baseline = workflow_hash(2)

    assert workflow_hash(4) == baseline
    assert workflow_hash(2, max_structures=6) != baseline


def test_ppiflow_app_version_changes_plan_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fingerprint() -> str:
        workflow = build_ppiflow_workflow(
            task_yaml_bytes=_task_yaml(enabled_steps="  PPIFlowStep: true\n"),
            steps_yaml_bytes=b"PPIFlowStep: {}\n",
        )
        return execution_plan(
            workflow.validate(),
            workload_run_key="run-1",
        ).workload_plan_fingerprint

    baseline = fingerprint()
    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        ppiflow_app.CONF.model_copy(update={"repo_commit_hash": "changed-ppiflow"}),
    )

    assert fingerprint() != baseline


def test_af3score_model_identity_changes_scientific_dag_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = ppiflow_workflow.Workflow("af3score-identity")
    workflow.add_node(
        ppiflow_workflow.AF3ScorePrepareNode("score", {}),
        id="score",
    )
    baseline = hashing.dag_hash(workflow.validate())

    monkeypatch.setattr(ppiflow_workflow, "DECLARED_MODEL_IDENTITY", "changed")

    assert hashing.dag_hash(workflow.validate()) != baseline


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
        "stage2-rosetta-fix-prepare",
        "stage2-rosetta-fix-workers",
        "stage2-rosetta-fix",
    ]
    assert isinstance(
        definition.nodes["stage2-existing-input"].node,
        ppiflow_workflow.ExistingStructuresNode,
    )
    assert definition.dependencies["stage2-rosetta-fix-prepare"] == {
        "stage2-existing-input"
    }
    assert definition.dependencies["stage2-rosetta-fix-workers"] == {
        "stage2-rosetta-fix-prepare"
    }
    assert definition.dependencies["stage2-rosetta-fix"] == {
        "stage2-rosetta-fix-prepare",
        "stage2-rosetta-fix-workers",
    }
    assert (
        definition.nodes["stage2-rosetta-fix-prepare"].inputs["candidate_manifest"].kind
        == ArtifactKind.TABLE
    )
    assert (
        definition.nodes["stage2-rosetta-fix-prepare"].inputs["candidate_manifest"].role
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
    assert "stage2-existing-input" in _manifest_ancestor_chain(
        definition,
        "stage2-rosetta-fix-prepare",
    )
    for node_id in [
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
        "stage2-ligandmpnn",
    ]:
        assert "stage2-rosetta-fix" in _manifest_ancestor_chain(
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


def test_submit_ppiflow_workflow_enables_external_checks(
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
    monkeypatch.setattr(
        ppiflow_workflow,
        "_stage_ppiflow_app_inputs",
        lambda **kwargs: kwargs["steps_doc"],
    )

    raw_f = ppiflow_workflow.submit_ppiflow_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        task_yaml=str(task_yaml),
        steps_yaml=str(steps_yaml),
        run_id="demo",
        wait=True,
    )

    assert calls["spawn"]["strict_external_artifact_checks"] is True
    assert (
        calls["spawn"]["external_artifact_checker_function_name"]
        == "check_ppiflow_external_artifact"
    )
