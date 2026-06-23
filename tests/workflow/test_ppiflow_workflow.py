"""Tests for the PPIFlow workflow definition."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import modal
import pytest

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
from biomodals.workflow.core import NodeRunContext
from biomodals.workflow.ppiflow_workflow import (
    CONF,
    PPIFlowModalNamespace,
    build_ppiflow_workflow,
)


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


def _fake_namespace(
    ppiflow_run: _FakePPIFlowFunction | None = None,
    ligandmpnn_run: _FakeModalFunction | None = None,
    flowpacker_run: _FakeModalFunction | None = None,
    af3score_manage_lock: _FakeModalFunction | None = None,
    af3score_prepare: _FakeModalFunction | None = None,
    af3score_run: _FakeModalFunction | None = None,
    af3score_postprocess: _FakeModalFunction | None = None,
    dockq_run: _FakeModalFunction | None = None,
    rosetta_run: _FakeModalFunction | None = None,
    alphafold3_search_msa: _FakeModalFunction | None = None,
    alphafold3_predict_structures: _FakeModalFunction | None = None,
    select_structures: _FakeModalFunction | None = None,
    copy_structures: _FakeModalFunction | None = None,
    filter_artifacts: _FakeModalFunction | None = None,
    derive_fixed_positions: _FakeModalFunction | None = None,
    rank_artifacts: _FakeModalFunction | None = None,
    stage_ppiflow_input: _FakeModalFunction | None = None,
    stage_af3score_inputs: _FakeModalFunction | None = None,
    stage_rosetta_inputs: _FakeModalFunction | None = None,
) -> PPIFlowModalNamespace:
    fake = cast(modal.Function, ppiflow_run or _FakePPIFlowFunction())
    fake_select = cast(
        modal.Function,
        select_structures
        or _FakeModalFunction("fc-select", [("model.pdb", b"ATOM\n")]),
    )
    return PPIFlowModalNamespace(
        ppiflow_run=fake,
        ligandmpnn_run=cast(modal.Function, ligandmpnn_run or fake),
        flowpacker_run=cast(modal.Function, flowpacker_run or fake),
        af3score_manage_lock=cast(modal.Function, af3score_manage_lock or fake),
        af3score_prepare=cast(modal.Function, af3score_prepare or fake),
        af3score_run=cast(modal.Function, af3score_run or fake),
        af3score_postprocess=cast(modal.Function, af3score_postprocess or fake),
        dockq_run=cast(modal.Function, dockq_run or fake),
        rosetta_run=cast(modal.Function, rosetta_run or fake),
        alphafold3_search_msa=cast(modal.Function, alphafold3_search_msa or fake),
        alphafold3_predict_structures=cast(
            modal.Function,
            alphafold3_predict_structures or fake,
        ),
        select_structures=fake_select,
        copy_structures=cast(
            modal.Function,
            copy_structures
            or _FakeModalFunction(
                "fc-copy", AppRunResult(status=AppRunStatus.SUCCEEDED)
            ),
        ),
        filter_artifacts=cast(
            modal.Function,
            filter_artifacts
            or _FakeModalFunction(
                "fc-filter", AppRunResult(status=AppRunStatus.SUCCEEDED)
            ),
        ),
        derive_fixed_positions=cast(
            modal.Function,
            derive_fixed_positions
            or _FakeModalFunction(
                "fc-fixed", AppRunResult(status=AppRunStatus.SUCCEEDED)
            ),
        ),
        rank_artifacts=cast(
            modal.Function,
            rank_artifacts
            or _FakeModalFunction(
                "fc-rank", AppRunResult(status=AppRunStatus.SUCCEEDED)
            ),
        ),
        stage_ppiflow_input=cast(
            modal.Function,
            stage_ppiflow_input
            or _FakeModalFunction("fc-stage-ppiflow", "/ppiflow/input.pdb"),
        ),
        stage_af3score_inputs=cast(
            modal.Function,
            stage_af3score_inputs or _FakeModalFunction("fc-stage-af3", ["model.pdb"]),
        ),
        stage_rosetta_inputs=cast(
            modal.Function,
            stage_rosetta_inputs
            or _FakeModalFunction(
                "fc-stage-rosetta",
                {
                    "run_name": "rosetta-run",
                    "run_id": "rosetta-id",
                    "run_root": "/rosetta/rosetta-run-rosetta-id",
                    "num_jobs": 1,
                },
            ),
        ),
    )


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


def test_ppiflow_app_step_uses_included_modal_namespace(tmp_path: Path) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = _fake_namespace(fake_function)
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
        modal_namespace=namespace,
    )

    definition = workflow.validate()
    spec = definition.nodes["stage1-ppiflow-design"]
    result = spec.node.run(
        NodeRunContext(
            run_id="run-1",
            node_id=spec.node_id,
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert fake_function.kwargs["run_name"] == "demo-run"
    assert isinstance(fake_function.kwargs["args"], ppiflow_app.PPIFlowArgs)
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


def test_ppiflow_app_step_submits_app_function_directly(tmp_path: Path) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = _fake_namespace(fake_function)
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
        modal_namespace=namespace,
    )

    spec = workflow.validate().nodes["stage1-ppiflow-design"]
    submission = spec.node.submit_remote(
        NodeRunContext(
            run_id="run-1",
            node_id=spec.node_id,
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    assert submission.function_name == "ppiflow_run"
    assert submission.function_call.object_id == "fc-ppiflow"
    assert fake_function.kwargs["run_name"] == "demo-run"
    assert isinstance(fake_function.kwargs["args"], ppiflow_app.PPIFlowArgs)


def test_ligandmpnn_step_selects_structure_and_submits_app_function(
    tmp_path: Path,
) -> None:
    selector = _FakeModalFunction("fc-select", [("selected.pdb", b"ATOM selected\n")])
    ligandmpnn = _FakeModalFunction(
        "fc-ligandmpnn",
        AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="LigandMPNN_outputs",
                    kind=ArtifactKind.ARCHIVE,
                    storage=InlineBytes(
                        data=b"archive",
                        filename="ligandmpnn.tar.zst",
                        media_type="application/zstd",
                    ),
                )
            ],
        ),
    )
    namespace = _fake_namespace(
        ligandmpnn_run=ligandmpnn,
        select_structures=selector,
    )
    node = ppiflow_workflow.LigandMPNNNode(
        "MPNNStep_stage1",
        namespace,
        {
            "run_name": "mpnn-run",
            "seeds": "1,2",
            "model_type": "protein_mpnn",
            "batch_size": 2,
            "number_of_batches": 3,
        },
    )
    context = NodeRunContext(
        run_id="run-1",
        node_id="stage1-ligandmpnn",
        attempt_id="attempt-1",
        cache_dir=tmp_path,
        inputs={"structures": [_upstream_structure_artifact()]},
    )

    submission = node.submit_remote(context)
    result = node.process_remote_result(
        submission.function_call.get(), submission.metadata
    )

    assert submission.function_name == "ligandmpnn_run"
    assert selector.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert ligandmpnn.kwargs["run_name"] == "mpnn-run"
    assert ligandmpnn.kwargs["script_mode"] == "run"
    assert ligandmpnn.kwargs["struct_bytes"] == b"ATOM selected\n"
    assert ligandmpnn.kwargs["seeds"] == [1, 2]
    assert ligandmpnn.kwargs["cli_args"]["--model_type"] == "protein_mpnn"
    assert ligandmpnn.kwargs["cli_args"]["--batch_size"] == "2"
    assert ligandmpnn.kwargs["cli_args"]["--number_of_batches"] == "3"
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].metadata["selected_structure"] == "selected.pdb"


def test_ligandmpnn_rejects_implicit_multi_structure_selection(
    tmp_path: Path,
) -> None:
    selector = _FakeModalFunction(
        "fc-select",
        [
            ("design-a.pdb", b"ATOM A\n"),
            ("design-b.pdb", b"ATOM B\n"),
        ],
    )
    node = ppiflow_workflow.LigandMPNNNode(
        "MPNNStep_stage1",
        _fake_namespace(select_structures=selector),
        {"run_name": "mpnn-run"},
    )

    with pytest.raises(ValueError, match="explicit structure_index"):
        node.submit_remote(
            NodeRunContext(
                run_id="run-1",
                node_id="stage1-ligandmpnn",
                attempt_id="attempt-1",
                cache_dir=tmp_path,
                inputs={"structures": [_upstream_structure_artifact()]},
            )
        )


def test_flowpacker_step_selects_structures_and_submits_app_function(
    tmp_path: Path,
) -> None:
    selected_structures = [
        ("design-a.pdb", b"ATOM A\n"),
        ("design-b.pdb", b"ATOM B\n"),
    ]
    selector = _FakeModalFunction("fc-select", selected_structures)
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
    namespace = _fake_namespace(
        flowpacker_run=flowpacker,
        select_structures=selector,
    )
    node = ppiflow_workflow.FlowPackerNode(
        "FlowpackerStep_stage1",
        namespace,
        {"run_name": "fp-run", "n_samples": 2, "seed": 7},
    )
    context = NodeRunContext(
        run_id="run-1",
        node_id="stage1-flowpacker",
        attempt_id="attempt-1",
        cache_dir=tmp_path,
        inputs={"structures": [_upstream_structure_artifact()]},
    )

    submission = node.submit_remote(context)
    result = node.process_remote_result(
        submission.function_call.get(), submission.metadata
    )

    assert submission.function_name == "run_flowpacker_workflow"
    assert selector.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert flowpacker.kwargs["input_files"] == selected_structures
    assert flowpacker.kwargs["run_name"] == "fp-run"
    assert flowpacker.kwargs["n_samples"] == 2
    assert flowpacker.kwargs["seed"] == 7
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].metadata["structure_count"] == 2


def test_af3score_step_runs_app_sequence_and_returns_metrics_artifact(
    tmp_path: Path,
) -> None:
    stage_inputs = _FakeModalFunction("fc-stage-af3", ["model.pdb"])
    prepare = _FakeModalFunction(
        "fc-af3-prepare",
        SimpleNamespace(chunk_specs=[]),
    )
    postprocess = _FakeModalFunction(
        "fc-af3-post",
        {
            "metrics_csv": (
                f"{ppiflow_workflow.AF3SCORE_OUTPUT_MOUNTPOINT}/af3-run/"
                "af3score_metrics.csv"
            ),
            "metrics_rows": 1,
            "processed": 1,
            "failed": 0,
        },
    )
    namespace = _fake_namespace(
        stage_af3score_inputs=stage_inputs,
        af3score_prepare=prepare,
        af3score_postprocess=postprocess,
    )
    node = ppiflow_workflow.AF3ScoreNode(
        "AF3scoreStep_stage1",
        namespace,
        {"run_name": "af3-run", "num_jobs": 4, "prepare_workers": 2},
    )
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-af3score",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert stage_inputs.kwargs["artifacts"] == [_upstream_structure_artifact()]
    assert prepare.kwargs["run_name"] == "af3-run"
    assert prepare.kwargs["input_files"] == ["model.pdb"]
    assert prepare.kwargs["num_jobs"] == 4
    assert prepare.kwargs["prepare_workers"] == 2
    assert postprocess.kwargs["input_files"] == ["model.pdb"]
    assert result.outputs[0].kind == ArtifactKind.SCORES
    assert result.outputs[0].storage == VolumePath(
        volume_name=ppiflow_workflow.AF3SCORE_OUTPUT_VOLUME_NAME,
        path="af3-run/af3score_metrics.csv",
    )
    assert stage_inputs.kwargs["patterns"] == ("*.pdb",)


def test_rosetta_step_stages_inputs_and_returns_output_directory(
    tmp_path: Path,
) -> None:
    stage_rosetta = _FakeModalFunction(
        "fc-stage-rosetta",
        {
            "run_name": "rosetta-run",
            "run_id": "rosetta-id",
            "run_root": f"{ppiflow_workflow.ROSETTA_OUTPUT_MOUNTPOINT}/rosetta-run-rosetta-id",
            "num_jobs": 1,
        },
    )
    rosetta_run = _FakeModalFunction("fc-rosetta", None)
    namespace = _fake_namespace(
        stage_rosetta_inputs=stage_rosetta,
        rosetta_run=rosetta_run,
    )
    node = ppiflow_workflow.RosettaRelaxNode(
        "RosettaRelaxStep",
        namespace,
        {"run_name": "rosetta-run", "rosetta_binary": "relax", "max_num_pods": 1},
    )
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-rosetta-relax",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert stage_rosetta.kwargs["rosetta_binary"] == "relax"
    assert rosetta_run.args == ("rosetta-run", "rosetta-id", 1)
    assert rosetta_run.kwargs == {}
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].storage == VolumePath(
        volume_name=ppiflow_workflow.ROSETTA_OUTPUT_VOLUME_NAME,
        path="rosetta-run-rosetta-id",
    )


def test_refold_step_derives_af3_config_and_runs_inference(tmp_path: Path) -> None:
    pdb_bytes = (
        b"ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n"
        b"ATOM      2  CA  CYS A   2      0.000   0.000   0.000  1.00  0.00           C\n"
    )
    selector = _FakeModalFunction("fc-select", [("model.pdb", pdb_bytes)])
    predict = _FakeModalFunction("fc-af3-predict", b"af3-tar-zst")
    namespace = _fake_namespace(
        select_structures=selector,
        alphafold3_predict_structures=predict,
    )
    node = ppiflow_workflow.ReFoldNode(
        "ReFoldStep",
        namespace,
        {"run_name": "refold-run", "model_seeds": [3], "recycle": 2, "sample": 1},
    )
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-alphafold3-refold",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert predict.kwargs["recycle"] == 2
    assert predict.kwargs["sample"] == 1
    assert predict.kwargs["model_seeds"] == [3]
    assert b'"sequence":"AC"' in predict.kwargs["json_bytes"]
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].storage.data == b"af3-tar-zst"


def test_refold_rejects_implicit_multi_structure_selection(tmp_path: Path) -> None:
    selector = _FakeModalFunction(
        "fc-select",
        [
            ("design-a.pdb", b"ATOM A\n"),
            ("design-b.pdb", b"ATOM B\n"),
        ],
    )
    node = ppiflow_workflow.ReFoldNode(
        "ReFoldStep",
        _fake_namespace(select_structures=selector),
        {"run_name": "refold-run"},
    )

    with pytest.raises(ValueError, match="explicit structure_index"):
        node.run(
            NodeRunContext(
                run_id="run-1",
                node_id="stage2-alphafold3-refold",
                attempt_id="attempt-1",
                cache_dir=tmp_path,
                inputs={"structures": [_upstream_structure_artifact()]},
            )
        )


def test_dockq_step_pairs_filtered_and_refolded_structures(tmp_path: Path) -> None:
    selector = _FakeModalFunction("fc-select", [("structure.pdb", b"ATOM\n")])
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
    namespace = _fake_namespace(select_structures=selector, dockq_run=dockq)
    node = ppiflow_workflow.DockQNode(
        "DockQStep",
        namespace,
        {"run_name": "dockq-run", "dockq_args": "--short"},
    )
    submission = node.submit_remote(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-dockq",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={
                "structures": [_upstream_structure_artifact()],
                "models": [_upstream_structure_artifact()],
            },
        )
    )
    result = node.process_remote_result(
        submission.function_call.get(), submission.metadata
    )

    assert submission.function_name == "run_dockq_workflow"
    assert dockq.kwargs["run_name"] == "dockq-run"
    assert dockq.kwargs["dockq_args"] == ["--short"]
    assert dockq.kwargs["pairs"][0]["model_bytes"] == b"ATOM\n"
    assert dockq.kwargs["pairs"][0]["reference_bytes"] == b"ATOM\n"
    assert result.outputs[0].kind == ArtifactKind.SCORES
    assert result.outputs[0].metadata["pair_count"] == 1


def test_dockq_rejects_unpaired_structure_counts(tmp_path: Path) -> None:
    class SequencedSelector:
        def __init__(self) -> None:
            self.calls = 0

        def remote(self, *args, **kwargs):
            _ = args, kwargs
            self.calls += 1
            if self.calls == 1:
                return [("reference.pdb", b"ATOM REF\n")]
            return [
                ("model-a.pdb", b"ATOM A\n"),
                ("model-b.pdb", b"ATOM B\n"),
            ]

    selector = cast(modal.Function, SequencedSelector())
    node = ppiflow_workflow.DockQNode(
        "DockQStep",
        _fake_namespace(select_structures=selector),
        {"run_name": "dockq-run"},
    )

    with pytest.raises(ValueError, match="same number"):
        node.submit_remote(
            NodeRunContext(
                run_id="run-1",
                node_id="stage2-dockq",
                attempt_id="attempt-1",
                cache_dir=tmp_path,
                inputs={
                    "structures": [_upstream_structure_artifact()],
                    "models": [_upstream_structure_artifact()],
                },
            )
        )


def test_filter_step_delegates_score_filtering(tmp_path: Path) -> None:
    filter_artifacts = _FakeModalFunction(
        "fc-filter", AppRunResult(status=AppRunStatus.SUCCEEDED)
    )
    node = ppiflow_workflow.FilterStructuresNode(
        "FilterStep_stage1",
        _fake_namespace(filter_artifacts=filter_artifacts),
        {"filters": {"iptm": "> 0.7"}},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-filter",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={
                "structures": [_upstream_structure_artifact()],
                "scores": [_upstream_structure_artifact(kind=ArtifactKind.SCORES)],
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert filter_artifacts.kwargs["step_name"] == "FilterStep_stage1"
    assert filter_artifacts.kwargs["config"] == {"filters": {"iptm": "> 0.7"}}


def test_fixed_positions_delegates_residue_energy_parsing(tmp_path: Path) -> None:
    derive_fixed_positions = _FakeModalFunction(
        "fc-fixed", AppRunResult(status=AppRunStatus.SUCCEEDED)
    )
    node = ppiflow_workflow.FixedPositionsNode(
        "FixedPositions",
        _fake_namespace(derive_fixed_positions=derive_fixed_positions),
        {"gentype": "binder", "energy_threshold": -5},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-fixed-positions",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert derive_fixed_positions.kwargs["config"] == {
        "gentype": "binder",
        "energy_threshold": -5,
    }


def test_rank_step_delegates_score_aware_ranking(tmp_path: Path) -> None:
    rank_artifacts = _FakeModalFunction(
        "fc-rank", AppRunResult(status=AppRunStatus.SUCCEEDED)
    )
    node = ppiflow_workflow.RankNode(
        "RankStep",
        _fake_namespace(rank_artifacts=rank_artifacts),
        {"gentype": "binder"},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-rank",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={"structures": [_upstream_structure_artifact()]},
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert rank_artifacts.kwargs["config"] == {"gentype": "binder"}


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
        attempt_id="attempt-1",
        step_name="FilterStep_stage1",
    )

    output_dir = workflow_root / result.outputs[0].storage.path
    assert [path.name for path in output_dir.iterdir()] == [
        "upstream-structures__design-1.pdb"
    ]
    assert result.outputs[0].metadata["retained_count"] == 1


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
        attempt_id="attempt-1",
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
        attempt_id="attempt-1",
        step_name="RankStep",
    )

    output_dir = workflow_root / result.outputs[0].storage.path
    assert [path.name for path in output_dir.iterdir()] == [
        "upstream-structures__design-1.pdb"
    ]
    assert result.outputs[1].metadata["rows"] == 1


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

    class UnexpectedWorkflowOrchestrator:
        def __init__(self) -> None:
            raise AssertionError("dry-run should not construct the orchestrator")

    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "WorkflowOrchestrator",
        UnexpectedWorkflowOrchestrator,
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
    assert "[workflow] DAG graph: node_id [placement; class] <- dependency" in stdout
    assert (
        "[workflow]   stage1-ppiflow-design [remote; PPIFlowDesignNode] <- -" in stdout
    )
    assert "ppiflow_workflow.PPIFlowDesignNode" not in stdout
    assert "Submitting PPIFlow workflow" not in stdout


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
        modal_namespace=_fake_namespace(),
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
    assert definition.dependencies["stage2-report"] == {"stage2-rank"}


def test_ppiflow_stage2_only_requires_existing_input() -> None:
    try:
        build_ppiflow_workflow(
            task_yaml_bytes=_task_yaml(enabled_steps="  RosettaFixStep: true\n"),
            steps_yaml_bytes=b"{}\n",
            stage=2,
            modal_namespace=_fake_namespace(),
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
        modal_namespace=_fake_namespace(),
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


def test_structure_consuming_steps_fail_clearly_without_inputs(
    tmp_path: Path,
) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = _fake_namespace(fake_function)
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  FlowpackerStep_stage1: true\n"),
        steps_yaml_bytes=b"FlowpackerStep_stage1: {}\n",
        modal_namespace=namespace,
    )

    spec = workflow.validate().nodes["stage1-flowpacker"]
    try:
        spec.node.run(
            NodeRunContext(
                run_id="run-1",
                node_id=spec.node_id,
                attempt_id="attempt-1",
                cache_dir=tmp_path,
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
        def remote(self, **kwargs):
            calls["remote"] = kwargs
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    class FakeWorkflowOrchestrator:
        def __init__(self) -> None:
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "WorkflowOrchestrator",
        FakeWorkflowOrchestrator,
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

    assert calls["remote"]["strict_external_artifact_checks"] is True
    checker = calls["remote"]["external_artifact_checker"]
    assert callable(checker)
    assert "check_ppiflow_external_artifact" in repr(checker)
