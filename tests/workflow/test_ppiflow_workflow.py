"""Tests for the PPIFlow workflow definition."""

# ruff: noqa: D103

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import modal
import orjson
import polars as pl

from biomodals.app.design import ppiflow_app
from biomodals.helper.shell import package_outputs
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow import ppiflow_workflow
from biomodals.workflow.core import NodeRunContext
from biomodals.workflow.ppiflow_workflow import (
    CONF,
    AF3ScoreWorkflowNode,
    DockQWorkflowNode,
    FilterStructuresNode,
    FixedPositionsNode,
    FlowPackerWorkflowNode,
    LigandMPNNWorkflowNode,
    PPIFlowDesignNode,
    PPIFlowModalNamespace,
    PPIFlowPartialWorkflowNode,
    RankAndReportNode,
    RankDesignsNode,
    ReFoldWorkflowNode,
    RosettaWorkflowNode,
    _active_ppiflow_app_steps,
    _ppiflow_rosetta_fix_update_xml,
    _stage_ppiflow_app_inputs,
    build_ppiflow_workflow,
)


class _FakePPIFlowFunction:
    def __init__(self) -> None:
        self.kwargs = {}

    def remote(self, **kwargs):
        self.kwargs = kwargs
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


class _FakeRemoteFunction:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        if callable(self.result):
            return self.result(**kwargs)
        return self.result


class _FakePPIFlowSequenceFunction:
    def __init__(self) -> None:
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ppiflow_outputs",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=ppiflow_app.CONF.output_volume_name,
                        path=kwargs["run_name"],
                    ),
                )
            ],
        )


def _task_yaml(*, enabled_steps: str) -> bytes:
    return f"""
task:
  gentype: binder
steps:
{enabled_steps}
""".encode()


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
    assert CONF.tags == {"depends_on": ".".join(CONF.depends_on_apps)}


def test_ppiflow_full_binder_workflow_matches_upstream_step_order() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(
            enabled_steps="""
  PPIFlowStep: true
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
        steps_yaml_bytes=b"""
PPIFlowStep:
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: /inputs/demo.pdb
    binder_chain: B
MPNNStep_stage1: {}
FlowpackerStep_stage1: {}
AF3scoreStep_stage1: {}
FilterStep_stage1:
  filters:
    iptm: "> 0.7"
RosettaFixStep: {}
PartialStep:
  args:
    name: demo-partial
    specified_hotspots: A1
    input_pdb: /inputs/demo-stage2.pdb
    fixed_positions: B1
    start_t: 0.7
MPNNStep_stage2: {}
FlowpackerStep_stage2: {}
AF3scoreStep_stage2: {}
FilterStep_stage2:
  filters:
    iptm: "> 0.8"
ReFoldStep: {}
DockQStep: {}
RosettaRelaxStep: {}
RankStep: {}
ReportStep: {}
""",
    )

    definition = workflow.validate()

    assert workflow.name == "ppiflow"
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
        "stage2-refold",
        "stage2-dockq",
        "stage2-rosetta-relax",
        "stage2-rank",
        "stage2-report",
    ]
    assert definition.dependencies["stage1-ligandmpnn"] == {"stage1-ppiflow-design"}
    assert definition.dependencies["stage1-flowpacker"] == {"stage1-ligandmpnn"}
    assert definition.dependencies["stage1-af3score"] == {"stage1-flowpacker"}
    assert definition.dependencies["stage1-filter"] == {
        "stage1-af3score",
        "stage1-flowpacker",
    }
    assert definition.dependencies["stage2-rosetta-fix"] == {"stage1-filter"}
    assert definition.dependencies["stage2-fixed-positions"] == {
        "stage1-filter",
        "stage2-rosetta-fix",
    }
    assert definition.dependencies["stage2-partial-ppiflow"] == {
        "stage2-fixed-positions"
    }
    assert definition.dependencies["stage2-ligandmpnn"] == {
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
    }
    assert definition.dependencies["stage2-flowpacker"] == {"stage2-ligandmpnn"}
    assert definition.dependencies["stage2-af3score"] == {"stage2-flowpacker"}
    assert definition.dependencies["stage2-filter"] == {
        "stage2-af3score",
        "stage2-flowpacker",
    }
    assert definition.dependencies["stage2-refold"] == {"stage2-filter"}
    assert definition.dependencies["stage2-dockq"] == {
        "stage2-filter",
        "stage2-refold",
    }
    assert definition.dependencies["stage2-rosetta-relax"] == {"stage2-filter"}
    assert definition.dependencies["stage2-rank"] == {
        "stage2-dockq",
        "stage2-rosetta-relax",
        "stage2-refold",
        "stage2-filter",
    }
    assert definition.dependencies["stage2-report"] == {
        "stage1-af3score",
        "stage2-af3score",
        "stage2-refold",
        "stage2-rank",
    }
    assert {
        node_id: spec.node.__class__.__name__
        for node_id, spec in definition.nodes.items()
    } == {
        "stage1-ppiflow-design": "PPIFlowDesignNode",
        "stage1-ligandmpnn": "LigandMPNNNode",
        "stage1-flowpacker": "FlowPackerNode",
        "stage1-af3score": "AF3ScoreNode",
        "stage1-filter": "FilterStructuresNode",
        "stage2-rosetta-fix": "RosettaFixNode",
        "stage2-fixed-positions": "FixedPositionsNode",
        "stage2-partial-ppiflow": "PPIFlowPartialNode",
        "stage2-ligandmpnn": "LigandMPNNNode",
        "stage2-flowpacker": "FlowPackerNode",
        "stage2-af3score": "AF3ScoreNode",
        "stage2-filter": "FilterStructuresNode",
        "stage2-refold": "ReFoldNode",
        "stage2-dockq": "DockQNode",
        "stage2-rosetta-relax": "RosettaRelaxNode",
        "stage2-rank": "RankNode",
        "stage2-report": "ReportNode",
    }
    assert definition.nodes["stage1-ligandmpnn"].node.config["name"] == "demo"


def test_ppiflow_stage2_only_requires_existing_filtered_inputs() -> None:
    try:
        build_ppiflow_workflow(
            task_yaml_bytes=_task_yaml(
                enabled_steps="""
  RosettaFixStep: true
  PartialStep: true
"""
            ),
            steps_yaml_bytes=b"RosettaFixStep: {}\nPartialStep: {}\n",
            stage=2,
        )
    except ValueError as exc:
        assert "Stage2Inputs.filtered_structures" in str(exc)
    else:
        raise AssertionError("stage 2-only workflow should require upstream inputs")


def test_ppiflow_stage2_only_uses_external_filtered_inputs() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(
            enabled_steps="""
  RosettaFixStep: true
  PartialStep: true
"""
        ),
        steps_yaml_bytes=b"""
Stage2Inputs:
  filtered_structures:
    path: ppiflow/previous/nodes/stage1-filter/cache/filtered_iptm07
RosettaFixStep: {}
PartialStep: {}
""",
        stage=2,
    )

    definition = workflow.validate()

    assert list(definition.nodes) == [
        "stage2-input-filtered",
        "stage2-rosetta-fix",
        "stage2-fixed-positions",
        "stage2-partial-ppiflow",
    ]
    assert definition.dependencies["stage2-input-filtered"] == set()
    assert definition.dependencies["stage2-rosetta-fix"] == {"stage2-input-filtered"}
    assert definition.dependencies["stage2-fixed-positions"] == {
        "stage2-input-filtered",
        "stage2-rosetta-fix",
    }


def test_ppiflow_app_step_uses_included_modal_namespace(tmp_path: Path) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = PPIFlowModalNamespace(
        ppiflow_run=cast(modal.Function, fake_function),
    )
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


def test_ppiflow_step_merges_task_fields_into_app_args(tmp_path: Path) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = PPIFlowModalNamespace(
        ppiflow_run=cast(modal.Function, fake_function),
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=b"""
task:
  name: demo
  gentype: binder
  input_pdb: /inputs/demo.pdb
  specified_hotspots: B25
steps:
  PPIFlowStep: true
""",
        steps_yaml_bytes=b"""
PPIFlowStep:
  config: /configs/inference_binder.yaml
""",
        modal_namespace=namespace,
    )

    spec = workflow.validate().nodes["stage1-ppiflow-design"]
    result = spec.node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-ppiflow-design",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    app_args = fake_function.kwargs["args"].args
    assert result.status == AppRunStatus.SUCCEEDED
    assert isinstance(app_args, ppiflow_app.SampleBinderConfig)
    assert app_args.name == "demo"
    assert str(app_args.input_pdb) == "/inputs/demo.pdb"
    assert app_args.target_chain == "B"
    assert app_args.binder_chain == "A"
    assert app_args.specified_hotspots == "B25"
    assert str(app_args.config) == "/configs/inference_binder.yaml"


def test_ppiflow_step_stages_app_volume_outputs_into_workflow_volume(
    tmp_path: Path,
) -> None:
    fake_ppiflow = _FakePPIFlowFunction()
    fake_stage_outputs = _FakeRemoteFunction([
        {
            "index": 0,
            "volume_name": "Workflow-outputs",
            "path": "ppiflow_app_outputs/run-1/stage1-ppiflow-design/ppiflow_outputs",
        }
    ])
    node = PPIFlowDesignNode(
        step_name="PPIFlowStep",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, fake_ppiflow),
            ppiflow_stage_outputs=cast(modal.Function, fake_stage_outputs),
        ),
        config={
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": "/inputs/demo.pdb",
                "binder_chain": "B",
            }
        },
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-ppiflow-design",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    assert fake_stage_outputs.calls == [
        {
            "run_id": "run-1",
            "node_id": "stage1-ppiflow-design",
            "outputs": [
                {
                    "index": 0,
                    "name": "ppiflow_outputs",
                    "volume_name": ppiflow_app.CONF.output_volume_name,
                    "path": "demo-run",
                }
            ],
        }
    ]
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="ppiflow_app_outputs/run-1/stage1-ppiflow-design/ppiflow_outputs",
    )


def test_ppiflow_flowpacker_step_has_workflow_adapter(tmp_path: Path) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = PPIFlowModalNamespace(
        ppiflow_run=cast(modal.Function, fake_function),
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  FlowpackerStep_stage1: true\n"),
        steps_yaml_bytes=b"FlowpackerStep_stage1: {}\n",
        modal_namespace=namespace,
    )

    spec = workflow.validate().nodes["stage1-flowpacker"]
    if spec.node.__class__.__name__ == "PPIFlowWorkflowNode":
        raise AssertionError("FlowpackerStep_stage1 must not use the PPIFlow adapter")
    if hasattr(spec.node, "run"):
        return
    else:
        spec.node.run(
            NodeRunContext(
                run_id="run-1",
                node_id=spec.node_id,
                attempt_id="attempt-1",
                cache_dir=tmp_path,
                inputs={},
            )
        )


def test_ppiflow_ligandmpnn_step_runs_once_per_input_pdb_with_bytes(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    structures_dir = volume_root / "ppiflow" / "mpnn_pdbs"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "ligandmpnn" / "cache"
    structures_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    pdb_bytes = b"ATOM design_a\n"
    structures_dir.joinpath("design_a.pdb").write_bytes(pdb_bytes)
    ligandmpnn_archive_root = tmp_path / "ligandmpnn-archive"
    seqs_dir = ligandmpnn_archive_root / "outputs" / "seed-3" / "seqs"
    seqs_dir.mkdir(parents=True)
    seqs_dir.joinpath("design_a.fa").write_text(
        ">native\nAAAA/BBBB\n>design1\nCCCC/DDDD\n",
        encoding="utf-8",
    )
    archive_bytes = package_outputs(ligandmpnn_archive_root)
    fake_ligandmpnn = _FakeRemoteFunction(archive_bytes)
    node = LigandMPNNWorkflowNode(
        step_name="MPNNStep_stage1",
        gentype="binder",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            ligandmpnn_run=cast(modal.Function, fake_ligandmpnn),
        ),
        config={
            "name": "demo",
            "seeds": "3,5",
            "num_seq_per_target": 8,
            "sampling_temp": 0.2,
            "batch_size": 2,
            "chains_to_design": "A",
        },
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-ligandmpnn",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="ppiflow-structures",
                        producing_node_id="stage1-ppiflow-design",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="ppiflow/mpnn_pdbs",
                        ),
                    )
                ]
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(fake_ligandmpnn.calls) == 1
    call = fake_ligandmpnn.calls[0]
    assert call["run_name"] == "run-1-MPNNStep_stage1-design_a"
    assert call["script_mode"] == "run"
    assert call["struct_bytes"] == pdb_bytes
    assert call["seeds"] == [3, 5]
    assert call["cli_args"]["--number_of_batches"] == "8"
    assert call["cli_args"]["--temperature"] == "0.2"
    assert call["cli_args"]["--chains_to_design"] == "A"
    archive_path = cache_dir / "ligandmpnn_outputs" / "design_a.tar.zst"
    assert archive_path.read_bytes() == archive_bytes
    mpnn_pdb = cache_dir / "ligandmpnn_outputs" / "mpnn_pdbs" / "demo_design_a.pdb"
    assert mpnn_pdb.is_symlink()
    assert mpnn_pdb.resolve() == (structures_dir / "design_a.pdb").resolve()
    assert (
        cache_dir.joinpath(
            "ligandmpnn_outputs", "mpnn_pdbs", "mpnn_seqs.csv"
        ).read_text(encoding="utf-8")
        == "link_name,sequence_dict,seq_idx\ndemo_design_a.pdb,\"{'A': 'CCCC', 'B': 'DDDD'}\",1\n"
    )
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/ligandmpnn/cache/ligandmpnn_outputs",
    )


def test_ppiflow_stage2_ligandmpnn_uses_fixed_positions_per_partial_pdb(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    structures_dir = volume_root / "partial_outputs"
    fixed_dir = volume_root / "fixed"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "ligandmpnn" / "cache"
    structures_dir.mkdir(parents=True)
    fixed_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    structures_dir.joinpath("cd3d_cd3d_3").mkdir()
    structures_dir.joinpath("cd3d_cd3d_4").mkdir()
    structures_dir.joinpath("cd3d_cd3d_3", "sample_a.pdb").write_bytes(
        b"ATOM design fixed\n"
    )
    structures_dir.joinpath("cd3d_cd3d_4", "sample_b.pdb").write_bytes(
        b"ATOM design base\n"
    )
    fixed_dir.joinpath("fixed_positions.csv").write_text(
        'filename,fixed_positions\ncd3d_cd3d_3.pdb,"A9,A16,A41"\n'
        "cd3d_cd3d_4.pdb,NONE\n",
        encoding="utf-8",
    )

    def ligandmpnn_archive(*, run_name: str, **_kwargs):
        pdb_stem = run_name.rsplit("-", 1)[-1]
        archive_root = tmp_path / f"{pdb_stem}-archive"
        seqs_dir = archive_root / "outputs" / "seed-0" / "seqs"
        seqs_dir.mkdir(parents=True)
        seqs_dir.joinpath(f"{pdb_stem}.fa").write_text(
            ">native\nAAAA/BBBB\n>design1\nCCCC/DDDD\n",
            encoding="utf-8",
        )
        return package_outputs(archive_root)

    fake_ligandmpnn = _FakeRemoteFunction(ligandmpnn_archive)
    node = LigandMPNNWorkflowNode(
        step_name="MPNNStep_stage2",
        gentype="binder",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            ligandmpnn_run=cast(modal.Function, fake_ligandmpnn),
        ),
        config={"seeds": "0", "num_seq_per_target": 1},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-ligandmpnn",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="partial-structures",
                        producing_node_id="stage2-partial-ppiflow",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="partial_outputs",
                        ),
                    )
                ],
                "fixed_positions": [
                    WorkflowArtifact(
                        artifact_id="fixed-positions",
                        producing_node_id="stage2-fixed-positions",
                        kind=ArtifactKind.TABLE,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="fixed/fixed_positions.csv",
                        ),
                    )
                ],
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(fake_ligandmpnn.calls) == 2
    calls_by_input = {
        call["run_name"].removeprefix("run-1-MPNNStep_stage2-"): call
        for call in fake_ligandmpnn.calls
    }
    assert calls_by_input["sample_a"]["cli_args"]["--fixed_residues"] == "9 16 41"
    assert "--fixed_residues" not in calls_by_input["sample_b"]["cli_args"]
    mpnn_pdbs_dir = cache_dir / "ligandmpnn_outputs" / "mpnn_pdbs"
    fixed_link = mpnn_pdbs_dir / "cd3d_cd3d_3_sample_a.pdb"
    base_link = mpnn_pdbs_dir / "cd3d_cd3d_4_sample_b.pdb"
    assert fixed_link.is_symlink()
    assert (
        fixed_link.resolve()
        == structures_dir.joinpath("cd3d_cd3d_3", "sample_a.pdb").resolve()
    )
    assert base_link.is_symlink()
    assert (
        base_link.resolve()
        == structures_dir.joinpath("cd3d_cd3d_4", "sample_b.pdb").resolve()
    )
    assert (
        mpnn_pdbs_dir.joinpath("mpnn_seqs.csv").read_text(encoding="utf-8")
        == "link_name,sequence_dict,seq_idx\n"
        "cd3d_cd3d_3_sample_a.pdb,\"{'A': 'CCCC', 'B': 'DDDD'}\",1\n"
        "cd3d_cd3d_4_sample_b.pdb,\"{'A': 'CCCC', 'B': 'DDDD'}\",1\n"
    )


def test_ppiflow_abmpnn_step_uses_checkpoint_chain_list_and_raw_fixed_positions(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    structures_dir = volume_root / "partial_outputs"
    fixed_dir = volume_root / "fixed"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "abmpnn" / "cache"
    structures_dir.mkdir(parents=True)
    fixed_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    structures_dir.joinpath("antibody_design.pdb").write_bytes(b"ATOM antibody\n")
    fixed_dir.joinpath("fixed_positions.csv").write_text(
        'filename,fixed_positions\nantibody_design.pdb,"H26,H27,L50"\n',
        encoding="utf-8",
    )

    archive_root = tmp_path / "abmpnn-archive"
    seqs_dir = archive_root / "outputs" / "seed-0" / "seqs"
    seqs_dir.mkdir(parents=True)
    seqs_dir.joinpath("antibody_design.fa").write_text(
        ">native\nHHHH/LLLL/CCCC\n>design1\nHHAA/LLAA/CCAA\n",
        encoding="utf-8",
    )
    fake_ligandmpnn = _FakeRemoteFunction(package_outputs(archive_root))
    node = LigandMPNNWorkflowNode(
        step_name="AbMPNNStep_stage2",
        gentype="antibody",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            ligandmpnn_run=cast(modal.Function, fake_ligandmpnn),
        ),
        config={
            "chains_to_design": None,
            "chain_list": "H,L",
            "num_seq_per_target": 2,
            "seeds": "0",
        },
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-abmpnn",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="partial-structures",
                        producing_node_id="stage2-partial-ppiflow",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="partial_outputs",
                        ),
                    )
                ],
                "fixed_positions": [
                    WorkflowArtifact(
                        artifact_id="fixed-positions",
                        producing_node_id="stage2-fixed-positions",
                        kind=ArtifactKind.TABLE,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="fixed/fixed_positions.csv",
                        ),
                    )
                ],
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(fake_ligandmpnn.calls) == 1
    cli_args = fake_ligandmpnn.calls[0]["cli_args"]
    assert cli_args["--model_type"] == "protein_mpnn"
    assert cli_args["--checkpoint_protein_mpnn"].endswith("/model_params/abmpnn.pt")
    assert cli_args["--chains_to_design"] == "H,L"
    assert cli_args["--fixed_residues"] == "H26,H27,L50"


def test_ppiflow_flowpacker_step_consumes_mpnn_pdbs_and_extracts_archive(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    ligandmpnn_dir = volume_root / "workflow" / "run-1" / "nodes" / "ligandmpnn"
    mpnn_pdbs_dir = ligandmpnn_dir / "mpnn_pdbs"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "flowpacker" / "cache"
    mpnn_pdbs_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    pdb_bytes = b"ATOM flowpacked input\n"
    mpnn_pdbs_dir.joinpath("design_a.pdb").write_bytes(pdb_bytes)
    mpnn_pdbs_dir.joinpath("mpnn_seqs.csv").write_text(
        "link_name,sequence_dict,seq_idx\n"
        "design_a.pdb,\"{'A': 'CCCC', 'B': 'DDDD'}\",1\n",
        encoding="utf-8",
    )

    flowpacker_archive_root = tmp_path / "flowpacker-archive"
    flowpacker_archive_root.joinpath("run_1").mkdir(parents=True)
    flowpacker_archive_root.joinpath("run_1", "design_a.pdb").write_bytes(
        b"ATOM flowpacked output\n"
    )
    fake_flowpacker = _FakeRemoteFunction(package_outputs(flowpacker_archive_root))
    node = FlowPackerWorkflowNode(
        step_name="FlowpackerStep_stage1",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            flowpacker_run=cast(modal.Function, fake_flowpacker),
        ),
        config={
            "model_name": "bc40",
            "use_confidence": True,
            "n_samples": 2,
            "num_steps": 8,
            "sample_coeff": 4.5,
            "use_gt_masks": True,
            "seed": 7,
        },
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-flowpacker",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="ligandmpnn-structures",
                        producing_node_id="stage1-ligandmpnn",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="workflow/run-1/nodes/ligandmpnn",
                        ),
                    )
                ]
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(fake_flowpacker.calls) == 1
    call = fake_flowpacker.calls[0]
    assert call["run_name"] == "run-1-FlowpackerStep_stage1"
    assert call["input_files"] == [("design_a.pdb", pdb_bytes)]
    assert call["model_name"] == "bc40"
    assert call["use_confidence"] is True
    assert call["n_samples"] == 2
    assert call["num_steps"] == 8
    assert call["sample_coeff"] == 4.5
    assert call["use_gt_masks"] is True
    assert call["seed"] == 7
    assert cache_dir.joinpath(
        "flowpacker_outputs",
        "run-1-FlowpackerStep_stage1.tar.zst",
    ).is_file()
    assert (
        cache_dir.joinpath(
            "flowpacker_outputs",
            "extracted",
            "run_1",
            "design_a.pdb",
        ).read_bytes()
        == b"ATOM flowpacked output\n"
    )
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/flowpacker/cache/flowpacker_outputs/extracted",
    )


def test_ppiflow_flowpacker_step_accepts_volume_path_archive_result(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    ligandmpnn_dir = volume_root / "workflow" / "run-1" / "nodes" / "ligandmpnn"
    mpnn_pdbs_dir = ligandmpnn_dir / "mpnn_pdbs"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "flowpacker" / "cache"
    mpnn_pdbs_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    mpnn_pdbs_dir.joinpath("design_a.pdb").write_bytes(b"ATOM input\n")
    mpnn_pdbs_dir.joinpath("mpnn_seqs.csv").write_text(
        "link_name,sequence_dict,seq_idx\n"
        "design_a.pdb,\"{'A': 'CCCC', 'B': 'DDDD'}\",1\n",
        encoding="utf-8",
    )
    flowpacker_archive_root = tmp_path / "flowpacker-archive"
    flowpacker_archive_root.joinpath("run_1").mkdir(parents=True)
    flowpacker_archive_root.joinpath("run_1", "design_a.pdb").write_bytes(
        b"ATOM flowpacked output\n"
    )
    archive_bytes = package_outputs(flowpacker_archive_root)
    fake_flowpacker = _FakeRemoteFunction(
        AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="flowpacker_outputs",
                    kind=ArtifactKind.ARCHIVE,
                    storage=VolumePath(
                        volume_name="FlowPacker-outputs",
                        path="workflow/run-1-FlowpackerStep_stage1/archive.tar.zst",
                    ),
                )
            ],
        )
    )
    fake_archive_stage = _FakeRemoteFunction(archive_bytes)
    node = FlowPackerWorkflowNode(
        step_name="FlowpackerStep_stage1",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            flowpacker_run=cast(modal.Function, fake_flowpacker),
            flowpacker_stage_archive=cast(modal.Function, fake_archive_stage),
        ),
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-flowpacker",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="ligandmpnn-structures",
                        producing_node_id="stage1-ligandmpnn",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="workflow/run-1/nodes/ligandmpnn",
                        ),
                    )
                ]
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert fake_archive_stage.calls == [
        {
            "source": {
                "volume_name": "FlowPacker-outputs",
                "path": "workflow/run-1-FlowpackerStep_stage1/archive.tar.zst",
            },
        }
    ]
    assert (
        cache_dir.joinpath(
            "flowpacker_outputs",
            "extracted",
            "run_1",
            "design_a.pdb",
        ).read_bytes()
        == b"ATOM flowpacked output\n"
    )


def test_ppiflow_refold_step_runs_alphafold3_handles(tmp_path: Path) -> None:
    volume_root = tmp_path / "workflow-volume"
    input_dir = volume_root / "filtered"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "refold" / "cache"
    input_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    input_dir.joinpath("demo.pdb").write_text(
        "\n".join([
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
            "ATOM      2  CA  CYS A   2       0.000   0.000   0.000  1.00 20.00           C",
            "ATOM      3  CA  ASP B   1       0.000   0.000   0.000  1.00 20.00           C",
        ])
        + "\n",
        encoding="utf-8",
    )
    artifact = WorkflowArtifact(
        artifact_id="filtered-structures",
        producing_node_id="stage2-filter",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(volume_name="Workflow-outputs", path="filtered"),
    )

    def data_pipeline_result(json_bytes):
        data = orjson.loads(json_bytes)
        for sequence in data["sequences"]:
            sequence["protein"]["templates"] = [{"template": "present"}]
        return orjson.dumps(data)

    fake_data = _FakeRemoteFunction(data_pipeline_result)
    fake_inference = _FakeRemoteFunction(b"archive bytes")
    node = ReFoldWorkflowNode(
        step_name="ReFoldStep",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            alphafold3_data=cast(modal.Function, fake_data),
            alphafold3_inference=cast(modal.Function, fake_inference),
        ),
        config={
            "seed_num": "2",
            "remove_template": True,
            "recycle": 3,
            "sample": 4,
        },
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-refold",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={"structures": [artifact]},
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(fake_data.calls) == 1
    assert len(fake_inference.calls) == 1
    generated_json = orjson.loads(fake_data.calls[0]["json_bytes"])
    assert generated_json["name"] == "demo"
    assert generated_json["modelSeeds"] == [1, 10]
    assert generated_json["sequences"] == [
        {
            "protein": {
                "id": "A",
                "sequence": "AC",
                "unpairedMsa": None,
                "pairedMsa": None,
                "templates": None,
            }
        },
        {
            "protein": {
                "id": "B",
                "sequence": "D",
                "unpairedMsa": None,
                "pairedMsa": None,
                "templates": None,
            }
        },
    ]
    inference_json = orjson.loads(fake_inference.calls[0]["json_bytes"])
    assert inference_json["sequences"][0]["protein"]["templates"] == []
    assert fake_inference.calls[0]["model_seeds"] == [1, 10]
    assert fake_inference.calls[0]["recycle"] == 3
    assert fake_inference.calls[0]["sample"] == 4
    assert cache_dir.joinpath("refold_outputs", "demo.tar.zst").read_bytes() == (
        b"archive bytes"
    )
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/refold/cache/refold_outputs",
    )


def test_ppiflow_refold_step_writes_af3_iptm_metrics_csv(tmp_path: Path) -> None:
    volume_root = tmp_path / "workflow-volume"
    input_dir = volume_root / "filtered"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "refold" / "cache"
    input_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    input_dir.joinpath("demo.pdb").write_text(
        "\n".join([
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
            "ATOM      2  CA  CYS B   1       0.000   0.000   0.000  1.00 20.00           C",
        ])
        + "\n",
        encoding="utf-8",
    )
    inference_archive_root = tmp_path / "af3-archive" / "demo"
    summary_dir = inference_archive_root / "seed-1_sample-0"
    summary_dir.mkdir(parents=True)
    summary_dir.joinpath("summary_confidences.json").write_bytes(
        orjson.dumps({
            "ptm": 0.65,
            "iptm": 0.85,
            "chain_ptm": [0.7, 0.6],
            "chain_iptm": [0.8, 0.75],
            "chain_pair_iptm": [[1.0, 0.82], [0.82, 1.0]],
        })
    )
    summary_dir.joinpath("model.cif").write_text("data_demo\n", encoding="utf-8")
    fake_data = _FakeRemoteFunction(lambda json_bytes: json_bytes)
    fake_inference = _FakeRemoteFunction(package_outputs(inference_archive_root))
    node = ReFoldWorkflowNode(
        step_name="ReFoldStep",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            alphafold3_data=cast(modal.Function, fake_data),
            alphafold3_inference=cast(modal.Function, fake_inference),
        ),
        config={"seed_num": "1"},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-refold",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="filtered-structures",
                        producing_node_id="stage2-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="filtered",
                        ),
                    )
                ]
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    metrics_csv = cache_dir / "refold_outputs" / "af3_iptm@1.csv"
    assert metrics_csv.read_text(encoding="utf-8") == (
        "pdb_name,seed,ptm,iptm,chain_A_ptm,chain_A_iptm,"
        "chain_B_ptm,chain_B_iptm,iptm_A_B\n"
        "demo,seed-1_sample-0,0.65,0.85,0.7,0.8,0.6,0.75,0.82\n"
    )


def test_ppiflow_af3score_step_returns_metrics_csv_volume_path(
    tmp_path: Path,
) -> None:
    fake_prepare = _FakeRemoteFunction(
        SimpleNamespace(input_files=["design_a.pdb"], chunk_specs=[])
    )
    fake_postprocess = _FakeRemoteFunction({
        "metrics_csv_exists": 1,
        "metrics_csv": "/mnt/AF3Score-outputs/run-1/af3score_metrics.csv",
        "metrics_rows": 1,
    })
    node = AF3ScoreWorkflowNode(
        "AF3scoreStep_stage1",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            af3score_prepare=cast(modal.Function, fake_prepare),
            af3score_postprocess=cast(modal.Function, fake_postprocess),
        ),
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-af3score",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].kind == ArtifactKind.SCORES
    assert result.outputs[0].storage == VolumePath(
        volume_name="AF3Score-outputs",
        path="run-1/af3score_metrics.csv",
    )


def test_ppiflow_af3score_step_stages_metrics_csv_to_workflow_volume(
    tmp_path: Path,
) -> None:
    fake_prepare = _FakeRemoteFunction(
        SimpleNamespace(input_files=["design_a.pdb"], chunk_specs=[])
    )
    fake_postprocess = _FakeRemoteFunction({
        "metrics_csv_exists": 1,
        "metrics_csv": "/mnt/AF3Score-outputs/run-1/af3score_metrics.csv",
        "metrics_rows": 1,
    })
    fake_stage_metrics = _FakeRemoteFunction({
        "volume_name": "Workflow-outputs",
        "path": "ppiflow_af3score/run-1-AF3scoreStep_stage1/af3score_metrics.csv",
    })
    node = AF3ScoreWorkflowNode(
        "AF3scoreStep_stage1",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            af3score_prepare=cast(modal.Function, fake_prepare),
            af3score_postprocess=cast(modal.Function, fake_postprocess),
            af3score_stage_metrics=cast(modal.Function, fake_stage_metrics),
        ),
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-af3score",
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    assert fake_stage_metrics.calls == [
        {
            "run_name": "run-1-AF3scoreStep_stage1",
            "metrics_csv": "/mnt/AF3Score-outputs/run-1/af3score_metrics.csv",
        }
    ]
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="ppiflow_af3score/run-1-AF3scoreStep_stage1/af3score_metrics.csv",
    )


def test_ppiflow_af3score_step_stages_flowpacker_pdbs_before_prepare(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    flowpacker_dir = (
        volume_root
        / "workflow"
        / "run-1"
        / "nodes"
        / "flowpacker"
        / "cache"
        / "flowpacker_outputs"
        / "extracted"
        / "run_1"
    )
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "af3score" / "cache"
    flowpacker_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    flowpacker_dir.joinpath("design_a.pdb").write_bytes(b"ATOM flowpacked output\n")
    fake_stage = _FakeRemoteFunction(["design_a.pdb"])
    fake_prepare = _FakeRemoteFunction(
        SimpleNamespace(
            input_files=["design_a.pdb"],
            chunk_specs=[
                SimpleNamespace(
                    batch_name="batch_0",
                    batch_json_dir="/mnt/AF3Score-outputs/run-1/prepare/json/batch_0",
                    batch_pdb_dir="/mnt/AF3Score-outputs/run-1/prepare/pdb/batch_0",
                )
            ],
        )
    )
    fake_run = _FakeRemoteFunction(None)
    fake_postprocess = _FakeRemoteFunction({
        "metrics_csv_exists": 1,
        "metrics_csv": "/mnt/AF3Score-outputs/run-1/af3score_metrics.csv",
        "metrics_rows": 1,
    })
    node = AF3ScoreWorkflowNode(
        "AF3scoreStep_stage1",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            af3score_stage_inputs=cast(modal.Function, fake_stage),
            af3score_prepare=cast(modal.Function, fake_prepare),
            af3score_run=cast(modal.Function, fake_run),
            af3score_postprocess=cast(modal.Function, fake_postprocess),
        ),
        config={"num_jobs": 3, "prepare_workers": 5},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-af3score",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="flowpacker-structures",
                        producing_node_id="stage1-flowpacker",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path=(
                                "workflow/run-1/nodes/flowpacker/cache/"
                                "flowpacker_outputs/extracted"
                            ),
                        ),
                    )
                ]
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert fake_stage.calls == [
        {
            "run_name": "run-1-AF3scoreStep_stage1",
            "source_paths": [
                (
                    "workflow/run-1/nodes/flowpacker/cache/"
                    "flowpacker_outputs/extracted/run_1/design_a.pdb"
                )
            ],
        }
    ]
    assert fake_prepare.calls[0]["run_name"] == "run-1-AF3scoreStep_stage1"
    assert fake_prepare.calls[0]["input_files"] == ["design_a.pdb"]
    assert fake_prepare.calls[0]["num_jobs"] == 3
    assert fake_prepare.calls[0]["prepare_workers"] == 5
    assert fake_run.calls == [
        {
            "run_name": "run-1-AF3scoreStep_stage1",
            "batch_name": "batch_0",
            "batch_json_dir": "/mnt/AF3Score-outputs/run-1/prepare/json/batch_0",
            "batch_pdb_dir": "/mnt/AF3Score-outputs/run-1/prepare/pdb/batch_0",
        }
    ]
    assert fake_postprocess.calls[0]["input_files"] == ["design_a.pdb"]


def test_ppiflow_dockq_step_pairs_refold_models_with_filtered_references(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    reference_dir = volume_root / "filtered_iptm08"
    model_dir = volume_root / "refold_outputs" / "models"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "dockq" / "cache"
    reference_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    reference_bytes = b"ATOM reference\n"
    model_bytes = b"ATOM model\n"
    reference_dir.joinpath("design_a.pdb").write_bytes(reference_bytes)
    model_dir.joinpath("design_a.pdb").write_bytes(model_bytes)

    dockq_archive_root = tmp_path / "dockq-archive"
    dockq_archive_root.mkdir()
    dockq_archive_root.joinpath("dockq_results.csv").write_text(
        "id,model,reference,dockq\ndesign_a,design_a.pdb,design_a.pdb,0.9\n",
        encoding="utf-8",
    )
    fake_dockq = _FakeRemoteFunction(package_outputs(dockq_archive_root))
    node = DockQWorkflowNode(
        "DockQStep",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            dockq_run=cast(modal.Function, fake_dockq),
        ),
        config={"dockq_args": ["--short", "--capri_peptide"]},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-dockq",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="filtered-structures",
                        producing_node_id="stage2-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="filtered_iptm08",
                        ),
                    )
                ],
                "models": [
                    WorkflowArtifact(
                        artifact_id="refold-structures",
                        producing_node_id="stage2-refold",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="refold_outputs/models",
                        ),
                    )
                ],
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert fake_dockq.calls == [
        {
            "pairs": [
                {
                    "id": "design_a",
                    "model_name": "design_a.pdb",
                    "model_bytes": model_bytes,
                    "reference_name": "design_a.pdb",
                    "reference_bytes": reference_bytes,
                }
            ],
            "run_name": "run-1-DockQStep",
            "dockq_args": ["--short", "--capri_peptide"],
        }
    ]
    assert cache_dir.joinpath("dockq_outputs", "dockq_results.csv").read_text(
        encoding="utf-8"
    ) == ("id,model,reference,dockq\ndesign_a,design_a.pdb,design_a.pdb,0.9\n")
    assert result.outputs[0].kind == ArtifactKind.SCORES
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/dockq/cache/dockq_outputs/dockq_results.csv",
    )


def test_ppiflow_rank_step_merges_scores_and_copies_selected_pdbs(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    filtered_dir = volume_root / "filtered_iptm08"
    dockq_dir = volume_root / "dockq_outputs"
    rosetta_dir = volume_root / "rosetta_relax_output"
    refold_dir = volume_root / "refold_outputs"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "rank" / "cache"
    for directory in (filtered_dir, dockq_dir, rosetta_dir, refold_dir, cache_dir):
        directory.mkdir(parents=True)
    selected_pdb = "\n".join([
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  GLY A   2       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  TYR B   1       0.000   0.000   0.000  1.00 20.00           C",
    ])
    filtered_dir.joinpath("design_a.pdb").write_text(
        selected_pdb + "\n",
        encoding="utf-8",
    )
    filtered_dir.joinpath("design_b.pdb").write_text(
        selected_pdb.replace("ALA", "SER") + "\n",
        encoding="utf-8",
    )
    dockq_dir.joinpath("dockq_results.csv").write_text(
        "id,model,reference,dockq,returncode\n"
        "design_a,design_a.pdb,design_a.pdb,0.9,0\n"
        "design_b,design_b.pdb,design_b.pdb,0.1,0\n",
        encoding="utf-8",
    )
    rosetta_dir.joinpath("rosetta_complex_0.csv").write_text(
        "pdb_name,interface_score\ndesign_a,-10\ndesign_b,-1\n",
        encoding="utf-8",
    )
    refold_dir.joinpath("af3_iptm@1.csv").write_text(
        "pdb_name,iptm,chain_A_ptm\ndesign_a,0.85,0.7\ndesign_b,0.95,0.9\n",
        encoding="utf-8",
    )
    node = RankDesignsNode(
        "RankStep",
        gentype="binder",
        config={"dockq_threshold": 0.49},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-rank",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="filtered-structures",
                        producing_node_id="stage2-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="filtered_iptm08",
                        ),
                    )
                ],
                "dockq": [
                    WorkflowArtifact(
                        artifact_id="dockq-results",
                        producing_node_id="stage2-dockq",
                        kind=ArtifactKind.SCORES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="dockq_outputs/dockq_results.csv",
                        ),
                    )
                ],
                "rosetta": [
                    WorkflowArtifact(
                        artifact_id="rosetta-relax",
                        producing_node_id="stage2-rosetta-relax",
                        kind=ArtifactKind.DIRECTORY,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="rosetta_relax_output",
                        ),
                    )
                ],
                "refold": [
                    WorkflowArtifact(
                        artifact_id="refold-metrics",
                        producing_node_id="stage2-refold",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="refold_outputs",
                        ),
                    )
                ],
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    ranked_csv = cache_dir / "design_output" / "ranked_designs.csv"
    assert ranked_csv.read_text(encoding="utf-8") == (
        "pdb_name,iptm,chain_A_ptm,target_name,reference_pdb,model_pdb,"
        "dockq_mean,interface_score,rank_score,binder_seq\n"
        "design_a,0.85,0.7,design_a,design_a.pdb,design_a.pdb,"
        "0.9,-10.0,75.0,AG\n"
    )
    assert (cache_dir / "design_output" / "pdbs" / "design_a.pdb").read_text(
        encoding="utf-8"
    ) == selected_pdb + "\n"
    assert not (cache_dir / "design_output" / "pdbs" / "design_b.pdb").exists()
    assert result.outputs[0].kind == ArtifactKind.TABLE
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/rank/cache/design_output/ranked_designs.csv",
    )


def test_ppiflow_report_step_generates_html_design_report(tmp_path: Path) -> None:
    volume_root = tmp_path / "workflow-volume"
    rank_output_dir = volume_root / "design_output"
    stage1_dir = volume_root / "stage1_af3score"
    stage2_dir = volume_root / "stage2_af3score"
    refold_dir = volume_root / "refold_outputs"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "report" / "cache"
    for directory in (
        rank_output_dir / "pdbs",
        stage1_dir,
        stage2_dir,
        refold_dir,
        cache_dir,
    ):
        directory.mkdir(parents=True)
    rank_output_dir.joinpath("ranked_designs.csv").write_text(
        "pdb_name,rank_score,interface_score,dockq_mean,binder_seq,iptm\n"
        "design_a,75.0,-10.0,0.9,AG,0.85\n",
        encoding="utf-8",
    )
    rank_output_dir.joinpath("pdbs", "design_a.pdb").write_text(
        "ATOM report top pdb\n",
        encoding="utf-8",
    )
    stage1_dir.joinpath("af3score_metrics.csv").write_text(
        "description,iptm\ns1_a.pdb,0.72\n",
        encoding="utf-8",
    )
    stage2_dir.joinpath("af3score_metrics.csv").write_text(
        "description,iptm\ns2_a.pdb,0.82\n",
        encoding="utf-8",
    )
    refold_dir.joinpath("af3_iptm@1.csv").write_text(
        "pdb_name,iptm,chain_A_ptm\ndesign_a,0.85,0.7\n",
        encoding="utf-8",
    )
    node = RankAndReportNode(
        "ReportStep",
        gentype="binder",
        config={"name": "demo-design"},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-report",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "ranked": [
                    WorkflowArtifact(
                        artifact_id="ranked-designs",
                        producing_node_id="stage2-rank",
                        kind=ArtifactKind.TABLE,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="design_output/ranked_designs.csv",
                        ),
                    )
                ],
                "stage1_af3score": [
                    WorkflowArtifact(
                        artifact_id="stage1-af3score",
                        producing_node_id="stage1-af3score",
                        kind=ArtifactKind.SCORES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="stage1_af3score/af3score_metrics.csv",
                        ),
                    )
                ],
                "stage2_af3score": [
                    WorkflowArtifact(
                        artifact_id="stage2-af3score",
                        producing_node_id="stage2-af3score",
                        kind=ArtifactKind.SCORES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="stage2_af3score/af3score_metrics.csv",
                        ),
                    )
                ],
                "refold": [
                    WorkflowArtifact(
                        artifact_id="refold-metrics",
                        producing_node_id="stage2-refold",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="refold_outputs",
                        ),
                    )
                ],
            },
        )
    )

    report_html = rank_output_dir / "design_report.html"
    html = report_html.read_text(encoding="utf-8")
    assert "PPIFlow Design Report" in html
    assert "demo-design" in html
    assert "RankStep" in html
    assert "design_a" in html
    assert "75.0" in html
    assert "AG" in html
    assert result.outputs[0].kind == ArtifactKind.REPORT
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="design_output/design_report.html",
        media_type="text/html",
    )


def test_ppiflow_rosetta_step_stages_runs_and_extracts_outputs(tmp_path: Path) -> None:
    volume_root = tmp_path / "workflow-volume"
    filtered_dir = volume_root / "filtered_iptm08"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "rosetta" / "cache"
    filtered_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    filtered_dir.joinpath("design_a.pdb").write_bytes(b"ATOM filtered\n")
    rosetta_archive_root = tmp_path / "rosetta-archive"
    rosetta_archive_root.joinpath("interface_energy_A_B").mkdir(parents=True)
    rosetta_archive_root.joinpath(
        "interface_energy_A_B",
        "residue_energy.csv",
    ).write_text(
        'pdbname,binder_energy\ndesign_a_1,"{1: -6.0}"\n',
        encoding="utf-8",
    )
    fake_stage = _FakeRemoteFunction({
        "run_name": "run-1-RosettaFixStep",
        "run_id": "stage2-rosetta-fix-attempt-1",
        "num_cpu_per_pod": 1,
        "root": "/mnt/Rosetta-outputs/run-1-RosettaFixStep-stage2-rosetta-fix-attempt-1",
    })
    fake_run = _FakeRemoteFunction(None)
    fake_package = _FakeRemoteFunction(package_outputs(rosetta_archive_root))
    node = RosettaWorkflowNode(
        "RosettaFixStep",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            rosetta_stage_tasks=cast(modal.Function, fake_stage),
            rosetta_run=cast(modal.Function, fake_run),
            rosetta_package=cast(modal.Function, fake_package),
        ),
        config={"num_cpu_per_pod": 2, "rosetta_bin": "rosetta_scripts"},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-rosetta-fix",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="filtered-structures",
                        producing_node_id="stage2-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="filtered_iptm08",
                        ),
                    )
                ]
            },
        )
    )

    assert fake_stage.calls == [
        {
            "run_name": "run-1-RosettaFixStep",
            "run_id": "stage2-rosetta-fix-attempt-1",
            "source_paths": ["filtered_iptm08/design_a.pdb"],
            "step_name": "RosettaFixStep",
            "config": {
                "gentype": "binder",
                "num_cpu_per_pod": 2,
                "rosetta_bin": "rosetta_scripts",
            },
        }
    ]
    assert fake_run.calls == [
        {
            "run_name": "run-1-RosettaFixStep",
            "run_id": "stage2-rosetta-fix-attempt-1",
            "num_cpu_per_pod": 1,
        }
    ]
    assert fake_package.calls == [
        {
            "root": "/mnt/Rosetta-outputs/run-1-RosettaFixStep-stage2-rosetta-fix-attempt-1"
        }
    ]
    assert result.status == AppRunStatus.SUCCEEDED
    assert cache_dir.joinpath(
        "rosetta_fix_output",
        "interface_energy_A_B",
        "residue_energy.csv",
    ).is_file()
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/rosetta/cache/rosetta_fix_output",
    )


def test_ppiflow_rosetta_fix_generates_update_xml_from_pdb_chains() -> None:
    pdb_text = "\n".join([
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  SER B  10       2.000   0.000   0.000  1.00 20.00           C",
        "ATOM      4  CA  TYR C  25       3.000   0.000   0.000  1.00 20.00           C",
    ])
    native_xml = (
        '<ROSETTASCRIPTS><RESIDUE_SELECTORS><Neighborhood name="nbrhood" '
        'resnums="29L-147L,18R-131R" distance="20.0"/></RESIDUE_SELECTORS>'
        '<MOVERS><FastRelax name="fast_relax"><MoveMap name="backbone">'
        '<Chain number="1" chi="1" bb="1"/><Chain number="2" chi="1" bb="0"/>'
        "</MoveMap></FastRelax></MOVERS></ROSETTASCRIPTS>"
    )

    update_xml = _ppiflow_rosetta_fix_update_xml(
        pdb_text=pdb_text,
        native_xml_text=native_xml,
        gentype="antibody",
    )

    assert 'resnums="1A-2A,10B-10B,25C-25C"' in update_xml
    assert '<Chain number="1" chi="1" bb="1"/>' in update_xml
    assert '<Chain number="2" chi="1" bb="1"/>' in update_xml
    assert '<Chain number="3" chi="1" bb="0"/>' in update_xml


def test_ppiflow_rosetta_fix_stages_upstream_default_flags(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1-RosettaFixStep-stage2-rosetta-fix-attempt-1"
    run_root.mkdir()

    remote_flags = ppiflow_workflow._stage_rosetta_flags_file(
        run_root=run_root,
        run_root_name=run_root.name,
        step_name="RosettaFixStep",
        config={},
    )

    assert remote_flags == f"{run_root.name}/_flags/ppiflow-rosetta-fix.flags"
    assert (
        run_root.joinpath("_flags", "ppiflow-rosetta-fix.flags").read_text(
            encoding="utf-8"
        )
        == "-overwrite\n-ignore_zero_occupancy false\n"
    )


def test_ppiflow_rosetta_stage_tasks_writes_app_task_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow_root = tmp_path / "workflow-volume"
    rosetta_root = tmp_path / "rosetta-volume"
    filtered_dir = workflow_root / "filtered_iptm08"
    filtered_dir.mkdir(parents=True)
    rosetta_root.mkdir(parents=True)
    filtered_dir.joinpath("design_a.pdb").write_text(
        "ATOM design A\n", encoding="utf-8"
    )
    filtered_dir.joinpath("design_b.pdb").write_text(
        "ATOM design B\n", encoding="utf-8"
    )
    queued = []

    class FakeVolume:
        def reload(self):
            return None

        def commit(self):
            return None

    class FakeQueue:
        def put(self, item):
            queued.append(item)

    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "CONF",
        SimpleNamespace(output_volume_mountpoint=str(workflow_root)),
    )
    monkeypatch.setattr(ppiflow_workflow.orchestrator, "OUT_VOLUME", FakeVolume())
    monkeypatch.setattr(
        ppiflow_workflow.rosetta_app,
        "CONF",
        SimpleNamespace(
            name="Rosetta",
            output_volume=FakeVolume(),
            output_volume_mountpoint=str(rosetta_root),
        ),
    )
    monkeypatch.setattr(
        ppiflow_workflow.modal,
        "Queue",
        SimpleNamespace(from_name=lambda *_args, **_kwargs: FakeQueue()),
    )

    stage_info = ppiflow_workflow.stage_ppiflow_rosetta_tasks.get_raw_f()(
        run_name="run-1-RosettaRelaxStep",
        run_id="stage2-rosetta-relax-attempt-1",
        source_paths=[
            "filtered_iptm08/design_a.pdb",
            "filtered_iptm08/design_b.pdb",
        ],
        step_name="RosettaRelaxStep",
        config={"rosetta_binary": "relax", "num_cpu_per_pod": 2},
    )

    tasks_path = (
        rosetta_root
        / "run-1-RosettaRelaxStep-stage2-rosetta-relax-attempt-1"
        / "tasks.parquet"
    )
    tasks = pl.read_parquet(tasks_path)
    assert stage_info["root"] == str(tasks_path.parent)
    assert tasks.select("index", "binary", "pdb").to_dicts() == [
        {
            "index": 1,
            "binary": "relax",
            "pdb": str(filtered_dir / "design_a.pdb"),
        },
        {
            "index": 2,
            "binary": "relax",
            "pdb": str(filtered_dir / "design_b.pdb"),
        },
    ]
    assert queued[0]["pdb"].endswith("/1/design_a.pdb")
    assert queued[1]["pdb"].endswith("/2/design_b.pdb")


def test_ppiflow_rosetta_fix_normalizes_resrese_logs_to_residue_energy_csv(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    filtered_dir = volume_root / "filtered_iptm08"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "rosetta" / "cache"
    filtered_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    filtered_dir.joinpath("design_a.pdb").write_text(
        "\n".join([
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
            "ATOM      2  CA  GLY B   1       0.000   0.000   6.000  1.00 20.00           C",
        ]),
        encoding="utf-8",
    )
    rosetta_archive_root = tmp_path / "rosetta-archive"
    rosetta_archive_root.joinpath("1").mkdir(parents=True)
    rosetta_archive_root.joinpath("1", "rosetta.log").write_text(
        "\n".join([
            "ResResE Res1 Res2 total",
            "ResResE weights weights 1.0",
            "ResResE pair_651 pair_661 -6.0",
        ]),
        encoding="utf-8",
    )
    fake_stage = _FakeRemoteFunction({
        "run_name": "run-1-RosettaFixStep",
        "run_id": "stage2-rosetta-fix-attempt-1",
        "num_cpu_per_pod": 1,
        "root": "/mnt/Rosetta-outputs/run-1-RosettaFixStep-stage2-rosetta-fix-attempt-1",
    })
    fake_run = _FakeRemoteFunction(None)
    fake_package = _FakeRemoteFunction(package_outputs(rosetta_archive_root))
    node = RosettaWorkflowNode(
        "RosettaFixStep",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            rosetta_stage_tasks=cast(modal.Function, fake_stage),
            rosetta_run=cast(modal.Function, fake_run),
            rosetta_package=cast(modal.Function, fake_package),
        ),
        config={"interface_dist": 12.0},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-rosetta-fix",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="filtered-structures",
                        producing_node_id="stage2-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="filtered_iptm08",
                        ),
                    )
                ]
            },
        )
    )

    residue_energy_csv = (
        cache_dir / "rosetta_fix_output" / "interface_energy_A_B" / "residue_energy.csv"
    )
    assert result.status == AppRunStatus.SUCCEEDED
    assert residue_energy_csv.is_file()
    assert "{1: -6.0}" in residue_energy_csv.read_text(encoding="utf-8")


def test_ppiflow_rosetta_relax_normalizes_scorefile_to_complex_csv(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    filtered_dir = volume_root / "filtered_iptm08"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "rosetta" / "cache"
    filtered_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    filtered_dir.joinpath("design_a.pdb").write_bytes(b"ATOM filtered\n")
    rosetta_archive_root = tmp_path / "rosetta-archive"
    rosetta_archive_root.joinpath("1").mkdir(parents=True)
    rosetta_archive_root.joinpath("1", "score.sc").write_text(
        "\n".join([
            "SCORE: total_score interface_score description",
            "SCORE: -123.4 -12.5 design_a",
        ]),
        encoding="utf-8",
    )
    fake_stage = _FakeRemoteFunction({
        "run_name": "run-1-RosettaRelaxStep",
        "run_id": "stage2-rosetta-relax-attempt-1",
        "num_cpu_per_pod": 1,
        "root": "/mnt/Rosetta-outputs/run-1-RosettaRelaxStep-stage2-rosetta-relax-attempt-1",
    })
    fake_run = _FakeRemoteFunction(None)
    fake_package = _FakeRemoteFunction(package_outputs(rosetta_archive_root))
    node = RosettaWorkflowNode(
        "RosettaRelaxStep",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, _FakePPIFlowFunction()),
            rosetta_stage_tasks=cast(modal.Function, fake_stage),
            rosetta_run=cast(modal.Function, fake_run),
            rosetta_package=cast(modal.Function, fake_package),
        ),
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-rosetta-relax",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="filtered-structures",
                        producing_node_id="stage2-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="filtered_iptm08",
                        ),
                    )
                ]
            },
        )
    )

    rosetta_complex_csv = cache_dir / "rosetta_relax_output" / "rosetta_complex_0.csv"
    assert result.status == AppRunStatus.SUCCEEDED
    assert rosetta_complex_csv.is_file()
    assert "pdb_name,total_score,interface_score" in rosetta_complex_csv.read_text(
        encoding="utf-8"
    )


def test_ppiflow_rosetta_relax_uses_wrapper_when_app_output_has_no_complex_csv(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    filtered_dir = volume_root / "filtered_iptm08"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "rosetta" / "cache"
    filtered_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    filtered_dir.joinpath("design_a.pdb").write_bytes(b"ATOM filtered\n")
    rosetta_archive_root = tmp_path / "rosetta-archive"
    rosetta_archive_root.joinpath("1").mkdir(parents=True)
    rosetta_archive_root.joinpath("1", "rosetta.log").write_text(
        "Generic Rosetta output without upstream relax_complex CSV\n",
        encoding="utf-8",
    )
    fake_stage = _FakeRemoteFunction({
        "run_name": "run-1-RosettaRelaxStep",
        "run_id": "stage2-rosetta-relax-attempt-1",
        "num_cpu_per_pod": 1,
        "root": "/mnt/Rosetta-outputs/run-1-RosettaRelaxStep-stage2-rosetta-relax-attempt-1",
    })
    fake_run = _FakeRemoteFunction(None)
    fake_package = _FakeRemoteFunction(package_outputs(rosetta_archive_root))
    fake_relax_score = _FakeRemoteFunction(
        b"pdb_name,interface_score,relaxed\ndesign_a,-9.5,100.0\n"
    )
    node = RosettaWorkflowNode(
        "RosettaRelaxStep",
        modal_namespace=cast(
            PPIFlowModalNamespace,
            SimpleNamespace(
                rosetta_stage_tasks=fake_stage,
                rosetta_run=fake_run,
                rosetta_package=fake_package,
                rosetta_relax_score=fake_relax_score,
            ),
        ),
        config={"batch_idx": "7", "dump_pdb": False},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-rosetta-relax",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="filtered-structures",
                        producing_node_id="stage2-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="filtered_iptm08",
                        ),
                    )
                ]
            },
        )
    )

    rosetta_complex_csv = cache_dir / "rosetta_relax_output" / "rosetta_complex_0.csv"
    batch_csv = cache_dir / "rosetta_relax_output" / "rosetta_complex_7.csv"
    assert result.status == AppRunStatus.SUCCEEDED
    assert fake_relax_score.calls == [
        {
            "pdb_inputs": [{"name": "design_a.pdb", "contents": b"ATOM filtered\n"}],
            "batch_idx": "7",
            "dump_pdb": False,
        }
    ]
    assert batch_csv.read_text(encoding="utf-8") == (
        "pdb_name,interface_score,relaxed\ndesign_a,-9.5,100.0\n"
    )
    assert rosetta_complex_csv.read_text(encoding="utf-8") == batch_csv.read_text(
        encoding="utf-8"
    )


def test_ppiflow_rosetta_relax_score_wrapper_initializes_pyrosetta_once(
    monkeypatch,
) -> None:
    init_calls = []

    class FakePose:
        def clone(self):
            return self

        def dump_pdb(self, _path: str) -> None:
            return None

    class FakeScoreFunction:
        def __init__(self) -> None:
            self.weights = []

        def set_weight(self, score_term, weight: float) -> None:
            self.weights.append((score_term, weight))

        def __call__(self, _pose: FakePose) -> float:
            return float(len(self.weights) or 10)

    class FakeFastRelax:
        def set_scorefxn(self, _score_function) -> None:
            return None

        def max_iter(self, _iterations: int) -> None:
            return None

        def apply(self, _pose: FakePose) -> None:
            return None

    class FakeInterfaceAnalyzerMover:
        def set_skip_reporting(self, _skip: bool) -> None:
            return None

        def apply(self, _pose: FakePose) -> None:
            return None

        def get_interface_dG(self) -> float:
            return -9.5

        def get_complexed_sasa(self) -> float:
            return 42.0

        def get_interface_delta_sasa(self) -> float:
            return 7.0

    pyrosetta_module = ModuleType("pyrosetta")
    rosetta_module = ModuleType("pyrosetta.rosetta")
    core_module = ModuleType("pyrosetta.rosetta.core")
    scoring_module = ModuleType("pyrosetta.rosetta.core.scoring")
    protocols_module = ModuleType("pyrosetta.rosetta.protocols")
    analysis_module = ModuleType("pyrosetta.rosetta.protocols.analysis")
    relax_module = ModuleType("pyrosetta.rosetta.protocols.relax")
    score_terms = (
        "dslf_fa13",
        "fa_atr",
        "fa_dun",
        "fa_elec",
        "fa_intra_rep",
        "fa_intra_sol",
        "fa_rep",
        "fa_sol",
        "hbond_bb_sc",
        "hbond_lr_bb",
        "hbond_sc",
        "hbond_sr_bb",
        "lk_ball_wtd",
        "omega",
        "p_aa_pp",
        "pro_close",
        "rama_prepro",
        "ref",
        "rg",
        "yhh_planarity",
    )
    for score_term in score_terms:
        setattr(scoring_module, score_term, score_term)
    pyrosetta_module.ScoreFunction = FakeScoreFunction
    pyrosetta_module.get_score_function = FakeScoreFunction
    pyrosetta_module.init = lambda options: init_calls.append(options)
    pyrosetta_module.pose_from_pdb = lambda _path: FakePose()
    pyrosetta_module.rosetta = rosetta_module
    rosetta_module.core = core_module
    rosetta_module.protocols = protocols_module
    core_module.scoring = scoring_module
    protocols_module.analysis = analysis_module
    protocols_module.relax = relax_module
    analysis_module.InterfaceAnalyzerMover = FakeInterfaceAnalyzerMover
    relax_module.FastRelax = FakeFastRelax
    for module in (
        pyrosetta_module,
        rosetta_module,
        core_module,
        scoring_module,
        protocols_module,
        analysis_module,
        relax_module,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ppiflow_workflow, "_PYROSETTA_INIT_DONE", False, raising=False)

    raw_wrapper = ppiflow_workflow.stage_ppiflow_rosetta_relax_scores.get_raw_f()
    first_csv = raw_wrapper(
        pdb_inputs=[{"name": "design_a.pdb", "contents": b"ATOM\n"}],
        batch_idx="0",
        dump_pdb=False,
    ).decode("utf-8")
    second_csv = raw_wrapper(
        pdb_inputs=[{"name": "design_b.pdb", "contents": b"ATOM\n"}],
        batch_idx="0",
        dump_pdb=False,
    ).decode("utf-8")

    assert len(init_calls) == 1
    assert "pdb_name,relaxed,interface_score" in first_csv
    assert "design_a,10.0,-9.5" in first_csv
    assert "design_b,10.0,-9.5" in second_csv


def test_ppiflow_filter_step_applies_af3score_metrics_and_links_pdbs(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    structures_dir = volume_root / "flowpacker" / "run_1"
    scores_dir = volume_root / "af3score"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "filter" / "cache"
    structures_dir.mkdir(parents=True)
    scores_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    for pdb_name in ("design_a.pdb", "design_b.pdb", "design_c.pdb"):
        structures_dir.joinpath(pdb_name).write_text(
            f"ATOM {pdb_name}\n",
            encoding="utf-8",
        )
    scores_dir.joinpath("af3score_metrics.csv").write_text(
        "\n".join([
            "description,iptm,pae",
            "design_a.pdb,0.75,8.0",
            "design_b.pdb,0.69,7.0",
            "design_c.pdb,0.82,12.0",
        ])
        + "\n",
        encoding="utf-8",
    )
    node = FilterStructuresNode(
        "FilterStep_stage1",
        config={"filters": {"iptm": "> 0.7", "pae": "<= 10"}},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage1-filter",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="flowpacker-structures",
                        producing_node_id="stage1-flowpacker",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="flowpacker/run_1",
                        ),
                    )
                ],
                "scores": [
                    WorkflowArtifact(
                        artifact_id="af3score-metrics",
                        producing_node_id="stage1-af3score",
                        kind=ArtifactKind.SCORES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="af3score/af3score_metrics.csv",
                        ),
                    )
                ],
            },
        )
    )

    output_dir = cache_dir / "filtered_iptm07"
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/filter/cache/filtered_iptm07",
    )
    assert output_dir.joinpath("design_a.pdb").is_symlink()
    assert (
        output_dir.joinpath("design_a.pdb").resolve()
        == (structures_dir / "design_a.pdb").resolve()
    )
    assert not output_dir.joinpath("design_b.pdb").exists()
    assert not output_dir.joinpath("design_c.pdb").exists()
    assert (
        output_dir.joinpath("filtered_iptm07.csv").read_text(encoding="utf-8")
        == "description,iptm,pae\ndesign_a.pdb,0.75,8.0\n"
    )


def test_ppiflow_fixed_positions_node_converts_rosetta_residue_energy(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    rosetta_dir = volume_root / "rosetta_fix"
    energy_dir = rosetta_dir / "interface_energy_A_B"
    filtered_dir = volume_root / "stage1" / "filtered_iptm07"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "fixed" / "cache"
    energy_dir.mkdir(parents=True)
    filtered_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    for pdb_name in ("cd3d_cd3d_3.pdb", "cd3d_cd3d_4.pdb", "unselected.pdb"):
        filtered_dir.joinpath(pdb_name).write_text(
            f"ATOM {pdb_name}\n",
            encoding="utf-8",
        )
    energy_dir.joinpath("residue_energy.csv").write_text(
        "\n".join([
            "pdbname,pdbpath,binder_energy",
            "cd3d_cd3d_3_8,,\"{'6': -4.5, '9': -6.1, '16': -8.0}\"",
            "cd3d_cd3d_3_9,,\"{'9': -7.2, '41': -5.1}\"",
            "cd3d_cd3d_4_1,,\"{'8': -3.0}\"",
        ])
        + "\n",
        encoding="utf-8",
    )
    node = FixedPositionsNode(
        gentype="binder",
        config={"energy_threshold": -5},
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-fixed-positions",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "rosetta": [
                    WorkflowArtifact(
                        artifact_id="rosetta-fix",
                        producing_node_id="stage2-rosetta-fix",
                        kind=ArtifactKind.DIRECTORY,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="rosetta_fix",
                        ),
                    )
                ],
                "structures": [
                    WorkflowArtifact(
                        artifact_id="stage1-filter",
                        producing_node_id="stage1-filter",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="stage1/filtered_iptm07",
                        ),
                    )
                ],
            },
        )
    )

    fixed_csv = cache_dir / "fixed_positions.csv"
    before_partial_dir = cache_dir / "before_partial_pdbs"
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].kind == ArtifactKind.TABLE
    assert result.outputs[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/fixed/cache/fixed_positions.csv",
    )
    assert fixed_csv.read_text(encoding="utf-8") == (
        'filename,fixed_positions\ncd3d_cd3d_3.pdb,"A9,A16,A41"\ncd3d_cd3d_4.pdb,NONE\n'
    )
    assert result.outputs[1].kind == ArtifactKind.STRUCTURES
    assert result.outputs[1].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="workflow/run-1/nodes/fixed/cache/before_partial_pdbs",
    )
    assert before_partial_dir.joinpath("cd3d_cd3d_3.pdb").is_symlink()
    assert (
        before_partial_dir.joinpath("cd3d_cd3d_3.pdb").resolve()
        == (filtered_dir / "cd3d_cd3d_3.pdb").resolve()
    )
    assert before_partial_dir.joinpath("cd3d_cd3d_4.pdb").is_symlink()
    assert not before_partial_dir.joinpath("unselected.pdb").exists()


def test_ppiflow_partial_step_runs_per_selected_pdb_with_fixed_positions(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "workflow-volume"
    selected_dir = volume_root / "before_partial_pdbs"
    fixed_dir = volume_root / "fixed"
    cache_dir = volume_root / "workflow" / "run-1" / "nodes" / "partial" / "cache"
    selected_dir.mkdir(parents=True)
    fixed_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    selected_dir.joinpath("cd3d_cd3d_3.pdb").write_text(
        "\n".join([
            "ATOM      1  CA  ALA B   7      11.104  13.207   9.309  1.00  2.00           C",
            "ATOM      2  CA  GLY B   8      11.104  13.207   9.309  1.00 20.00           C",
            "ATOM      3  CA  SER B   9      11.104  13.207   9.309  1.00  2.00           C",
        ])
        + "\n",
        encoding="utf-8",
    )
    fixed_dir.joinpath("fixed_positions.csv").write_text(
        'filename,fixed_positions\ncd3d_cd3d_3.pdb,"A9,A16,A41"\n',
        encoding="utf-8",
    )
    fake_ppiflow = _FakePPIFlowSequenceFunction()
    node = PPIFlowPartialWorkflowNode(
        step_name="PartialStep",
        gentype="binder",
        modal_namespace=PPIFlowModalNamespace(
            ppiflow_run=cast(modal.Function, fake_ppiflow),
        ),
        config={
            "target_chain": "B",
            "binder_chain": "A",
            "start_t": 0.7,
            "sample_hotspot_rate_min": 0.2,
            "sample_hotspot_rate_max": 0.5,
        },
    )

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="stage2-partial-ppiflow",
            attempt_id="attempt-1",
            cache_dir=cache_dir,
            inputs={
                "structures": [
                    WorkflowArtifact(
                        artifact_id="before-partial",
                        producing_node_id="stage2-fixed-positions",
                        kind=ArtifactKind.STRUCTURES,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="before_partial_pdbs",
                        ),
                    )
                ],
                "fixed_positions": [
                    WorkflowArtifact(
                        artifact_id="fixed-positions",
                        producing_node_id="stage2-fixed-positions",
                        kind=ArtifactKind.TABLE,
                        storage=VolumePath(
                            volume_name="Workflow-outputs",
                            path="fixed/fixed_positions.csv",
                        ),
                    )
                ],
            },
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(fake_ppiflow.calls) == 1
    call = fake_ppiflow.calls[0]
    assert call["run_name"] == "run-1-partial-cd3d_cd3d_3"
    assert isinstance(call["args"], ppiflow_app.PPIFlowArgs)
    args = call["args"].args
    assert isinstance(args, ppiflow_app.SampleBinderPartialConfig)
    assert args.input_pdb == str(selected_dir / "cd3d_cd3d_3.pdb")
    assert args.fixed_positions == "A9,A16,A41"
    assert args.specified_hotspots == "B7,B9"
    assert args.target_chain == "B"
    assert args.binder_chain == "A"
    assert args.samples_per_target == 10
    assert result.outputs[0].kind == ArtifactKind.STRUCTURES
    assert result.outputs[0].storage == VolumePath(
        volume_name=ppiflow_app.CONF.output_volume_name,
        path="run-1-partial-cd3d_cd3d_3",
    )


def test_ppiflow_entrypoint_stages_local_app_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )

    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": str(input_pdb),
                "binder_chain": "B",
            }
        }
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=("PPIFlowStep",),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/input_pdb/input.pdb"
    )
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]


def test_ppiflow_entrypoint_stages_antibody_design_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    antigen_pdb = tmp_path / "antigen.pdb"
    framework_pdb = tmp_path / "framework.pdb"
    antigen_pdb.write_text("ATOM antigen\n", encoding="utf-8")
    framework_pdb.write_text("ATOM framework\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )

    staged = _stage_ppiflow_app_inputs(
        steps_doc={"PPIFlowStep": {"config": "/configs/inference_nanobody.yaml"}},
        task_doc={
            "task": {
                "name": "demo-ab",
                "gentype": "antibody",
                "specified_hotspots": "C42",
                "antigen_pdb": str(antigen_pdb),
                "antigen_chain": "C",
                "framework_pdb": str(framework_pdb),
                "heavy_chain": "H",
                "light_chain": "L",
            },
            "steps": {"PPIFlowStep": True},
        },
        run_id="run-1",
        app_steps=("PPIFlowStep",),
    )

    args = staged["PPIFlowStep"]["args"]
    assert args["antigen_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/antigen_pdb/antigen.pdb"
    )
    assert args["framework_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/framework_pdb/framework.pdb"
    )
    assert uploaded == [
        (antigen_pdb, "/run-1/PPIFlowStep/antigen_pdb/antigen.pdb"),
        (framework_pdb, "/run-1/PPIFlowStep/framework_pdb/framework.pdb"),
    ]


def test_ppiflow_staging_uses_active_stage_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    task_doc = {
        "steps": {
            "PPIFlowStep": True,
            "PartialStep": True,
        }
    }
    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": str(input_pdb),
                "binder_chain": "B",
            }
        },
        "PartialStep": {
            "args": {
                "name": "demo-partial",
                "specified_hotspots": "A1",
                "input_pdb": str(tmp_path / "stage2-not-local.pdb"),
                "fixed_positions": "B1",
                "start_t": 0.5,
            }
        },
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=_active_ppiflow_app_steps(task_doc, stage=1),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"].endswith(
        "/PPIFlowStep/input_pdb/input.pdb"
    )
    assert staged["PartialStep"]["args"]["input_pdb"].endswith("stage2-not-local.pdb")
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]


def test_ppiflow_staging_skips_runtime_partial_step_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    task_doc = {
        "steps": {
            "PPIFlowStep": True,
            "PartialStep": True,
        }
    }
    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": str(input_pdb),
                "binder_chain": "B",
            }
        },
        "PartialStep": {
            "model_weights": "binder.ckpt",
            "samples_per_target": 10,
            "start_t": 0.7,
        },
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=_active_ppiflow_app_steps(task_doc, stage=None),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"].endswith(
        "/PPIFlowStep/input_pdb/input.pdb"
    )
    assert staged["PartialStep"] == steps_doc["PartialStep"]
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]


def test_ppiflow_staging_merges_task_fields_before_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    task_doc = {
        "task": {
            "name": "demo",
            "gentype": "binder",
            "input_pdb": str(input_pdb),
            "specified_hotspots": "B25",
        },
        "steps": {
            "PPIFlowStep": True,
        },
    }
    steps_doc = {"PPIFlowStep": {"config": "/configs/inference_binder.yaml"}}

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        task_doc=task_doc,
        run_id="run-1",
        app_steps=_active_ppiflow_app_steps(task_doc, stage=None),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/input_pdb/input.pdb"
    )
    assert staged["PPIFlowStep"]["args"]["config"] == "/configs/inference_binder.yaml"
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]


def test_ppiflow_staging_keeps_same_basename_inputs_distinct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    antigen_pdb = tmp_path / "antigen" / "input.pdb"
    framework_pdb = tmp_path / "framework" / "input.pdb"
    antigen_pdb.parent.mkdir()
    framework_pdb.parent.mkdir()
    antigen_pdb.write_text("ATOM antigen\n", encoding="utf-8")
    framework_pdb.write_text("ATOM framework\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "antigen_pdb": str(antigen_pdb),
                "antigen_chain": "A",
                "framework_pdb": str(framework_pdb),
                "heavy_chain": "H",
            }
        }
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=("PPIFlowStep",),
    )

    assert staged["PPIFlowStep"]["args"]["antigen_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/antigen_pdb/input.pdb"
    )
    assert staged["PPIFlowStep"]["args"]["framework_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/framework_pdb/input.pdb"
    )
    assert uploaded == [
        (antigen_pdb, "/run-1/PPIFlowStep/antigen_pdb/input.pdb"),
        (framework_pdb, "/run-1/PPIFlowStep/framework_pdb/input.pdb"),
    ]
