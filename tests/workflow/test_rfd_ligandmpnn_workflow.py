"""Tests for the RFdiffusion to LigandMPNN workflow definition."""

# ruff: noqa: D103

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import modal
import pytest

from biomodals.app.design import ligandmpnn_app, rfdiffusion_app
from biomodals.helper.styling import strip_ansi
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    NodePlacement,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE
from biomodals.workflow import rfd_ligandmpnn_workflow
from biomodals.workflow.core.nodes import NodeRunContext
from biomodals.workflow.rfd_ligandmpnn_workflow import (
    LigandMPNNDesignNode,
    LigandMPNNDesignSettings,
    RFdiffusionTrajectoryNode,
    RFDLigandMPNNSummaryNode,
    WorkflowModalNamespace,
    build_rfd_ligandmpnn_workflow,
    select_rfdiffusion_design,
)


class UnexpectedRemoteFunction:
    """Sentinel remote object for paths a test must not call."""

    def remote(self, *args: object, **kwargs: object) -> object:
        """Fail if the sentinel is invoked."""
        pytest.fail(f"Unexpected remote call: args={args}, kwargs={kwargs}")

    def spawn(self, *args: object, **kwargs: object) -> object:
        """Fail if the sentinel is spawned."""
        pytest.fail(f"Unexpected spawn call: args={args}, kwargs={kwargs}")


class FakeFunctionCall:
    """Small FunctionCall stand-in for direct submission tests."""

    def __init__(self, object_id: str, result: AppRunResult | None = None) -> None:
        """Initialize the fake call with a stable Modal object id and result."""
        self.object_id = object_id
        self.result = result or AppRunResult(status=AppRunStatus.SUCCEEDED)

    def get(self, timeout: float | int | None = None) -> AppRunResult:
        """Return a successful fake app result."""
        _ = timeout
        return self.result


UNEXPECTED_REMOTE = cast(modal.Function, UnexpectedRemoteFunction())


def _context(
    *,
    node_id: str = "node",
    inputs: dict[str, list[WorkflowArtifact]] | None = None,
    tmp_path: Path,
) -> NodeRunContext:
    return NodeRunContext(
        run_id="run-1",
        node_id=node_id,
        attempt_id="attempt-1",
        cache_dir=tmp_path / "cache",
        inputs=inputs or {},
    )


def test_rfd_ligandmpnn_uses_dependency_app_metadata() -> None:
    assert rfd_ligandmpnn_workflow.CONF.depends_on_apps == (
        "rfdiffusion",
        "ligandmpnn",
    )
    assert rfd_ligandmpnn_workflow.CONF.tags == {"depends_on": "rfdiffusion-ligandmpnn"}
    assert (
        rfd_ligandmpnn_workflow.RFDIFFUSION_OUTPUT_MOUNTPOINT
        == rfdiffusion_app.CONF.output_volume_mountpoint
    )
    assert (
        rfd_ligandmpnn_workflow.RFDIFFUSION_OUTPUT_VOLUME
        is rfdiffusion_app.CONF.output_volume
    )
    assert (
        rfd_ligandmpnn_workflow.RFDIFFUSION_OUTPUT_VOLUME_NAME
        == rfdiffusion_app.CONF.output_volume_name
    )


def test_build_rfd_ligandmpnn_workflow_models_trajectory_design_fanout() -> None:
    workflow = build_rfd_ligandmpnn_workflow(
        input_pdb=("input.pdb", b"ATOM\n"),
        run_namespace="demo",
        contigs="100-150/0 E333-526",
        hotspot_res="E405,E408",
        num_rfdiffusion_trajectories=2,
        num_rfdiffusion_designs=2,
        model_type="protein_mpnn",
        seeds=[7, 11],
        batch_size=4,
        number_of_batches=3,
        sc_num_samples=7,
        number_of_packs_per_design=5,
        max_parallel=8,
    )

    definition = workflow.validate()

    assert workflow.name == "rfd_ligandmpnn"
    assert set(definition.nodes) == {
        "rfd-demo-rfd001",
        "rfd-demo-rfd002",
        "ligandmpnn-demo-rfd001-d000",
        "ligandmpnn-demo-rfd001-d001",
        "ligandmpnn-demo-rfd002-d000",
        "ligandmpnn-demo-rfd002-d001",
        "summary",
    }
    assert definition.dependencies["ligandmpnn-demo-rfd001-d000"] == {"rfd-demo-rfd001"}
    assert definition.dependencies["ligandmpnn-demo-rfd001-d001"] == {"rfd-demo-rfd001"}
    assert definition.dependencies["ligandmpnn-demo-rfd002-d000"] == {"rfd-demo-rfd002"}
    assert definition.dependencies["summary"] == {
        "ligandmpnn-demo-rfd001-d000",
        "ligandmpnn-demo-rfd001-d001",
        "ligandmpnn-demo-rfd002-d000",
        "ligandmpnn-demo-rfd002-d001",
    }

    rfd_node = definition.nodes["rfd-demo-rfd001"].node
    mpnn_node = definition.nodes["ligandmpnn-demo-rfd001-d000"].node
    summary_node = definition.nodes["summary"].node

    assert isinstance(rfd_node, RFdiffusionTrajectoryNode)
    assert rfd_node.placement == NodePlacement.REMOTE
    assert rfd_node.pdb_content == b"ATOM\n"
    assert rfd_node.run_name == "demo-rfd001"
    assert rfd_node.contigs == "100-150/0 E333-526"
    assert rfd_node.hotspot_res == "E405,E408"
    assert rfd_node.num_designs == 2
    assert isinstance(rfd_node.modal_namespace, WorkflowModalNamespace)

    assert isinstance(mpnn_node, LigandMPNNDesignNode)
    assert mpnn_node.placement == NodePlacement.REMOTE
    assert mpnn_node.rfd_run_name == "demo-rfd001"
    assert mpnn_node.design_index == 0
    assert mpnn_node.run_name == "demo-rfd001-d000-mpnn"
    assert mpnn_node.settings == LigandMPNNDesignSettings(
        model_type="protein_mpnn",
        seeds=(7, 11),
        batch_size=4,
        number_of_batches=3,
        sc_num_samples=7,
        number_of_packs_per_design=5,
    )
    assert mpnn_node.modal_namespace is rfd_node.modal_namespace

    assert isinstance(summary_node, RFDLigandMPNNSummaryNode)
    assert summary_node.max_parallel == 8


def test_rfdiffusion_node_calls_app_function_with_hydra_overrides(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    class FakeRFdiffusionFunction:
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            calls.update(kwargs)
            return FakeFunctionCall(
                "fc-rfd-run",
                AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        AppOutput(
                            name="RFdiffusion_outputs",
                            kind=ArtifactKind.DIRECTORY,
                            storage=VolumePath(
                                volume_name=rfdiffusion_app.CONF.output_volume_name,
                                path="demo-rfd001/rfd-scaffolds",
                            ),
                            metadata={"run_name": "demo-rfd001"},
                        )
                    ],
                ),
            )

    node = RFdiffusionTrajectoryNode(
        pdb_content=b"ATOM\n",
        input_pdb_name="input.pdb",
        run_name="../demo-rfd001",
        contigs="100-150/0 E333-526",
        hotspot_res="E405 E408",
        num_designs=2,
        modal_namespace=WorkflowModalNamespace(
            rfdiffusion_infer=cast(modal.Function, FakeRFdiffusionFunction()),
            ligandmpnn_run=UNEXPECTED_REMOTE,
            select_rfd_design=UNEXPECTED_REMOTE,
        ),
    )

    result = node.run(_context(tmp_path=tmp_path))

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["input_pdb_bytes"] == b"ATOM\n"
    assert calls["input_pdb_name"] == "input.pdb"
    assert calls["run_name"] == "demo-rfd001"
    assert calls[
        "hydra_overrides"
    ] == rfdiffusion_app.build_rfdiffusion_hydra_overrides(
        contigs="100-150/0 E333-526",
        num_designs=2,
        hotspot_res="E405 E408",
    )


def test_rfdiffusion_node_submits_app_function_directly(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    class FakeRFdiffusionFunction:
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            calls.update(kwargs)
            return FakeFunctionCall("fc-rfd")

    node = RFdiffusionTrajectoryNode(
        pdb_content=b"ATOM\n",
        input_pdb_name="input.pdb",
        run_name="../demo-rfd001",
        contigs="100-150/0 E333-526",
        hotspot_res="E405 E408",
        num_designs=2,
        modal_namespace=WorkflowModalNamespace(
            rfdiffusion_infer=cast(modal.Function, FakeRFdiffusionFunction()),
            ligandmpnn_run=UNEXPECTED_REMOTE,
            select_rfd_design=UNEXPECTED_REMOTE,
        ),
    )

    submission = node.submit_remote(_context(tmp_path=tmp_path))

    assert submission.function_name == "rfdiffusion_infer"
    assert submission.function_call.object_id == "fc-rfd"
    assert calls["input_pdb_bytes"] == b"ATOM\n"
    assert calls["run_name"] == "demo-rfd001"
    assert calls[
        "hydra_overrides"
    ] == rfdiffusion_app.build_rfdiffusion_hydra_overrides(
        contigs="100-150/0 E333-526",
        num_designs=2,
        hotspot_res="E405 E408",
    )


def test_select_rfdiffusion_design_reads_pdb_trb_and_infers_redesigned_residues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffolds_dir = tmp_path / "demo-rfd001" / "rfd-scaffolds"
    scaffolds_dir.mkdir(parents=True)
    pdb_bytes = (
        b"ATOM      1  N   GLY A   1      0.000   0.000   0.000  1.00  0.00           N\n"
        b"ATOM      2  CA  GLY A   2      0.000   0.000   0.000  1.00 42.00           C\n"
        b"ATOM      3  N   GLY A   3      0.000   0.000   0.000  1.00 42.00           N\n"
        b"ATOM      4  CA  GLY A   4      0.000   0.000   0.000  1.00  0.00           C\n"
        b"ATOM      5  N   GLY B  10      0.000   0.000   0.000  1.00 42.00           N\n"
        b"ATOM      6  CA  GLY B  11      0.000   0.000   0.000  1.00  0.00           C\n"
    )
    scaffolds_dir.joinpath("demo-rfd001_0.pdb").write_bytes(pdb_bytes)
    scaffolds_dir.joinpath("demo-rfd001_0.trb").write_bytes(
        pickle.dumps({"mask_1d": [0, 1, 1, 0, 1, 0]})
    )

    class FakeVolume:
        def __init__(self) -> None:
            self.reloaded = False

        def reload(self) -> None:
            self.reloaded = True

    fake_volume = FakeVolume()
    monkeypatch.setattr(
        rfd_ligandmpnn_workflow,
        "RFDIFFUSION_OUTPUT_MOUNTPOINT",
        str(tmp_path),
    )
    monkeypatch.setattr(
        rfd_ligandmpnn_workflow,
        "RFDIFFUSION_OUTPUT_VOLUME",
        fake_volume,
    )

    selected = select_rfdiffusion_design.get_raw_f()(
        rfd_output_storage_path="demo-rfd001/rfd-scaffolds",
        rfd_run_name="demo-rfd001",
        design_index=0,
    )

    assert fake_volume.reloaded is True
    assert selected == {
        "pdb_name": "demo-rfd001_0.pdb",
        "pdb_bytes": pdb_bytes,
        "trb_name": "demo-rfd001_0.trb",
        "redesigned_residues": "A1 A4 B11",
    }


def test_select_rfdiffusion_design_uses_mask_1d_without_complex_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffolds_dir = tmp_path / "demo-rfd001" / "rfd-scaffolds"
    scaffolds_dir.mkdir(parents=True)
    pdb_bytes = (
        b"ATOM      1  N   GLY A   1      0.000   0.000   0.000  1.00  0.00           N\n"
        b"ATOM      2  CA  GLY A   2      0.000   0.000   0.000  1.00 42.00           C\n"
        b"ATOM      3  N   GLY A   3      0.000   0.000   0.000  1.00  0.00           N\n"
    )
    scaffolds_dir.joinpath("demo-rfd001_0.pdb").write_bytes(pdb_bytes)
    scaffolds_dir.joinpath("demo-rfd001_0.trb").write_bytes(
        pickle.dumps({"mask_1d": [0, 1, 0]})
    )

    class FakeVolume:
        def reload(self) -> None:
            return None

    monkeypatch.setattr(
        rfd_ligandmpnn_workflow,
        "RFDIFFUSION_OUTPUT_MOUNTPOINT",
        str(tmp_path),
    )
    monkeypatch.setattr(
        rfd_ligandmpnn_workflow,
        "RFDIFFUSION_OUTPUT_VOLUME",
        FakeVolume(),
    )

    selected = select_rfdiffusion_design.get_raw_f()(
        rfd_output_storage_path="demo-rfd001/rfd-scaffolds",
        rfd_run_name="demo-rfd001",
        design_index=0,
    )

    assert selected["redesigned_residues"] == "A1 A3"


def test_ligandmpnn_node_selects_rfd_output_and_calls_ligandmpnn(
    tmp_path: Path,
) -> None:
    select_calls: dict[str, Any] = {}
    ligandmpnn_calls: dict[str, Any] = {}

    class FakeSelectorFunction:
        def remote(self, **kwargs: object) -> dict[str, object]:
            select_calls.update(kwargs)
            return {
                "pdb_name": "demo-rfd001_0.pdb",
                "pdb_bytes": b"ATOM\n",
                "trb_name": "demo-rfd001_0.trb",
                "redesigned_residues": "A1 A2",
            }

    class FakeLigandMPNNFunction:
        def _record(self, **kwargs: object) -> AppRunResult:
            ligandmpnn_calls.update(kwargs)
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="LigandMPNN_outputs",
                        kind=ArtifactKind.ARCHIVE,
                        storage=InlineBytes(
                            data=b"tarball",
                            filename="demo-rfd001-d000-mpnn_LigandMPNN.tar.zst",
                            media_type=ZSTD_MEDIA_TYPE,
                        ),
                    )
                ],
            )

        def remote(self, **kwargs: object) -> AppRunResult:
            return self._record(**kwargs)

        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            return FakeFunctionCall("fc-ligandmpnn-run", self._record(**kwargs))

    node = LigandMPNNDesignNode(
        rfd_run_name="demo-rfd001",
        design_index=0,
        run_name="../demo-rfd001-d000-mpnn",
        modal_namespace=WorkflowModalNamespace(
            rfdiffusion_infer=UNEXPECTED_REMOTE,
            ligandmpnn_run=cast(modal.Function, FakeLigandMPNNFunction()),
            select_rfd_design=cast(modal.Function, FakeSelectorFunction()),
        ),
        settings=LigandMPNNDesignSettings(
            model_type="protein_mpnn",
            seeds=(7, 11),
            batch_size=4,
            number_of_batches=3,
            sc_num_samples=7,
            number_of_packs_per_design=5,
        ),
    )
    rfd_artifact = WorkflowArtifact(
        artifact_id="rfd-output",
        producing_node_id="rfd-demo-rfd001",
        kind=ArtifactKind.DIRECTORY,
        storage=VolumePath(
            volume_name=rfdiffusion_app.CONF.output_volume_name,
            path="demo-rfd001/rfd-scaffolds",
        ),
        metadata={"run_name": "demo-rfd001"},
    )

    result = node.run(
        _context(
            node_id="ligandmpnn-demo-rfd001-d000",
            inputs={"rfd_output": [rfd_artifact]},
            tmp_path=tmp_path,
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert select_calls == {
        "rfd_output_storage_path": "demo-rfd001/rfd-scaffolds",
        "rfd_run_name": "demo-rfd001",
        "design_index": 0,
    }
    assert ligandmpnn_calls["run_name"] == "demo-rfd001-d000-mpnn"
    assert ligandmpnn_calls["script_mode"] == "run"
    assert ligandmpnn_calls["struct_bytes"] == b"ATOM\n"
    assert ligandmpnn_calls["seeds"] == [7, 11]
    assert ligandmpnn_calls["cli_args"] == ligandmpnn_app.build_ligandmpnn_cli_args(
        script_mode="run",
        model_type="protein_mpnn",
        batch_size=4,
        number_of_batches=3,
        parse_atoms_with_zero_occupancy=True,
        pack_side_chains=True,
        number_of_packs_per_design=5,
        sc_num_samples=7,
        repack_everything=True,
        redesigned_residues="A1 A2",
    )
    assert result.outputs[0].name == "LigandMPNN_outputs"


def test_ligandmpnn_node_submits_app_function_directly_and_processes_metadata(
    tmp_path: Path,
) -> None:
    select_calls: dict[str, Any] = {}
    ligandmpnn_calls: dict[str, Any] = {}

    class FakeSelectorFunction:
        def remote(self, **kwargs: object) -> dict[str, object]:
            select_calls.update(kwargs)
            return {
                "pdb_name": "demo-rfd001_0.pdb",
                "pdb_bytes": b"ATOM\n",
                "trb_name": "demo-rfd001_0.trb",
                "redesigned_residues": "A1 A2",
            }

    class FakeLigandMPNNFunction:
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            ligandmpnn_calls.update(kwargs)
            return FakeFunctionCall("fc-ligandmpnn")

    node = LigandMPNNDesignNode(
        rfd_run_name="demo-rfd001",
        design_index=0,
        run_name="../demo-rfd001-d000-mpnn",
        modal_namespace=WorkflowModalNamespace(
            rfdiffusion_infer=UNEXPECTED_REMOTE,
            ligandmpnn_run=cast(modal.Function, FakeLigandMPNNFunction()),
            select_rfd_design=cast(modal.Function, FakeSelectorFunction()),
        ),
        settings=LigandMPNNDesignSettings(
            model_type="protein_mpnn",
            seeds=(7, 11),
            batch_size=4,
            number_of_batches=3,
            sc_num_samples=7,
            number_of_packs_per_design=5,
        ),
    )
    rfd_artifact = WorkflowArtifact(
        artifact_id="rfd-output",
        producing_node_id="rfd-demo-rfd001",
        kind=ArtifactKind.DIRECTORY,
        storage=VolumePath(
            volume_name=rfdiffusion_app.CONF.output_volume_name,
            path="demo-rfd001/rfd-scaffolds",
        ),
        metadata={"run_name": "demo-rfd001"},
    )

    submission = node.submit_remote(
        _context(
            node_id="ligandmpnn-demo-rfd001-d000",
            inputs={"rfd_output": [rfd_artifact]},
            tmp_path=tmp_path,
        )
    )

    assert submission.function_name == "ligandmpnn_run"
    assert submission.function_call.object_id == "fc-ligandmpnn"
    assert submission.metadata == {
        "rfd_run_name": "demo-rfd001",
        "design_index": "0",
        "redesigned_residues": "A1 A2",
    }
    assert select_calls == {
        "rfd_output_storage_path": "demo-rfd001/rfd-scaffolds",
        "rfd_run_name": "demo-rfd001",
        "design_index": 0,
    }
    assert ligandmpnn_calls["run_name"] == "demo-rfd001-d000-mpnn"
    assert ligandmpnn_calls["script_mode"] == "run"
    assert ligandmpnn_calls["struct_bytes"] == b"ATOM\n"
    assert ligandmpnn_calls["seeds"] == [7, 11]
    assert ligandmpnn_calls["cli_args"] == ligandmpnn_app.build_ligandmpnn_cli_args(
        script_mode="run",
        model_type="protein_mpnn",
        batch_size=4,
        number_of_batches=3,
        parse_atoms_with_zero_occupancy=True,
        pack_side_chains=True,
        number_of_packs_per_design=5,
        sc_num_samples=7,
        repack_everything=True,
        redesigned_residues="A1 A2",
    )

    processed = node.process_remote_result(
        AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="LigandMPNN_outputs",
                    kind=ArtifactKind.ARCHIVE,
                    storage=InlineBytes(
                        data=b"tarball",
                        filename="demo-rfd001-d000-mpnn_LigandMPNN.tar.zst",
                        media_type=ZSTD_MEDIA_TYPE,
                    ),
                )
            ],
        ),
        submission.metadata,
    )

    assert processed.outputs[0].metadata == {
        "rfd_run_name": "demo-rfd001",
        "design_index": "0",
        "redesigned_residues": "A1 A2",
    }


def test_rfd_ligandmpnn_summary_reports_design_artifacts(tmp_path: Path) -> None:
    node = RFDLigandMPNNSummaryNode(
        num_rfdiffusion_trajectories=1,
        num_rfdiffusion_designs=2,
        max_parallel=4,
    )
    artifact = WorkflowArtifact(
        artifact_id="mpnn-output",
        producing_node_id="ligandmpnn-demo-rfd001-d000",
        kind=ArtifactKind.ARCHIVE,
        storage=VolumePath(
            volume_name="Workflow-outputs",
            path="attempt/mpnn-output",
            media_type=ZSTD_MEDIA_TYPE,
        ),
        metadata={
            "rfd_run_name": "demo-rfd001",
            "design_index": "0",
            "run_name": "demo-rfd001-d000-mpnn",
        },
    )

    result = node.run(_context(inputs={"mpnn": [artifact]}, tmp_path=tmp_path))

    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "rfd_ligandmpnn_summary"
    assert result.outputs[0].kind == ArtifactKind.REPORT
    assert isinstance(result.outputs[0].storage, InlineBytes)
    report = result.outputs[0].storage.data.decode("utf-8")
    assert "# RFdiffusion + LigandMPNN Workflow Summary" in report
    assert "| demo-rfd001 | 0 | demo-rfd001-d000-mpnn | Workflow-outputs |" in report


def test_submit_rfd_ligandmpnn_workflow_uses_orchestrator_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    calls: dict[str, Any] = {}

    class FakeOrchestratorMethod:
        def remote(self, **kwargs: object) -> AppRunResult:
            calls["remote"] = kwargs
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def spawn(self, **kwargs: object) -> str:
            calls["spawn"] = kwargs
            return "call-1"

    class FakeWorkflowOrchestrator:
        def __init__(self) -> None:
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        rfd_ligandmpnn_workflow.orchestrator,
        "WorkflowOrchestrator",
        FakeWorkflowOrchestrator,
    )

    raw_f = rfd_ligandmpnn_workflow.submit_rfd_ligandmpnn_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        input_pdb=str(input_pdb),
        contigs="100-150/0 E333-526",
        hotspot_res="E405,E408",
        run_id="demo",
        num_rfdiffusion_trajectories=1,
        num_rfdiffusion_designs=2,
        model_type="protein_mpnn",
        seeds="7,11",
        batch_size=4,
        number_of_batches=3,
        sc_num_samples=7,
        number_of_packs_per_design=5,
        wait=False,
        max_parallel=3,
    )

    assert "remote" not in calls
    assert calls["spawn"]["workflow"].name == "rfd_ligandmpnn"
    definition = calls["spawn"]["workflow"].validate()
    rfd_node = definition.nodes["rfd-demo-rfd001"].node
    mpnn_node = definition.nodes["ligandmpnn-demo-rfd001-d000"].node
    assert isinstance(rfd_node, RFdiffusionTrajectoryNode)
    assert isinstance(mpnn_node, LigandMPNNDesignNode)
    assert rfd_node.run_name == "demo-rfd001"
    assert mpnn_node.settings.seeds == (7, 11)
    assert calls["spawn"]["run_id"] == "demo"
    assert calls["spawn"]["force"] is False
    assert calls["spawn"]["max_ready_workers"] == 3
    stdout = strip_ansi(capsys.readouterr().out)
    assert "Submitting RFDLigandMPNNWorkflow 'demo'" in stdout
    assert "1 RFdiffusion trajector" in stdout


def test_submit_rfd_ligandmpnn_workflow_dry_run_prints_dag_without_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")

    class UnexpectedWorkflowOrchestrator:
        def __init__(self) -> None:
            pytest.fail("dry-run should not construct the orchestrator")

    monkeypatch.setattr(
        rfd_ligandmpnn_workflow.orchestrator,
        "WorkflowOrchestrator",
        UnexpectedWorkflowOrchestrator,
    )

    raw_f = rfd_ligandmpnn_workflow.submit_rfd_ligandmpnn_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        input_pdb=str(input_pdb),
        contigs="100-150/0 E333-526",
        hotspot_res="E405,E408",
        run_id="demo",
        num_rfdiffusion_trajectories=1,
        num_rfdiffusion_designs=2,
        dry_run=True,
    )

    stdout = capsys.readouterr().out
    assert "[workflow] DAG graph: node_id [placement; class] <- dependency" in stdout
    assert (
        "[workflow]   rfd-demo-rfd001 [remote; RFdiffusionTrajectoryNode] <- -"
        in stdout
    )
    assert (
        "[workflow]   ligandmpnn-demo-rfd001-d000 "
        "[remote; LigandMPNNDesignNode] <- rfd-demo-rfd001" in stdout
    )
    assert "rfd_ligandmpnn_workflow.RFdiffusionTrajectoryNode" not in stdout
    assert "Submitting RFDLigandMPNNWorkflow" not in stdout
