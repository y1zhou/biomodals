"""Tests for the RFdiffusion to LigandMPNN workflow definition."""

# ruff: noqa: D103

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast
from uuid import UUID

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
RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _context(
    *,
    node_id: str = "node",
    inputs: dict[str, list[WorkflowArtifact]] | None = None,
    tmp_path: Path,
) -> NodeRunContext:
    return NodeRunContext(
        execution_run_id=RUN_ID,
        workload_run_key="run-1",
        node_id=node_id,
        task_key="node",
        work_dir=tmp_path / "result",
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
    assert rfd_node.pdb_content == b"ATOM\n"
    assert rfd_node.run_name == "demo-rfd001"
    assert rfd_node.contigs == "100-150/0 E333-526"
    assert rfd_node.hotspot_res == "E405,E408"
    assert rfd_node.num_designs == 2
    assert isinstance(rfd_node.modal_namespace, WorkflowModalNamespace)

    assert isinstance(mpnn_node, LigandMPNNDesignNode)
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


def test_rfdiffusion_node_prepares_kernel_call_with_hydra_overrides(
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
                                path="demo-rfd001/outputs/rfd-scaffolds",
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

    invocation = node.prepare_remote(_context(tmp_path=tmp_path))

    assert invocation.function_name == "rfdiffusion_infer"
    assert invocation.uses_gpu is True
    assert invocation.kwargs["input_pdb_bytes"] == b"ATOM\n"
    assert invocation.kwargs["input_pdb_name"] == "input.pdb"
    assert invocation.kwargs["run_name"] == "demo-rfd001"
    assert invocation.kwargs[
        "hydra_overrides"
    ] == rfdiffusion_app.build_rfdiffusion_hydra_overrides(
        contigs="100-150/0 E333-526",
        num_designs=2,
        hotspot_res="E405 E408",
    )
    assert calls == {}


def test_select_rfdiffusion_design_reads_pdb_trb_and_infers_redesigned_residues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffolds_dir = tmp_path / "demo-rfd001" / "outputs" / "rfd-scaffolds"
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
        rfd_output_storage_path="demo-rfd001/outputs/rfd-scaffolds",
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
    scaffolds_dir = tmp_path / "demo-rfd001" / "outputs" / "rfd-scaffolds"
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
        rfd_output_storage_path="demo-rfd001/outputs/rfd-scaffolds",
        rfd_run_name="demo-rfd001",
        design_index=0,
    )

    assert selected["redesigned_residues"] == "A1 A3"


def test_select_rfdiffusion_design_rejects_mask_length_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffolds_dir = tmp_path / "demo-rfd001" / "outputs" / "rfd-scaffolds"
    scaffolds_dir.mkdir(parents=True)
    scaffolds_dir.joinpath("demo-rfd001_0.pdb").write_bytes(
        b"ATOM      1  N   GLY A   1      0.000   0.000   0.000  1.00  0.00           N\n"
        b"ATOM      2  CA  GLY A   2      0.000   0.000   0.000  1.00 42.00           C\n"
        b"ATOM      3  N   GLY A   3      0.000   0.000   0.000  1.00  0.00           N\n"
    )
    scaffolds_dir.joinpath("demo-rfd001_0.trb").write_bytes(
        pickle.dumps({"mask_1d": [0, 1]})
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

    with pytest.raises(ValueError, match="mask_1d length 2 does not match 3"):
        select_rfdiffusion_design.get_raw_f()(
            rfd_output_storage_path="demo-rfd001/outputs/rfd-scaffolds",
            rfd_run_name="demo-rfd001",
            design_index=0,
        )


def test_ligandmpnn_node_selects_rfd_output_and_prepares_kernel_call(
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
            path="demo-rfd001/outputs/rfd-scaffolds",
        ),
        metadata={"run_name": "demo-rfd001"},
    )

    invocation = node.prepare_remote(
        _context(
            node_id="ligandmpnn-demo-rfd001-d000",
            inputs={"rfd_output": [rfd_artifact]},
            tmp_path=tmp_path,
        )
    )

    assert invocation.function_name == "ligandmpnn_run"
    assert invocation.uses_gpu is True
    assert select_calls == {
        "rfd_output_storage_path": "demo-rfd001/outputs/rfd-scaffolds",
        "rfd_run_name": "demo-rfd001",
        "design_index": 0,
    }
    assert invocation.kwargs["run_name"] == "demo-rfd001-d000-mpnn"
    assert invocation.kwargs["script_mode"] == "run"
    assert invocation.kwargs["struct_bytes"] == b"ATOM\n"
    assert invocation.kwargs["seeds"] == [7, 11]
    assert invocation.kwargs["cli_args"] == ligandmpnn_app.build_ligandmpnn_cli_args(
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
    assert ligandmpnn_calls == {}


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
            path="demo-rfd001/outputs/rfd-scaffolds",
        ),
        metadata={"run_name": "demo-rfd001"},
    )

    invocation = node.prepare_remote(
        _context(
            node_id="ligandmpnn-demo-rfd001-d000",
            inputs={"rfd_output": [rfd_artifact]},
            tmp_path=tmp_path,
        )
    )

    assert invocation.function_name == "ligandmpnn_run"
    assert invocation.metadata == {
        "rfd_run_name": "demo-rfd001",
        "design_index": "0",
        "redesigned_residues": "A1 A2",
    }
    assert select_calls == {
        "rfd_output_storage_path": "demo-rfd001/outputs/rfd-scaffolds",
        "rfd_run_name": "demo-rfd001",
        "design_index": 0,
    }
    assert invocation.kwargs["run_name"] == "demo-rfd001-d000-mpnn"
    assert invocation.kwargs["script_mode"] == "run"
    assert invocation.kwargs["struct_bytes"] == b"ATOM\n"
    assert invocation.kwargs["seeds"] == [7, 11]
    assert invocation.kwargs["cli_args"] == ligandmpnn_app.build_ligandmpnn_cli_args(
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
        invocation.metadata,
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
            path="results/mpnn-output",
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
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            calls["spawn"] = kwargs
            return FakeFunctionCall("call-1")

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

    assert calls["spawn"]["workflow"].name == "rfd_ligandmpnn"
    definition = calls["spawn"]["workflow"].validate()
    rfd_node = definition.nodes["rfd-demo-rfd001"].node
    mpnn_node = definition.nodes["ligandmpnn-demo-rfd001-d000"].node
    assert isinstance(rfd_node, RFdiffusionTrajectoryNode)
    assert isinstance(mpnn_node, LigandMPNNDesignNode)
    assert rfd_node.run_name == "demo-rfd001"
    assert mpnn_node.settings.seeds == (7, 11)
    UUID(str(calls["spawn"]["execution_run_id"]))
    assert calls["spawn"]["workload_run_key"] == "demo"
    assert calls["spawn"]["deployment_environment"] == "development"
    assert calls["spawn"]["deployment_name"] == rfd_ligandmpnn_workflow.CONF.name
    assert calls["spawn"]["deployment_version"] == 1
    assert calls["spawn"]["max_active_provider_calls"] == 3
    assert calls["spawn"]["max_active_gpu_provider_calls"] == 3
    assert set(calls["spawn"]["development_function_handles"]) == {
        "rfdiffusion_infer",
        "ligandmpnn_run",
    }
    stdout = strip_ansi(capsys.readouterr().out)
    assert "Submitting RFDLigandMPNNWorkflow 'demo'" in stdout
    assert "1 RFdiffusion trajector" in stdout


def test_submit_rfd_ligandmpnn_workflow_can_enable_strict_external_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakeOrchestratorMethod:
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            calls["spawn"] = kwargs
            return FakeFunctionCall("call-1")

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
        num_rfdiffusion_designs=1,
        wait=False,
        strict_artifact_checks=True,
    )

    spawn_kwargs = calls["spawn"]
    assert spawn_kwargs["strict_external_artifact_checks"] is True
    checker = spawn_kwargs["external_artifact_checker"]
    assert callable(checker)
    assert "check_rfd_ligandmpnn_external_artifact" in repr(checker)


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
    assert "[workflow] DAG graph: node_id [execution; class] <- dependency" in stdout
    assert (
        "[workflow]   rfd-demo-rfd001 [provider; RFdiffusionTrajectoryNode] <- -"
        in stdout
    )
    assert (
        "[workflow]   ligandmpnn-demo-rfd001-d000 "
        "[provider; LigandMPNNDesignNode] <- rfd-demo-rfd001" in stdout
    )
    assert "rfd_ligandmpnn_workflow.RFdiffusionTrajectoryNode" not in stdout
    assert "Submitting RFDLigandMPNNWorkflow" not in stdout
