"""Tests for the ShortMD workflow definition."""

# ruff: noqa: D103

import pickle
from pathlib import Path
from uuid import UUID

import pytest

from biomodals.app.bioinfo import gromacs_app
from biomodals.helper.styling import strip_ansi
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow import shortmd_workflow
from biomodals.workflow.core.execution import execution_plan
from biomodals.workflow.core.nodes import NodeRunContext
from biomodals.workflow.shortmd_workflow import (
    ShortMDAnalysisNode,
    ShortMDClearNode,
    ShortMDCloneNode,
    ShortMDGromacsSettings,
    ShortMDPrepNode,
    ShortMDReplicateNode,
    ShortMDSummaryNode,
    build_shortmd_workflow,
    clone_prepared_shortmd_run,
    discover_pdb_inputs,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _context(
    tmp_path: Path,
    *,
    node_id: str,
    inputs: dict[str, list[WorkflowArtifact]] | None = None,
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


class FakeFunctionCall:
    """Small FunctionCall stand-in for direct submission tests."""

    def __init__(self, object_id: str, result: object | None = None) -> None:
        """Initialize the fake call with a stable object id and result."""
        self.object_id = object_id
        self.result = result

    def get(self, timeout: float | int | None = None) -> object:
        """Return the fake result."""
        _ = timeout
        return self.result


def test_shortmd_uses_gromacs_app_volume_metadata() -> None:
    assert shortmd_workflow.CONF.depends_on_apps == ("gromacs",)
    assert shortmd_workflow.CONF.tags == {"depends_on": "gromacs"}
    assert (
        shortmd_workflow.GROMACS_OUTPUT_MOUNTPOINT
        == gromacs_app.CONF.output_volume_mountpoint
    )
    assert shortmd_workflow.GROMACS_OUTPUT_VOLUME is gromacs_app.CONF.output_volume
    assert (
        shortmd_workflow.GROMACS_OUTPUT_VOLUME_NAME
        == gromacs_app.CONF.output_volume_name
    )


def test_discover_pdb_inputs_globs_pdb_files(tmp_path: Path) -> None:
    tmp_path.joinpath("b.pdb").write_text("B\n", encoding="utf-8")
    tmp_path.joinpath("a.pdb").write_text("A\n", encoding="utf-8")
    tmp_path.joinpath("ignore.txt").write_text("x\n", encoding="utf-8")

    discovered = discover_pdb_inputs(tmp_path)

    assert set(discovered) == {("a.pdb", b"A\n"), ("b.pdb", b"B\n")}


def test_discover_pdb_inputs_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No PDB files"):
        discover_pdb_inputs(tmp_path)


def test_build_shortmd_workflow_models_production_analysis_dependencies() -> None:
    workflow = build_shortmd_workflow(
        input_pdbs=[("alpha.pdb", b"ATOM\n"), ("beta.pdb", b"ATOM\n")],
        replicates=2,
        simulation_time_ns=2,
        cpu_only=True,
        max_parallel=8,
    )

    definition = workflow.validate()

    assert workflow.name == "shortmd"
    assert set(definition.nodes) == {
        "prep-alpha",
        "clone-alpha-r001",
        "clone-alpha-r002",
        "replicate-alpha-r001",
        "replicate-alpha-r002",
        "analysis-alpha-r001",
        "analysis-alpha-r002",
        "prep-beta",
        "clone-beta-r001",
        "clone-beta-r002",
        "replicate-beta-r001",
        "replicate-beta-r002",
        "analysis-beta-r001",
        "analysis-beta-r002",
        "summary",
    }
    assert definition.dependencies["clone-alpha-r001"] == {"prep-alpha"}
    assert definition.dependencies["clone-alpha-r002"] == {"prep-alpha"}
    assert definition.dependencies["replicate-alpha-r001"] == {"clone-alpha-r001"}
    assert definition.dependencies["replicate-alpha-r002"] == {"clone-alpha-r002"}
    assert definition.dependencies["analysis-alpha-r001"] == {"replicate-alpha-r001"}
    assert definition.dependencies["analysis-alpha-r002"] == {"replicate-alpha-r002"}
    assert definition.dependencies["summary"] == {
        "analysis-alpha-r001",
        "analysis-alpha-r002",
        "analysis-beta-r001",
        "analysis-beta-r002",
    }

    prep_node = definition.nodes["prep-alpha"].node
    clone_node = definition.nodes["clone-alpha-r001"].node
    replicate_node = definition.nodes["replicate-alpha-r001"].node
    analysis_node = definition.nodes["analysis-alpha-r001"].node
    summary_node = definition.nodes["summary"].node

    assert isinstance(prep_node, ShortMDPrepNode)
    assert prep_node.run_name == "alpha"
    assert prep_node.pdb_content == b"ATOM\n"
    assert {
        "app_name",
        "prep_cpu_function",
        "prep_gpu_function",
        "prep_cpu_function_name",
        "prep_gpu_function_name",
    }.isdisjoint(prep_node.__dict__)
    assert isinstance(clone_node, ShortMDCloneNode)
    assert clone_node.source_run_name == "alpha"
    assert clone_node.replicate_run_name == "alpha-r001"
    assert "clone_function" not in clone_node.__dict__
    assert isinstance(replicate_node, ShortMDReplicateNode)
    assert replicate_node.source_run_name == "alpha"
    assert replicate_node.replicate_run_name == "alpha-r001"
    assert replicate_node.gromacs.simulation_time_ns == 2
    assert replicate_node.gromacs.cpu_only is True
    assert {
        "app_name",
        "production_cpu_function",
        "production_gpu_function",
        "stats_function",
        "production_cpu_function_name",
        "production_gpu_function_name",
        "stats_function_name",
    }.isdisjoint(replicate_node.__dict__)
    assert isinstance(analysis_node, ShortMDAnalysisNode)
    assert analysis_node.source_run_name == "alpha"
    assert analysis_node.replicate_run_name == "alpha-r001"

    assert isinstance(summary_node, ShortMDSummaryNode)
    assert summary_node.max_parallel == 8
    restored = pickle.loads(pickle.dumps(workflow))  # noqa: S301
    assert restored.validate() == definition


def test_build_shortmd_force_adds_tracked_cleanup_dependencies() -> None:
    workflow = build_shortmd_workflow(
        input_pdbs=[("alpha.pdb", b"ATOM\n")],
        replicates=1,
        overwrite_existing=True,
    )

    definition = workflow.validate()

    assert isinstance(definition.nodes["clear-alpha"].node, ShortMDClearNode)
    assert definition.dependencies["prep-alpha"] == {"clear-alpha"}


def test_shortmd_node_parallelism_is_not_scientific_identity() -> None:
    def fingerprint(max_parallel: int) -> str:
        workflow = build_shortmd_workflow(
            input_pdbs=[("alpha.pdb", b"ATOM\n")],
            replicates=1,
            max_parallel=max_parallel,
        )
        return execution_plan(
            workflow.validate(),
            workload_run_key="run-1",
        ).workload_plan_fingerprint

    assert fingerprint(1) == fingerprint(8)


def test_build_shortmd_workflow_rejects_duplicate_sanitized_stems() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        build_shortmd_workflow(
            input_pdbs=[("../a.pdb", b"A\n"), ("a.pdb", b"B\n")],
            replicates=1,
        )


def test_shortmd_clear_node_prepares_tracked_provider_call(tmp_path: Path) -> None:
    node = ShortMDClearNode(run_name="../source")
    invocation = node.prepare_remote(_context(tmp_path, node_id="clear-source"))
    result = node.process_remote_result(None, invocation.metadata)

    assert invocation.function_name == "clear_shortmd_gromacs_run"
    assert invocation.uses_gpu is False
    assert invocation.kwargs == {"run_name": "source"}
    assert invocation.metadata == {"stage": "clear", "run_name": "source"}
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs == []


def test_shortmd_prep_node_prepares_kernel_call_and_processes_result(
    tmp_path: Path,
) -> None:
    node = ShortMDPrepNode(
        pdb_content=b"ATOM\n",
        run_name="../source",
        gromacs=ShortMDGromacsSettings(
            simulation_time_ns=2,
            run_pdbfixer=True,
            cpu_only=True,
            num_threads=8,
            use_openmp_threads=True,
            ld_seed=11,
            gen_seed=12,
            genion_seed=13,
        ),
    )
    invocation = node.prepare_remote(_context(tmp_path, node_id="prep-source"))
    result = node.process_remote_result(
        f"{gromacs_app.CONF.output_volume_mountpoint}/prepared/source",
        invocation.metadata,
    )

    assert invocation.function_name == "prepare_tpr_cpu"
    assert invocation.uses_gpu is False
    assert invocation.kwargs == {
        "pdb_content": b"ATOM\n",
        "run_name": "source",
        "simulation_time_ns": 2,
        "run_pdbfixer": True,
        "num_threads": 8,
        "use_openmp_threads": True,
        "ld_seed": 11,
        "gen_seed": 12,
        "genion_seed": 13,
    }
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "prepared_gromacs_run"
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.CONF.output_volume_name,
        path="prepared/source",
    )
    assert result.outputs[0].metadata == {
        "stage": "prep",
        "run_name": "source",
        "files": [
            {"path": "source.pdb", "role": "input_structure"},
            {"path": "production_source.tpr", "role": "production_topology"},
            {"path": "production.mdp", "role": "production_parameters"},
        ],
    }


def test_shortmd_prep_node_records_metadata_without_submitting(
    tmp_path: Path,
) -> None:
    node = ShortMDPrepNode(
        pdb_content=b"ATOM\n",
        run_name="../source",
        gromacs=ShortMDGromacsSettings(cpu_only=True),
    )

    invocation = node.prepare_remote(_context(tmp_path, node_id="prep-source"))

    assert invocation.function_name == "prepare_tpr_cpu"
    assert invocation.metadata == {"stage": "prep", "run_name": "source"}
    assert invocation.kwargs["run_name"] == "source"


def test_shortmd_prep_node_rejects_workdir_outside_gromacs_mount(
    tmp_path: Path,
) -> None:
    node = ShortMDPrepNode(
        pdb_content=b"ATOM\n",
        run_name="source",
    )
    with pytest.raises(ValueError, match="outside"):
        node.process_remote_result(
            "/outside-gromacs-output",
            {"stage": "prep", "run_name": "source"},
        )


def test_clone_prepared_shortmd_run_copies_prepared_inputs_into_replicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeOutputVolume:
        def __init__(self) -> None:
            self.commit_count = 0
            self.reload_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            self.reload_count += 1

    output_volume = FakeOutputVolume()
    source_dir = tmp_path / "prepared" / "source"
    source_dir.mkdir(parents=True)
    source_dir.joinpath("source.pdb").write_text("ATOM\n", encoding="utf-8")
    source_dir.joinpath("production_source.tpr").write_text("tpr\n", encoding="utf-8")
    source_dir.joinpath("production_source.xtc").write_text("stale\n", encoding="utf-8")
    source_dir.joinpath("npt_source.gro").write_text("npt\n", encoding="utf-8")

    monkeypatch.setattr(shortmd_workflow, "GROMACS_OUTPUT_MOUNTPOINT", str(tmp_path))
    monkeypatch.setattr(shortmd_workflow, "GROMACS_OUTPUT_VOLUME", output_volume)

    result = clone_prepared_shortmd_run.get_raw_f()(
        source_storage_path="prepared/source",
        source_run_name="source",
        replicate_run_name="source-r001",
    )

    replicate_dir = tmp_path / "source-r001"
    assert result == str(replicate_dir)
    assert replicate_dir.joinpath("source-r001.pdb").read_text(encoding="utf-8") == (
        "ATOM\n"
    )
    assert (
        replicate_dir.joinpath("production_source-r001.tpr").read_text(encoding="utf-8")
        == "tpr\n"
    )
    assert not replicate_dir.joinpath("production_source.xtc").exists()
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_shortmd_clone_node_prepares_kernel_call_and_processes_result(
    tmp_path: Path,
) -> None:
    node = ShortMDCloneNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        overwrite_clone=True,
    )
    context = _context(
        tmp_path,
        node_id="clone-source-r001",
        inputs={
            "prepared": [
                WorkflowArtifact(
                    artifact_id="source",
                    producing_node_id="prep-source",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=gromacs_app.CONF.output_volume_name,
                        path="prepared/source",
                    ),
                    metadata={"stage": "prep", "run_name": "source"},
                )
            ]
        },
    )
    invocation = node.prepare_remote(context)
    result = node.process_remote_result(
        f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001",
        invocation.metadata,
    )

    assert invocation.function_name == "clone_prepared_shortmd_run"
    assert invocation.uses_gpu is False
    assert invocation.kwargs == {
        "source_storage_path": "prepared/source",
        "source_run_name": "source",
        "replicate_run_name": "source-r001",
        "overwrite": True,
    }
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "cloned_gromacs_run"
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.CONF.output_volume_name,
        path="source-r001",
    )
    assert result.outputs[0].metadata == {
        "stage": "clone",
        "run_name": "source-r001",
        "source_run_name": "source",
        "files": [
            {"path": "source-r001.pdb", "role": "input_structure"},
            {
                "path": "production_source-r001.tpr",
                "role": "production_topology",
            },
            {"path": "production.mdp", "role": "production_parameters"},
        ],
    }


def test_shortmd_clone_node_does_not_submit_during_preparation(
    tmp_path: Path,
) -> None:
    node = ShortMDCloneNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        overwrite_clone=True,
    )

    invocation = node.prepare_remote(
        _context(
            tmp_path,
            node_id="clone-source-r001",
            inputs={
                "prepared": [
                    WorkflowArtifact(
                        artifact_id="source",
                        producing_node_id="prep-source",
                        kind=ArtifactKind.DIRECTORY,
                        storage=VolumePath(
                            volume_name=gromacs_app.CONF.output_volume_name,
                            path="prepared/source",
                        ),
                        metadata={"stage": "prep", "run_name": "source"},
                    )
                ]
            },
        )
    )

    assert invocation.function_name == "clone_prepared_shortmd_run"
    assert invocation.metadata == {
        "stage": "clone",
        "run_name": "source-r001",
        "source_run_name": "source",
    }
    assert invocation.kwargs == {
        "source_storage_path": "prepared/source",
        "source_run_name": "source",
        "replicate_run_name": "source-r001",
        "overwrite": True,
    }


def test_shortmd_replicate_node_prepares_and_publishes_raw_production(
    tmp_path: Path,
) -> None:
    node = ShortMDReplicateNode(
        source_run_name="source",
        replicate_run_name="source-r001",
    )
    context = _context(
        tmp_path,
        node_id="replicate-source-r001",
        inputs={
            "cloned": [
                WorkflowArtifact(
                    artifact_id="source-r001",
                    producing_node_id="clone-source-r001",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=gromacs_app.CONF.output_volume_name,
                        path="source-r001",
                    ),
                    metadata={
                        "stage": "clone",
                        "run_name": "source-r001",
                        "source_run_name": "source",
                    },
                )
            ]
        },
    )
    invocation = node.prepare_remote(context)
    result = node.process_remote_result(
        f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001",
        invocation.metadata,
    )

    assert invocation.function_name == "production_run_gpu"
    assert invocation.uses_gpu is True
    assert invocation.kwargs == {
        "run_name": "source-r001",
        "simulation_time_ns": 2,
        "num_threads": 16,
        "use_openmp_threads": False,
    }
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "gromacs_production_raw"
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.CONF.output_volume_name,
        path="source-r001",
    )
    assert result.outputs[0].metadata["run_name"] == "source-r001"
    assert result.outputs[0].metadata["source_run_name"] == "source"
    assert result.outputs[0].metadata["files"] == [
        {"path": "production_source-r001.xtc", "role": "trajectory"},
        {"path": "production_source-r001.tpr", "role": "production_topology"},
    ]


def test_shortmd_replicate_node_selects_cpu_function(
    tmp_path: Path,
) -> None:
    node = ShortMDReplicateNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        gromacs=ShortMDGromacsSettings(cpu_only=True),
    )

    invocation = node.prepare_remote(
        _context(
            tmp_path,
            node_id="replicate-source-r001",
            inputs={
                "cloned": [
                    WorkflowArtifact(
                        artifact_id="source-r001",
                        producing_node_id="clone-source-r001",
                        kind=ArtifactKind.DIRECTORY,
                        storage=VolumePath(
                            volume_name=gromacs_app.CONF.output_volume_name,
                            path="source-r001",
                        ),
                        metadata={
                            "stage": "clone",
                            "run_name": "source-r001",
                            "source_run_name": "source",
                        },
                    )
                ]
            },
        )
    )

    assert invocation.function_name == "production_run_cpu"
    assert invocation.metadata == {
        "stage": "production",
        "run_name": "source-r001",
        "source_run_name": "source",
    }
    assert invocation.kwargs == {
        "run_name": "source-r001",
        "simulation_time_ns": 2,
        "num_threads": 16,
        "use_openmp_threads": False,
    }


def test_shortmd_analysis_node_prepares_and_publishes_analyzed_output(
    tmp_path: Path,
) -> None:
    node = ShortMDAnalysisNode(
        source_run_name="source",
        replicate_run_name="source-r001",
    )
    invocation = node.prepare_remote(
        _context(
            tmp_path,
            node_id="analysis-source-r001",
            inputs={
                "production": [
                    WorkflowArtifact(
                        artifact_id="source-r001",
                        producing_node_id="replicate-source-r001",
                        kind=ArtifactKind.DIRECTORY,
                        storage=VolumePath(
                            volume_name=gromacs_app.CONF.output_volume_name,
                            path="source-r001",
                        ),
                        metadata={
                            "stage": "production",
                            "run_name": "source-r001",
                            "source_run_name": "source",
                        },
                    )
                ]
            },
        )
    )
    result = node.process_remote_result(
        f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001",
        invocation.metadata,
    )

    assert invocation.function_name == "collect_traj_stats"
    assert invocation.uses_gpu is False
    assert invocation.kwargs == {
        "traj_prefix": "production_",
        "run_name": "source-r001",
        "save_processed_traj": True,
        "make_figures": True,
    }
    assert invocation.metadata == {
        "stage": "analysis",
        "run_name": "source-r001",
        "source_run_name": "source",
    }
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "gromacs_production"
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.CONF.output_volume_name,
        path="source-r001",
    )
    assert result.outputs[0].metadata["files"] == [
        {"path": "production_source-r001.xtc", "role": "trajectory"},
        {"path": "production_source-r001.tpr", "role": "production_topology"},
        {
            "path": "production_source-r001_nopbc_centered.pdb",
            "role": "centered_structure",
        },
        {"path": "rmsd_production_source-r001.csv", "role": "rmsd"},
        {
            "path": "rg_production_source-r001.csv",
            "role": "radius_of_gyration",
        },
        {"path": "rmsf_production_source-r001.csv", "role": "rmsf"},
    ]


def test_shortmd_summary_node_emits_markdown_manifest(tmp_path: Path) -> None:
    node = ShortMDSummaryNode(replicates=2, max_parallel=4)
    context = _context(
        tmp_path,
        node_id="summary",
        inputs={
            "alpha-r001": [
                WorkflowArtifact(
                    artifact_id="alpha-r001",
                    producing_node_id="replicate-alpha-r001",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=gromacs_app.CONF.output_volume_name,
                        path="alpha-r001",
                    ),
                    metadata={
                        "source_run_name": "alpha",
                        "run_name": "alpha-r001",
                    },
                )
            ],
            "alpha-r002": [
                WorkflowArtifact(
                    artifact_id="alpha-r002",
                    producing_node_id="replicate-alpha-r002",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=gromacs_app.CONF.output_volume_name,
                        path="alpha-r002",
                    ),
                    metadata={
                        "source_run_name": "alpha",
                        "run_name": "alpha-r002",
                    },
                )
            ],
        },
    )

    result = node.run(context)

    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.name == "shortmd_summary"
    assert output.kind == ArtifactKind.REPORT
    assert isinstance(output.storage, InlineBytes)
    assert output.storage.filename == "shortmd-summary.md"
    report = output.storage.data.decode("utf-8")
    assert "# ShortMD Workflow Summary" in report
    assert (
        f"| alpha | alpha-r001 | {gromacs_app.CONF.output_volume_name} | alpha-r001 |"
        in report
    )
    assert (
        f"| alpha | alpha-r002 | {gromacs_app.CONF.output_volume_name} | alpha-r002 |"
        in report
    )


def test_shortmd_app_includes_orchestrator_class() -> None:
    functions = shortmd_workflow.app._local_state.functions

    assert "ExecutionCoordinator.*" in functions
    assert "prepare_tpr_cpu" in functions
    assert "prepare_tpr_gpu" in functions
    assert "production_run_cpu" in functions
    assert "production_run_gpu" in functions
    assert "collect_traj_stats" in functions


def test_submit_shortmd_workflow_uses_included_orchestrator_class_boundary(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "pdbs"
    input_dir.mkdir()
    input_dir.joinpath("alpha.pdb").write_text("ATOM\n", encoding="utf-8")
    calls = {}

    class FakeOrchestratorMethod:
        def spawn(self, **kwargs):
            calls["spawn"] = kwargs
            return FakeFunctionCall(
                "call-1",
                AppRunResult(status=AppRunStatus.SUCCEEDED),
            )

    class FakeExecutionCoordinator:
        def __init__(self, **kwargs) -> None:
            calls["coordinator"] = kwargs
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "ExecutionCoordinator",
        FakeExecutionCoordinator,
    )

    raw_f = shortmd_workflow.submit_shortmd_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        input_dir=str(input_dir),
        run_id="shortmd-run",
        replicates=1,
        wait=False,
        max_parallel=3,
    )

    assert calls["spawn"]["workflow"].name == "shortmd"
    definition = calls["spawn"]["workflow"].validate()
    prep_node = definition.nodes["prep-shortmd-run-alpha"].node
    replicate_node = definition.nodes["replicate-shortmd-run-alpha-r001"].node
    analysis_node = definition.nodes["analysis-shortmd-run-alpha-r001"].node

    assert prep_node.run_name == "shortmd-run-alpha"
    assert replicate_node.source_run_name == "shortmd-run-alpha"
    assert replicate_node.replicate_run_name == "shortmd-run-alpha-r001"
    assert analysis_node.replicate_run_name == "shortmd-run-alpha-r001"
    assert {"prep_cpu_function", "prep_gpu_function"}.isdisjoint(prep_node.__dict__)
    assert {
        "production_cpu_function",
        "production_gpu_function",
        "stats_function",
    }.isdisjoint(replicate_node.__dict__)
    UUID(str(calls["coordinator"]["execution_run_id"]))
    assert calls["spawn"]["workload_run_key"] == "shortmd-run"
    assert calls["coordinator"]["deployment_environment"] == "development"
    assert calls["coordinator"]["deployment_name"] == shortmd_workflow.CONF.name
    assert calls["coordinator"]["deployment_version"] == 1
    assert calls["spawn"]["max_parallel_nodes"] == 3
    assert calls["spawn"]["max_active_provider_calls"] == 3
    assert calls["spawn"]["max_active_gpu_provider_calls"] == 3
    assert set(calls["spawn"]["development_function_handles"]) == {
        "clear_shortmd_gromacs_run",
        "prepare_tpr_cpu",
        "prepare_tpr_gpu",
        "clone_prepared_shortmd_run",
        "production_run_cpu",
        "production_run_gpu",
        "collect_traj_stats",
        "check_shortmd_external_artifact",
    }
    stdout = strip_ansi(capsys.readouterr().out)
    assert "Submitting ShortMD workflow 'shortmd-run'" in stdout
    assert "1 input PDB(s)" in stdout
    assert "1 replicate(s)" in stdout


def test_submit_shortmd_workflow_enables_external_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "pdbs"
    input_dir.mkdir()
    input_dir.joinpath("alpha.pdb").write_text("ATOM\n", encoding="utf-8")
    calls = {}

    class FakeOrchestratorMethod:
        def spawn(self, **kwargs):
            calls["spawn"] = kwargs
            return FakeFunctionCall(
                "call-1",
                AppRunResult(status=AppRunStatus.SUCCEEDED),
            )

    class FakeExecutionCoordinator:
        def __init__(self, **kwargs) -> None:
            calls["coordinator"] = kwargs
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "ExecutionCoordinator",
        FakeExecutionCoordinator,
    )

    raw_f = shortmd_workflow.submit_shortmd_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        input_dir=str(input_dir),
        run_id="shortmd-run",
        replicates=1,
        wait=True,
    )

    assert calls["spawn"]["strict_external_artifact_checks"] is True
    assert (
        calls["spawn"]["external_artifact_checker_function_name"]
        == "check_shortmd_external_artifact"
    )


def test_submit_shortmd_workflow_uses_exact_deployed_coordinator_without_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "pdbs"
    input_dir.mkdir()
    input_dir.joinpath("alpha.pdb").write_text("ATOM\n", encoding="utf-8")
    calls = {}

    class FakeOrchestratorMethod:
        def spawn(self, **kwargs):
            calls["spawn"] = kwargs
            return FakeFunctionCall("call-1")

    class FakeCoordinator:
        run = FakeOrchestratorMethod()

    def fake_coordinator_handle(**kwargs):
        calls["coordinator"] = kwargs
        return FakeCoordinator()

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "execution_coordinator_handle",
        fake_coordinator_handle,
    )
    raw_f = shortmd_workflow.submit_shortmd_workflow.info.raw_f
    assert raw_f is not None

    raw_f(
        input_dir=str(input_dir),
        run_id="shortmd-run",
        replicates=1,
        wait=False,
        use_deployed_coordinator=True,
        deployment_environment="production",
        deployment_name="shortmd-prod",
        deployment_version=7,
    )

    deployment = calls["coordinator"]["deployment"]
    assert deployment.environment == "production"
    assert deployment.deployment_name == "shortmd-prod"
    assert deployment.deployment_version == 7
    assert calls["coordinator"]["use_deployed_coordinator"] is True
    assert "development_function_handles" not in calls["spawn"]


def test_submit_shortmd_workflow_uses_successor_operation_for_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "pdbs"
    input_dir.mkdir()
    input_dir.joinpath("alpha.pdb").write_text("ATOM\n", encoding="utf-8")
    calls = {}
    predecessor = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    class UnexpectedRunMethod:
        def spawn(self, **_kwargs):
            raise AssertionError("restart must not create a root run")

    class FakeRestartMethod:
        def spawn(self, **kwargs):
            calls["restart"] = kwargs
            return FakeFunctionCall("call-1")

    class FakeCoordinator:
        run = UnexpectedRunMethod()
        restart_from = FakeRestartMethod()

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "execution_coordinator_handle",
        lambda **kwargs: calls.setdefault("coordinator", kwargs) and FakeCoordinator(),
    )
    raw_f = shortmd_workflow.submit_shortmd_workflow.info.raw_f
    assert raw_f is not None

    raw_f(
        input_dir=str(input_dir),
        run_id="shortmd-run",
        replicates=1,
        wait=False,
        use_deployed_coordinator=True,
        deployment_environment="production",
        deployment_name="shortmd-prod",
        deployment_version=7,
        restart_from=predecessor,
    )

    assert calls["restart"]["predecessor_execution_run_id"] == predecessor
    assert calls["restart"]["workload_run_key"] == "shortmd-run"
    assert calls["restart"]["workflow"].name == "shortmd"


def test_submit_shortmd_workflow_dry_run_prints_dag_without_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "pdbs"
    input_dir.mkdir()
    input_dir.joinpath("alpha.pdb").write_text("ATOM\n", encoding="utf-8")

    class UnexpectedExecutionCoordinator:
        def __init__(self, **_kwargs) -> None:
            pytest.fail("dry-run should not construct the orchestrator")

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "ExecutionCoordinator",
        UnexpectedExecutionCoordinator,
    )

    raw_f = shortmd_workflow.submit_shortmd_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        input_dir=str(input_dir),
        run_id="shortmd-run",
        replicates=1,
        dry_run=True,
    )

    stdout = capsys.readouterr().out
    assert "[workflow] DAG graph: node_id [execution; class] <- dependency" in stdout
    assert (
        "[workflow]   prep-shortmd-run-alpha [provider; ShortMDPrepNode] <- -" in stdout
    )
    assert (
        "[workflow]   clone-shortmd-run-alpha-r001 "
        "[provider; ShortMDCloneNode] <- prep-shortmd-run-alpha" in stdout
    )
    assert (
        "[workflow]   analysis-shortmd-run-alpha-r001 "
        "[provider; ShortMDAnalysisNode] <- replicate-shortmd-run-alpha-r001" in stdout
    )
    assert "shortmd_workflow.ShortMDPrepNode" not in stdout
    assert "Submitting ShortMD workflow" not in stdout


def test_submit_shortmd_workflow_propagates_force_to_gromacs_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "pdbs"
    input_dir.mkdir()
    input_dir.joinpath("alpha.pdb").write_text("ATOM\n", encoding="utf-8")
    calls = {}

    class FakeOrchestratorMethod:
        def spawn(self, **kwargs):
            calls["spawn"] = kwargs
            return FakeFunctionCall("call-1")

    class FakeExecutionCoordinator:
        def __init__(self, **kwargs) -> None:
            calls["coordinator"] = kwargs
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "ExecutionCoordinator",
        FakeExecutionCoordinator,
    )

    raw_f = shortmd_workflow.submit_shortmd_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        input_dir=str(input_dir),
        run_id="shortmd-run",
        replicates=1,
        force=True,
        wait=False,
    )

    definition = calls["spawn"]["workflow"].validate()
    clear_node = definition.nodes["clear-shortmd-run-alpha"].node
    clone_node = definition.nodes["clone-shortmd-run-alpha-r001"].node

    assert isinstance(clear_node, ShortMDClearNode)
    assert definition.dependencies["prep-shortmd-run-alpha"] == {
        "clear-shortmd-run-alpha"
    }
    assert clone_node.overwrite_clone is True
    assert "clone_function" not in clone_node.__dict__
    assert "force" not in calls["spawn"]
