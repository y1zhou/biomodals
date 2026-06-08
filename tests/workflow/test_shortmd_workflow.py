"""Tests for the ShortMD workflow definition."""

# ruff: noqa: D103

from pathlib import Path
from typing import cast

import modal
import pytest

from biomodals.app.bioinfo import gromacs_app
from biomodals.helper.styling import strip_ansi
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    NodePlacement,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow import shortmd_workflow
from biomodals.workflow.core.nodes import NodeRunContext
from biomodals.workflow.shortmd_workflow import (
    ShortMDCloneNode,
    ShortMDGromacsSettings,
    ShortMDPrepNode,
    ShortMDReplicateNode,
    ShortMDSummaryNode,
    WorkflowModalNamespace,
    build_shortmd_workflow,
    clone_prepared_shortmd_run,
    discover_pdb_inputs,
)


class UnexpectedRemoteFunction:
    """Sentinel remote object for paths a test must not call."""

    def remote(self, *args: object, **kwargs: object) -> object:
        """Fail if the sentinel is invoked."""
        pytest.fail(f"Unexpected remote call: args={args}, kwargs={kwargs}")

    def spawn(self, *args: object, **kwargs: object) -> object:
        """Fail if the sentinel is spawned."""
        pytest.fail(f"Unexpected spawn call: args={args}, kwargs={kwargs}")


UNEXPECTED_REMOTE = cast(modal.Function, UnexpectedRemoteFunction())


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


def test_build_shortmd_workflow_models_prep_replicate_summary_dependencies() -> None:
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
        "prep-beta",
        "clone-beta-r001",
        "clone-beta-r002",
        "replicate-beta-r001",
        "replicate-beta-r002",
        "summary",
    }
    assert definition.dependencies["clone-alpha-r001"] == {"prep-alpha"}
    assert definition.dependencies["clone-alpha-r002"] == {"prep-alpha"}
    assert definition.dependencies["replicate-alpha-r001"] == {"clone-alpha-r001"}
    assert definition.dependencies["replicate-alpha-r002"] == {"clone-alpha-r002"}
    assert definition.dependencies["summary"] == {
        "replicate-alpha-r001",
        "replicate-alpha-r002",
        "replicate-beta-r001",
        "replicate-beta-r002",
    }

    prep_node = definition.nodes["prep-alpha"].node
    clone_node = definition.nodes["clone-alpha-r001"].node
    replicate_node = definition.nodes["replicate-alpha-r001"].node
    summary_node = definition.nodes["summary"].node

    assert isinstance(prep_node, ShortMDPrepNode)
    assert prep_node.placement == NodePlacement.REMOTE
    assert prep_node.run_name == "alpha"
    assert prep_node.pdb_content == b"ATOM\n"
    assert {
        "app_name",
        "prep_cpu_function",
        "prep_gpu_function",
        "prep_cpu_function_name",
        "prep_gpu_function_name",
    }.isdisjoint(prep_node.__dict__)
    assert isinstance(prep_node.modal_namespace, WorkflowModalNamespace)

    assert isinstance(clone_node, ShortMDCloneNode)
    assert clone_node.placement == NodePlacement.REMOTE
    assert clone_node.source_run_name == "alpha"
    assert clone_node.replicate_run_name == "alpha-r001"
    assert "clone_function" not in clone_node.__dict__
    assert clone_node.modal_namespace is prep_node.modal_namespace

    assert isinstance(replicate_node, ShortMDReplicateNode)
    assert replicate_node.placement == NodePlacement.REMOTE
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
    assert replicate_node.modal_namespace is prep_node.modal_namespace

    assert isinstance(summary_node, ShortMDSummaryNode)
    assert summary_node.max_parallel == 8


def test_build_shortmd_workflow_rejects_duplicate_sanitized_stems() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        build_shortmd_workflow(
            input_pdbs=[("../a.pdb", b"A\n"), ("a.pdb", b"B\n")],
            replicates=1,
        )


def test_shortmd_prep_node_runs_gromacs_prepare_and_returns_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepare_kwargs = {}
    clear_kwargs = {}
    events = []

    class FakePrepareFunction:
        def _record(self, **kwargs: object) -> str:
            events.append("prepare")
            prepare_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/prepared/source"

        def remote(self, **kwargs: object) -> object:
            return self._record(**kwargs)

        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            return FakeFunctionCall("fc-prepare", self._record(**kwargs))

    class FakeClearFunction:
        def remote(self, **kwargs: object) -> object:
            events.append("clear")
            clear_kwargs.update(kwargs)
            return None

    modal_namespace = WorkflowModalNamespace(
        clear=cast(modal.Function, FakeClearFunction()),
        clone=UNEXPECTED_REMOTE,
        prepare_cpu=cast(modal.Function, FakePrepareFunction()),
        prepare_gpu=UNEXPECTED_REMOTE,
        production_cpu=UNEXPECTED_REMOTE,
        production_gpu=UNEXPECTED_REMOTE,
        collect_stats=UNEXPECTED_REMOTE,
    )
    monkeypatch.setattr(shortmd_workflow.modal.Function, "from_name", pytest.fail)
    monkeypatch.setattr(
        shortmd_workflow,
        "clear_shortmd_gromacs_run",
        pytest.fail,
    )
    monkeypatch.setattr(
        shortmd_workflow.gromacs_app,
        "prepare_tpr_cpu",
        pytest.fail,
    )

    node = ShortMDPrepNode(
        pdb_content=b"ATOM\n",
        run_name="../source",
        modal_namespace=modal_namespace,
        overwrite_existing=True,
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
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="prep-source",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
            inputs={},
        )
    )

    assert prepare_kwargs == {
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
    assert clear_kwargs == {"run_name": "source"}
    assert events == ["clear", "prepare"]
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "prepared_gromacs_run"
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.CONF.output_volume_name,
        path="prepared/source",
    )
    assert result.outputs[0].metadata == {"stage": "prep", "run_name": "source"}


def test_shortmd_prep_node_submits_gromacs_prepare_directly(tmp_path: Path) -> None:
    prepare_kwargs = {}
    clear_kwargs = {}
    events = []

    class FakePrepareFunction:
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            events.append("prepare")
            prepare_kwargs.update(kwargs)
            return FakeFunctionCall(
                "fc-prepare",
                f"{gromacs_app.CONF.output_volume_mountpoint}/prepared/source",
            )

    class FakeClearFunction:
        def remote(self, **kwargs: object) -> object:
            events.append("clear")
            clear_kwargs.update(kwargs)
            return None

    node = ShortMDPrepNode(
        pdb_content=b"ATOM\n",
        run_name="../source",
        modal_namespace=WorkflowModalNamespace(
            clear=cast(modal.Function, FakeClearFunction()),
            clone=UNEXPECTED_REMOTE,
            prepare_cpu=cast(modal.Function, FakePrepareFunction()),
            prepare_gpu=UNEXPECTED_REMOTE,
            production_cpu=UNEXPECTED_REMOTE,
            production_gpu=UNEXPECTED_REMOTE,
            collect_stats=UNEXPECTED_REMOTE,
        ),
        overwrite_existing=True,
        gromacs=ShortMDGromacsSettings(cpu_only=True),
    )

    submission = node.submit_remote(
        NodeRunContext(
            run_id="run-1",
            node_id="prep-source",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
            inputs={},
        )
    )

    assert submission.function_name == "prepare_tpr_cpu"
    assert submission.function_call.object_id == "fc-prepare"
    assert submission.metadata == {"stage": "prep", "run_name": "source"}
    assert clear_kwargs == {"run_name": "source"}
    assert prepare_kwargs["run_name"] == "source"
    assert events == ["clear", "prepare"]


def test_shortmd_prep_node_rejects_workdir_outside_gromacs_mount(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakePrepareFunction:
        def remote(self, **kwargs: object) -> object:
            return "/outside-gromacs-output"

        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            return FakeFunctionCall("fc-outside", self.remote(**kwargs))

    modal_namespace = WorkflowModalNamespace(
        clear=UNEXPECTED_REMOTE,
        clone=UNEXPECTED_REMOTE,
        prepare_cpu=UNEXPECTED_REMOTE,
        prepare_gpu=cast(modal.Function, FakePrepareFunction()),
        production_cpu=UNEXPECTED_REMOTE,
        production_gpu=UNEXPECTED_REMOTE,
        collect_stats=UNEXPECTED_REMOTE,
    )
    monkeypatch.setattr(
        shortmd_workflow.modal.Function,
        "from_name",
        pytest.fail,
    )
    monkeypatch.setattr(
        shortmd_workflow.gromacs_app,
        "prepare_tpr_gpu",
        FakePrepareFunction(),
    )

    node = ShortMDPrepNode(
        pdb_content=b"ATOM\n",
        run_name="source",
        modal_namespace=modal_namespace,
    )
    with pytest.raises(ValueError, match="outside"):
        node.run(
            NodeRunContext(
                run_id="run-1",
                node_id="prep-source",
                attempt_id="attempt-1",
                cache_dir=tmp_path / "cache",
                inputs={},
            )
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


def test_shortmd_clone_node_clones_prepared_run_and_returns_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clone_kwargs = {}

    class FakeCloneFunction:
        def _record(self, **kwargs: object) -> str:
            clone_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001"

        def remote(self, **kwargs: object) -> object:
            return self._record(**kwargs)

        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            return FakeFunctionCall("fc-clone", self._record(**kwargs))

    modal_namespace = WorkflowModalNamespace(
        clear=UNEXPECTED_REMOTE,
        clone=cast(modal.Function, FakeCloneFunction()),
        prepare_cpu=UNEXPECTED_REMOTE,
        prepare_gpu=UNEXPECTED_REMOTE,
        production_cpu=UNEXPECTED_REMOTE,
        production_gpu=UNEXPECTED_REMOTE,
        collect_stats=UNEXPECTED_REMOTE,
    )
    monkeypatch.setattr(shortmd_workflow.modal.Function, "from_name", pytest.fail)
    monkeypatch.setattr(
        shortmd_workflow,
        "clone_prepared_shortmd_run",
        pytest.fail,
    )

    node = ShortMDCloneNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        modal_namespace=modal_namespace,
        overwrite_clone=True,
    )
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="clone-source-r001",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
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

    assert clone_kwargs == {
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
    }


def test_shortmd_clone_node_submits_clone_directly(tmp_path: Path) -> None:
    clone_kwargs = {}

    class FakeCloneFunction:
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            clone_kwargs.update(kwargs)
            return FakeFunctionCall(
                "fc-clone",
                f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001",
            )

    node = ShortMDCloneNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        modal_namespace=WorkflowModalNamespace(
            clear=UNEXPECTED_REMOTE,
            clone=cast(modal.Function, FakeCloneFunction()),
            prepare_cpu=UNEXPECTED_REMOTE,
            prepare_gpu=UNEXPECTED_REMOTE,
            production_cpu=UNEXPECTED_REMOTE,
            production_gpu=UNEXPECTED_REMOTE,
            collect_stats=UNEXPECTED_REMOTE,
        ),
        overwrite_clone=True,
    )

    submission = node.submit_remote(
        NodeRunContext(
            run_id="run-1",
            node_id="clone-source-r001",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
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

    assert submission.function_name == "clone_prepared_shortmd_run"
    assert submission.function_call.object_id == "fc-clone"
    assert submission.metadata == {
        "stage": "clone",
        "run_name": "source-r001",
        "source_run_name": "source",
    }
    assert clone_kwargs == {
        "source_storage_path": "prepared/source",
        "source_run_name": "source",
        "replicate_run_name": "source-r001",
        "overwrite": True,
    }


def test_shortmd_replicate_node_runs_gromacs_production(
    tmp_path: Path,
    monkeypatch,
) -> None:
    production_kwargs = {}
    stats_kwargs = {}

    class FakeProductionFunction:
        def _record(self, **kwargs: object) -> str:
            production_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001"

        def remote(self, **kwargs: object) -> object:
            return self._record(**kwargs)

        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            return FakeFunctionCall("fc-production", self._record(**kwargs))

    class FakeStatsFunction:
        def remote(self, traj_prefix: str, **kwargs: object) -> object:
            stats_kwargs["traj_prefix"] = traj_prefix
            stats_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/production/source-r001"

    modal_namespace = WorkflowModalNamespace(
        clear=UNEXPECTED_REMOTE,
        clone=UNEXPECTED_REMOTE,
        prepare_cpu=UNEXPECTED_REMOTE,
        prepare_gpu=UNEXPECTED_REMOTE,
        production_cpu=UNEXPECTED_REMOTE,
        production_gpu=cast(modal.Function, FakeProductionFunction()),
        collect_stats=cast(modal.Function, FakeStatsFunction()),
    )
    monkeypatch.setattr(shortmd_workflow.modal.Function, "from_name", pytest.fail)
    monkeypatch.setattr(
        shortmd_workflow.gromacs_app,
        "production_run_gpu",
        pytest.fail,
    )
    monkeypatch.setattr(
        shortmd_workflow.gromacs_app,
        "collect_traj_stats",
        pytest.fail,
    )

    node = ShortMDReplicateNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        modal_namespace=modal_namespace,
    )
    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="replicate-source-r001",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
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

    assert production_kwargs == {
        "run_name": "source-r001",
        "simulation_time_ns": 2,
        "num_threads": 16,
        "use_openmp_threads": False,
    }
    assert stats_kwargs == {
        "traj_prefix": "production_",
        "run_name": "source-r001",
        "save_processed_traj": True,
        "make_figures": True,
    }
    assert result.status == AppRunStatus.SUCCEEDED
    assert result.outputs[0].name == "gromacs_production"
    assert result.outputs[0].kind == ArtifactKind.DIRECTORY
    assert result.outputs[0].storage == VolumePath(
        volume_name=gromacs_app.CONF.output_volume_name,
        path="production/source-r001",
    )
    assert result.outputs[0].metadata["run_name"] == "source-r001"
    assert result.outputs[0].metadata["source_run_name"] == "source"


def test_shortmd_replicate_node_submits_production_directly(tmp_path: Path) -> None:
    production_kwargs = {}

    class FakeProductionFunction:
        def spawn(self, **kwargs: object) -> FakeFunctionCall:
            production_kwargs.update(kwargs)
            return FakeFunctionCall(
                "fc-production",
                f"{gromacs_app.CONF.output_volume_mountpoint}/source-r001",
            )

    node = ShortMDReplicateNode(
        source_run_name="source",
        replicate_run_name="source-r001",
        modal_namespace=WorkflowModalNamespace(
            clear=UNEXPECTED_REMOTE,
            clone=UNEXPECTED_REMOTE,
            prepare_cpu=UNEXPECTED_REMOTE,
            prepare_gpu=UNEXPECTED_REMOTE,
            production_cpu=cast(modal.Function, FakeProductionFunction()),
            production_gpu=UNEXPECTED_REMOTE,
            collect_stats=UNEXPECTED_REMOTE,
        ),
        gromacs=ShortMDGromacsSettings(cpu_only=True),
    )

    submission = node.submit_remote(
        NodeRunContext(
            run_id="run-1",
            node_id="replicate-source-r001",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
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

    assert submission.function_name == "production_run_cpu"
    assert submission.function_call.object_id == "fc-production"
    assert submission.metadata == {
        "stage": "production",
        "run_name": "source-r001",
        "source_run_name": "source",
    }
    assert production_kwargs == {
        "run_name": "source-r001",
        "simulation_time_ns": 2,
        "num_threads": 16,
        "use_openmp_threads": False,
    }


def test_shortmd_summary_node_emits_markdown_manifest(tmp_path: Path) -> None:
    node = ShortMDSummaryNode(replicates=2, max_parallel=4)
    context = NodeRunContext(
        run_id="run-1",
        node_id="summary",
        attempt_id="attempt-1",
        cache_dir=tmp_path / "cache",
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

    assert "WorkflowOrchestrator.*" in functions
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
        def remote(self, **kwargs):
            calls["remote"] = kwargs
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def spawn(self, **kwargs):
            calls["spawn"] = kwargs
            return "call-1"

    class FakeWorkflowOrchestrator:
        def __init__(self) -> None:
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "WorkflowOrchestrator",
        FakeWorkflowOrchestrator,
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

    assert "remote" not in calls
    assert calls["spawn"]["workflow"].name == "shortmd"
    definition = calls["spawn"]["workflow"].validate()
    prep_node = definition.nodes["prep-shortmd-run-alpha"].node
    replicate_node = definition.nodes["replicate-shortmd-run-alpha-r001"].node

    assert prep_node.run_name == "shortmd-run-alpha"
    assert replicate_node.source_run_name == "shortmd-run-alpha"
    assert replicate_node.replicate_run_name == "shortmd-run-alpha-r001"
    assert {"prep_cpu_function", "prep_gpu_function"}.isdisjoint(prep_node.__dict__)
    assert {
        "production_cpu_function",
        "production_gpu_function",
        "stats_function",
    }.isdisjoint(replicate_node.__dict__)
    assert prep_node.modal_namespace.clear is shortmd_workflow.clear_shortmd_gromacs_run
    assert prep_node.modal_namespace.prepare_cpu is gromacs_app.prepare_tpr_cpu
    assert prep_node.modal_namespace.prepare_gpu is gromacs_app.prepare_tpr_gpu
    assert (
        replicate_node.modal_namespace.production_cpu is gromacs_app.production_run_cpu
    )
    assert (
        replicate_node.modal_namespace.production_gpu is gromacs_app.production_run_gpu
    )
    assert (
        replicate_node.modal_namespace.collect_stats is gromacs_app.collect_traj_stats
    )
    assert calls["spawn"]["run_id"] == "shortmd-run"
    assert calls["spawn"]["force"] is False
    assert calls["spawn"]["max_ready_workers"] == 3
    stdout = strip_ansi(capsys.readouterr().out)
    assert "Submitting ShortMD workflow 'shortmd-run'" in stdout
    assert "1 input PDB(s)" in stdout
    assert "1 replicate(s)" in stdout


def test_submit_shortmd_workflow_dry_run_prints_dag_without_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "pdbs"
    input_dir.mkdir()
    input_dir.joinpath("alpha.pdb").write_text("ATOM\n", encoding="utf-8")

    class UnexpectedWorkflowOrchestrator:
        def __init__(self) -> None:
            pytest.fail("dry-run should not construct the orchestrator")

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "WorkflowOrchestrator",
        UnexpectedWorkflowOrchestrator,
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
    assert "[workflow] DAG graph: node_id [placement; class] <- dependency" in stdout
    assert (
        "[workflow]   prep-shortmd-run-alpha [remote; ShortMDPrepNode] <- -" in stdout
    )
    assert (
        "[workflow]   clone-shortmd-run-alpha-r001 "
        "[remote; ShortMDCloneNode] <- prep-shortmd-run-alpha" in stdout
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
            return "call-1"

    class FakeWorkflowOrchestrator:
        def __init__(self) -> None:
            self.run = FakeOrchestratorMethod()

    monkeypatch.setattr(
        shortmd_workflow.orchestrator,
        "WorkflowOrchestrator",
        FakeWorkflowOrchestrator,
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
    prep_node = definition.nodes["prep-shortmd-run-alpha"].node
    clone_node = definition.nodes["clone-shortmd-run-alpha-r001"].node

    assert prep_node.overwrite_existing is True
    assert clone_node.overwrite_clone is True
    assert "clone_function" not in clone_node.__dict__
    assert prep_node.modal_namespace.clear is shortmd_workflow.clear_shortmd_gromacs_run
    assert (
        clone_node.modal_namespace.clone is shortmd_workflow.clone_prepared_shortmd_run
    )
    assert calls["spawn"]["force"] is True
