"""Pure GROMACS operation-plan contracts."""

# ruff: noqa: D101,D103

import pytest

from biomodals.app.bioinfo.gromacs_execution import (
    GROMACS_SCIENTIFIC_VERSION,
    PREPARE_RESULT,
    REQUIRED_FUNCTIONS,
    execution_plan,
    modal_invocation,
    operation_provider_binding,
    operation_task_plan,
)
from biomodals.execution import ProviderBinding
from biomodals.service.workloads import GROMACS_WORKLOAD


def test_preparation_fans_out_and_production_analysis_joins_production() -> None:
    plan = execution_plan(
        cpu_only=False,
        workload_run_key="simulation",
        pdb_sha256="abc123",
        simulation_time_ns=5,
        run_pdbfixer=False,
    )
    assert {
        node.node_key: tuple(item.node_key for item in node.dependencies)
        for node in plan.nodes[:-1]
    } == {
        "prepare_tpr_gpu": (),
        "collect_traj_stats:nvt_": ("prepare_tpr_gpu",),
        "collect_traj_stats:npt_": ("prepare_tpr_gpu",),
        "production_run_gpu": ("prepare_tpr_gpu",),
        "collect_traj_stats:production_": ("production_run_gpu",),
    }


@pytest.mark.parametrize("cpu_only", [False, True])
def test_every_planned_operation_has_public_stage_metadata(
    cpu_only: bool,
) -> None:
    plan = execution_plan(
        cpu_only=cpu_only,
        workload_run_key="simulation",
        pdb_sha256="abc123",
        simulation_time_ns=5,
        run_pdbfixer=False,
    )

    for operation_name in plan.node_keys[:-1]:
        stage = GROMACS_WORKLOAD.stage(operation_name)
        assert stage is not None
        assert stage.function_name in REQUIRED_FUNCTIONS


def test_invocations_reproduce_the_established_modal_function_contract() -> None:
    production = modal_invocation(
        "production_run_cpu",
        cpu_only=True,
        run_name="simulation-1",
        simulation_time_ns=25,
    )
    analysis = modal_invocation(
        "collect_traj_stats:production_",
        cpu_only=True,
        run_name="simulation-1",
        simulation_time_ns=25,
    )

    assert production.function_name == "production_run_cpu"
    assert production.kwargs == {
        "run_name": "simulation-1",
        "simulation_time_ns": 25,
    }
    assert analysis.function_name == "collect_traj_stats"
    assert analysis.kwargs == {
        "traj_prefix": "production_",
        "run_name": "simulation-1",
        "save_processed_traj": True,
    }


def test_execution_plan_preserves_parallel_gromacs_dag() -> None:
    plan = execution_plan(
        cpu_only=False,
        workload_run_key="protein-md-1234",
        pdb_sha256="abc123",
        simulation_time_ns=20,
        run_pdbfixer=True,
    )

    assert plan.workload_name == "gromacs"
    assert plan.workload_run_key == "protein-md-1234"
    assert plan.node_keys == (
        "prepare_tpr_gpu",
        "collect_traj_stats:nvt_",
        "collect_traj_stats:npt_",
        "production_run_gpu",
        "collect_traj_stats:production_",
        PREPARE_RESULT,
    )
    assert {dependency.node_key for dependency in plan.nodes[-1].dependencies} == {
        "collect_traj_stats:nvt_",
        "collect_traj_stats:npt_",
        "collect_traj_stats:production_",
    }
    assert plan.terminal_node_keys == (PREPARE_RESULT,)
    assert plan.scientific_payload == {
        "cpu_only": False,
        "gen_seed": -1,
        "genion_seed": 0,
        "ld_seed": -1,
        "pdb_sha256": "abc123",
        "run_pdbfixer": True,
        "simulation_time_ns": 20,
    }
    assert plan.scientific_versions["gromacs"] == GROMACS_SCIENTIFIC_VERSION


def test_execution_plan_fingerprint_excludes_workload_run_name() -> None:
    first = execution_plan(
        cpu_only=True,
        workload_run_key="first-name",
        pdb_sha256="abc123",
        simulation_time_ns=5,
        run_pdbfixer=False,
    )
    second = execution_plan(
        cpu_only=True,
        workload_run_key="another-name",
        pdb_sha256="abc123",
        simulation_time_ns=5,
        run_pdbfixer=False,
    )

    assert first.workload_plan_fingerprint == second.workload_plan_fingerprint


@pytest.mark.parametrize(
    ("operation_name", "uses_gpu", "function_name"),
    [
        ("prepare_tpr_gpu", True, "prepare_tpr_gpu"),
        ("prepare_tpr_cpu", False, "prepare_tpr_cpu"),
        ("production_run_gpu", True, "production_run_gpu"),
        ("collect_traj_stats:nvt_", False, "collect_traj_stats"),
    ],
)
def test_operation_binding_keeps_resources_out_of_scientific_identity(
    operation_name: str,
    uses_gpu: bool,
    function_name: str,
) -> None:
    binding = operation_provider_binding(
        operation_name,
        environment="production",
        app_name="Gromacs",
        app_version=23,
    )

    assert binding == ProviderBinding(
        environment="production",
        app_name="Gromacs",
        app_version=23,
        function_name=function_name,
        uses_gpu=uses_gpu,
        runtime_image_key=("gromacs-gpu" if uses_gpu else "gromacs-cpu"),
    )
    assert operation_task_plan(operation_name).scientific_payload == {
        "operation": operation_name
    }
