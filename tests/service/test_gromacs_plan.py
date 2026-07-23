"""Pure GROMACS operation-plan contracts."""

# ruff: noqa: D101,D103

from uuid import UUID

import pytest

from biomodals.service.gromacs.plan import (
    REQUIRED_FUNCTIONS,
    all_operations_completed,
    modal_invocation,
    operation_dependencies,
    ready_operations,
)
from biomodals.service.store import (
    JobOperationExecutor,
    JobOperationRecord,
    JobOperationState,
)
from biomodals.service.workloads import GROMACS_WORKLOAD

JOB_ID = UUID("11111111-1111-4111-8111-111111111111")


def operation(
    name: str,
    state: JobOperationState,
    ordinal: int,
) -> JobOperationRecord:
    return JobOperationRecord(
        job_id=JOB_ID,
        operation=name,
        ordinal=ordinal,
        executor=JobOperationExecutor.MODAL,
        modal_call_id=f"fc-{ordinal}",
        state=state,
        submission_token=None,
        submission_lease_until=None,
        started_at=ordinal,
        completed_at=(ordinal + 1 if state == JobOperationState.COMPLETED else None),
    )


def test_preparation_fans_out_and_production_analysis_joins_production() -> None:
    assert operation_dependencies(cpu_only=False) == {
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
    operations = operation_dependencies(cpu_only=cpu_only)

    for operation_name in operations:
        stage = GROMACS_WORKLOAD.stage(operation_name)
        assert stage is not None
        assert stage.function_name in REQUIRED_FUNCTIONS


def test_ready_operations_preserve_parallel_plan_order() -> None:
    completed_prepare = [operation("prepare_tpr_gpu", JobOperationState.COMPLETED, 0)]

    assert ready_operations(
        cpu_only=False,
        operations=completed_prepare,
    ) == [
        "collect_traj_stats:nvt_",
        "collect_traj_stats:npt_",
        "production_run_gpu",
    ]
    assert not all_operations_completed(
        cpu_only=False,
        operations=completed_prepare,
    )


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
