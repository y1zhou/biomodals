"""Pure GROMACS operation graph and deployed-function invocation plan."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from biomodals.service.store import JobOperationRecord, JobOperationState

NVT_ANALYSIS = "collect_traj_stats:nvt_"
NPT_ANALYSIS = "collect_traj_stats:npt_"
PRODUCTION_ANALYSIS = "collect_traj_stats:production_"
FINAL_OPERATION = PRODUCTION_ANALYSIS
REQUIRED_FUNCTIONS = (
    "prepare_tpr_cpu",
    "prepare_tpr_gpu",
    "collect_traj_stats",
    "production_run_cpu",
    "production_run_gpu",
)


@dataclass(frozen=True, slots=True)
class ModalInvocation:
    """One deployed function name and its established keyword arguments."""

    function_name: str
    kwargs: dict[str, object]


def prepare_operation(*, cpu_only: bool) -> str:
    """Select the established preparation function for one request."""
    return "prepare_tpr_cpu" if cpu_only else "prepare_tpr_gpu"


def operation_dependencies(*, cpu_only: bool) -> dict[str, tuple[str, ...]]:
    """Return the fixed GROMACS graph in stable display/submission order."""
    prepare = prepare_operation(cpu_only=cpu_only)
    production = "production_run_cpu" if cpu_only else "production_run_gpu"
    return {
        prepare: (),
        NVT_ANALYSIS: (prepare,),
        NPT_ANALYSIS: (prepare,),
        production: (prepare,),
        PRODUCTION_ANALYSIS: (production,),
    }


def ready_operations(
    *,
    cpu_only: bool,
    operations: Iterable[JobOperationRecord],
) -> list[str]:
    """Return every dependency-satisfied operation not yet in the ledger."""
    dependencies = operation_dependencies(cpu_only=cpu_only)
    operation_list = list(operations)
    known = {operation.operation for operation in operation_list}
    completed = {
        operation.operation
        for operation in operation_list
        if operation.state == JobOperationState.COMPLETED
    }
    return [
        operation
        for operation, requirements in tuple(dependencies.items())[1:]
        if operation not in known and all(item in completed for item in requirements)
    ]


def all_operations_completed(
    *,
    cpu_only: bool,
    operations: Iterable[JobOperationRecord],
) -> bool:
    """Return whether every operation in the fixed plan completed."""
    completed = {
        operation.operation
        for operation in operations
        if operation.state == JobOperationState.COMPLETED
    }
    return set(operation_dependencies(cpu_only=cpu_only)).issubset(completed)


def modal_invocation(
    operation: str,
    *,
    cpu_only: bool,
    run_name: str,
    simulation_time_ns: int,
) -> ModalInvocation:
    """Build established Modal function arguments for one successor operation."""
    production_operation = "production_run_cpu" if cpu_only else "production_run_gpu"
    if operation not in {
        NVT_ANALYSIS,
        NPT_ANALYSIS,
        production_operation,
        PRODUCTION_ANALYSIS,
    }:
        raise ValueError(f"Unsupported GROMACS operation: {operation}")

    function_name, _, traj_prefix = operation.partition(":")
    kwargs: dict[str, object]
    if function_name.startswith("production_run_"):
        kwargs = {
            "run_name": run_name,
            "simulation_time_ns": simulation_time_ns,
        }
    elif operation == PRODUCTION_ANALYSIS:
        kwargs = {
            "traj_prefix": traj_prefix,
            "run_name": run_name,
            "save_processed_traj": True,
        }
    else:
        kwargs = {
            "traj_prefix": traj_prefix,
            "run_name": run_name,
        }
    return ModalInvocation(function_name=function_name, kwargs=kwargs)
