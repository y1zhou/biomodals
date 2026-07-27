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


@dataclass(frozen=True, slots=True)
class PlannedOperation:
    """One operation's identity, dependency, and invocation metadata."""

    operation: str
    dependencies: tuple[str, ...]
    function_name: str
    traj_prefix: str | None = None
    include_simulation_time: bool = False
    save_processed_traj: bool = False


def _operation_plan(*, cpu_only: bool) -> tuple[PlannedOperation, ...]:
    """Build the selected fixed plan from one CPU/GPU decision."""
    prepare = "prepare_tpr_cpu" if cpu_only else "prepare_tpr_gpu"
    production = "production_run_cpu" if cpu_only else "production_run_gpu"
    return (
        PlannedOperation(prepare, (), prepare),
        PlannedOperation(NVT_ANALYSIS, (prepare,), "collect_traj_stats", "nvt_"),
        PlannedOperation(NPT_ANALYSIS, (prepare,), "collect_traj_stats", "npt_"),
        PlannedOperation(
            production,
            (prepare,),
            production,
            include_simulation_time=True,
        ),
        PlannedOperation(
            PRODUCTION_ANALYSIS,
            (production,),
            "collect_traj_stats",
            "production_",
            save_processed_traj=True,
        ),
    )


def prepare_operation(*, cpu_only: bool) -> str:
    """Select the established preparation function for one request."""
    return _operation_plan(cpu_only=cpu_only)[0].operation


def operation_dependencies(*, cpu_only: bool) -> dict[str, tuple[str, ...]]:
    """Return the fixed GROMACS graph in stable display/submission order."""
    return {
        operation.operation: operation.dependencies
        for operation in _operation_plan(cpu_only=cpu_only)
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
    planned = next(
        (
            candidate
            for candidate in _operation_plan(cpu_only=cpu_only)[1:]
            if candidate.operation == operation
        ),
        None,
    )
    if planned is None:
        raise ValueError(f"Unsupported GROMACS operation: {operation}")

    kwargs: dict[str, object] = {"run_name": run_name}
    if planned.traj_prefix is not None:
        kwargs["traj_prefix"] = planned.traj_prefix
    if planned.include_simulation_time:
        kwargs["simulation_time_ns"] = simulation_time_ns
    if planned.save_processed_traj:
        kwargs["save_processed_traj"] = True
    return ModalInvocation(function_name=planned.function_name, kwargs=kwargs)
