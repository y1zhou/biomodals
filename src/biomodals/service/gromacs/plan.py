"""Pure GROMACS operation graph and deployed-function invocation plan."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from biomodals.execution import (
    ExecutionPlan,
    NodeDependency,
    NodePlan,
    ProviderBinding,
    TaskPlan,
)
from biomodals.service.store import JobOperationRecord, JobOperationState

NVT_ANALYSIS = "collect_traj_stats:nvt_"
NPT_ANALYSIS = "collect_traj_stats:npt_"
PRODUCTION_ANALYSIS = "collect_traj_stats:production_"
FINAL_OPERATION = PRODUCTION_ANALYSIS
PREPARE_RESULT = "prepare_result"
REQUIRED_FUNCTIONS = (
    "prepare_tpr_cpu",
    "prepare_tpr_gpu",
    "collect_traj_stats",
    "production_run_cpu",
    "production_run_gpu",
)
_EXECUTION_PLAN_SCHEMA_VERSION = "1"


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


def execution_plan(
    *,
    cpu_only: bool,
    workload_run_key: str,
    pdb_sha256: str,
    simulation_time_ns: int,
    run_pdbfixer: bool,
) -> ExecutionPlan:
    """Express the established service workflow as one immutable kernel plan."""
    operations = _operation_plan(cpu_only=cpu_only)
    analysis_nodes = (
        NVT_ANALYSIS,
        NPT_ANALYSIS,
        PRODUCTION_ANALYSIS,
    )
    nodes = tuple(
        NodePlan(
            node_key=operation.operation,
            dependencies=tuple(
                NodeDependency(node_key=dependency)
                for dependency in operation.dependencies
            ),
        )
        for operation in operations
    ) + (
        NodePlan(
            node_key=PREPARE_RESULT,
            dependencies=tuple(
                NodeDependency(node_key=dependency) for dependency in analysis_nodes
            ),
        ),
    )
    return ExecutionPlan(
        workload_name="gromacs",
        workload_run_key=workload_run_key,
        nodes=nodes,
        scientific_payload={
            "cpu_only": cpu_only,
            "pdb_sha256": pdb_sha256,
            "run_pdbfixer": run_pdbfixer,
            "simulation_time_ns": simulation_time_ns,
        },
        scientific_versions={
            "biomodals.gromacs.execution_plan": _EXECUTION_PLAN_SCHEMA_VERSION,
        },
    )


def operation_task_plan(operation: str) -> TaskPlan:
    """Represent one GROMACS service stage as one scientific Task."""
    return TaskPlan(
        task_key="operation",
        scientific_payload={"operation": operation},
    )


def operation_provider_binding(
    operation: str,
    *,
    environment: str,
    app_name: str,
    app_version: int,
) -> ProviderBinding:
    """Bind one remote operation to its exact deployed GROMACS function."""
    function_name = operation.partition(":")[0]
    if function_name not in REQUIRED_FUNCTIONS:
        raise ValueError(f"Unsupported GROMACS operation: {operation}")
    uses_gpu = function_name.endswith("_gpu")
    return ProviderBinding(
        environment=environment,
        app_name=app_name,
        app_version=app_version,
        function_name=function_name,
        uses_gpu=uses_gpu,
        runtime_image_key="gromacs-gpu" if uses_gpu else "gromacs-cpu",
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
