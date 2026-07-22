"""Fixed metadata that lets each API workload own its service contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkloadStageDefinition:
    """Map one provider operation to its stable public timeline fields."""

    operation: str
    code: str
    function_name: str | None


@dataclass(frozen=True, slots=True)
class WorkloadDefinition:
    """Static identity, Runtime Setting keys, and stages for one workload."""

    name: str
    display_name: str
    modal_app_name_environment: str
    modal_app_version_environment: str
    active_job_limit_environment: str
    stages: tuple[WorkloadStageDefinition, ...]

    def stage(self, operation: str) -> WorkloadStageDefinition | None:
        """Return the public stage owned by a provider operation."""
        return next(
            (stage for stage in self.stages if stage.operation == operation),
            None,
        )


GROMACS_WORKLOAD = WorkloadDefinition(
    name="gromacs",
    display_name="GROMACS MD simulation",
    modal_app_name_environment="BIOMODALS_GROMACS_APP",
    modal_app_version_environment="BIOMODALS_GROMACS_APP_VERSION",
    active_job_limit_environment="BIOMODALS_GROMACS_ACTIVE_LIMIT",
    stages=(
        WorkloadStageDefinition(
            "prepare_tpr_cpu",
            "prepare_simulation",
            "prepare_tpr_cpu",
        ),
        WorkloadStageDefinition(
            "prepare_tpr_gpu",
            "prepare_simulation",
            "prepare_tpr_gpu",
        ),
        WorkloadStageDefinition(
            "collect_traj_stats:nvt_",
            "analyze_nvt",
            "collect_traj_stats",
        ),
        WorkloadStageDefinition(
            "collect_traj_stats:npt_",
            "analyze_npt",
            "collect_traj_stats",
        ),
        WorkloadStageDefinition(
            "production_run_cpu",
            "run_production",
            "production_run_cpu",
        ),
        WorkloadStageDefinition(
            "production_run_gpu",
            "run_production",
            "production_run_gpu",
        ),
        WorkloadStageDefinition(
            "collect_traj_stats:production_",
            "analyze_production",
            "collect_traj_stats",
        ),
        WorkloadStageDefinition("result_packaging", "prepare_result", None),
    ),
)

WORKLOAD_DEFINITIONS = {GROMACS_WORKLOAD.name: GROMACS_WORKLOAD}
