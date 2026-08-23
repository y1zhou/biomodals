"""Static API Tool definitions and semantic stage projection."""

from __future__ import annotations

from dataclasses import dataclass

from biomodals.execution import ExecutionOverview, NodeStatus, TaskStatus


@dataclass(frozen=True, slots=True)
class ToolStageDefinition:
    """Map kernel Nodes into one user-facing scientific stage."""

    code: str
    label: str
    node_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Fixed API metadata for one explicitly registered Tool."""

    key: str
    display_name: str
    modal_app_name_environment: str
    modal_app_version_environment: str
    active_job_limit_environment: str
    default_modal_app_name: str
    default_modal_app_version: int
    default_active_job_limit: int
    default_max_active_provider_calls: int
    default_max_active_gpu_provider_calls: int
    stages: tuple[ToolStageDefinition, ...]
    job_logs_visible_to_owner_default: bool = False


GROMACS_TOOL = ToolDefinition(
    key="gromacs",
    display_name="GROMACS MD simulation",
    modal_app_name_environment="BIOMODALS_GROMACS_APP",
    modal_app_version_environment="BIOMODALS_GROMACS_APP_VERSION",
    active_job_limit_environment="BIOMODALS_GROMACS_ACTIVE_LIMIT",
    default_modal_app_name="Gromacs",
    default_modal_app_version=1,
    default_active_job_limit=2,
    default_max_active_provider_calls=3,
    default_max_active_gpu_provider_calls=1,
    stages=(
        ToolStageDefinition(
            "prepare_simulation",
            "Prepare simulation",
            ("prepare_tpr_cpu", "prepare_tpr_gpu"),
        ),
        ToolStageDefinition("analyze_nvt", "Analyze NVT", ("collect_traj_stats:nvt_",)),
        ToolStageDefinition("analyze_npt", "Analyze NPT", ("collect_traj_stats:npt_",)),
        ToolStageDefinition(
            "run_production",
            "Run production",
            ("production_run_cpu", "production_run_gpu"),
        ),
        ToolStageDefinition(
            "analyze_production",
            "Analyze production",
            ("collect_traj_stats:production_",),
        ),
        ToolStageDefinition("prepare_result", "Prepare result", ("prepare_result",)),
    ),
    job_logs_visible_to_owner_default=True,
)

ALPHAFOLD3_TOOL = ToolDefinition(
    key="alphafold3",
    display_name="AlphaFold3 structure prediction",
    modal_app_name_environment="BIOMODALS_ALPHAFOLD3_APP",
    modal_app_version_environment="BIOMODALS_ALPHAFOLD3_APP_VERSION",
    active_job_limit_environment="BIOMODALS_ALPHAFOLD3_ACTIVE_LIMIT",
    default_modal_app_name="AlphaFold3",
    default_modal_app_version=1,
    default_active_job_limit=2,
    default_max_active_provider_calls=4,
    default_max_active_gpu_provider_calls=1,
    stages=(
        ToolStageDefinition("prepare_input", "Prepare input", ("stage-request-input",)),
        ToolStageDefinition(
            "search_sequence_databases",
            "Search sequence databases",
            ("raw-database-searches", "combined-msa-publications"),
        ),
        ToolStageDefinition(
            "search_templates",
            "Search templates",
            ("protein-template-searches",),
        ),
        ToolStageDefinition(
            "predict_structures",
            "Predict structures",
            ("stage-inference-input", "seed-predictions"),
        ),
        ToolStageDefinition(
            "prepare_results",
            "Prepare results",
            ("inference-summary", "request-publication"),
        ),
    ),
)


def project_overview(
    definition: ToolDefinition,
    overview: ExecutionOverview,
) -> dict[str, object]:
    """Build the bounded semantic-stage projection cached by the service."""
    nodes = {node.node_key: node for node in overview.nodes}
    task_counts = {
        counts.node_key: {
            status.value: getattr(counts, status.value) for status in TaskStatus
        }
        for counts in overview.node_task_status_counts
    }
    stages: list[dict[str, object]] = []
    for stage in definition.stages:
        selected = [nodes[key] for key in stage.node_keys if key in nodes]
        if not selected:
            continue
        statuses = {node.status for node in selected}
        if NodeStatus.FAILED in statuses:
            outcome = "failed"
        elif statuses & {NodeStatus.CANCELLED, NodeStatus.SKIPPED}:
            outcome = "cancelled"
        elif statuses and all(status == NodeStatus.SUCCEEDED for status in statuses):
            outcome = "completed"
        else:
            outcome = None
        counts = {status.value: 0 for status in TaskStatus}
        for node in selected:
            for status, count in task_counts.get(node.node_key, {}).items():
                counts[status] += count
        stages.append({
            "code": stage.code,
            "label": stage.label,
            "node_keys": list(stage.node_keys),
            "started_at": min(
                (node.started_at for node in selected if node.started_at is not None),
                default=None,
            ),
            "ended_at": max(
                (
                    node.completed_at
                    for node in selected
                    if node.completed_at is not None
                ),
                default=None,
            ),
            "outcome": outcome,
            "task_counts": counts,
        })
    return {
        "stages": stages,
        "active_provider_calls": {
            "total": overview.active_provider_calls.total,
            "gpu": overview.active_provider_calls.gpu,
        },
        "warnings": [],
    }


TOOLS = (GROMACS_TOOL, ALPHAFOLD3_TOOL)
