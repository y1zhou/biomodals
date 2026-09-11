"""Static API Tool definitions and semantic stage projection."""

from __future__ import annotations

from dataclasses import dataclass

from biomodals.execution import (
    ExecutionOverview,
    NodeStatus,
    RunStatus,
    TaskStatus,
    WorkStatusReason,
)


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
    stages=(
        ToolStageDefinition(
            "prepare_simulation",
            "Prepare simulation",
            ("prepare_tpr_cpu", "prepare_tpr_gpu"),
        ),
        ToolStageDefinition(
            "analyze_nvt",
            "Analyze NVT",
            ("collect_traj_stats:nvt_",),
        ),
        ToolStageDefinition(
            "analyze_npt",
            "Analyze NPT",
            ("collect_traj_stats:npt_",),
        ),
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
        ToolStageDefinition(
            "prepare_result",
            "Prepare result",
            ("prepare_result",),
        ),
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
    stages=(
        ToolStageDefinition(
            "prepare_input",
            "Prepare input",
            ("stage-request-input",),
        ),
        ToolStageDefinition(
            "prepare_environment",
            "Prepare environment",
            ("prepare-environment",),
        ),
        ToolStageDefinition(
            "search_sequence_databases",
            "Search MSAs",
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
    *,
    queued_provider_call_handles: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Build the bounded semantic-stage projection cached by the service."""
    nodes = {node.node_key: node for node in overview.nodes}
    task_counts = {
        counts.node_key: {
            status.value: getattr(counts, status.value) for status in TaskStatus
        }
        for counts in overview.node_task_status_counts
    }
    active_calls = {
        call.node_key: call
        for call in overview.representative_provider_calls
        if not call.status.is_terminal
    }
    stages: list[dict[str, object]] = []
    for stage in definition.stages:
        selected = [nodes[key] for key in stage.node_keys if key in nodes]
        if not selected:
            continue
        statuses = {node.status for node in selected}
        cache_satisfied = all(
            node.status == NodeStatus.SUCCEEDED
            or (
                node.status == NodeStatus.SKIPPED
                and node.status_reason == WorkStatusReason.RESULT_ALREADY_SATISFIED
            )
            for node in selected
        )
        if NodeStatus.FAILED in statuses:
            outcome = "failed"
        elif NodeStatus.PARTIAL in statuses:
            outcome = "partial"
        elif cache_satisfied:
            outcome = "completed"
        elif statuses & {NodeStatus.CANCELLED, NodeStatus.SKIPPED}:
            outcome = "cancelled"
        else:
            outcome = None
        counts = {status.value: 0 for status in TaskStatus}
        for node in selected:
            for status, count in task_counts.get(node.node_key, {}).items():
                counts[status] += count
        stage_calls = [
            active_calls[node_key]
            for node_key in stage.node_keys
            if node_key in active_calls
        ]
        provider_state = (
            "queued"
            if stage_calls
            and all(
                call.provider_call_handle_id in queued_provider_call_handles
                for call in stage_calls
            )
            else "queued"
            if not stage_calls and counts[TaskStatus.PENDING.value] > 0
            else "running"
            if stage_calls
            else None
        )
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
            "provider_state": provider_state,
        })
    warnings = (
        ["Some results could not be produced"]
        if overview.run.status == RunStatus.PARTIAL
        else []
    )
    return {
        "stages": stages,
        "active_provider_calls": {
            "total": overview.active_provider_calls.total,
            "gpu": overview.active_provider_calls.gpu,
        },
        "warnings": warnings,
    }


HUMANIZATION_TOOL = ToolDefinition(
    key="humanization",
    display_name="Antibody humanization",
    modal_app_name_environment="BIOMODALS_HUMANIZATION_APP",
    modal_app_version_environment="BIOMODALS_HUMANIZATION_APP_VERSION",
    active_job_limit_environment="BIOMODALS_HUMANIZATION_ACTIVE_LIMIT",
    default_modal_app_name="HumanizationWorkflow",
    default_modal_app_version=1,
    default_active_job_limit=2,
    stages=(
        ToolStageDefinition("generate_sapiens", "Sapiens", ("generate_sapiens",)),
        ToolStageDefinition("generate_humatch", "Humatch", ("generate_humatch",)),
        ToolStageDefinition(
            "generate_pabnativ2", "p-AbNatiV2", ("generate_pabnativ2",)
        ),
        ToolStageDefinition("generate_hudiff_ab", "HuDiff", ("generate_hudiff_ab",)),
        # Previously deployed plans contain only this aggregate Node.
        ToolStageDefinition("generate", "Generate candidates", ("generate",)),
        ToolStageDefinition("union", "Collect unique candidates", ("union",)),
        ToolStageDefinition("evaluate", "Evaluate and rank candidates", ("evaluate",)),
    ),
)

TOOLS = (GROMACS_TOOL, ALPHAFOLD3_TOOL, HUMANIZATION_TOOL)
