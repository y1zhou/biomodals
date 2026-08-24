"""Static Tool stage projection contracts."""

# ruff: noqa: D103

from types import SimpleNamespace

from biomodals.execution import (
    ActiveProviderCallCounts,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    WorkStatusReason,
)
from biomodals.service.tools import GROMACS_TOOL, project_overview

CALL_HANDLE = "fc-provider"


def test_projection_names_only_active_remote_functions() -> None:
    overview = SimpleNamespace(
        run=SimpleNamespace(status=RunStatus.RUNNING, status_message=None),
        nodes=(
            SimpleNamespace(
                node_key="collect_traj_stats:nvt_",
                status=NodeStatus.RUNNING,
                status_reason=None,
                error_message=None,
                started_at=10,
                completed_at=None,
            ),
            SimpleNamespace(
                node_key="prepare_result",
                status=NodeStatus.PENDING,
                status_reason=None,
                error_message=None,
                started_at=None,
                completed_at=None,
            ),
        ),
        representative_provider_calls=(
            SimpleNamespace(
                provider_call_handle_id=CALL_HANDLE,
                node_key="collect_traj_stats:nvt_",
                status=ProviderCallStatus.RUNNING,
            ),
        ),
        active_provider_calls=ActiveProviderCallCounts(total=1, gpu=0),
        node_task_status_counts=(),
    )

    projection = project_overview(GROMACS_TOOL, overview)

    stages = {stage["code"]: stage for stage in projection["stages"]}
    assert stages["analyze_nvt"]["running_functions"] == ["collect_traj_stats"]
    assert stages["analyze_nvt"]["provider_state"] == "running"
    assert stages["prepare_result"]["running_functions"] == []
    assert stages["prepare_result"]["provider_state"] is None


def test_projection_presents_unassigned_provider_call_as_queued() -> None:
    overview = SimpleNamespace(
        run=SimpleNamespace(status=RunStatus.RUNNING, status_message=None),
        nodes=(
            SimpleNamespace(
                node_key="collect_traj_stats:nvt_",
                status=NodeStatus.RUNNING,
                status_reason=None,
                error_message=None,
                started_at=10,
                completed_at=None,
            ),
        ),
        representative_provider_calls=(
            SimpleNamespace(
                provider_call_handle_id=CALL_HANDLE,
                node_key="collect_traj_stats:nvt_",
                status=ProviderCallStatus.RUNNING,
            ),
        ),
        active_provider_calls=ActiveProviderCallCounts(total=1, gpu=0),
        node_task_status_counts=(),
    )

    projection = project_overview(
        GROMACS_TOOL,
        overview,
        queued_provider_call_handles=frozenset({CALL_HANDLE}),
    )

    [stage] = projection["stages"]
    assert stage["provider_state"] == "queued"


def test_projection_treats_cache_hits_and_partial_nodes_as_terminal() -> None:
    overview = SimpleNamespace(
        run=SimpleNamespace(
            status=RunStatus.PARTIAL,
            status_message="One prediction could not be produced",
        ),
        nodes=(
            SimpleNamespace(
                node_key="prepare_tpr_cpu",
                status=NodeStatus.SKIPPED,
                status_reason=WorkStatusReason.RESULT_ALREADY_SATISFIED,
                error_message=None,
                started_at=None,
                completed_at=10,
            ),
            SimpleNamespace(
                node_key="production_run_cpu",
                status=NodeStatus.PARTIAL,
                status_reason=None,
                error_message="One task failed",
                started_at=10,
                completed_at=20,
            ),
        ),
        representative_provider_calls=(),
        active_provider_calls=ActiveProviderCallCounts(total=0, gpu=0),
        node_task_status_counts=(),
    )

    projection = project_overview(GROMACS_TOOL, overview)
    stages = {stage["code"]: stage for stage in projection["stages"]}

    assert stages["prepare_simulation"]["outcome"] == "completed"
    assert stages["run_production"]["outcome"] == "partial"
    assert projection["warnings"] == ["Some results could not be produced"]
    assert "One prediction" not in str(projection)
    assert "One task failed" not in str(projection)
