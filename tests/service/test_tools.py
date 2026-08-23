"""Static Tool stage projection contracts."""

# ruff: noqa: D103

from types import SimpleNamespace

from biomodals.execution import (
    ActiveProviderCallCounts,
    NodeStatus,
    ProviderCallStatus,
)
from biomodals.service.tools import GROMACS_TOOL, project_overview


def test_projection_names_only_active_remote_functions() -> None:
    overview = SimpleNamespace(
        nodes=(
            SimpleNamespace(
                node_key="collect_traj_stats:nvt_",
                status=NodeStatus.RUNNING,
                started_at=10,
                completed_at=None,
            ),
            SimpleNamespace(
                node_key="prepare_result",
                status=NodeStatus.PENDING,
                started_at=None,
                completed_at=None,
            ),
        ),
        representative_provider_calls=(
            SimpleNamespace(
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
    assert stages["prepare_result"]["running_functions"] == []
