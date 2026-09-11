"""Static Tool stage projection contracts."""

# ruff: noqa: D103

from types import SimpleNamespace

import pytest

from biomodals.execution import (
    ActiveProviderCallCounts,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    WorkStatusReason,
)
from biomodals.service.tools import GROMACS_TOOL, HUMANIZATION_TOOL, project_overview

CALL_HANDLE = "fc-provider"


def test_historical_humanization_plan_keeps_its_recorded_aggregate_stage():
    overview = SimpleNamespace(
        run=SimpleNamespace(status=RunStatus.RUNNING),
        nodes=(
            SimpleNamespace(
                node_key="generate",
                status=NodeStatus.RUNNING,
                status_reason=None,
                started_at=10,
                completed_at=None,
            ),
        ),
        representative_provider_calls=(),
        active_provider_calls=ActiveProviderCallCounts(total=0, gpu=0),
        node_task_status_counts=(),
    )
    [stage] = project_overview(HUMANIZATION_TOOL, overview)["stages"]
    assert (stage["code"], stage["label"], stage["started_at"]) == (
        "generate",
        "Generate candidates",
        10,
    )


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (NodeStatus.RUNNING, None),
        (NodeStatus.PARTIAL, "partial"),
        (NodeStatus.FAILED, "failed"),
        (NodeStatus.CANCELLED, "cancelled"),
    ],
)
def test_humanization_methods_keep_independent_status_timing_and_counts(
    status, outcome
):
    states = [
        NodeStatus.SUCCEEDED,
        NodeStatus.SUCCEEDED,
        status,
        NodeStatus.RUNNING,
        NodeStatus.PENDING,
        NodeStatus.PENDING,
    ]
    codes = [
        "generate_sapiens",
        "generate_humatch",
        "generate_pabnativ2",
        "generate_hudiff_ab",
        "union",
        "evaluate",
    ]
    overview = SimpleNamespace(
        run=SimpleNamespace(status=RunStatus.RUNNING),
        nodes=tuple(
            SimpleNamespace(
                node_key=code,
                status=state,
                status_reason=None,
                started_at=10 if index < 4 else None,
                completed_at=20 + index if state.is_terminal else None,
            )
            for index, (code, state) in enumerate(zip(codes, states, strict=True))
        ),
        representative_provider_calls=tuple(
            SimpleNamespace(
                node_key=code,
                provider_call_handle_id=f"fc-{code}",
                status=ProviderCallStatus.RUNNING,
            )
            for code, state in zip(codes, states, strict=True)
            if state == NodeStatus.RUNNING
        ),
        active_provider_calls=ActiveProviderCallCounts(total=2, gpu=2),
        node_task_status_counts=tuple(
            SimpleNamespace(
                node_key=code,
                pending=0,
                running=int(state == NodeStatus.RUNNING),
                succeeded=int(state == NodeStatus.SUCCEEDED),
                failed=0,
                cancelled=0,
                skipped=0,
            )
            for code, state in zip(codes, states, strict=True)
        ),
    )
    stages = project_overview(
        HUMANIZATION_TOOL,
        overview,
        queued_provider_call_handles=frozenset({"fc-generate_hudiff_ab"}),
    )["stages"]
    assert [stage["code"] for stage in stages] == codes
    assert [stage["label"] for stage in stages[:4]] == [
        "Sapiens",
        "Humatch",
        "p-AbNatiV2",
        "HuDiff",
    ]
    assert [stage["ended_at"] for stage in stages[:2]] == [20, 21]
    assert all(stage["outcome"] == "completed" for stage in stages[:2])
    assert stages[2]["outcome"] == outcome
    assert stages[2]["ended_at"] == (22 if status.is_terminal else None)
    assert stages[3]["provider_state"] == "queued"
    assert stages[3]["task_counts"]["running"] == 1
    assert stages[4]["started_at"] is None
    assert stages[5]["outcome"] is None
    assert [stage["node_keys"] for stage in stages] == [[code] for code in codes]


def test_projection_marks_only_active_remote_stages() -> None:
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
    assert stages["analyze_nvt"]["provider_state"] == "running"
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


def test_projection_presents_pending_remote_tasks_as_queued() -> None:
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
        representative_provider_calls=(),
        active_provider_calls=ActiveProviderCallCounts(total=0, gpu=0),
        node_task_status_counts=(
            SimpleNamespace(
                node_key="collect_traj_stats:nvt_",
                pending=1,
                running=0,
                succeeded=0,
                failed=0,
                cancelled=0,
                skipped=0,
            ),
        ),
    )

    [stage] = project_overview(GROMACS_TOOL, overview)["stages"]

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
