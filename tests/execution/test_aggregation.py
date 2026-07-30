"""Node Task aggregation tests."""

# ruff: noqa: D103

from biomodals.execution import (
    NodeAggregationPolicy,
    NodeStatus,
    TaskStatus,
    aggregate_task_outcome,
)


def test_task_aggregation_policies_are_strict_and_non_vacuous() -> None:
    assert (
        aggregate_task_outcome(
            NodeAggregationPolicy.COLLECT_ALL,
            (TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED),
        )
        == NodeStatus.SUCCEEDED
    )
    assert (
        aggregate_task_outcome(
            NodeAggregationPolicy.COLLECT_ALL,
            (TaskStatus.SUCCEEDED, TaskStatus.FAILED),
        )
        == NodeStatus.FAILED
    )
    assert (
        aggregate_task_outcome(
            NodeAggregationPolicy.ALLOW_PARTIAL,
            (TaskStatus.SUCCEEDED, TaskStatus.FAILED),
        )
        == NodeStatus.PARTIAL
    )
    assert (
        aggregate_task_outcome(
            NodeAggregationPolicy.ALLOW_PARTIAL,
            (TaskStatus.FAILED, TaskStatus.SKIPPED),
        )
        == NodeStatus.FAILED
    )
    assert (
        aggregate_task_outcome(
            NodeAggregationPolicy.FAIL_FAST,
            (TaskStatus.FAILED, TaskStatus.RUNNING),
        )
        is None
    )
    assert aggregate_task_outcome(NodeAggregationPolicy.COLLECT_ALL, ()) is None
