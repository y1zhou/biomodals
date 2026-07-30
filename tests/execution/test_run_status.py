"""Execution Run outcome tests."""

# ruff: noqa: D103

from biomodals.execution import (
    ExecutionPlan,
    NodeDependency,
    NodePlan,
    NodeStatus,
    RunStatus,
    terminal_run_outcome,
)


def _branched_plan() -> ExecutionPlan:
    return ExecutionPlan(
        workload_name="branched",
        nodes=(
            NodePlan(node_key="source"),
            NodePlan(
                node_key="result-a",
                dependencies=(NodeDependency(node_key="source"),),
            ),
            NodePlan(
                node_key="result-b",
                dependencies=(NodeDependency(node_key="source"),),
            ),
        ),
    )


def test_run_outcome_is_strictly_driven_by_terminal_nodes() -> None:
    plan = _branched_plan()

    assert (
        terminal_run_outcome(
            plan,
            {
                "source": NodeStatus.FAILED,
                "result-a": NodeStatus.SUCCEEDED,
                "result-b": NodeStatus.PARTIAL,
            },
        )
        == RunStatus.PARTIAL
    )
    assert (
        terminal_run_outcome(
            plan,
            {
                "source": NodeStatus.SUCCEEDED,
                "result-a": NodeStatus.SUCCEEDED,
                "result-b": NodeStatus.FAILED,
            },
        )
        == RunStatus.FAILED
    )
    assert (
        terminal_run_outcome(
            plan,
            {
                "source": NodeStatus.SUCCEEDED,
                "result-a": NodeStatus.CANCELLED,
                "result-b": NodeStatus.FAILED,
            },
        )
        == RunStatus.CANCELLED
    )
