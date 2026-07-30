"""Result-driven DAG boundary tests."""

# ruff: noqa: D103

from biomodals.execution import (
    AvailabilityStatus,
    ExecutionPlan,
    NodeDependency,
    NodePlan,
    required_node_keys,
)


def _linear_plan() -> ExecutionPlan:
    return ExecutionPlan(
        workload_name="linear",
        nodes=(
            NodePlan(node_key="input"),
            NodePlan(
                node_key="compute",
                dependencies=(NodeDependency(node_key="input"),),
            ),
            NodePlan(
                node_key="summary",
                dependencies=(NodeDependency(node_key="compute"),),
            ),
        ),
    )


def test_required_closure_walks_backward_and_stops_at_available_results() -> None:
    plan = _linear_plan()

    assert (
        required_node_keys(
            plan,
            {
                "summary": AvailabilityStatus.AVAILABLE,
            },
        )
        == ()
    )
    assert required_node_keys(
        plan,
        {
            "summary": AvailabilityStatus.MISSING,
            "compute": AvailabilityStatus.AVAILABLE,
        },
    ) == ("summary",)
    assert required_node_keys(
        plan,
        {
            "summary": AvailabilityStatus.MISSING,
            "compute": AvailabilityStatus.MISSING,
            "input": AvailabilityStatus.MISSING,
        },
    ) == ("input", "compute", "summary")


def test_unknown_result_observation_authorizes_no_work() -> None:
    assert (
        required_node_keys(
            _linear_plan(),
            {
                "summary": AvailabilityStatus.UNKNOWN,
            },
        )
        is None
    )
