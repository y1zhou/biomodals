"""Execution plan behavior tests."""

# ruff: noqa: D103

import pytest

from biomodals.execution import (
    ExecutionPlan,
    NodeAggregationPolicy,
    NodeDependency,
    NodePlan,
    NodeStatus,
    propagated_skip_node_keys,
    ready_node_keys,
)


def test_plan_preserves_node_order_and_identifies_terminal_nodes() -> None:
    plan = ExecutionPlan(
        workload_name="short-md",
        nodes=(
            NodePlan(node_key="prepare"),
            NodePlan(
                node_key="production",
                dependencies=(NodeDependency(node_key="prepare"),),
            ),
            NodePlan(
                node_key="summary",
                dependencies=(NodeDependency(node_key="production"),),
            ),
        ),
    )

    assert plan.node_keys == ("prepare", "production", "summary")
    assert plan.terminal_node_keys == ("summary",)
    assert plan.nodes[0].aggregation_policy == NodeAggregationPolicy.COLLECT_ALL


def test_plan_rejects_duplicate_node_keys() -> None:
    with pytest.raises(ValueError, match="duplicate Node key 'prepare'"):
        ExecutionPlan(
            workload_name="short-md",
            nodes=(
                NodePlan(node_key="prepare"),
                NodePlan(node_key="prepare"),
            ),
        )


def test_plan_rejects_unknown_dependency() -> None:
    with pytest.raises(
        ValueError,
        match="Node 'summary' depends on unknown Node 'production'",
    ):
        ExecutionPlan(
            workload_name="short-md",
            nodes=(
                NodePlan(
                    node_key="summary",
                    dependencies=(NodeDependency(node_key="production"),),
                ),
            ),
        )


def test_plan_rejects_dependency_cycles() -> None:
    with pytest.raises(ValueError, match="execution plan contains a cycle"):
        ExecutionPlan(
            workload_name="cycle",
            nodes=(
                NodePlan(
                    node_key="a",
                    dependencies=(NodeDependency(node_key="c"),),
                ),
                NodePlan(
                    node_key="b",
                    dependencies=(NodeDependency(node_key="a"),),
                ),
                NodePlan(
                    node_key="c",
                    dependencies=(NodeDependency(node_key="b"),),
                ),
            ),
        )


def test_workload_plan_fingerprint_uses_canonical_scientific_content() -> None:
    plan = ExecutionPlan(
        workload_name="demo",
        workload_run_key="display-name-is-operational",
        nodes=(NodePlan(node_key="prepare"),),
        scientific_payload={"input_sha256": "abc"},
        scientific_versions={"tool": "2", "schema": "1"},
    )

    assert plan.workload_plan_fingerprint == (
        "a22d4d6c6e0706ec551b42a21d0588c89510209e8f8028ba1a6cd51328d8a3f1"
    )


def test_workload_plan_fingerprint_includes_node_aggregation_policy() -> None:
    strict = ExecutionPlan(
        workload_name="demo",
        nodes=(
            NodePlan(
                node_key="inference",
                aggregation_policy=NodeAggregationPolicy.COLLECT_ALL,
            ),
        ),
    )
    partial = ExecutionPlan(
        workload_name="demo",
        nodes=(
            NodePlan(
                node_key="inference",
                aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
            ),
        ),
    )

    assert strict.workload_plan_fingerprint != partial.workload_plan_fingerprint


def test_ready_nodes_require_accepted_dependency_outcomes() -> None:
    plan = ExecutionPlan(
        workload_name="partial-input",
        nodes=(
            NodePlan(node_key="search"),
            NodePlan(
                node_key="strict-summary",
                dependencies=(NodeDependency(node_key="search"),),
            ),
            NodePlan(
                node_key="partial-summary",
                dependencies=(NodeDependency(node_key="search", accept_partial=True),),
            ),
        ),
    )

    assert ready_node_keys(
        plan,
        {
            "search": NodeStatus.PENDING,
            "strict-summary": NodeStatus.PENDING,
            "partial-summary": NodeStatus.PENDING,
        },
    ) == ("search",)
    assert ready_node_keys(
        plan,
        {
            "search": NodeStatus.PARTIAL,
            "strict-summary": NodeStatus.PENDING,
            "partial-summary": NodeStatus.PENDING,
        },
    ) == ("partial-summary",)


def test_unacceptable_terminal_dependency_propagates_skips() -> None:
    plan = ExecutionPlan(
        workload_name="failed-chain",
        nodes=(
            NodePlan(node_key="search"),
            NodePlan(
                node_key="assemble",
                dependencies=(NodeDependency(node_key="search"),),
            ),
            NodePlan(
                node_key="summary",
                dependencies=(NodeDependency(node_key="assemble"),),
            ),
        ),
    )

    assert propagated_skip_node_keys(
        plan,
        {
            "search": NodeStatus.FAILED,
            "assemble": NodeStatus.PENDING,
            "summary": NodeStatus.PENDING,
        },
    ) == ("assemble", "summary")
