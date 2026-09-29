"""Tests for the Python executable-graph builder."""

# ruff: noqa: D101,D102,D103

from dataclasses import dataclass, field

import pytest

from biomodals.execution import ExecutionPlanMetadata, NodeAggregationPolicy
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.nodes import CoordinatorNode
from biomodals.workflow import ExecutionGraph


class DummyNode(CoordinatorNode):
    def run(self, context):  # pragma: no cover - builder tests do not execute nodes
        raise NotImplementedError


@dataclass
class ConfiguredDummyNode(CoordinatorNode):
    config: dict[str, object] = field(metadata={"dag_hash_exclude_keys": ("workers",)})

    def run(self, context):  # pragma: no cover - builder tests do not execute nodes
        raise NotImplementedError


def test_duplicate_node_ids_raise_value_error() -> None:
    workflow = ExecutionGraph("demo")
    workflow.add_node(DummyNode(), id="design")

    with pytest.raises(ValueError, match="Duplicate Execution Node id"):
        workflow.add_node(DummyNode(), id="design")


def test_empty_sanitized_workflow_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="safe filename"):
        ExecutionGraph("///")


def test_cycles_raise_value_error() -> None:
    workflow = ExecutionGraph("demo")
    first = workflow.add_node(DummyNode(), id="first")
    second = workflow.add_node(DummyNode(), id="second", depends_on=[first])
    workflow.add_control_edge(second, first)

    with pytest.raises(ValueError, match="cycle"):
        workflow.validate()


def test_workflow_definition_maps_to_execution_plan_in_encounter_order() -> None:
    workflow = ExecutionGraph("demo", scientific_versions={"model": "v1"})
    first = workflow.add_node(DummyNode(), id="first")
    workflow.add_node(
        DummyNode(),
        id="second",
        depends_on=[first],
        accept_partial_from=[first],
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
        allow_empty_result=True,
    )
    definition = workflow.validate()

    plan = execution_plan(definition, workload_run_key="run-1")

    assert plan.workload_name == "workflow:demo"
    assert plan.workload_run_key == "run-1"
    assert plan.node_keys == ("first", "second")
    assert [dependency.node_key for dependency in plan.nodes[1].dependencies] == [
        "first"
    ]
    assert plan.nodes[1].dependencies[0].accept_partial is True
    assert plan.nodes[1].aggregation_policy == NodeAggregationPolicy.ALLOW_PARTIAL
    assert plan.nodes[1].allow_empty_result is True
    assert plan.scientific_payload["dag_hash"]
    assert plan.scientific_versions == {
        "biomodals.workflow.execution_plan": "1",
        "model": "v1",
    }


def test_explicit_plan_metadata_preserves_direct_app_identity() -> None:
    graph = ExecutionGraph(
        "direct-app",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="example",
            scientific_payload={"input_sha256": "abc"},
            scientific_versions={"example": "v4"},
        ),
    )
    graph.add_node(DummyNode(), id="run")

    plan = execution_plan(graph.validate(), workload_run_key="sample")

    assert plan.workload_name == "example"
    assert plan.workload_run_key == "sample"
    assert plan.scientific_payload == {"input_sha256": "abc"}
    assert plan.scientific_versions == {"example": "v4"}


def test_workflow_hash_can_exclude_declared_operational_config_keys() -> None:
    def fingerprint(*, workers: int, threshold: float) -> str:
        workflow = ExecutionGraph("demo")
        workflow.add_node(
            ConfiguredDummyNode({
                "workers": workers,
                "threshold": threshold,
                "nested": {"workers": 99},
            }),
            id="configured",
        )
        return execution_plan(
            workflow.validate(),
            workload_run_key="run-1",
        ).workload_plan_fingerprint

    baseline = fingerprint(workers=1, threshold=0.5)

    assert fingerprint(workers=8, threshold=0.5) == baseline
    assert fingerprint(workers=1, threshold=0.8) != baseline


def test_partial_acceptance_must_name_an_actual_dependency() -> None:
    workflow = ExecutionGraph("demo")
    first = workflow.add_node(DummyNode(), id="first")
    workflow.add_node(
        DummyNode(),
        id="second",
        accept_partial_from=[first],
    )

    with pytest.raises(ValueError, match="must name a Node dependency"):
        workflow.validate()
