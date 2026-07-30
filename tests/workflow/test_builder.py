"""Tests for the Python workflow builder."""

# ruff: noqa: D101,D102,D103

from dataclasses import fields

import pytest

import biomodals.workflow as workflow_api
from biomodals.execution import NodeAggregationPolicy
from biomodals.schema import ArtifactKind
from biomodals.workflow import Workflow
from biomodals.workflow.core.builder import NodeHandle
from biomodals.workflow.core.execution import (
    execution_plan,
    node_task_plan,
)
from biomodals.workflow.core.nodes import WorkflowNativeNode


class DummyNode(WorkflowNativeNode):
    def run(self, context):  # pragma: no cover - builder tests do not execute nodes
        raise NotImplementedError


def test_selector_input_creates_data_dependency() -> None:
    workflow = Workflow("demo")
    upstream = workflow.add_node(DummyNode(), id="design")
    downstream = workflow.add_node(
        DummyNode(),
        id="score",
        inputs={
            "structures": upstream.outputs(
                kind=ArtifactKind.STRUCTURES,
                pattern="**/*.pdb",
            )
        },
    )

    definition = workflow.validate()

    assert definition.dependencies["score"] == {"design"}
    assert definition.nodes["score"].inputs["structures"].producing_node_id == "design"
    assert downstream.node_id == "score"


def test_node_handle_exposes_only_node_id_and_selector_api() -> None:
    assert [field.name for field in fields(NodeHandle)] == ["node_id"]
    assert not hasattr(workflow_api, "NodeOutputRef")


def test_depends_on_creates_control_edge() -> None:
    workflow = Workflow("demo")
    ranked = workflow.add_node(DummyNode(), id="ranked")
    packaged = workflow.add_node(DummyNode(), id="package", depends_on=[ranked])

    definition = workflow.validate()

    assert definition.dependencies["package"] == {"ranked"}
    assert definition.nodes["package"].control_dependencies == {"ranked"}
    assert packaged.node_id == "package"


def test_duplicate_node_ids_raise_value_error() -> None:
    workflow = Workflow("demo")
    workflow.add_node(DummyNode(), id="design")

    with pytest.raises(ValueError, match="Duplicate workflow node id"):
        workflow.add_node(DummyNode(), id="design")


def test_empty_sanitized_workflow_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="safe filename"):
        Workflow("///")


def test_cycles_raise_value_error() -> None:
    workflow = Workflow("demo")
    first = workflow.add_node(DummyNode(), id="first")
    second = workflow.add_node(DummyNode(), id="second", depends_on=[first])
    workflow.add_control_edge(second, first)

    with pytest.raises(ValueError, match="cycle"):
        workflow.validate()


def test_workflow_definition_maps_to_execution_plan_in_encounter_order() -> None:
    workflow = Workflow("demo")
    first = workflow.add_node(DummyNode(), id="first")
    workflow.add_node(
        DummyNode(),
        id="second",
        depends_on=[first],
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
    assert plan.nodes[1].aggregation_policy == NodeAggregationPolicy.ALLOW_PARTIAL
    assert plan.nodes[1].allow_empty_result is True
    assert plan.scientific_payload["dag_hash"]


def test_workflow_node_is_one_scientifically_identified_task() -> None:
    task = node_task_plan("score")

    assert task.task_key == "node"
    assert task.scientific_payload == {"workflow_node_id": "score"}
    assert task.execution_payload is None
