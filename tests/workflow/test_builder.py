"""Tests for the Python workflow builder."""

# ruff: noqa: D101,D102,D103

import pytest

from biomodals.schema import ArtifactKind
from biomodals.workflow import Workflow
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


def test_cycles_raise_value_error() -> None:
    workflow = Workflow("demo")
    first = workflow.add_node(DummyNode(), id="first")
    second = workflow.add_node(DummyNode(), id="second", depends_on=[first])
    workflow.add_control_edge(second, first)

    with pytest.raises(ValueError, match="cycle"):
        workflow.validate()


def test_sibling_nodes_are_ready_after_shared_upstream_completes() -> None:
    workflow = Workflow("demo")
    upstream = workflow.add_node(DummyNode(), id="design")
    workflow.add_node(
        DummyNode(),
        id="score-a",
        inputs={"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)},
    )
    workflow.add_node(
        DummyNode(),
        id="score-b",
        inputs={"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)},
    )

    ready = workflow.ready_nodes(completed_node_ids={"design"})

    assert ready == ["score-a", "score-b"]
