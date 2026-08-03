"""Python-first workflow DAG builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from biomodals.execution import NodeAggregationPolicy
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import ArtifactKind, ArtifactSelector
from biomodals.workflow.core.nodes import WorkflowNode


@dataclass(frozen=True)
class NodeHandle:
    """Stable handle returned after adding a node to a workflow."""

    node_id: str

    def outputs(
        self,
        kind: ArtifactKind | None = None,
        pattern: str | None = None,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactSelector:
        """Select artifacts produced by this node."""
        return ArtifactSelector(
            producing_node_id=self.node_id,
            kind=kind,
            pattern=pattern,
            role=role,
            metadata=metadata or {},
        )


@dataclass
class WorkflowNodeSpec:
    """Builder-time node metadata."""

    node_id: str
    node: WorkflowNode
    aggregation_policy: NodeAggregationPolicy = NodeAggregationPolicy.COLLECT_ALL
    allow_empty_result: bool = False
    inputs: dict[str, ArtifactSelector] = field(default_factory=dict)
    control_dependencies: set[str] = field(default_factory=set)
    partial_dependencies: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class WorkflowDefinition:
    """Validated workflow DAG definition."""

    name: str
    nodes: dict[str, WorkflowNodeSpec]
    dependencies: dict[str, set[str]]
    scientific_versions: dict[str, str]


class Workflow:
    """Python-first workflow DAG builder."""

    def __init__(
        self,
        name: str,
        *,
        scientific_versions: Mapping[str, str] | None = None,
    ):
        """Initialize an empty workflow definition."""
        self.name = sanitize_filename(name)
        self.scientific_versions = dict(scientific_versions or {})
        self._nodes: dict[str, WorkflowNodeSpec] = {}

    def add_node(
        self,
        node: WorkflowNode,
        *,
        id: str,
        inputs: dict[str, ArtifactSelector] | None = None,
        depends_on: list[NodeHandle | str] | None = None,
        accept_partial_from: list[NodeHandle | str] | None = None,
        aggregation_policy: NodeAggregationPolicy = NodeAggregationPolicy.COLLECT_ALL,
        allow_empty_result: bool = False,
    ) -> NodeHandle:
        """Add one node to the workflow and return its handle."""
        node_id = sanitize_filename(id)
        if node_id in self._nodes:
            raise ValueError(f"Duplicate workflow node id: {node_id}")

        control_dependencies = {
            dependency.node_id if isinstance(dependency, NodeHandle) else dependency
            for dependency in depends_on or []
        }
        partial_dependencies = {
            dependency.node_id if isinstance(dependency, NodeHandle) else dependency
            for dependency in accept_partial_from or []
        }
        self._nodes[node_id] = WorkflowNodeSpec(
            node_id=node_id,
            node=node,
            aggregation_policy=aggregation_policy,
            allow_empty_result=allow_empty_result,
            inputs=inputs or {},
            control_dependencies=control_dependencies,
            partial_dependencies=partial_dependencies,
        )
        return NodeHandle(node_id=node_id)

    def add_control_edge(
        self,
        upstream: NodeHandle | str,
        downstream: NodeHandle | str,
    ) -> None:
        """Add an ordering-only dependency between two existing nodes."""
        upstream_id = upstream.node_id if isinstance(upstream, NodeHandle) else upstream
        downstream_id = (
            downstream.node_id if isinstance(downstream, NodeHandle) else downstream
        )
        self._nodes[downstream_id].control_dependencies.add(upstream_id)

    def validate(self) -> WorkflowDefinition:
        """Validate the workflow DAG and return an immutable definition."""
        dependencies = self._dependencies()
        missing = {
            dependency
            for node_dependencies in dependencies.values()
            for dependency in node_dependencies
            if dependency not in self._nodes
        }
        if missing:
            raise ValueError(f"Unknown workflow node dependencies: {sorted(missing)}")
        invalid_partial = {
            node_id: sorted(spec.partial_dependencies - dependencies[node_id])
            for node_id, spec in self._nodes.items()
            if spec.partial_dependencies - dependencies[node_id]
        }
        if invalid_partial:
            raise ValueError(
                "Partial-result acceptance must name a Node dependency: "
                f"{invalid_partial}"
            )
        self._raise_for_cycles(dependencies)
        return WorkflowDefinition(
            name=self.name,
            nodes=dict(self._nodes),
            dependencies=dependencies,
            scientific_versions=dict(self.scientific_versions),
        )

    def _dependencies(self) -> dict[str, set[str]]:
        dependencies: dict[str, set[str]] = {}
        for node_id, spec in self._nodes.items():
            input_dependencies = {
                selector.producing_node_id for selector in spec.inputs.values()
            }
            dependencies[node_id] = input_dependencies | spec.control_dependencies
        return dependencies

    @staticmethod
    def _raise_for_cycles(dependencies: dict[str, set[str]]) -> None:
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise ValueError("Workflow DAG contains a cycle")
            temporary.add(node_id)
            for dependency in dependencies.get(node_id, set()):
                if dependency in dependencies:
                    visit(dependency)
            temporary.remove(node_id)
            permanent.add(node_id)

        for node_id in dependencies:
            visit(node_id)
