"""Workflow-specific console display helpers."""

from __future__ import annotations

from biomodals.execution.graph import WorkflowDefinition
from biomodals.execution.nodes import RemoteTaskWorkflowNode, RemoteWorkflowNode

__all__ = ["print_workflow_dag"]


def print_workflow_dag(definition: WorkflowDefinition) -> None:
    """Print a compact workflow DAG graph."""
    print("[workflow] DAG graph: node_id [execution; class] <- dependency", flush=True)
    for node_id, spec in definition.nodes.items():
        dependencies = sorted(definition.dependencies[node_id])
        dependency_text = ", ".join(dependencies) if dependencies else "-"
        node_class = spec.node.__class__.__qualname__
        execution = (
            "provider"
            if isinstance(spec.node, RemoteWorkflowNode | RemoteTaskWorkflowNode)
            else "coordinator-local"
        )
        print(
            f"[workflow]   {node_id} [{execution}; {node_class}] <- {dependency_text}",
            flush=True,
        )
