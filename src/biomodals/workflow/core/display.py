"""Workflow console display helpers."""

from __future__ import annotations

import sys

from biomodals.workflow.core.builder import WorkflowDefinition
from biomodals.workflow.core.nodes import RemoteWorkflowNode

__all__ = ["print_workflow_dag", "print_workflow_message"]


def print_workflow_message(renderable: object, *, style: str | None = None) -> None:
    """Print a workflow runtime message."""
    del style
    sys.stdout.write(f"{renderable}\n")
    sys.stdout.flush()


def print_workflow_dag(definition: WorkflowDefinition) -> None:
    """Print a compact workflow DAG graph."""
    print_workflow_message(
        "[workflow] DAG graph: node_id [execution; class] <- dependency",
        style="bold blue",
    )
    for node_id, spec in definition.nodes.items():
        dependencies = sorted(definition.dependencies[node_id])
        dependency_text = ", ".join(dependencies) if dependencies else "-"
        node_class = spec.node.__class__.__qualname__
        execution = (
            "provider"
            if isinstance(spec.node, RemoteWorkflowNode)
            else "coordinator-local"
        )
        print_workflow_message(
            f"[workflow]   {node_id} [{execution}; {node_class}] <- {dependency_text}"
        )
