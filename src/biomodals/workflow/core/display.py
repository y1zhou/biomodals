"""Workflow console display helpers."""

from __future__ import annotations

import sys

from biomodals.workflow.core.builder import WorkflowDefinition

__all__ = ["print_workflow_dag", "print_workflow_message"]


def print_workflow_message(renderable: object, *, style: str | None = None) -> None:
    """Print a workflow runtime message."""
    del style
    sys.stdout.write(f"{renderable}\n")
    sys.stdout.flush()


def print_workflow_dag(definition: WorkflowDefinition) -> None:
    """Print a compact workflow DAG graph."""
    print_workflow_message(
        "[workflow] DAG graph: node_id [placement; class] <- dependency",
        style="bold blue",
    )
    for node_id, spec in definition.nodes.items():
        dependencies = sorted(definition.dependencies[node_id])
        dependency_text = ", ".join(dependencies) if dependencies else "-"
        node_class = spec.node.__class__.__qualname__
        print_workflow_message(
            "[workflow]   "
            f"{node_id} [{spec.node.placement.value}; {node_class}] <- "
            f"{dependency_text}"
        )
