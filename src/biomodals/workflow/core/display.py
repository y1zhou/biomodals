"""Workflow console display helpers."""

from __future__ import annotations

from biomodals.helper.styling import print_rich, styled_text
from biomodals.workflow.core.builder import WorkflowDefinition

__all__ = ["print_workflow_dag", "print_workflow_message"]


def print_workflow_message(renderable: object, *, style: str | None = None) -> None:
    """Print a workflow runtime message with workflow-specific color controls."""
    print_rich(renderable, style=style, color_env_var="BIOMODALS_WORKFLOW_COLOR")


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
            styled_text(
                ("[workflow]   ", "grey50"),
                (node_id, "yellow" if dependencies else "bold yellow"),
                (" [", "grey50"),
                (spec.node.placement.value, "bold"),
                ("; ", "grey50"),
                (node_class, "bold"),
                ("] <- ", "grey50"),
                (dependency_text, "grey50"),
            )
        )
