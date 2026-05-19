"""Pure workflow orchestrator boundary helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from biomodals.schema import AppRunResult
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNode
from biomodals.workflow.core.runtime import (
    RemoteNodeRunner,
    WorkflowRuntime,
    WorkflowVolume,
)


def run_workflow_definition(
    *,
    workflow_name: str,
    run_id: str,
    workflow_definition: Workflow | dict[str, object],
    volume_root: Path,
    workflow_volume_name: str,
    force: bool = False,
    workflow_volume: WorkflowVolume | None = None,
    remote_node_runner: RemoteNodeRunner | None = None,
) -> AppRunResult:
    """Run one workflow definition through the workflow runtime."""
    runtime = WorkflowRuntime.from_definition(
        workflow_name=workflow_name,
        workflow_definition=workflow_definition,
        volume_root=volume_root,
        workflow_volume_name=workflow_volume_name,
        workflow_volume=workflow_volume,
        remote_node_runner=remote_node_runner,
    )
    return runtime.run(run_id=run_id, force=force)


def run_remote_node_with_volume(
    *,
    node: WorkflowNode,
    context: NodeRunContext,
    workflow_volume: WorkflowVolume,
) -> AppRunResult:
    """Run one remote workflow node and commit the volume in a finally block."""
    workflow_volume.reload()
    try:
        return node.run(context)
    finally:
        workflow_volume.commit()


def load_workflow_definition(factory_ref: str) -> Workflow:
    """Load a Python workflow definition from ``module:function``."""
    module_name, separator, function_name = factory_ref.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("workflow factory must use 'module:function' syntax")

    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Workflow factory is not callable: {factory_ref}")

    workflow_definition = factory()
    if isinstance(workflow_definition, Workflow):
        return workflow_definition
    raise TypeError("Workflow factory must return a Workflow object")


def submit_workflow_run(
    *,
    orchestrator_function: Any,
    workflow_name: str,
    run_id: str,
    workflow_definition: Workflow | dict[str, object],
    force: bool = False,
    wait: bool = True,
) -> AppRunResult | str:
    """Submit one workflow run to a remote orchestrator function."""
    kwargs = {
        "workflow_name": workflow_name,
        "run_id": run_id,
        "workflow_definition": workflow_definition,
        "force": force,
    }
    if wait:
        return AppRunResult.model_validate(orchestrator_function.remote(**kwargs))

    function_call = orchestrator_function.spawn(**kwargs)
    return str(getattr(function_call, "object_id", function_call))
