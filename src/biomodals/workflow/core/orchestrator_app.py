"""Remote Modal orchestrator for Biomodals workflow runs."""

from __future__ import annotations

import os
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MAX_TIMEOUT
from biomodals.helper import patch_image_for_helper
from biomodals.schema import AppRunResult
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNode
from biomodals.workflow.core.orchestrator import (
    load_workflow_definition,
    run_workflow_definition,
    submit_workflow_run,
)

CONF = AppConfig(
    tags={"group": "workflow"},
    name="WorkflowOrchestrator",
    package_name="biomodals-workflow-orchestrator",
    version="0.1.0",
    python_version="3.12",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)
OUT_VOLUME = CONF.get_out_volume()
OUT_VOLUME_NAME = f"{CONF.name}-outputs"

runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version).env(CONF.default_env)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


def _run_workflow_orchestrator(
    workflow_name: str,
    run_id: str,
    workflow_definition: Workflow | dict[str, object],
    force: bool = False,
) -> AppRunResult:
    return run_workflow_definition(
        workflow_name=workflow_name,
        run_id=run_id,
        workflow_definition=workflow_definition,
        volume_root=Path(CONF.output_volume_mountpoint),
        workflow_volume_name=OUT_VOLUME_NAME,
        workflow_volume=OUT_VOLUME,
        remote_node_runner=run_remote_workflow_node.remote,
        force=force,
    )


@app.function(
    cpu=(1.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
def run_workflow_orchestrator(
    workflow_name: str,
    run_id: str,
    workflow_definition: Workflow | dict[str, object],
    force: bool = False,
) -> AppRunResult:
    """Run one workflow definition through the workflow runtime."""
    return _run_workflow_orchestrator(
        workflow_name=workflow_name,
        run_id=run_id,
        workflow_definition=workflow_definition,
        force=force,
    )


@app.function(
    cpu=(1.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
def run_remote_workflow_node(
    node: WorkflowNode,
    context: NodeRunContext,
) -> AppRunResult:
    """Run one failure-isolated workflow node in a separate Modal function."""
    OUT_VOLUME.reload()
    result = node.run(context)
    OUT_VOLUME.commit()
    return result


@app.local_entrypoint()
def submit_workflow_orchestrator_task(
    workflow_factory: str,
    run_id: str,
    workflow_name: str | None = None,
    force: bool = False,
    wait: bool = True,
) -> None:
    """Submit a Python workflow definition factory to the remote orchestrator.

    Args:
        workflow_factory: Import path in `module:function` form. The function
            must take no arguments and return a `Workflow` object.
        run_id: Stable workflow run id used for durable ledger paths.
        workflow_name: Optional workflow name. Defaults to the factory
            workflow's name.
        force: Replace an existing run ledger instead of resuming it.
        wait: Wait locally for the remote orchestrator result. Disable to print
            the Modal function call id for asynchronous collection.
    """
    workflow_definition = load_workflow_definition(workflow_factory)
    resolved_workflow_name = workflow_name or workflow_definition.name

    result = submit_workflow_run(
        orchestrator_function=run_workflow_orchestrator,
        workflow_name=resolved_workflow_name,
        run_id=run_id,
        workflow_definition=workflow_definition,
        force=force,
        wait=wait,
    )
    if isinstance(result, AppRunResult):
        print(f"Workflow run finished with status: {result.status}")
    else:
        print(f"Workflow run submitted. FunctionCall id: {result}")
