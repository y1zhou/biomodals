"""Workflow orchestrator helpers and Modal class boundary."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import cast

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MAX_TIMEOUT
from biomodals.helper import patch_image_for_helper
from biomodals.helper.catalog import BiomodalsApp, get_catalog
from biomodals.schema import AppRunResult
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNode
from biomodals.workflow.core.runtime import RemoteFunctionCall, WorkflowRuntime

CONF = AppConfig(
    tags={"group": "workflow"},
    name="WorkflowOrchestrator",
    package_name="biomodals-workflow-orchestrator",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)
OUT_VOLUME = CONF.get_out_volume()
OUT_VOLUME_NAME = f"{CONF.name}-outputs"
REMOTE_NODE_FUNCTION_NAME = "WorkflowOrchestrator.run_remote_workflow_node"

runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version).env(CONF.default_env)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


def load_workflow_definition(factory_ref: str) -> Workflow:
    """Load a Python workflow definition from ``module:function``."""
    module_name, separator, function_name = factory_ref.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("workflow factory must use 'module:function' syntax")

    module = _import_workflow_factory_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Workflow factory is not callable: {factory_ref}")

    workflow_definition = factory()
    if isinstance(workflow_definition, Workflow):
        return workflow_definition
    raise TypeError("Workflow factory must return a Workflow object")


def _import_workflow_factory_module(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        catalog_module = _catalog_workflow_module(module_name)
        if catalog_module is None:
            raise
        return importlib.import_module(catalog_module)


def _catalog_workflow_module(module_name: str) -> str | None:
    all_workflows = get_catalog("workflow")
    module_path = Path(module_name).expanduser()
    if module_name not in all_workflows and not module_path.exists():
        return None
    return BiomodalsApp(module_name, all_workflows).module


def submit_workflow_run(
    *,
    orchestrator_function: modal.Function,
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


def _spawn_remote_workflow_node(
    remote_method: modal.Function,
    node: WorkflowNode,
    context: NodeRunContext,
) -> RemoteFunctionCall:
    return cast(RemoteFunctionCall, remote_method.spawn(node, context))


@app.cls(
    cpu=(1.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
class WorkflowOrchestrator:
    """Modal-hosted coordinator for one workflow run."""

    @modal.enter()
    def enter(self) -> None:
        """Refresh the workflow volume before serving orchestrator methods."""
        OUT_VOLUME.reload()

    @modal.method()
    def run_workflow_orchestrator(
        self,
        workflow_name: str,
        run_id: str,
        workflow_definition: Workflow | dict[str, object],
        force: bool = False,
    ) -> AppRunResult:
        """Run one workflow definition through the workflow runtime."""
        orchestrator_handle = WorkflowOrchestrator()

        def remote_node_runner(
            node: WorkflowNode,
            context: NodeRunContext,
        ) -> RemoteFunctionCall:
            return _spawn_remote_workflow_node(
                orchestrator_handle.run_remote_workflow_node,
                node,
                context,
            )

        runtime = WorkflowRuntime.from_definition(
            workflow_name=workflow_name,
            workflow_definition=workflow_definition,
            volume_root=Path(CONF.output_volume_mountpoint),
            workflow_volume_name=OUT_VOLUME_NAME,
            workflow_volume=OUT_VOLUME,
            remote_node_runner=remote_node_runner,
            remote_node_function_name=REMOTE_NODE_FUNCTION_NAME,
            function_call_resolver=modal.FunctionCall.from_id,
        )
        return runtime.run(run_id=run_id, force=force)

    @modal.method()
    def run_remote_workflow_node(
        self,
        node: WorkflowNode,
        context: NodeRunContext,
    ) -> AppRunResult:
        """Run one failure-isolated workflow node in a separate Modal method call."""
        OUT_VOLUME.reload()
        try:
            return node.run(context)
        finally:
            OUT_VOLUME.commit()

    @modal.exit()
    def exit(self) -> None:
        """Persist any pending workflow volume writes on container shutdown."""
        OUT_VOLUME.commit()


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
    orchestrator_handle = WorkflowOrchestrator()

    result = submit_workflow_run(
        orchestrator_function=orchestrator_handle.run_workflow_orchestrator,
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
