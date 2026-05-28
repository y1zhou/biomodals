"""Workflow orchestrator helpers and Modal class boundary."""

from __future__ import annotations

import os
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import (
    MAX_TIMEOUT,
    WORKFLOW_ORCHESTRATOR_VOLUME,
    WORKFLOW_ORCHESTRATOR_VOLUME_NAME,
)
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
OUT_VOLUME = WORKFLOW_ORCHESTRATOR_VOLUME
OUT_VOLUME_NAME = WORKFLOW_ORCHESTRATOR_VOLUME_NAME
REMOTE_NODE_FUNCTION_NAME = "run_node"

runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version).env(CONF.default_env),
    include_workflow_modules=True,
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


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
        self._close_runtime()
        self._exit_cleanup_done = False
        OUT_VOLUME.reload()

    @modal.method()
    def run(
        self,
        workflow: Workflow,
        run_id: str,
        force: bool = False,
        max_ready_workers: int = 32,
    ) -> AppRunResult:
        """Run one workflow definition through the workflow runtime."""
        if not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow object")

        def remote_node_runner(
            node: WorkflowNode,
            context: NodeRunContext,
        ) -> RemoteFunctionCall:
            function_call = self.run_node.spawn(node, context)
            if not hasattr(function_call, "object_id") or not hasattr(
                function_call, "get"
            ):
                raise TypeError(
                    "Remote workflow node spawn did not return a FunctionCall"
                )
            return function_call

        OUT_VOLUME.reload()
        self._runtime = WorkflowRuntime(
            workflow=workflow,
            volume_root=Path(CONF.output_volume_mountpoint),
            workflow_volume_name=OUT_VOLUME_NAME,
            workflow_volume=OUT_VOLUME,
            remote_node_runner=remote_node_runner,
            remote_node_function_name=REMOTE_NODE_FUNCTION_NAME,
            function_call_resolver=modal.FunctionCall.from_id,
            max_ready_workers=max_ready_workers,
        )
        try:
            return self._runtime.run(run_id=run_id, force=force)
        finally:
            self._close_runtime()
            OUT_VOLUME.commit()

    @modal.method()
    def run_node(
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
        if not getattr(self, "_exit_cleanup_done", False):
            self._exit_cleanup_done = True
            runtime = getattr(self, "_runtime", None)
            if runtime is not None:
                cancel_active_remote_calls = getattr(
                    runtime,
                    "cancel_active_remote_calls",
                    None,
                )
                if cancel_active_remote_calls is not None:
                    try:
                        cancel_active_remote_calls(terminate_containers=True)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            "[workflow] Remote call cleanup failed during "
                            f"orchestrator exit: {exc}",
                            flush=True,
                        )
        self._close_runtime()
        OUT_VOLUME.commit()

    def _close_runtime(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
            self._runtime = None
