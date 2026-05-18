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
from biomodals.workflow.core.runtime import WorkflowRuntime

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
    runtime = WorkflowRuntime.from_definition(
        workflow_name=workflow_name,
        workflow_definition=workflow_definition,
        volume_root=Path(CONF.output_volume_mountpoint),
        workflow_volume_name=OUT_VOLUME_NAME,
    )
    return runtime.run(run_id=run_id, force=force)


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
