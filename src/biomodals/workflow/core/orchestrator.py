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
from biomodals.workflow.core.artifact_availability import ExternalArtifactChecker
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.runtime import WorkflowRuntime

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

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


@app.cls(
    cpu=(0.125, 16.125),
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
        OUT_VOLUME.reload()

    @modal.method()
    def run(
        self,
        workflow: Workflow,
        run_id: str,
        force: bool = False,
        max_ready_workers: int = 32,
        strict_external_artifact_checks: bool = False,
        external_artifact_checker: ExternalArtifactChecker | None = None,
    ) -> AppRunResult:
        """Run one workflow definition through the workflow runtime."""
        if not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow object")

        OUT_VOLUME.reload()
        self._runtime = WorkflowRuntime(
            workflow=workflow,
            volume_root=Path(CONF.output_volume_mountpoint),
            workflow_volume_name=OUT_VOLUME_NAME,
            workflow_volume=OUT_VOLUME,
            function_call_resolver=modal.FunctionCall.from_id,
            max_ready_workers=max_ready_workers,
            strict_external_artifact_checks=strict_external_artifact_checks,
            external_artifact_checker=external_artifact_checker,
        )
        try:
            return self._runtime.run(run_id=run_id, force=force)
        finally:
            self._close_runtime()
            OUT_VOLUME.commit()

    @modal.exit()
    def exit(self) -> None:
        """Persist any pending workflow volume writes on container shutdown."""
        self._close_runtime()
        OUT_VOLUME.commit()

    def _close_runtime(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
            self._runtime = None
