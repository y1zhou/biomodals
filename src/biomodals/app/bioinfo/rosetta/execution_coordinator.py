"""Deployment-local coordinator for direct Rosetta App Runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from biomodals.app.bioinfo.rosetta.execution_request import (
    RosettaExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.app.bioinfo.rosetta.execution_runtime import rosetta_execution_graph
from biomodals.execution import DeploymentIdentity, ExecutionGraph, PullTaskClaim
from biomodals.execution.modal import ExecutionDefinitionCoordinatorLifecycle
from biomodals.schema import AppRunResult


class RosettaExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind one run-scoped writer to Rosetta's pull-worker publications."""

    _request_loader = staticmethod(load_execution_request)
    _request_persister = staticmethod(persist_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        output_volume_name: str,
        provider_driver: Any,
        pull_worker_coordinator: Any,
        app_version: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only deployment resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={"rosetta": app_version},
            pull_worker_coordinator=pull_worker_coordinator,
            poll_interval_seconds=poll_interval_seconds,
        )

    def claim_tasks(
        self,
        provider_call_id: UUID,
        *,
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        """Checkpoint one worker claim through the serialized writer."""
        with self._volume_io_lock, self._writer_lock:
            runtime = self._open_current_runtime(recover=True)
            runtime.attach()
            return runtime.claim_pull_tasks(
                provider_call_id,
                request_id=request_id,
                capacity=capacity,
            )

    def complete_tasks_and_claim(
        self,
        provider_call_id: UUID,
        completions: tuple[tuple[str, str, AppRunResult], ...],
        *,
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        """Publish one microbatch and return its checkpointed successor claim."""
        with self._volume_io_lock, self._writer_lock:
            runtime = self._open_current_runtime(recover=True)
            runtime.refresh_publications()
            return runtime.complete_pull_tasks_and_claim(
                provider_call_id,
                completions,
                request_id=request_id,
                capacity=capacity,
            )

    def _graph(
        self,
        request: RosettaExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return rosetta_execution_graph(request, output_root=self.volume_root)
