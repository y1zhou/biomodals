"""Deployment-local coordinator adapter for direct AlphaFold3 App Runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.app.fold.alphafold3.execution_runtime import (
    alphafold3_execution_graph,
)
from biomodals.app.fold.alphafold3.inference_inputs import ALPHAFOLD3_APP_VERSION
from biomodals.app.fold.alphafold3.msa_search import SearchRuntime
from biomodals.app.fold.alphafold3.profiles import ALPHAFOLD3_COMMIT
from biomodals.app.fold.alphafold3.seed_predictions import InferenceRuntime
from biomodals.app.fold.alphafold3.template_search import TemplateRuntime
from biomodals.execution import DeploymentIdentity, ExecutionGraph
from biomodals.execution.modal import ExecutionDefinitionCoordinatorLifecycle


class AlphaFold3ExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind one run-scoped writer to AlphaFold3-owned state and publications."""

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
        search_runtime: SearchRuntime,
        template_runtime: TemplateRuntime,
        inference_runtime: InferenceRuntime,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only the host resources needed by this deployment adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={
                "alphafold3_app": ALPHAFOLD3_APP_VERSION,
                "alphafold3_upstream": ALPHAFOLD3_COMMIT,
            },
            poll_interval_seconds=poll_interval_seconds,
        )
        self.search_runtime = search_runtime
        self.template_runtime = template_runtime
        self.inference_runtime = inference_runtime

    def _graph(
        self,
        request: AlphaFold3ExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return alphafold3_execution_graph(
            request=request,
            execution_run_id=self.execution_run_id,
            output_volume=self.output_volume,
            search_runtime=self.search_runtime,
            template_runtime=self.template_runtime,
            inference_runtime=self.inference_runtime,
        )
