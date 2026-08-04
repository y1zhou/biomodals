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
    AlphaFold3ExecutionRuntime,
)
from biomodals.app.fold.alphafold3.inference_inputs import ALPHAFOLD3_APP_VERSION
from biomodals.app.fold.alphafold3.msa_search import SearchRuntime
from biomodals.app.fold.alphafold3.profiles import ALPHAFOLD3_COMMIT
from biomodals.app.fold.alphafold3.seed_predictions import InferenceRuntime
from biomodals.app.fold.alphafold3.template_search import TemplateRuntime
from biomodals.execution import (
    DeploymentIdentity,
)
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
)


class AlphaFold3ExecutionCoordinator(ExecutionCoordinatorLifecycle):
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
        modal_driver: Any,
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
            target_scientific_versions={
                "alphafold3_app": ALPHAFOLD3_APP_VERSION,
                "alphafold3_upstream": ALPHAFOLD3_COMMIT,
            },
        )
        self.output_volume = output_volume
        self.modal_driver = modal_driver
        self.search_runtime = search_runtime
        self.template_runtime = template_runtime
        self.inference_runtime = inference_runtime
        self.poll_interval_seconds = poll_interval_seconds

    def prepare_restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        candidate_request: AlphaFold3ExecutionRequest | None = None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> None:
        """Validate and persist a Successor request without driving it."""
        if candidate_request is not None and (
            max_active_provider_calls is not None
            or max_active_gpu_provider_calls is not None
        ):
            raise ValueError(
                "Candidate request and generic restart overrides are mutually exclusive"
            )
        with self._drive_lock:
            with self._writer_lock:
                self.output_volume.reload()
                with self._open_successor_source(
                    predecessor_execution_run_id,
                    predecessor_deployment=predecessor_deployment,
                ) as (predecessor, predecessor_request, _):
                    request = candidate_request
                if request is None:
                    request = _restart_request(
                        predecessor_request,
                        predecessor_max_active_provider_calls=(
                            predecessor.max_active_provider_calls
                        ),
                        predecessor_max_active_gpu_provider_calls=(
                            predecessor.max_active_gpu_provider_calls
                        ),
                        max_active_provider_calls=max_active_provider_calls,
                        max_active_gpu_provider_calls=max_active_gpu_provider_calls,
                    )
                self._require_successor_plan_match(predecessor, request)
                self._persist_successor_request(
                    request,
                    predecessor_execution_run_id,
                )

    def _create_runtime(
        self,
        request: AlphaFold3ExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> AlphaFold3ExecutionRuntime:
        return AlphaFold3ExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            search_runtime=self.search_runtime,
            template_runtime=self.template_runtime,
            inference_runtime=self.inference_runtime,
            poll_interval_seconds=self.poll_interval_seconds,
        )


def _restart_request(
    request: AlphaFold3ExecutionRequest,
    *,
    predecessor_max_active_provider_calls: int,
    predecessor_max_active_gpu_provider_calls: int,
    max_active_provider_calls: int | None,
    max_active_gpu_provider_calls: int | None,
) -> AlphaFold3ExecutionRequest:
    """Apply only operational restart overrides to an immutable request."""
    if max_active_provider_calls is None and max_active_gpu_provider_calls is None:
        return request
    total = (
        predecessor_max_active_provider_calls
        if max_active_provider_calls is None
        else max_active_provider_calls
    )
    gpu = (
        predecessor_max_active_gpu_provider_calls
        if max_active_gpu_provider_calls is None
        else max_active_gpu_provider_calls
    )
    if gpu > total:
        raise ValueError(
            "max_active_gpu_provider_calls cannot exceed max_active_provider_calls"
        )
    return AlphaFold3ExecutionRequest.prepare(
        request.config,
        search_msa=request.search_msa,
        search_protein_templates=request.search_protein_templates,
        max_parallel_search_workers=total,
        max_num_gpus=gpu,
        recycle=request.recycle,
        sample=request.sample,
    )
