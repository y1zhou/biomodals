"""Deployment-local coordinator adapter for direct BoltzGen App Runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from biomodals.app.design.boltzgen.execution_request import (
    DESIGN_RUNS_NODE,
    BoltzGenExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.app.design.boltzgen.execution_runtime import (
    boltzgen_execution_graph,
)
from biomodals.execution import DeploymentIdentity, ExecutionGraph
from biomodals.execution.modal import ExecutionDefinitionCoordinatorLifecycle
from biomodals.execution.store import ExecutionRunStore


class BoltzGenExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind one run-scoped writer to BoltzGen publications."""

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
        app_version: str,
        repo_commit_hash: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only deployment resources needed by this app adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={
                "boltzgen": app_version,
                "boltzgen_repository": repo_commit_hash,
            },
            poll_interval_seconds=poll_interval_seconds,
        )

    def _graph(
        self,
        request: BoltzGenExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return boltzgen_execution_graph(request, output_root=self.volume_root)

    def _prepare_successor_request(
        self,
        request: BoltzGenExecutionRequest,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_store: ExecutionRunStore,
    ) -> BoltzGenExecutionRequest:
        del predecessor_execution_run_id
        return replace(
            request,
            replace_claim_owners=_replaceable_claim_owners(
                predecessor_store,
                request,
            ),
        )


def _replaceable_claim_owners(
    store: ExecutionRunStore,
    request: BoltzGenExecutionRequest,
) -> tuple[tuple[str, str], ...]:
    """Bind failed design Tasks to their conclusively terminal old calls."""
    owners = []
    for task in store.execution.list_tasks(
        store.execution_run_id,
        DESIGN_RUNS_NODE,
    ):
        if task.task_key not in request.run_ids:
            raise ValueError("Predecessor contains an unknown BoltzGen design Task")
        if task.provider_call_id is not None:
            owners.append((task.task_key, str(task.provider_call_id)))
    return tuple(owners)
