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
    BoltzGenExecutionRuntime,
)
from biomodals.execution import (
    DeploymentIdentity,
    ExecutionSnapshot,
)
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRunStore,
    persist_execution_launch,
)


class BoltzGenExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one run-scoped writer to BoltzGen-owned publications."""

    _request_loader = staticmethod(load_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        modal_driver: Any,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only deployment resources needed by this app adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
        )
        self.output_volume = output_volume
        self.modal_driver = modal_driver
        self.poll_interval_seconds = poll_interval_seconds

    def restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive a compatible Successor from conclusive state."""
        self.prepare_restart(
            predecessor_execution_run_id=predecessor_execution_run_id,
            predecessor_deployment=predecessor_deployment,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            expected_workload_plan_fingerprint=expected_workload_plan_fingerprint,
        )
        return self.drive_prepared()

    def prepare_restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
    ) -> None:
        """Validate and persist a Successor request without driving it."""
        with self._drive_lock:
            with self._writer_lock:
                self.output_volume.reload()
                with self._open_successor_source(
                    predecessor_execution_run_id,
                    predecessor_deployment=predecessor_deployment,
                    expected_workload_plan_fingerprint=(
                        expected_workload_plan_fingerprint
                    ),
                ) as (predecessor, predecessor_request, predecessor_store):
                    claim_owners = _replaceable_claim_owners(
                        predecessor_store,
                        predecessor_execution_run_id,
                    )

                total = (
                    predecessor.max_active_provider_calls
                    if max_active_provider_calls is None
                    else max_active_provider_calls
                )
                gpu = (
                    predecessor.max_active_gpu_provider_calls
                    if max_active_gpu_provider_calls is None
                    else max_active_gpu_provider_calls
                )
                request = replace(
                    predecessor_request,
                    max_active_provider_calls=total,
                    max_active_gpu_provider_calls=gpu,
                    replace_claim_owners=claim_owners,
                )
                if (
                    request.execution_plan.workload_plan_fingerprint
                    != predecessor.plan.workload_plan_fingerprint
                ):
                    raise ValueError(
                        "Target deployment changed the Workload Plan Fingerprint"
                    )
                persist_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                    request,
                )
                persist_execution_launch(
                    self.volume_root,
                    self.execution_run_id,
                    predecessor_execution_run_id,
                )
                self.output_volume.commit()

    def _open_runtime(
        self,
        request: BoltzGenExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> BoltzGenExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active BoltzGen runtime does not match request")
            return runtime
        runtime = BoltzGenExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            output_root=self.volume_root,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime


def _replaceable_claim_owners(
    store: ExecutionRunStore,
    execution_run_id: UUID,
) -> tuple[tuple[str, str], ...]:
    """Bind failed design Tasks to their conclusively terminal old calls."""
    owners = []
    for task in store.execution.list_tasks(
        execution_run_id,
        DESIGN_RUNS_NODE,
    ):
        if task.provider_call_id is not None:
            owners.append((task.task_key, str(task.provider_call_id)))
    return tuple(owners)
