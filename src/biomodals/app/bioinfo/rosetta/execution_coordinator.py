"""Deployment-local coordinator for direct Rosetta App Runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from biomodals.app.bioinfo.rosetta.execution_request import (
    RosettaExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.app.bioinfo.rosetta.execution_runtime import (
    RosettaExecutionRuntime,
)
from biomodals.execution import (
    DeploymentIdentity,
    ExecutionSnapshot,
    PullTaskClaim,
)
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    persist_execution_launch,
)


class RosettaExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one single-writer App Run ledger to Rosetta publications."""

    _request_loader = staticmethod(load_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        modal_driver: Any,
        pull_worker_coordinator: Any,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only deployment resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
        )
        self.output_volume = output_volume
        self.modal_driver = modal_driver
        self.pull_worker_coordinator = pull_worker_coordinator
        self.poll_interval_seconds = poll_interval_seconds

    def claim_tasks(
        self,
        provider_call_id: UUID,
        *,
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        """Checkpoint one worker claim through the serialized writer."""
        with self._writer_lock:
            request = load_execution_request(
                self.volume_root,
                self.execution_run_id,
            )
            runtime = self._open_runtime(
                request,
                predecessor_execution_run_id=self._existing_predecessor(),
            )
            runtime.attach()
            return runtime.claim_pull_tasks(
                provider_call_id,
                request_id=request_id,
                capacity=capacity,
            )

    def complete_task(
        self,
        provider_call_id: UUID,
        task_key: str,
        *,
        request_id: str,
        result: dict[str, object],
    ):
        """Validate and checkpoint one worker completion."""
        with self._writer_lock:
            request = load_execution_request(
                self.volume_root,
                self.execution_run_id,
            )
            runtime = self._open_runtime(
                request,
                predecessor_execution_run_id=self._existing_predecessor(),
            )
            runtime.refresh_publications()
            return runtime.complete_pull_task(
                provider_call_id,
                task_key,
                request_id=request_id,
                result=result,
            )

    def restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        claim_capacity: int | None = None,
        max_parallel_per_worker: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
    ) -> ExecutionSnapshot:
        """Create a compatible Successor from conclusive predecessor state."""
        self.prepare_restart(
            predecessor_execution_run_id=predecessor_execution_run_id,
            predecessor_deployment=predecessor_deployment,
            max_active_provider_calls=max_active_provider_calls,
            claim_capacity=claim_capacity,
            max_parallel_per_worker=max_parallel_per_worker,
            expected_workload_plan_fingerprint=expected_workload_plan_fingerprint,
        )
        return self.drive_prepared()

    def prepare_restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        claim_capacity: int | None = None,
        max_parallel_per_worker: int | None = None,
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
                ) as source:
                    predecessor, predecessor_request, _ = source

                request = replace(
                    predecessor_request,
                    max_active_provider_calls=(
                        predecessor.max_active_provider_calls
                        if max_active_provider_calls is None
                        else max_active_provider_calls
                    ),
                    claim_capacity=(
                        predecessor_request.claim_capacity
                        if claim_capacity is None
                        else claim_capacity
                    ),
                    max_parallel_per_worker=(
                        predecessor_request.max_parallel_per_worker
                        if max_parallel_per_worker is None
                        else max_parallel_per_worker
                    ),
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
        request: RosettaExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> RosettaExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active Rosetta runtime does not match request")
            return runtime
        runtime = RosettaExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            output_root=self.volume_root,
            pull_worker_coordinator=self.pull_worker_coordinator,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime
