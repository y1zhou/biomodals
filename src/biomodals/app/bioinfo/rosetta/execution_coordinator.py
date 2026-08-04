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
    PullTaskClaim,
)
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
)


class RosettaExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one single-writer App Run ledger to Rosetta publications."""

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
        pull_worker_coordinator: Any,
        app_version: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only deployment resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            target_scientific_versions={"rosetta": app_version},
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

    def complete_tasks(
        self,
        provider_call_id: UUID,
        completions: tuple[
            tuple[str, str, dict[str, object]],
            ...,
        ],
    ):
        """Validate and checkpoint one worker completion microbatch."""
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
        return runtime.complete_pull_tasks(
            provider_call_id,
            completions,
        )

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
                ) as (predecessor, predecessor_request, _):
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
                self._require_successor_plan_match(predecessor, request)
                self._persist_successor_request(
                    request,
                    predecessor_execution_run_id,
                )

    def _create_runtime(
        self,
        request: RosettaExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> RosettaExecutionRuntime:
        return RosettaExecutionRuntime(
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
