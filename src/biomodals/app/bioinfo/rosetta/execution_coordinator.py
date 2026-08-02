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
    ExecutionRunNotFoundError,
    ExecutionSnapshot,
    PullTaskClaim,
)
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRunStore,
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

    def _open_current_runtime(self, *, recover: bool) -> RosettaExecutionRuntime:
        request = self._request_loader(self.volume_root, self.execution_run_id)
        return self._open_runtime(
            request,
            predecessor_execution_run_id=(
                self._existing_predecessor() if recover else None
            ),
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
        if predecessor_execution_run_id == self.execution_run_id:
            raise ValueError("Successor Execution Run ID must be new")
        with self._drive_lock:
            with self._writer_lock:
                self.output_volume.reload()
                predecessor_store = ExecutionRunStore(
                    self.volume_root,
                    predecessor_execution_run_id,
                )
                if not predecessor_store.ledger_path.is_file():
                    raise ExecutionRunNotFoundError(str(predecessor_execution_run_id))
                try:
                    predecessor = predecessor_store.execution.validate_successor_source(
                        predecessor_execution_run_id
                    )
                    if (
                        predecessor_deployment is not None
                        and predecessor.deployment != predecessor_deployment
                    ):
                        raise ValueError(
                            "Predecessor Deployment Identity does not match "
                            "Execution Run"
                        )
                    if (
                        expected_workload_plan_fingerprint is not None
                        and predecessor.plan.workload_plan_fingerprint
                        != expected_workload_plan_fingerprint
                    ):
                        raise ValueError(
                            "Restart arguments changed the Workload Plan Fingerprint"
                        )
                    predecessor_request = load_execution_request(
                        self.volume_root,
                        predecessor_execution_run_id,
                    )
                finally:
                    predecessor_store.close()

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
                self.output_volume.commit()
                runtime = self._open_runtime(
                    request,
                    predecessor_execution_run_id=predecessor_execution_run_id,
                )
            return self._drive(runtime, resume=False)

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

    def _existing_predecessor(self) -> UUID | None:
        store = self._run_store()
        if not store.ledger_path.is_file():
            return None
        try:
            return store.execution.get_run(
                self.execution_run_id
            ).predecessor_execution_run_id
        finally:
            store.close()
