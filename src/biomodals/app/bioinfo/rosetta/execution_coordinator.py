"""Deployment-local coordinator for direct Rosetta App Runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from threading import Lock, RLock
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
from biomodals.helper.app_execution import AppExecutionRunStore


class RosettaExecutionCoordinator:
    """Bind one single-writer App Run ledger to Rosetta publications."""

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
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.volume_root = Path(volume_root)
        self.output_volume = output_volume
        self.modal_driver = modal_driver
        self.pull_worker_coordinator = pull_worker_coordinator
        self.poll_interval_seconds = poll_interval_seconds
        self._writer_lock = RLock()
        self._drive_lock = Lock()
        self._runtime: RosettaExecutionRuntime | None = None

    def run(self) -> ExecutionSnapshot:
        """Load the staged request and drive a root Run."""
        with self._drive_lock:
            with self._writer_lock:
                request = load_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                )
                runtime = self._open_runtime(request)
            return self._drive(runtime, resume=False)

    def status(self) -> ExecutionSnapshot:
        """Read one verified snapshot without advancing work."""
        with self._writer_lock:
            runtime = self._runtime
            if runtime is not None:
                snapshot = runtime.store.execution.snapshot(self.execution_run_id)
            else:
                store = self._run_store()
                if not store.ledger_path.is_file():
                    raise ExecutionRunNotFoundError(str(self.execution_run_id))
                try:
                    snapshot = store.execution.snapshot(self.execution_run_id)
                finally:
                    store.close()
            self._verify_snapshot(snapshot)
            return snapshot

    def cancel(self) -> ExecutionSnapshot:
        """Request idempotent cancellation through the shared runtime."""
        with self._writer_lock:
            request = load_execution_request(
                self.volume_root,
                self.execution_run_id,
            )
            snapshot = self._open_runtime(
                request,
                predecessor_execution_run_id=self._existing_predecessor(),
            ).cancel()
            self._verify_snapshot(snapshot)
            return snapshot

    def resume(self) -> ExecutionSnapshot:
        """Resume this same Run without retrying failed Tasks."""
        with self._drive_lock:
            with self._writer_lock:
                request = load_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                )
                runtime = self._open_runtime(
                    request,
                    predecessor_execution_run_id=self._existing_predecessor(),
                )
            return self._drive(runtime, resume=True)

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
            runtime.attach()
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
                predecessor_store = AppExecutionRunStore(
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

    def close(self) -> None:
        """Checkpoint state without cancelling attached Provider Calls."""
        with self._writer_lock:
            self._close_runtime()
            self.output_volume.commit()

    def synchronize(self) -> AbstractContextManager[object]:
        """Return the serialized SQLite writer boundary."""
        return self._writer_lock

    def _drive(
        self,
        runtime: RosettaExecutionRuntime,
        *,
        resume: bool,
    ) -> ExecutionSnapshot:
        try:
            snapshot = (
                runtime.resume(synchronize=self.synchronize)
                if resume
                else runtime.run(synchronize=self.synchronize)
            )
            self._verify_snapshot(snapshot)
            return snapshot
        finally:
            with self._writer_lock:
                self._close_runtime()
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

    def _run_store(self) -> AppExecutionRunStore:
        return AppExecutionRunStore(self.volume_root, self.execution_run_id)

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

    def _verify_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        if snapshot.run.execution_run_id != self.execution_run_id:
            raise ValueError("Execution Run ID does not match coordinator")
        if snapshot.run.deployment != self.deployment:
            raise ValueError("Deployment Identity does not match Execution Run")

    def _close_runtime(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            runtime.close()
            self._runtime = None
