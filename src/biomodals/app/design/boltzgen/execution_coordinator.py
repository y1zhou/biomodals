"""Deployment-local coordinator adapter for direct BoltzGen App Runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from threading import Lock, RLock
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
    ExecutionRunNotFoundError,
    ExecutionSnapshot,
)
from biomodals.helper.app_execution import ExecutionRunStore


class BoltzGenExecutionCoordinator:
    """Bind one run-scoped writer to BoltzGen-owned publications."""

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
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.volume_root = Path(volume_root)
        self.output_volume = output_volume
        self.modal_driver = modal_driver
        self.poll_interval_seconds = poll_interval_seconds
        self._writer_lock = RLock()
        self._drive_lock = Lock()
        self._runtime: BoltzGenExecutionRuntime | None = None

    def run(self) -> ExecutionSnapshot:
        """Load the staged request and drive one root Run."""
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
        """Idempotently request cancellation without replacing unknown work."""
        with self._writer_lock:
            request = load_execution_request(
                self.volume_root,
                self.execution_run_id,
            )
            runtime = self._open_runtime(request)
            snapshot = runtime.cancel()
            self._verify_snapshot(snapshot)
        if snapshot.run.status.is_terminal:
            return snapshot
        return self._drive(runtime, resume=False)

    def resume(self) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive Task failures."""
        with self._drive_lock:
            with self._writer_lock:
                request = load_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                )
                runtime = self._open_runtime(request)
            return self._drive(runtime, resume=True)

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
                        expected_workload_plan_fingerprint is not None
                        and predecessor.plan.workload_plan_fingerprint
                        != expected_workload_plan_fingerprint
                    ):
                        raise ValueError(
                            "Restart arguments changed the Workload Plan Fingerprint"
                        )
                    if (
                        predecessor_deployment is not None
                        and predecessor.deployment != predecessor_deployment
                    ):
                        raise ValueError(
                            "Predecessor Deployment Identity does not match "
                            "Execution Run"
                        )
                    predecessor_request = load_execution_request(
                        self.volume_root,
                        predecessor_execution_run_id,
                    )
                    claim_owners = _replaceable_claim_owners(
                        predecessor_store,
                        predecessor_execution_run_id,
                    )
                finally:
                    predecessor_store.close()

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
                self.output_volume.commit()
                runtime = self._open_runtime(
                    request,
                    predecessor_execution_run_id=predecessor_execution_run_id,
                )
            return self._drive(runtime, resume=False)

    def close(self) -> None:
        """Close local state without cancelling child calls."""
        with self._writer_lock:
            self._close_runtime()

    def synchronize(self) -> AbstractContextManager[object]:
        """Return the serialized writer boundary between coordinator cycles."""
        return self._writer_lock

    def _drive(
        self,
        runtime: BoltzGenExecutionRuntime,
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

    def _run_store(self) -> ExecutionRunStore:
        return ExecutionRunStore(self.volume_root, self.execution_run_id)

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
