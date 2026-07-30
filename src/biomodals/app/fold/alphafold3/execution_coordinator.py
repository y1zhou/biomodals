"""Deployment-local coordinator adapter for direct AlphaFold3 App Runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock, RLock
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
from biomodals.app.fold.alphafold3.msa_search import SearchRuntime
from biomodals.app.fold.alphafold3.seed_predictions import InferenceRuntime
from biomodals.app.fold.alphafold3.template_search import TemplateRuntime
from biomodals.execution import (
    DeploymentIdentity,
    ExecutionRunNotFoundError,
    ExecutionSnapshot,
)
from biomodals.helper.app_execution import AppExecutionRunStore


class AlphaFold3ExecutionCoordinator:
    """Bind one run-scoped writer to AlphaFold3-owned state and publications."""

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
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.volume_root = Path(volume_root)
        self.output_volume = output_volume
        self.modal_driver = modal_driver
        self.search_runtime = search_runtime
        self.template_runtime = template_runtime
        self.inference_runtime = inference_runtime
        self.poll_interval_seconds = poll_interval_seconds
        self._writer_lock = RLock()
        self._drive_lock = Lock()
        self._runtime: AlphaFold3ExecutionRuntime | None = None

    def initialize_container(self) -> None:
        """Reload durable state before this parameterized container serves calls."""
        with self._writer_lock:
            self.output_volume.reload()

    def run(self) -> ExecutionSnapshot:
        """Load the staged immutable request and drive a root Run."""
        with self._drive_lock:
            with self._writer_lock:
                request = load_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                )
                runtime = self._open_runtime(request)
            return self._drive(runtime, resume=False)

    def status(self) -> ExecutionSnapshot:
        """Read one verified snapshot without advancing or retrying work."""
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
            return snapshot

    def resume(self) -> ExecutionSnapshot:
        """Resume this same Run without retrying conclusive Task failures."""
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
        candidate_request: AlphaFold3ExecutionRequest | None = None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive a compatible Successor Run from conclusive state."""
        if predecessor_execution_run_id == self.execution_run_id:
            raise ValueError("Successor Execution Run ID must be new")
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
                    predecessor_request = load_execution_request(
                        self.volume_root,
                        predecessor_execution_run_id,
                    )
                finally:
                    predecessor_store.close()

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
        """Checkpoint local state on exit without cancelling child calls."""
        with self._writer_lock:
            self._close_runtime()
            self.output_volume.commit()

    def synchronize(self) -> AbstractContextManager[object]:
        """Return the single-writer lock used between coordinator cycles."""
        return self._writer_lock

    def _drive(
        self,
        runtime: AlphaFold3ExecutionRuntime,
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
        request: AlphaFold3ExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> AlphaFold3ExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError(
                    "Active AlphaFold3 runtime does not match coordinator request"
                )
            return runtime
        runtime = AlphaFold3ExecutionRuntime(
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
        self._runtime = runtime
        return runtime

    def _run_store(self) -> AppExecutionRunStore:
        return AppExecutionRunStore(self.volume_root, self.execution_run_id)

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
