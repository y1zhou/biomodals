"""Service-owned GROMACS hosting for the shared executable graph."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, Protocol, TypeVar
from uuid import UUID

from biomodals.app.bioinfo.gromacs_execution import PREPARE_RESULT
from biomodals.app.bioinfo.gromacs_execution_runtime import (
    GromacsExecutionRequest,
    gromacs_execution_graph,
)
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    GraphExecutionRunStore,
    RunStatus,
    RunStatusReason,
)
from biomodals.execution.artifact_availability import ArtifactAvailability
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.modal import (
    ProviderCallObservation,
    ProviderDefiniteSubmissionError,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    ExecutionArtifact,
    VolumePath,
)
from biomodals.service.gromacs.archive import GROMACS_ARCHIVE_SCHEMA_VERSION
from biomodals.service.gromacs.contracts import GromacsJobOptions
from biomodals.service.gromacs.results import (
    ArchiveNotReadyError,
    FinalArchive,
    GromacsResultInvalidError,
    ResultIdentityMismatchError,
)
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.store import JobRecord, ServiceStore

LOGGER = logging.getLogger(__name__)
_SERVICE_ARTIFACT_VOLUME = "Biomodals-service-execution"
_GROMACS_THREADS = 16
_T = TypeVar("_T")


class GromacsExecutionAdapter(Protocol):
    """Provider calls and Result publication required by GROMACS."""

    output_volume_name: str

    async def resolve(self, binding: Any) -> Any:
        """Resolve one exact deployed operation."""

    async def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Spawn one detached provider call."""

    async def observe(self, provider_call_handle_id: str) -> ProviderCallObservation:
        """Observe one attached provider call."""

    async def cancel(self, provider_call_handle_id: str) -> None:
        """Cancel one attached provider call."""

    async def publish_archive(
        self,
        job: JobRecord,
        *,
        completed_at: int,
    ) -> FinalArchive:
        """Publish and validate the user-facing Result archive."""

    async def recover_archive(self, job: JobRecord) -> FinalArchive:
        """Recover an already published immutable Result archive."""

    async def cleanup_intermediates(self, job: JobRecord) -> None:
        """Remove remote files reconstructible from durable publications."""


class _EventLoopBridge:
    """Run host-owned async operations from one graph-runtime worker thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def call(self, awaitable: Coroutine[Any, Any, _T]) -> _T:
        return asyncio.run_coroutine_threadsafe(awaitable, self.loop).result()


class _AsyncProviderBridge:
    """Present the service's async adapter through the kernel's provider seam."""

    def __init__(
        self,
        bridge: _EventLoopBridge,
        adapter: GromacsExecutionAdapter,
    ) -> None:
        self.bridge = bridge
        self.adapter = adapter
        self.definite_submission_error: ProviderDefiniteSubmissionError | None = None

    def resolve(self, binding: Any) -> Any:
        return self.bridge.call(self.adapter.resolve(binding))

    def spawn(
        self,
        operation: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        try:
            return self.bridge.call(
                self.adapter.spawn(operation, args=args, kwargs=kwargs)
            )
        except ProviderDefiniteSubmissionError as error:
            self.definite_submission_error = error
            raise

    def observe(self, provider_call_handle_id: str) -> ProviderCallObservation:
        return self.bridge.call(self.adapter.observe(provider_call_handle_id))

    def cancel(self, provider_call_handle_id: str) -> None:
        self.bridge.call(self.adapter.cancel(provider_call_handle_id))


class _ServiceGromacsPublications:
    """Adapt the service-owned final ZIP to GROMACS graph publications."""

    _CONCLUSIVE_ERRORS: tuple[type[Exception], ...] = (
        GromacsResultInvalidError,
        ResultIdentityMismatchError,
        ValueError,
    )

    def __init__(
        self,
        *,
        store: ServiceStore,
        adapter: GromacsExecutionAdapter,
        bridge: _EventLoopBridge,
        job_id: UUID,
        now: Callable[[], int],
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.bridge = bridge
        self.job_id = job_id
        self.now = now
        self.archive: FinalArchive | None = None
        self.archive_observation: AvailabilityStatus | None = None
        self.conclusive_error: Exception | None = None
        self.publication_error: Exception | None = None

    def recover_result(self, node_key: str) -> AppRunResult | None:
        if node_key != PREPARE_RESULT:
            return None
        if self.archive is not None:
            return self._archive_result(self.archive)
        if self.archive_observation == AvailabilityStatus.MISSING:
            return None
        if not self._result_needs_recovery():
            return None
        try:
            self.archive = self.bridge.call(self.adapter.recover_archive(self._job()))
        except ArchiveNotReadyError:
            self.archive_observation = AvailabilityStatus.MISSING
            return None
        except self._CONCLUSIVE_ERRORS as error:
            self.conclusive_error = error
            self.archive_observation = AvailabilityStatus.MISSING
            return None
        except Exception:
            self.archive_observation = AvailabilityStatus.UNKNOWN
            raise
        self.archive_observation = AvailabilityStatus.AVAILABLE
        return self._archive_result(self.archive)

    def result(
        self,
        node_key: str,
        *,
        files: tuple[ArtifactFile, ...] | None = None,
    ) -> AppRunResult:
        del files
        if node_key != PREPARE_RESULT:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)
        if self.conclusive_error is not None:
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[str(self.conclusive_error)],
            )
        try:
            self.archive = self.bridge.call(
                self.adapter.publish_archive(
                    self._job(),
                    completed_at=self.now(),
                )
            )
        except self._CONCLUSIVE_ERRORS as error:
            self.conclusive_error = error
            self.archive_observation = AvailabilityStatus.MISSING
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[str(error)],
            )
        except Exception as error:  # provider/storage availability is inconclusive
            self.publication_error = error
            self.archive_observation = AvailabilityStatus.UNKNOWN
            return self._archive_result(None)
        self.archive_observation = AvailabilityStatus.AVAILABLE
        return self._archive_result(self.archive)

    def commit(
        self,
        node_key: str,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus:
        del artifacts
        return self.observe(node_key)

    def observe(self, node_key: str) -> AvailabilityStatus:
        if node_key != PREPARE_RESULT:
            return AvailabilityStatus.AVAILABLE
        if self.archive is not None:
            return AvailabilityStatus.AVAILABLE
        if self.conclusive_error is not None:
            return AvailabilityStatus.MISSING
        if self.publication_error is not None:
            return AvailabilityStatus.UNKNOWN
        if self.archive_observation is not None:
            return self.archive_observation
        try:
            self.archive = self.bridge.call(self.adapter.recover_archive(self._job()))
        except ArchiveNotReadyError:
            self.archive_observation = AvailabilityStatus.MISSING
        except self._CONCLUSIVE_ERRORS as error:
            self.conclusive_error = error
            self.archive_observation = AvailabilityStatus.MISSING
        except Exception:
            self.archive_observation = AvailabilityStatus.UNKNOWN
        else:
            self.archive_observation = AvailabilityStatus.AVAILABLE
        return self.archive_observation

    def check_artifact(self, artifact: ExecutionArtifact) -> ArtifactAvailability:
        expected_path = self._archive_path()
        if (
            artifact.storage.volume_name != self.adapter.output_volume_name
            or artifact.storage.path != expected_path
        ):
            return ArtifactAvailability(
                artifact_id=artifact.artifact_id,
                status=AvailabilityStatus.UNKNOWN,
                unknown_reason="Artifact is outside the GROMACS Result publication",
            )
        status = self.observe(PREPARE_RESULT)
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=status,
            errors=("Published GROMACS Result is unavailable",)
            if status == AvailabilityStatus.MISSING
            else (),
            unknown_reason=(
                "Published GROMACS Result could not be inspected"
                if status == AvailabilityStatus.UNKNOWN
                else None
            ),
        )

    def _archive_result(self, archive: FinalArchive | None) -> AppRunResult:
        job = self._job()
        path = self._archive_path()
        metadata: dict[str, object] = {}
        if archive is not None:
            metadata["files"] = [
                ArtifactFile(
                    path=archive.filename,
                    role="result_archive",
                    size_bytes=archive.size_bytes,
                    content_sha256=archive.sha256,
                ).model_dump(mode="json", exclude_none=True)
            ]
            path = archive.path
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="result-archive",
                    kind=ArtifactKind.ARCHIVE,
                    storage=VolumePath(
                        volume_name=self.adapter.output_volume_name,
                        path=path,
                    ),
                    metadata=metadata,
                )
            ],
            metrics={"job_id": str(job.job_id)},
        )

    def _archive_path(self) -> str:
        job = self._job()
        if job.run_name is None:
            raise ValueError("GROMACS Job has no run name")
        return f"api-results/{job.run_name}/result.zip"

    def _job(self) -> JobRecord:
        job = self.store.get_job_by_id(self.job_id)
        if job is None:
            raise LookupError(f"Job not found: {self.job_id}")
        return job

    def _result_needs_recovery(self) -> bool:
        job = self._job()
        if job.execution_run_id is None:
            return False
        with self.store.execution_repository() as repository:
            node = repository.get_node(job.execution_run_id, PREPARE_RESULT)
            if node.result_observation != AvailabilityStatus.MISSING:
                return True
            return any(
                task.local_owned and not task.status.is_terminal
                for task in repository.list_tasks(
                    job.execution_run_id,
                    PREPARE_RESULT,
                )
            )


class GromacsExecutionCoordinator:
    """Advance service Jobs through the app-owned shared execution graph."""

    def __init__(
        self,
        store: ServiceStore,
        adapter: GromacsExecutionAdapter,
        *,
        lifecycle_locks: JobLifecycleLocks | None = None,
        now: Callable[[], int] | None = None,
        intermediate_retention_days: int | None = None,
        max_concurrent_jobs: int = 4,
    ) -> None:
        """Bind service state and async provider operations."""
        self.store = store
        self.adapter = adapter
        self.lifecycle_locks = lifecycle_locks or JobLifecycleLocks()
        self._now = now or (lambda: int(time.time()))
        if intermediate_retention_days is not None and intermediate_retention_days < 1:
            raise ValueError("intermediate_retention_days must be positive")
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be positive")
        self.max_concurrent_jobs = max_concurrent_jobs
        self.intermediate_retention_seconds = (
            intermediate_retention_days * 24 * 60 * 60
            if intermediate_retention_days is not None
            else None
        )

    async def reconcile(self) -> None:
        """Advance every active GROMACS Job once."""
        jobs = iter(self.store.list_reconcilable_jobs("gromacs"))

        async def worker() -> None:
            for job in jobs:
                try:
                    await self.advance(job.job_id)
                except Exception:
                    LOGGER.exception("Could not reconcile GROMACS job %s", job.job_id)

        await asyncio.gather(*(worker() for _ in range(self.max_concurrent_jobs)))
        await self._cleanup_intermediates()

    async def cancel_job(self, job_id: UUID) -> None:
        """Durably cancel one Run and its attached provider calls."""
        async with self.lifecycle_locks.for_job(job_id):
            bridge = _EventLoopBridge(asyncio.get_running_loop())
            await asyncio.to_thread(self._cancel_sync, job_id, bridge)

    async def advance(self, job_id: UUID) -> None:
        """Advance one Job and suspend unexpected coordinator failures."""
        try:
            async with self.lifecycle_locks.for_job(job_id):
                bridge = _EventLoopBridge(asyncio.get_running_loop())
                await asyncio.to_thread(self._advance_sync, job_id, bridge)
        except ProviderDefiniteSubmissionError:
            raise
        except Exception as error:
            self._suspend_after_coordinator_error(job_id, error)
            raise

    def _advance_sync(self, job_id: UUID, bridge: _EventLoopBridge) -> None:
        job = self._job(job_id)
        if job.execution_run_id is None:
            raise ValueError("Job is not linked to an Execution Run")
        with self.store.execution_repository() as repository:
            run = repository.get_run(job.execution_run_id)
        if run.status.is_terminal:
            self.store.clear_job_input(job_id)
            self._finish_terminal_job(job, run.status, bridge)
            return
        if run.status == RunStatus.SUSPENDED:
            self._project_terminal_or_running_job(job, run.status)
            return

        publications = _ServiceGromacsPublications(
            store=self.store,
            adapter=self.adapter,
            bridge=bridge,
            job_id=job_id,
            now=self._now,
        )
        provider = _AsyncProviderBridge(bridge, self.adapter)
        runtime = self._runtime(job, run, publications, provider)
        try:
            runtime.attach(workload_run_key=self._run_name(job))
            if run.cancellation_is_durable:
                runtime.cancel()
            runtime.advance_once()
            overview = runtime.store.execution.overview(job.execution_run_id)
        finally:
            runtime.close()

        if overview.run.status.is_terminal:
            self.store.clear_job_input(job_id)
        if publications.archive is not None:
            self._complete_job(self._job(job_id), publications.archive)
        else:
            self._project_terminal_or_running_job(job, overview.run.status)
        if provider.definite_submission_error is not None:
            raise provider.definite_submission_error
        if publications.publication_error is not None:
            raise publications.publication_error

    def _cancel_sync(self, job_id: UUID, bridge: _EventLoopBridge) -> None:
        job = self._job(job_id)
        if job.execution_run_id is None:
            raise ValueError("Job is not linked to an Execution Run")
        with self.store.execution_repository() as repository:
            run = repository.get_run(job.execution_run_id)
        if run.status.is_terminal:
            return
        publications = _ServiceGromacsPublications(
            store=self.store,
            adapter=self.adapter,
            bridge=bridge,
            job_id=job_id,
            now=self._now,
        )
        runtime = self._runtime(
            job,
            run,
            publications,
            _AsyncProviderBridge(bridge, self.adapter),
        )
        try:
            runtime.attach(workload_run_key=self._run_name(job))
            runtime.cancel()
            with runtime.store.transaction():
                status = runtime.store.execution.finalize_run_from_results(
                    job.execution_run_id,
                    now=self._now(),
                ).status
        finally:
            runtime.close()
        if status.is_terminal:
            self.store.clear_job_input(job_id)
        self._project_terminal_or_running_job(job, status)

    def _runtime(
        self,
        job: JobRecord,
        run: Any,
        publications: _ServiceGromacsPublications,
        provider: _AsyncProviderBridge,
    ) -> ExecutionGraphRuntime:
        execution_run_id = job.execution_run_id
        if execution_run_id is None:
            raise ValueError("Job is not linked to an Execution Run")
        input_content = self.store.load_job_input(job.job_id)
        if input_content is None:
            raise RuntimeError("Staged GROMACS input is unavailable")
        options = GromacsJobOptions.model_validate_json(job.parameters_json)
        payload = run.plan.scientific_payload
        if not isinstance(payload, Mapping):
            raise TypeError("GROMACS plan scientific payload is invalid")
        request = GromacsExecutionRequest(
            run_name=self._run_name(job),
            pdb_content=input_content,
            simulation_time_ns=options.simulation_time_ns,
            run_pdbfixer=options.run_pdbfixer,
            cpu_only=options.cpu_only,
            num_threads=_GROMACS_THREADS,
            use_openmp_threads=False,
            ld_seed=_persisted_seed(payload, "ld_seed"),
            gen_seed=_persisted_seed(payload, "gen_seed"),
            genion_seed=_persisted_seed(payload, "genion_seed"),
            max_active_provider_calls=run.max_active_provider_calls,
            max_active_gpu_provider_calls=run.max_active_gpu_provider_calls,
        )
        graph = gromacs_execution_graph(request, publications)
        execution_root = self.store.path.parent / "execution"
        graph_store = GraphExecutionRunStore(
            execution_root,
            execution_run_id,
            database_path=self.store.path,
            output_root=execution_root / "runs" / str(execution_run_id),
        )
        return ExecutionGraphRuntime(
            graph=graph,
            execution_run_id=execution_run_id,
            deployment=DeploymentIdentity(
                job.modal_environment,
                job.modal_app_name,
                job.modal_app_version,
            ),
            volume_root=execution_root,
            artifact_volume_name=_SERVICE_ARTIFACT_VOLUME,
            workload_run_key=request.run_name,
            request=request,
            provider_driver=provider,
            max_active_provider_calls=run.max_active_provider_calls,
            max_active_gpu_provider_calls=run.max_active_gpu_provider_calls,
            strict_external_artifact_checks=True,
            external_artifact_checker=publications.check_artifact,
            store=graph_store,
            now=self._now,
        )

    def _finish_terminal_job(
        self,
        job: JobRecord,
        status: RunStatus,
        bridge: _EventLoopBridge,
    ) -> None:
        if status not in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}:
            self._project_terminal_or_running_job(job, status)
            return
        if job.result_filename is not None:
            return
        try:
            archive = bridge.call(self.adapter.recover_archive(job))
        except ArchiveNotReadyError:
            archive = bridge.call(
                self.adapter.publish_archive(job, completed_at=self._now())
            )
        self._complete_job(job, archive)

    def _suspend_after_coordinator_error(
        self,
        job_id: UUID,
        error: Exception,
    ) -> None:
        try:
            job = self.store.get_job_by_id(job_id)
            if job is None or job.execution_run_id is None:
                return
            with self.store.execution_repository() as repository:
                run = repository.get_run(job.execution_run_id)
                if run.status not in {
                    RunStatus.PENDING,
                    RunStatus.RUNNING,
                    RunStatus.SUSPENDED,
                }:
                    return
                repository.suspend_run(
                    job.execution_run_id,
                    reason=RunStatusReason.COORDINATOR_ERROR,
                    message=str(error) or type(error).__name__,
                    now=self._now(),
                )
        except Exception:
            LOGGER.exception(
                "Could not persist coordinator suspension for GROMACS job %s",
                job_id,
            )

    def _complete_job(self, job: JobRecord, archive: FinalArchive) -> None:
        try:
            self.store.complete_job(
                job.job_id,
                state=archive.state,
                result_volume_name=archive.volume_name,
                result_volume_path=archive.path,
                result_filename=archive.filename,
                result_size_bytes=archive.size_bytes,
                result_sha256=archive.sha256,
                result_archive_schema_version=GROMACS_ARCHIVE_SCHEMA_VERSION,
                warnings_json=archive.warnings_json,
                result_cached=archive.cache_lease is not None,
                now=self._now(),
            )
        finally:
            if archive.cache_lease is not None:
                archive.cache_lease.close()

    def _project_terminal_or_running_job(
        self,
        job: JobRecord,
        status: RunStatus,
    ) -> None:
        if status == RunStatus.FAILED:
            self.store.fail_job(
                job.job_id,
                error_code="compute_failed",
                error_message="GROMACS could not complete the simulation.",
                now=self._now(),
            )

    def _job(self, job_id: UUID) -> JobRecord:
        job = self.store.get_job_by_id(job_id)
        if job is None:
            raise LookupError(f"Job not found: {job_id}")
        return job

    @staticmethod
    def _run_name(job: JobRecord) -> str:
        if job.run_name is None:
            raise ValueError("GROMACS Job has no run name")
        return job.run_name

    async def _cleanup_intermediates(self) -> None:
        if self.intermediate_retention_seconds is None:
            return
        now = self._now()
        jobs = self.store.list_intermediate_cleanup_candidates(
            "gromacs",
            completed_before=now - self.intermediate_retention_seconds,
        )
        for job in jobs:
            try:
                await self.adapter.cleanup_intermediates(job)
            except Exception:
                LOGGER.exception("Could not clean intermediates for job %s", job.job_id)
                continue
            self.store.mark_intermediates_cleaned(job.job_id, now=now)


def _persisted_seed(scientific_payload: Mapping[Any, object], key: str) -> int:
    value = scientific_payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"GROMACS plan {key} is invalid")
    return value
