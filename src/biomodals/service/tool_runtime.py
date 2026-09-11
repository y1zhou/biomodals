"""Thin service boundary around deployed Tool coordinators."""

from __future__ import annotations

import asyncio
import errno
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID
from weakref import WeakValueDictionary

import modal.exception

from biomodals.execution import DeploymentIdentity, ExecutionOverview, RunStatus
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.remote_execution import (
    ExecutionLocator,
    RemoteDeploymentUnavailableError,
    RemoteExecutionClient,
    RemoteExecutionIdentityMismatchError,
    RemoteExecutionNotInitializedError,
    RemoteSubmissionOutcomeUnknownError,
)
from biomodals.service.store import JobRecord, JobState, ServiceStore
from biomodals.service.tools import ToolDefinition, project_overview

LOGGER = logging.getLogger(__name__)
_TRANSIENT_RESULT_ERRORS = (
    ConnectionError,
    TimeoutError,
    modal.exception.ConnectionError,
    modal.exception.InternalError,
    modal.exception.ResourceExhaustedError,
    modal.exception.ServiceError,
    modal.exception.TimeoutError,
)
_RECOVERABLE_STORAGE_ERRNOS = frozenset({errno.EDQUOT, errno.ENOSPC})
_RECONCILIATION_CONCURRENCY = 4


class ResultIntegrityError(RuntimeError):
    """An exact previously published Result could not be restored."""


class EnvironmentPreparationError(RuntimeError):
    """Required runtime dependencies need operator intervention before launch."""


@dataclass(frozen=True, slots=True)
class PreparedResult:
    """Verified locally cached result metadata."""

    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    archive_schema: str


@dataclass(frozen=True, slots=True)
class SubmissionWait:
    """A service-owned reason to defer launching one remote coordinator."""

    reason: str
    message: str
    retry_after_seconds: int = 60


class ToolAdapter(Protocol):
    """Only the Tool-specific service operations."""

    async def stage(self, job: JobRecord) -> SubmissionWait | None:
        """Idempotently stage one admitted request or defer its launch."""

    async def discard_pending(self, job: JobRecord) -> None:
        """Remove request staging after verified remote publication."""

    async def prepare_result(
        self,
        job: JobRecord,
        cache: ArtifactCache,
        *,
        completed_at: int,
    ) -> PreparedResult:
        """Build or restore one verified browser download."""


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """One explicit Tool definition and its narrow service adapter."""

    definition: ToolDefinition
    adapter: ToolAdapter


class JobLifecycle:
    """Project remote execution without becoming an execution scheduler."""

    def __init__(
        self,
        store: ServiceStore,
        remote: RemoteExecutionClient,
        registrations: Sequence[ToolRegistration],
        cache: ArtifactCache,
    ) -> None:
        """Bind service persistence to exact remote coordinator clients."""
        self.store = store
        self.remote = remote
        self.cache = cache
        self.registrations = {
            registration.definition.key: registration for registration in registrations
        }
        if len(self.registrations) != len(registrations):
            raise ValueError("Tool registrations must be unique")
        self._locks: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()
        self._restore_tasks: dict[UUID, asyncio.Task[PreparedResult]] = {}

    async def advance(
        self,
        job_id: UUID,
        *,
        force_refresh: bool = False,
        finalize: bool = False,
        background: bool = False,
    ) -> JobRecord:
        """Perform at most one idempotent service-side lifecycle pass."""
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        if lock.locked() and not force_refresh:
            return self._required_job(job_id)
        async with lock:
            job = self._required_job(job_id)
            registration = self.registrations[job.tool]
            now = int(time.time())
            if job.state == JobState.QUEUED and job.root_function_call_id is None:
                try:
                    waiting = await registration.adapter.stage(job)
                except EnvironmentPreparationError as error:
                    LOGGER.warning("Runtime preparation failed", exc_info=True)
                    return self.store.fail_job(
                        job_id,
                        error_code="environment_preparation_failed",
                        error_message=str(error),
                        now=now,
                    )
                if waiting is not None:
                    return self.store.defer_submission(
                        job_id,
                        reason=waiting.reason,
                        message=waiting.message,
                        retry_at=now + waiting.retry_after_seconds,
                        now=now,
                    )
                await registration.adapter.discard_pending(job)
                self.store.mark_request_staged(job_id, now=now)
                self.store.mark_submission_in_progress(job_id, now=now)
                locator = _locator(job)
                try:
                    call_id = await self.remote.launch(locator)
                except RemoteSubmissionOutcomeUnknownError:
                    LOGGER.warning("Remote launch outcome is unknown", exc_info=True)
                    return self.store.mark_state_unknown(
                        job_id,
                        reason="submission_outcome_unknown",
                        message="The remote launch outcome could not be confirmed",
                        now=now,
                    )
                return self.store.record_launch(
                    job_id,
                    function_call_id=call_id,
                    now=now,
                )
            if job.state == JobState.CANCEL_REQUESTED:
                if not background:
                    return job
                try:
                    overview = await self.remote.cancel(
                        _locator(job), root_function_call_id=job.root_function_call_id
                    )
                except RemoteExecutionNotInitializedError:
                    return self._finish_uninitialized(job, now=now)
                except RemoteExecutionIdentityMismatchError:
                    LOGGER.warning(
                        "Remote execution identity is unknown", exc_info=True
                    )
                    return self.store.mark_state_unknown(
                        job_id,
                        reason="provider_outcome_unknown",
                        message="The remote execution identity could not be confirmed",
                        now=now,
                    )
                except RemoteDeploymentUnavailableError:
                    LOGGER.warning("Remote deployment is unavailable", exc_info=True)
                    return self.store.mark_state_unknown(
                        job_id,
                        reason="deployment_unavailable",
                        message="The deployed Tool could not be reached",
                        now=now,
                    )
                return await self._observe(job, overview, registration, now=now)
            if job.state == JobState.BLOCKED and job.result_state is not None:
                if finalize:
                    return await self._finalize(
                        job,
                        registration,
                        result_state=JobState(job.result_state),
                    )
                return job
            if job.state in {
                JobState.RUNNING,
                JobState.BLOCKED,
                JobState.STATE_UNKNOWN,
            }:
                if (
                    job.state == JobState.STATE_UNKNOWN
                    and job.root_function_call_id is None
                ):
                    return job
                if (
                    not force_refresh
                    and not background
                    and job.projection_observed_at is not None
                    and now - job.projection_observed_at < 60
                ):
                    return job
                try:
                    overview = None
                    active_root = False
                    if (
                        background
                        or force_refresh
                        or job.state == JobState.STATE_UNKNOWN
                    ) and job.root_function_call_id is not None:
                        overview = await self.remote.poll_root(
                            _locator(job), job.root_function_call_id
                        )
                        active_root = overview is None
                        if active_root and not force_refresh:
                            return self.store.touch_job(job_id, now=now)
                    if overview is None:
                        overview = await self.remote.status(_locator(job))
                except RemoteExecutionNotInitializedError:
                    return self._finish_uninitialized(job, now=now)
                except RemoteExecutionIdentityMismatchError:
                    LOGGER.warning(
                        "Remote execution identity is unknown", exc_info=True
                    )
                    return self.store.mark_state_unknown(
                        job_id,
                        reason="provider_outcome_unknown",
                        message="The remote execution identity could not be confirmed",
                        now=now,
                    )
                except RemoteDeploymentUnavailableError:
                    LOGGER.warning("Remote deployment is unavailable", exc_info=True)
                    return self.store.mark_state_unknown(
                        job_id,
                        reason="deployment_unavailable",
                        message="The deployed Tool could not be reached",
                        now=now,
                    )
                return await self._observe(
                    job,
                    overview,
                    registration,
                    now=now,
                    finalize=finalize,
                    resume_recoverable=force_refresh and not active_root,
                )
            if finalize and job.state == JobState.FINALIZING:
                return await self._finalize(
                    job,
                    registration,
                    result_state=JobState(job.result_state or JobState.SUCCEEDED),
                )
            return job

    def _finish_uninitialized(self, job: JobRecord, *, now: int) -> JobRecord:
        """Release admission only when no previously observed Run is missing."""
        if job.projection_observed_at is not None:
            return self.store.mark_state_unknown(
                job.job_id,
                reason="remote_execution_missing",
                message="A previously observed execution could not be found",
                now=now,
            )
        if job.cancel_requested_at is not None:
            return self.store.replace_projection(
                job.job_id, state=JobState.CANCELLED, projection={}, observed_at=now
            )
        return self.store.fail_job(
            job.job_id,
            error_code="execution_initialization_failed",
            error_message=(
                "The remote execution failed before initialization. "
                "Contact an administrator before submitting a new Job."
            ),
            now=now,
        )

    async def resolve_state_unknown(
        self,
        job_id: UUID,
        *,
        resolution: str,
        function_call_id: str | None,
    ) -> JobRecord:
        """Serialize an Administrator resolution with every Job side effect."""
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            return self.store.resolve_state_unknown(
                job_id,
                resolution=resolution,
                function_call_id=function_call_id,
                now=int(time.time()),
            )

    async def cancel(self, job_id: UUID) -> JobRecord:
        """Acknowledge durable intent; reconciliation delivers remote cancellation."""
        job = self._required_job(job_id)
        if job.state == JobState.CANCEL_REQUESTED:
            return job
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self.store.request_cancel(job_id, now=int(time.time()))
            if job.state == JobState.CANCELLED:
                await self.registrations[job.tool].adapter.discard_pending(job)
            return job

    async def _observe(
        self,
        job: JobRecord,
        overview: ExecutionOverview,
        registration: ToolRegistration,
        *,
        now: int,
        finalize: bool = False,
        resume_recoverable: bool = False,
    ) -> JobRecord:
        queued_call_handles = (
            await self.remote.queued_provider_call_handles(
                job.root_function_call_id,
                overview,
            )
            if overview.representative_provider_calls
            else frozenset()
        )
        projection = project_overview(
            registration.definition,
            overview,
            queued_provider_call_handles=queued_call_handles,
        )
        if overview.run.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}:
            result_state = JobState(overview.run.status.value)
            job = self.store.begin_finalization(
                job.job_id,
                result_state=result_state,
                projection=projection,
                now=now,
            )
            if finalize:
                return await self._finalize(
                    job,
                    registration,
                    result_state=result_state,
                )
            return job
        state = {
            RunStatus.PENDING: JobState.RUNNING,
            RunStatus.RUNNING: JobState.RUNNING,
            RunStatus.CANCEL_REQUESTED: JobState.CANCEL_REQUESTED,
            RunStatus.SUSPENDED: JobState.BLOCKED,
            RunStatus.STATE_UNKNOWN: JobState.STATE_UNKNOWN,
            RunStatus.FAILED: JobState.FAILED,
            RunStatus.CANCELLED: JobState.CANCELLED,
        }[overview.run.status]
        projected = self.store.replace_projection(
            job.job_id,
            state=state,
            projection=projection,
            observed_at=now,
        )
        if overview.run.status == RunStatus.SUSPENDED:
            projected = self.store.block_job(
                job.job_id,
                category="remote_execution_suspended",
                message="Remote execution is suspended and can be resumed",
                retry_at=None,
                now=now,
            )
        elif overview.run.status == RunStatus.STATE_UNKNOWN:
            projected = self.store.mark_state_unknown(
                job.job_id,
                reason="remote_execution_state_unknown",
                message="Remote execution requires explicit reconciliation",
                now=now,
            )
        if (
            resume_recoverable
            and overview.run.status in {RunStatus.SUSPENDED, RunStatus.STATE_UNKNOWN}
            and job.root_function_call_id is not None
        ):
            try:
                call_id = await self.remote.resume(_locator(job))
            except RemoteSubmissionOutcomeUnknownError:
                LOGGER.warning("Remote resume outcome is unknown", exc_info=True)
                return self.store.mark_state_unknown(
                    job.job_id,
                    reason="resume_outcome_unknown",
                    message="The remote resume outcome could not be confirmed",
                    now=now,
                )
            return self.store.record_resume(
                job.job_id,
                previous_function_call_id=job.root_function_call_id,
                function_call_id=call_id,
                now=now,
            )
        if state == JobState.FAILED:
            return self.store.fail_job(
                job.job_id,
                error_code="remote_execution_failed",
                error_message="Remote execution failed",
                now=now,
            )
        return projected

    async def _finalize(
        self,
        job: JobRecord,
        registration: ToolRegistration,
        *,
        result_state: JobState,
    ) -> JobRecord:
        try:
            result = await registration.adapter.prepare_result(
                job,
                self.cache,
                completed_at=job.finalization_started_at or int(time.time()),
            )
        except _TRANSIENT_RESULT_ERRORS:
            now = int(time.time())
            LOGGER.warning(
                "Result preparation is temporarily unavailable for Job %s",
                job.job_id,
                exc_info=True,
            )
            return self.store.block_job(
                job.job_id,
                category="result_preparation_failed",
                message="Result preparation is temporarily unavailable",
                retry_at=now + 60,
                now=now,
            )
        except OSError as error:
            now = int(time.time())
            if error.errno in _RECOVERABLE_STORAGE_ERRNOS:
                LOGGER.warning(
                    "Local Result storage is temporarily unavailable for Job %s",
                    job.job_id,
                    exc_info=True,
                )
                return self.store.block_job(
                    job.job_id,
                    category="result_preparation_failed",
                    message="Result preparation is temporarily unavailable",
                    retry_at=now + 60,
                    now=now,
                )
            LOGGER.exception("Could not prepare Result for Job %s", job.job_id)
            return self.store.fail_job(
                job.job_id,
                error_code="result_preparation_failed",
                error_message="The Result archive could not be prepared",
                now=now,
            )
        except Exception:
            now = int(time.time())
            LOGGER.exception("Could not prepare Result for Job %s", job.job_id)
            return self.store.fail_job(
                job.job_id,
                error_code="result_preparation_failed",
                error_message="The Result archive could not be prepared",
                now=now,
            )
        return self.store.complete_job(
            job.job_id,
            result_state=result_state,
            result_filename=result.filename,
            result_media_type=result.media_type,
            result_size_bytes=result.size_bytes,
            result_sha256=result.sha256,
            result_archive_schema=result.archive_schema,
            now=int(time.time()),
        )

    def _required_job(self, job_id: UUID) -> JobRecord:
        job = self.store.get_job_by_id(job_id)
        if job is None:
            raise LookupError(f"Job not found: {job_id}")
        return job

    async def restore_result(self, job: JobRecord) -> PreparedResult:
        """Join one cancellation-safe exact Result restoration per Job."""
        task = self._restore_tasks.get(job.job_id)
        if task is None:
            task = asyncio.create_task(
                self._restore_result(job.job_id),
                name=f"biomodals-result-restore-{job.job_id}",
            )
            self._restore_tasks[job.job_id] = task
            task.add_done_callback(
                lambda completed, job_id=job.job_id: self._restore_finished(
                    job_id, completed
                )
            )
        return await asyncio.shield(task)

    async def _restore_result(self, job_id: UUID) -> PreparedResult:
        job = self._required_job(job_id)
        recorded = _recorded_result(job)
        lease = await self.cache.acquire_async(
            str(job_id),
            size_bytes=recorded.size_bytes,
            sha256=recorded.sha256,
        )
        if lease is not None:
            lease.close()
            return recorded
        result = await self.registrations[job.tool].adapter.prepare_result(
            job,
            self.cache,
            completed_at=job.finalization_started_at
            or job.completed_at
            or job.updated_at,
        )
        if (
            result.size_bytes != job.result_size_bytes
            or result.sha256 != job.result_sha256
            or result.archive_schema != job.result_archive_schema
        ):
            await self.cache.discard_async(str(job.job_id))
            self.store.block_job(
                job.job_id,
                category="result_integrity",
                message="The published Result could not be restored exactly",
                retry_at=None,
                now=int(time.time()),
            )
            raise ResultIntegrityError(
                "Rebuilt Result does not match its recorded identity"
            )
        self.store.restore_result_cached(job.job_id, now=int(time.time()))
        return result

    def _restore_finished(
        self,
        job_id: UUID,
        task: asyncio.Task[PreparedResult],
    ) -> None:
        if not task.cancelled():
            task.exception()
        if self._restore_tasks.get(job_id) is task:
            self._restore_tasks.pop(job_id, None)


def _locator(job: JobRecord) -> ExecutionLocator:
    return ExecutionLocator(
        execution_run_id=job.job_id,
        deployment=DeploymentIdentity(
            environment=job.modal_environment,
            deployment_name=job.modal_app_name,
            deployment_version=job.modal_app_version,
        ),
    )


async def reconciliation_loop(
    lifecycle: JobLifecycle,
    *,
    interval_seconds: float,
    stop: asyncio.Event,
    wake: asyncio.Event,
) -> None:
    """Keep slow Jobs in their own bounded slots across admission wakeups."""
    active: dict[UUID, asyncio.Task[None]] = {}
    pending: deque[JobRecord] = deque()
    loop = asyncio.get_running_loop()
    next_scan = loop.time()
    wake_task = asyncio.create_task(wake.wait())

    async def reconcile(job: JobRecord) -> None:
        try:
            await lifecycle.advance(job.job_id, finalize=True, background=True)
        except Exception:
            LOGGER.exception("Could not reconcile Job %s", job.job_id)

    try:
        while not stop.is_set():
            if wake_task.done() or loop.time() >= next_scan:
                wake.clear()
                if wake_task.done():
                    wake_task = asyncio.create_task(wake.wait())
                pending = deque(
                    job
                    for job in lifecycle.store.list_reconcilable_jobs(
                        now=int(time.time())
                    )
                    if job.job_id not in active
                )
                next_scan = loop.time() + interval_seconds
            while pending and len(active) < _RECONCILIATION_CONCURRENCY:
                job = pending.popleft()
                active[job.job_id] = asyncio.create_task(reconcile(job))
            completed, _ = await asyncio.wait(
                [wake_task, *active.values()],
                timeout=max(0, next_scan - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            active = {
                job_id: task for job_id, task in active.items() if task not in completed
            }
    finally:
        for task in (wake_task, *active.values()):
            task.cancel()
        await asyncio.gather(wake_task, *active.values(), return_exceptions=True)


def _recorded_result(job: JobRecord) -> PreparedResult:
    if (
        job.result_filename is None
        or job.result_media_type is None
        or job.result_size_bytes is None
        or job.result_sha256 is None
        or job.result_archive_schema is None
    ):
        raise ResultIntegrityError("Job has no complete recorded Result identity")
    return PreparedResult(
        filename=job.result_filename,
        media_type=job.result_media_type,
        size_bytes=job.result_size_bytes,
        sha256=job.result_sha256,
        archive_schema=job.result_archive_schema,
    )
