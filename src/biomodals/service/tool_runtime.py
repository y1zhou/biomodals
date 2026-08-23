"""Thin service boundary around deployed Tool coordinators."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from biomodals.execution import DeploymentIdentity, ExecutionOverview, RunStatus
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.remote_execution import (
    ExecutionLocator,
    RemoteExecutionClient,
    RemoteSubmissionOutcomeUnknownError,
)
from biomodals.service.store import JobRecord, JobState, ServiceStore
from biomodals.service.tools import ToolDefinition, project_overview

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedResult:
    """Verified locally cached result metadata."""

    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    archive_schema: str


class ToolAdapter(Protocol):
    """Only the Tool-specific service operations."""

    async def stage(self, job: JobRecord) -> None:
        """Idempotently stage one admitted request."""

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
        self._locks: dict[UUID, asyncio.Lock] = {}

    async def advance(self, job_id: UUID, *, force_refresh: bool = False) -> JobRecord:
        """Perform at most one idempotent service-side lifecycle pass."""
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self._required_job(job_id)
            registration = self.registrations[job.tool]
            now = int(time.time())
            if job.state == JobState.QUEUED and job.root_function_call_id is None:
                await registration.adapter.stage(job)
                await registration.adapter.discard_pending(job)
                self.store.mark_request_staged(job_id, now=now)
                locator = _locator(job)
                try:
                    call_id = await self.remote.launch(locator)
                except RemoteSubmissionOutcomeUnknownError as error:
                    return self.store.mark_state_unknown(
                        job_id,
                        reason="submission_outcome_unknown",
                        message=str(error),
                        now=now,
                    )
                return self.store.record_launch(
                    job_id,
                    function_call_id=call_id,
                    now=now,
                )
            if job.state == JobState.CANCEL_REQUESTED:
                overview = await self.remote.cancel(_locator(job))
                return await self._observe(job, overview, registration, now=now)
            if job.state == JobState.BLOCKED and job.result_state is not None:
                return await self._finalize(
                    job,
                    registration,
                    result_state=JobState(job.result_state),
                    now=now,
                )
            if job.state in {JobState.RUNNING, JobState.BLOCKED}:
                if (
                    not force_refresh
                    and job.projection_observed_at is not None
                    and now - job.projection_observed_at < 60
                ):
                    return job
                overview = (
                    await self.remote.poll_root(job.root_function_call_id)
                    if job.root_function_call_id is not None
                    else None
                )
                if overview is None:
                    overview = await self.remote.status(_locator(job))
                return await self._observe(job, overview, registration, now=now)
            if job.state == JobState.FINALIZING:
                return await self._finalize(
                    job,
                    registration,
                    result_state=JobState(job.result_state or JobState.SUCCEEDED),
                    now=now,
                )
            return job

    async def _observe(
        self,
        job: JobRecord,
        overview: ExecutionOverview,
        registration: ToolRegistration,
        *,
        now: int,
    ) -> JobRecord:
        projection = project_overview(registration.definition, overview)
        if overview.run.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}:
            result_state = JobState(overview.run.status.value)
            job = self.store.begin_finalization(
                job.job_id,
                result_state=result_state,
                projection=projection,
                now=now,
            )
            return await self._finalize(
                job,
                registration,
                result_state=result_state,
                now=now,
            )
        state = {
            RunStatus.PENDING: JobState.RUNNING,
            RunStatus.RUNNING: JobState.RUNNING,
            RunStatus.CANCEL_REQUESTED: JobState.CANCEL_REQUESTED,
            RunStatus.SUSPENDED: JobState.BLOCKED,
            RunStatus.STATE_UNKNOWN: JobState.STATE_UNKNOWN,
            RunStatus.FAILED: JobState.FAILED,
            RunStatus.CANCELLED: JobState.CANCELLED,
        }[overview.run.status]
        return self.store.replace_projection(
            job.job_id,
            state=state,
            projection=projection,
            observed_at=now,
        )

    async def _finalize(
        self,
        job: JobRecord,
        registration: ToolRegistration,
        *,
        result_state: JobState,
        now: int,
    ) -> JobRecord:
        try:
            result = await registration.adapter.prepare_result(
                job,
                self.cache,
                completed_at=job.finalization_started_at or now,
            )
        except Exception as error:
            return self.store.block_job(
                job.job_id,
                category="result_preparation_failed",
                message=str(error),
                retry_at=now + 60,
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
            now=now,
        )

    def _required_job(self, job_id: UUID) -> JobRecord:
        job = self.store.get_job_by_id(job_id)
        if job is None:
            raise LookupError(f"Job not found: {job_id}")
        return job

    async def discard_pending(self, job: JobRecord) -> None:
        """Remove a queued request cancelled before remote launch."""
        await self.registrations[job.tool].adapter.discard_pending(job)

    async def restore_result(self, job: JobRecord) -> PreparedResult:
        """Rebuild a cleared local archive from its remote publication."""
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
            raise RuntimeError("Rebuilt Result does not match its recorded identity")
        return result


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
) -> None:
    """Retry bounded service work; remote coordinators keep executing alone."""
    while not stop.is_set():
        now = int(time.time())
        for job in lifecycle.store.list_reconcilable_jobs(now=now):
            try:
                await lifecycle.advance(job.job_id)
            except Exception:
                LOGGER.exception("Could not reconcile Job %s", job.job_id)
                continue
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            pass
