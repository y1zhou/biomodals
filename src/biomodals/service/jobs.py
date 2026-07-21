"""Shared API job models and workload registration."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, cast
from uuid import UUID
from weakref import WeakValueDictionary

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from biomodals.service.store import JobRecord, JobStageRecord, JobState
from biomodals.service.workloads import WORKLOAD_DEFINITIONS, WorkloadDefinition

LOGGER = logging.getLogger(__name__)
JobErrorCode = Literal["compute_failed", "result_invalid"]
JobStageCode = Literal[
    "prepare_simulation",
    "analyze_nvt",
    "analyze_npt",
    "run_production",
    "analyze_production",
    "prepare_result",
]
JobStageOutcome = Literal["completed", "failed", "cancelled"]
DeployedFunctionName = Literal[
    "prepare_tpr_cpu",
    "prepare_tpr_gpu",
    "collect_traj_stats",
    "production_run_cpu",
    "production_run_gpu",
]


class JobLifecycleLocks:
    """Serialize paid provider transitions with durable cancellation per Job.

    The MVP runs one API process and one reconciler, so an in-process lock is
    the narrow synchronization boundary needed around the database transition
    and its corresponding provider call. Durable state remains in SQLite.
    """

    def __init__(self) -> None:
        """Create an initially empty lifecycle-lock registry."""
        self._locks: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()

    def for_job(self, job_id: UUID) -> asyncio.Lock:
        """Return the event-loop lock shared by HTTP and reconciliation work."""
        return self._locks.setdefault(job_id, asyncio.Lock())


class JobStageView(BaseModel):
    """Safe execution stage timing without a provider call identifier."""

    model_config = ConfigDict(frozen=True)

    code: JobStageCode
    function_name: DeployedFunctionName | None = None
    started_at: datetime
    ended_at: datetime | None = None
    outcome: JobStageOutcome | None = None


def _stage_view(
    workload: str,
    provider_operation: str,
    event: JobStageRecord | None = None,
) -> JobStageView | None:
    definition = WORKLOAD_DEFINITIONS.get(workload)
    stage = definition.stage(provider_operation) if definition is not None else None
    if stage is None or event is None:
        return None
    return JobStageView(
        code=cast(JobStageCode, stage.code),
        function_name=cast(DeployedFunctionName | None, stage.function_name),
        started_at=datetime.fromtimestamp(event.started_at, UTC),
        ended_at=(
            datetime.fromtimestamp(event.completed_at, UTC)
            if event.completed_at is not None
            else None
        ),
        outcome=cast(JobStageOutcome | None, event.outcome),
    )


def _job_stage(
    record: JobRecord,
    history: Sequence[JobStageRecord],
) -> JobStageView | None:
    if record.workload not in WORKLOAD_DEFINITIONS:
        return None
    if record.state in {
        JobState.FINALIZING,
        JobState.BLOCKED,
        JobState.SUCCEEDED,
        JobState.PARTIAL,
    } or (
        record.state == JobState.FAILED
        and record.error_code == "result_invalid"
        and record.provider_operation in {None, "collect_traj_stats:production_"}
    ):
        provider_operation = "result_packaging"
    else:
        provider_operation = record.provider_operation or ""
    event = next(
        (
            candidate
            for candidate in reversed(history)
            if candidate.provider_operation == provider_operation
        ),
        None,
    )
    return _stage_view(record.workload, provider_operation, event)


def _job_stage_history(
    record: JobRecord,
    history: Sequence[JobStageRecord],
) -> list[JobStageView]:
    if record.workload not in WORKLOAD_DEFINITIONS:
        return []
    return [
        stage
        for event in history
        if (
            stage := _stage_view(
                record.workload,
                event.provider_operation,
                event,
            )
        )
        is not None
    ]


class JobView(BaseModel):
    """Provider-neutral job details returned to a submitter."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    workload: str
    display_name: str
    state: JobState
    stage: JobStageView | None = None
    stage_history: list[JobStageView] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    state_unknown_at: datetime | None = None
    blocked_at: datetime | None = None
    next_retry_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    error_code: JobErrorCode | None = None
    error_message: str | None = None
    download_url: str | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> JobView:
        """Build a safe public view without exposing Modal identifiers or paths."""
        warnings = record.warnings
        if (
            record.state == JobState.CANCEL_REQUESTED
            and record.cancel_requested_at is not None
            and int(time.time()) - record.cancel_requested_at >= 15 * 60
        ):
            warnings.append("Cancellation is taking longer than expected.")
        stage_history = record.stage_history
        return cls(
            job_id=str(record.job_id),
            workload=record.workload,
            display_name=record.display_name,
            state=record.state,
            stage=_job_stage(record, stage_history),
            stage_history=_job_stage_history(record, stage_history),
            created_at=datetime.fromtimestamp(record.created_at, UTC),
            updated_at=datetime.fromtimestamp(record.updated_at, UTC),
            completed_at=(
                datetime.fromtimestamp(record.completed_at, UTC)
                if record.completed_at is not None
                else None
            ),
            cancel_requested_at=(
                datetime.fromtimestamp(record.cancel_requested_at, UTC)
                if record.cancel_requested_at is not None
                else None
            ),
            state_unknown_at=(
                datetime.fromtimestamp(record.state_unknown_at, UTC)
                if record.state_unknown_at is not None
                else None
            ),
            blocked_at=(
                datetime.fromtimestamp(record.blocked_at, UTC)
                if record.blocked_at is not None
                else None
            ),
            next_retry_at=(
                datetime.fromtimestamp(record.next_retry_at, UTC)
                if record.next_retry_at is not None
                else None
            ),
            warnings=warnings,
            error_code=cast(JobErrorCode | None, record.error_code),
            error_message=record.error_message,
            download_url=(
                f"/api/v1/jobs/{record.job_id}/download"
                if record.state in {JobState.SUCCEEDED, JobState.PARTIAL}
                else None
            ),
        )


class JobPageView(BaseModel):
    """One bounded page of private Job history."""

    model_config = ConfigDict(frozen=True)

    jobs: list[JobView]
    next_cursor: UUID | None = None


class Reconciler(Protocol):
    """Background lifecycle reconciler supplied by a workload module."""

    async def reconcile(self) -> None:
        """Refresh all locally active jobs for this workload once."""
        ...


CancelJob = Callable[[JobRecord], Awaitable[None]]
ReadArtifact = Callable[[JobRecord], AsyncIterable[bytes]]
PreflightWorkload = Callable[[str, str, int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class WorkloadRegistration:
    """Explicit contribution made by one app or workflow service module."""

    definition: WorkloadDefinition
    router: APIRouter
    lifecycle_locks: JobLifecycleLocks
    reconciler: Reconciler | None = None
    cancel: CancelJob | None = None
    read_artifact: ReadArtifact | None = None
    rebuild_artifact: ReadArtifact | None = None
    preflight: PreflightWorkload | None = None
    max_body_bytes: int = 1024 * 1024

    @property
    def name(self) -> str:
        """Return the stable workload key used by routing and persistence."""
        return self.definition.name


async def reconciliation_loop(
    workloads: Sequence[WorkloadRegistration],
    *,
    interval_seconds: float,
    stop: asyncio.Event,
) -> None:
    """Refresh active jobs until application shutdown."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    while not stop.is_set():
        for workload in workloads:
            if workload.reconciler is None:
                continue
            try:
                await workload.reconciler.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Could not reconcile %s jobs", workload.name)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            pass
