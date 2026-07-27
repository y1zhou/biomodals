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

from biomodals.service.store import (
    JobOperationRecord,
    JobOperationState,
    JobRecord,
    JobStageRecord,
    JobState,
    ServiceStore,
)
from biomodals.service.workloads import WorkloadDefinition

LOGGER = logging.getLogger(__name__)
JobErrorCode = Literal["compute_failed", "result_invalid"]
JobStageOutcome = Literal["completed", "failed", "cancelled"]
OperationLogMode = Literal["live", "historical"]


@dataclass(frozen=True, slots=True)
class OperationLogRequest:
    """Provider-independent selection of live or time-bounded operation logs."""

    mode: OperationLogMode
    since: datetime | None = None
    until: datetime | None = None

    def __post_init__(self) -> None:
        """Reject combinations that no provider adapter can interpret safely."""
        if self.mode == "live" and (self.since is not None or self.until is not None):
            raise ValueError("Live operation logs cannot use a time range")
        if (self.since is None) != (self.until is None):
            raise ValueError("Operation log ranges require both since and until")
        if self.since is not None and self.until is not None:
            if self.mode != "historical":
                raise ValueError("Time-bounded operation logs must be historical")
            if (
                self.since.tzinfo is None
                or self.until.tzinfo is None
                or self.since >= self.until
            ):
                raise ValueError("Operation log ranges require valid aware timestamps")


def operation_log_mode(state: JobOperationState) -> OperationLogMode | None:
    """Classify operation states that retain inspectable provider logs."""
    if state in {JobOperationState.RUNNING, JobOperationState.STATE_UNKNOWN}:
        return "live"
    if state in {
        JobOperationState.COMPLETED,
        JobOperationState.FAILED,
        JobOperationState.CANCELLED,
    }:
        return "historical"
    return None


def can_view_job_logs(
    *,
    is_admin: bool,
    owner_visibility_enabled: bool,
    logs_supported: bool,
) -> bool:
    """Return whether a caller may inspect logs exposed by one workload."""
    return logs_supported and (is_admin or owner_visibility_enabled)


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

    code: str = Field(description="Stable workload stage code.")
    function_name: str | None = Field(
        default=None,
        description="Deployed provider function for this stage, when applicable.",
    )
    started_at: datetime = Field(description="Time this stage began.")
    ended_at: datetime | None = Field(
        default=None,
        description="Time this stage ended; absent while the stage is active.",
    )
    outcome: JobStageOutcome | None = Field(
        default=None,
        description="Terminal stage outcome; absent while the stage is active.",
    )


def _stage_view(
    definition: WorkloadDefinition | None,
    operation: str,
    event: JobStageRecord | None = None,
) -> JobStageView | None:
    stage = definition.stage(operation) if definition is not None else None
    if stage is None or event is None:
        return None
    return JobStageView(
        code=stage.code,
        function_name=stage.function_name,
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
    definition: WorkloadDefinition | None,
) -> JobStageView | None:
    active = [event for event in history if event.completed_at is None]
    if active:
        event = active[-1]
    elif record.state == JobState.FAILED:
        event = next(
            (
                candidate
                for candidate in reversed(history)
                if candidate.outcome == "failed"
            ),
            history[-1] if history else None,
        )
    elif record.state == JobState.CANCELLED:
        event = next(
            (
                candidate
                for candidate in reversed(history)
                if candidate.outcome == "cancelled"
            ),
            history[-1] if history else None,
        )
    else:
        event = history[-1] if history else None
    return (
        _stage_view(definition, event.operation, event) if event is not None else None
    )


def job_stage_history(
    record: JobRecord,
    history: Sequence[JobStageRecord],
    definition: WorkloadDefinition | None,
) -> list[JobStageView]:
    """Project durable operation events into safe public Stage views."""
    return [
        stage
        for event in history
        if (
            stage := _stage_view(
                definition,
                event.operation,
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
    can_view_logs: bool = Field(
        description=(
            "Whether the authenticated caller may inspect retained provider logs "
            "for started remote stages of this Job."
        ),
    )
    state: JobState
    stage: JobStageView | None = Field(
        default=None,
        description=(
            "Representative active or most recently relevant execution stage; "
            "absent before execution starts or for an unknown workload."
        ),
    )
    active_stages: list[JobStageView] = Field(
        default_factory=list,
        description="All execution stages that are currently active.",
    )
    stage_history: list[JobStageView] = Field(
        default_factory=list,
        description="Recorded execution stages in lifecycle order.",
    )
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = Field(
        default=None,
        description="Terminal completion time; absent for non-terminal Jobs.",
    )
    cancel_requested_at: datetime | None = Field(
        default=None,
        description="Time cancellation was requested, when applicable.",
    )
    state_unknown_at: datetime | None = Field(
        default=None,
        description=(
            "Time remote execution state first became unknown; retained after "
            "administrator resolution."
        ),
    )
    blocked_at: datetime | None = Field(
        default=None,
        description="Time recoverable finalization first became blocked.",
    )
    next_retry_at: datetime | None = Field(
        default=None,
        description="Scheduled time for the next recoverable finalization retry.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Safe owner-visible lifecycle warnings.",
    )
    error_code: JobErrorCode | None = Field(
        default=None,
        description="Stable failure code; present only for failed Jobs.",
    )
    error_message: str | None = Field(
        default=None,
        description="Safe failure explanation; present only for failed Jobs.",
    )
    download_url: str | None = Field(
        default=None,
        description="Result download path for succeeded or partial Jobs only.",
    )

    @classmethod
    def from_record(
        cls,
        record: JobRecord,
        *,
        definition: WorkloadDefinition | None,
        can_view_logs: bool = False,
    ) -> JobView:
        """Build a safe public view without exposing Modal identifiers or paths."""
        warnings = record.warnings
        if (
            record.state == JobState.CANCEL_REQUESTED
            and record.cancel_requested_at is not None
            and int(time.time()) - record.cancel_requested_at >= 15 * 60
        ):
            warnings.append("Cancellation is taking longer than expected.")
        stage_history = record.stage_history
        active_stages = [
            stage
            for event in stage_history
            if event.completed_at is None
            and (
                stage := _stage_view(
                    definition,
                    event.operation,
                    event,
                )
            )
            is not None
        ]
        return cls(
            job_id=str(record.job_id),
            workload=record.workload,
            display_name=record.display_name,
            can_view_logs=can_view_logs,
            state=record.state,
            stage=_job_stage(record, stage_history, definition),
            active_stages=active_stages,
            stage_history=job_stage_history(record, stage_history, definition),
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


CancelJob = Callable[[ServiceStore, JobRecord], Awaitable[None]]
ReadArtifact = Callable[[JobRecord], AsyncIterable[bytes]]
OpenOperationLogs = Callable[
    [JobRecord, JobOperationRecord, OperationLogRequest],
    Awaitable[AsyncIterable[bytes]],
]
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
    open_operation_logs: OpenOperationLogs | None = None
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
