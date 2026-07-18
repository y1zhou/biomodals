"""Shared API job models and workload registration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, cast

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from biomodals.service.store import JobRecord, JobState

LOGGER = logging.getLogger(__name__)
JobErrorCode = Literal["compute_failed", "result_invalid", "result_unavailable"]
JobStageCode = Literal[
    "preparation",
    "nvt_analysis",
    "npt_analysis",
    "production",
    "production_analysis",
    "result_packaging",
]
DeployedFunctionName = Literal[
    "prepare_tpr_cpu",
    "prepare_tpr_gpu",
    "collect_traj_stats",
    "production_run_cpu",
    "production_run_gpu",
]


class JobStageView(BaseModel):
    """Safe current execution stage without a provider call identifier."""

    model_config = ConfigDict(frozen=True)

    code: JobStageCode
    function_name: DeployedFunctionName | None = None


_GROMACS_STAGES: dict[str, tuple[JobStageCode, DeployedFunctionName]] = {
    "prepare_tpr_cpu": ("preparation", "prepare_tpr_cpu"),
    "prepare_tpr_gpu": ("preparation", "prepare_tpr_gpu"),
    "collect_traj_stats:nvt_": ("nvt_analysis", "collect_traj_stats"),
    "collect_traj_stats:npt_": ("npt_analysis", "collect_traj_stats"),
    "production_run_cpu": ("production", "production_run_cpu"),
    "production_run_gpu": ("production", "production_run_gpu"),
    "collect_traj_stats:production_": (
        "production_analysis",
        "collect_traj_stats",
    ),
}


def _job_stage(record: JobRecord) -> JobStageView | None:
    if record.workload != "gromacs":
        return None
    if record.state in {
        JobState.FINALIZING,
        JobState.SUCCEEDED,
        JobState.PARTIAL,
    } or (
        record.state == JobState.FAILED
        and record.error_code in {"result_invalid", "result_unavailable"}
        and record.provider_operation in {None, "collect_traj_stats:production_"}
    ):
        return JobStageView(code="result_packaging")
    stage = _GROMACS_STAGES.get(record.provider_operation or "")
    return JobStageView(code=stage[0], function_name=stage[1]) if stage else None


class JobView(BaseModel):
    """Provider-neutral job details returned to a submitter."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    workload: str
    display_name: str
    state: JobState
    stage: JobStageView | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    error_code: JobErrorCode | None = None
    error_message: str | None = None
    download_url: str | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> JobView:
        """Build a safe public view without exposing Modal identifiers or paths."""
        warnings = record.warnings
        return cls(
            job_id=str(record.job_id),
            workload=record.workload,
            display_name=record.display_name,
            state=record.state,
            stage=_job_stage(record),
            created_at=datetime.fromtimestamp(record.created_at, UTC),
            updated_at=datetime.fromtimestamp(record.updated_at, UTC),
            completed_at=(
                datetime.fromtimestamp(record.completed_at, UTC)
                if record.completed_at is not None
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


class Reconciler(Protocol):
    """Background lifecycle reconciler supplied by a workload module."""

    async def reconcile(self) -> None:
        """Refresh all locally active jobs for this workload once."""
        ...


CancelJob = Callable[[JobRecord], Awaitable[None]]
ReadArtifact = Callable[[JobRecord], AsyncIterable[bytes]]


@dataclass(frozen=True, slots=True)
class WorkloadRegistration:
    """Explicit contribution made by one app or workflow service module."""

    name: str
    router: APIRouter
    reconciler: Reconciler | None = None
    cancel: CancelJob | None = None
    read_artifact: ReadArtifact | None = None
    max_body_bytes: int = 1024 * 1024


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
