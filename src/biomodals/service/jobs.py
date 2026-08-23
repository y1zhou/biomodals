"""Public service Job projections."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from biomodals.service.store import JobRecord, JobState


class StageTaskCounts(BaseModel):
    """Bounded Task counts aggregated across one semantic stage."""

    model_config = ConfigDict(frozen=True)

    pending: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    skipped: int = 0


class JobStageView(BaseModel):
    """One Tool-owned semantic stage."""

    model_config = ConfigDict(frozen=True)

    code: str
    label: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    outcome: Literal["completed", "failed", "cancelled"] | None = None
    task_counts: StageTaskCounts = Field(default_factory=StageTaskCounts)
    running_functions: list[str] = Field(default_factory=list)


class JobView(BaseModel):
    """Owner-safe local projection of one remote execution Run."""

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    tool: str
    display_name: str
    state: JobState
    can_view_logs: bool
    stages: list[JobStageView]
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    download_url: str | None = None

    @classmethod
    def from_record(
        cls,
        record: JobRecord,
        *,
        can_view_logs: bool,
    ) -> JobView:
        """Decode the bounded disposable projection."""
        stages = [
            JobStageView(
                code=str(stage["code"]),
                label=str(stage.get("label", stage["code"])),
                started_at=_timestamp(stage.get("started_at")),
                ended_at=_timestamp(stage.get("ended_at")),
                outcome=stage.get("outcome"),
                task_counts=StageTaskCounts.model_validate(
                    stage.get("task_counts", {})
                ),
                running_functions=[
                    str(value)
                    for value in stage.get("running_functions", [])
                    if isinstance(value, str)
                ],
            )
            for stage in record.projection.get("stages", [])
            if isinstance(stage, dict) and isinstance(stage.get("code"), str)
        ]
        return cls(
            job_id=record.job_id,
            tool=record.tool,
            display_name=record.display_name,
            state=record.state,
            can_view_logs=can_view_logs,
            stages=stages,
            created_at=datetime.fromtimestamp(record.created_at, UTC),
            updated_at=datetime.fromtimestamp(record.updated_at, UTC),
            completed_at=_timestamp(record.completed_at),
            cancel_requested_at=_timestamp(record.cancel_requested_at),
            warnings=record.warnings,
            error_code=record.error_code,
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


def _timestamp(value: object) -> datetime | None:
    if isinstance(value, int):
        return datetime.fromtimestamp(value, UTC)
    return None
