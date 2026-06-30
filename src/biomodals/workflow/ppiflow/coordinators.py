"""Candidate-wide coordinator helpers for the PPIFlow workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from pydantic import BaseModel

from biomodals.helper.task_budget import bounded_map
from biomodals.schema import AppRunStatus
from biomodals.workflow.ppiflow import manifests

DEFAULT_CANDIDATE_CONCURRENCY = 4


@dataclass(frozen=True)
class CandidateTask:
    """One candidate unit of work for a candidate-wide stage."""

    candidate_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateOutcome:
    """Result for one candidate inside a candidate-wide stage."""

    candidate_id: str
    status: AppRunStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def candidate_concurrency_from_config(
    *configs: Mapping[str, Any],
    default: int = DEFAULT_CANDIDATE_CONCURRENCY,
) -> int:
    """Resolve shared or per-stage candidate concurrency."""
    for config in configs:
        if "candidate_concurrency" in config:
            value = int(config["candidate_concurrency"])
            if value < 1:
                raise ValueError("candidate_concurrency must be at least 1")
            return value
    return default


def run_candidate_tasks(
    tasks: Sequence[CandidateTask],
    submit: Callable[[CandidateTask], CandidateOutcome],
    *,
    candidate_concurrency: int = DEFAULT_CANDIDATE_CONCURRENCY,
) -> list[CandidateOutcome]:
    """Run candidate tasks with bounded local concurrency."""
    if candidate_concurrency < 1:
        raise ValueError("candidate_concurrency must be at least 1")
    if not tasks:
        return []

    def safe_submit(task: CandidateTask) -> CandidateOutcome:
        try:
            return submit(task)
        except Exception as exc:  # noqa: BLE001
            return CandidateOutcome(
                candidate_id=task.candidate_id,
                status=AppRunStatus.FAILED,
                error=str(exc),
            )

    outcomes = bounded_map(tasks, safe_submit, max_parallel=candidate_concurrency)
    return sorted(outcomes, key=lambda outcome: outcome.candidate_id)


def status_from_candidate_outcomes(
    outcomes: Sequence[CandidateOutcome],
) -> AppRunStatus:
    """Return stage status from per-candidate outcomes."""
    succeeded = sum(
        1 for outcome in outcomes if outcome.status == AppRunStatus.SUCCEEDED
    )
    if outcomes and succeeded == len(outcomes):
        return AppRunStatus.SUCCEEDED
    if succeeded:
        return AppRunStatus.PARTIAL
    return AppRunStatus.FAILED


def pending_candidate_tasks(
    manifest_frame: pl.DataFrame,
    *,
    reusable_candidate_ids: Iterable[str],
) -> list[CandidateTask]:
    """Build tasks for manifest rows that are not reusable."""
    reusable = set(reusable_candidate_ids)
    tasks = []
    for row in manifest_frame.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        if candidate_id in reusable:
            continue
        tasks.append(CandidateTask(candidate_id=candidate_id, payload=dict(row)))
    return tasks


def outcome_manifest_rows(
    *,
    stage_name: str,
    stage_role: str,
    operation_mode: str,
    outcomes: Sequence[CandidateOutcome],
) -> list[dict[str, object]]:
    """Convert candidate outcomes to manifest rows."""
    return [
        manifests.candidate_manifest_row(
            candidate_id=outcome.candidate_id,
            stage_name=stage_name,
            stage_role=stage_role,
            operation_mode=operation_mode,
            candidate_status=outcome.status.value,
            error=outcome.error,
            summary=_jsonable(outcome.outputs),
            files=outcome.outputs.get("files", [])
            if isinstance(outcome.outputs.get("files"), Sequence)
            else [],
        )
        for outcome in outcomes
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


def merge_candidate_manifest_rows(
    reusable_rows: pl.DataFrame,
    outcome_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Merge reusable manifest rows with newly produced outcome rows."""
    rows = reusable_rows.to_dicts()
    rows.extend(dict(row) for row in outcome_rows)
    rows.sort(key=lambda row: str(row["candidate_id"]))
    return rows
