"""Reusable run-scoped coordinator loop mechanics."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from uuid import UUID

from biomodals.execution.model import (
    ExecutionRunRecord,
    ExecutionSnapshot,
    RunStatus,
    RunStatusReason,
)
from biomodals.execution.sqlite import SqliteExecutionRepository

COORDINATOR_SCALEDOWN_WINDOW_SECONDS = 2
_DRIVABLE_STATUSES = {
    RunStatus.PENDING,
    RunStatus.RUNNING,
    RunStatus.CANCEL_REQUESTED,
}
LOGGER = logging.getLogger(__name__)


def drive_execution_run(
    repository: SqliteExecutionRepository,
    execution_run_id: UUID,
    *,
    advance_once: Callable[[], None],
    checkpoint: Callable[[], SqliteExecutionRepository | None],
    current_repository: Callable[[], SqliteExecutionRepository] | None = None,
    now: Callable[[], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 1.0,
    synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> ExecutionSnapshot:
    """Advance one Run without holding its writer across workload or provider I/O."""
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds cannot be negative")
    clock = now or (lambda: int(time.time()))

    while True:
        with synchronize():
            if current_repository is not None:
                repository = current_repository()
            run = repository.get_run(execution_run_id)
            if run.status not in _DRIVABLE_STATUSES:
                snapshot = repository.snapshot(execution_run_id)
                reason = (
                    run.status_reason.value if run.status_reason is not None else None
                )
                log = (
                    LOGGER.warning
                    if run.status in {RunStatus.SUSPENDED, RunStatus.STATE_UNKNOWN}
                    else LOGGER.info
                )
                log(
                    "Execution coordinator input returning for Run %s: "
                    "status=%s, reason=%s, message=%r; automatic scheduling stopped; "
                    "durable occupied Provider Call slots=%d (%d GPU). These "
                    "are ownership records, not confirmed live Modal containers.",
                    execution_run_id,
                    run.status.value,
                    reason,
                    run.status_message,
                    snapshot.active_provider_calls.total,
                    snapshot.active_provider_calls.gpu,
                )
                return snapshot
        try:
            advance_once()
        except Exception as exc:
            with synchronize():
                if current_repository is not None:
                    repository = current_repository()
                _suspend_after_application_error(
                    repository,
                    execution_run_id,
                    message=str(exc) or type(exc).__name__,
                    checkpoint=checkpoint,
                    now=clock(),
                )
            raise
        with synchronize():
            if current_repository is not None:
                repository = current_repository()
            keep_driving = (
                repository.get_run(execution_run_id).status in _DRIVABLE_STATUSES
            )
            if not keep_driving:
                replacement = checkpoint()
                if replacement is not None:
                    repository = replacement
        if keep_driving:
            sleep(poll_interval_seconds)


def resume_execution_run(
    repository: SqliteExecutionRepository,
    execution_run_id: UUID,
    *,
    reconcile_once: Callable[[], None],
    checkpoint: Callable[[], SqliteExecutionRepository | None],
    now: int,
    current_repository: Callable[[], SqliteExecutionRepository] | None = None,
    synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> ExecutionRunRecord:
    """Resume one Run without holding its writer during provider reconciliation."""
    with synchronize():
        if current_repository is not None:
            repository = current_repository()
        run = repository.get_run(execution_run_id)
        if run.status == RunStatus.SUSPENDED:
            repository.resume_run(execution_run_id, now=now)
            replacement = checkpoint()
            if replacement is not None:
                repository = replacement
            return repository.get_run(execution_run_id)
        if run.status != RunStatus.STATE_UNKNOWN:
            raise ValueError(
                "only a suspended or state_unknown Run can be explicitly resumed"
            )

    reconcile_once()
    with synchronize():
        if current_repository is not None:
            repository = current_repository()
        replacement = checkpoint()
        if replacement is not None:
            repository = replacement
        return repository.get_run(execution_run_id)


def _suspend_after_application_error(
    repository: SqliteExecutionRepository,
    execution_run_id: UUID,
    *,
    message: str,
    checkpoint: Callable[[], SqliteExecutionRepository | None],
    now: int,
) -> None:
    """Best-effort persistence for an uncaught coordinator application error."""
    try:
        try:
            run = repository.get_run(execution_run_id)
        except sqlite3.ProgrammingError:
            replacement = checkpoint()
            if replacement is None:
                raise
            repository = replacement
            run = repository.get_run(execution_run_id)
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        repository.transition_run(
            execution_run_id,
            RunStatus.SUSPENDED,
            reason=RunStatusReason.COORDINATOR_ERROR,
            message=message,
            now=now,
        )
        checkpoint()
    except Exception:
        # Preserve the original coordinator failure. A replacement attempt will
        # recover whichever earlier checkpoint is actually durable.
        LOGGER.exception("Could not persist coordinator suspension")
