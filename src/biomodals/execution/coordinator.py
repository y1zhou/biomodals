"""Reusable run-scoped coordinator loop mechanics."""

from __future__ import annotations

import logging
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
    now: Callable[[], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 1.0,
    synchronize: Callable[[], AbstractContextManager[None]] = nullcontext,
) -> ExecutionSnapshot:
    """Advance one Run, releasing its optional host lock between cycles."""
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds cannot be negative")
    clock = now or (lambda: int(time.time()))

    while True:
        with synchronize():
            if repository.get_run(execution_run_id).status not in _DRIVABLE_STATUSES:
                return repository.snapshot(execution_run_id)
            try:
                advance_once()
                replacement = checkpoint()
                if replacement is not None:
                    repository = replacement
            except Exception as exc:
                _suspend_after_application_error(
                    repository,
                    execution_run_id,
                    message=str(exc) or type(exc).__name__,
                    checkpoint=checkpoint,
                    now=clock(),
                )
                raise
            keep_driving = (
                repository.get_run(execution_run_id).status in _DRIVABLE_STATUSES
            )
        if keep_driving:
            sleep(poll_interval_seconds)


def resume_execution_run(
    repository: SqliteExecutionRepository,
    execution_run_id: UUID,
    *,
    checkpoint: Callable[[], SqliteExecutionRepository | None],
    now: int,
) -> ExecutionRunRecord:
    """Explicitly resume one suspended Run and cross its durability boundary."""
    run = repository.resume_run(execution_run_id, now=now)
    replacement = checkpoint()
    if replacement is not None:
        run = replacement.get_run(execution_run_id)
    return run


def _suspend_after_application_error(
    repository: SqliteExecutionRepository,
    execution_run_id: UUID,
    *,
    message: str,
    checkpoint: Callable[[], SqliteExecutionRepository | None],
    now: int,
) -> None:
    """Best-effort persistence for an uncaught coordinator application error."""
    run = repository.get_run(execution_run_id)
    if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
        return
    try:
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
