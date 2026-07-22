"""Exactly-once attachment boundary for paid Modal Job operations."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.store import (
    JobRecord,
    JobState,
    JobStateUnknownReason,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)


class SubmissionOutcomeUnknownError(RuntimeError):
    """Raised when Modal may have started work without returning its call ID."""


class SubmittedModalOperation(Protocol):
    """Provider identifiers returned after a detached Modal spawn."""

    @property
    def modal_call_id(self) -> str:
        """Return the durable Modal call identifier."""
        ...

    @property
    def run_name(self) -> str:
        """Return the stable workload run name."""
        ...

    @property
    def operation(self) -> str:
        """Return the directly submitted Job operation."""
        ...


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Durable Job snapshot and whether this attempt attached a new call."""

    job: JobRecord
    attached: bool


SpawnModalOperation = Callable[[JobRecord], Awaitable[SubmittedModalOperation]]
CancelModalCall = Callable[[str], Awaitable[None]]


class ModalJobSubmitter:
    """Own the one shared database-to-Modal submission state machine."""

    def __init__(
        self,
        store: ServiceStore,
        lifecycle_locks: JobLifecycleLocks,
        *,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind durable state, per-Job serialization, and a testable clock."""
        self.store = store
        self.lifecycle_locks = lifecycle_locks
        self._now = now or (lambda: int(time.time()))

    async def submit(
        self,
        job: JobRecord,
        *,
        operation: str,
        run_name: str,
        submission_token: str,
        spawn: SpawnModalOperation,
        cancel: CancelModalCall,
    ) -> SubmissionResult:
        """Claim, spawn, and attach one Modal operation without duplicate work."""
        if job.operations or job.state != JobState.QUEUED:
            return SubmissionResult(job=job, attached=False)

        async with self.lifecycle_locks.for_job(job.job_id):
            claimed = self.store.claim_modal_operation(
                job.job_id,
                operation=operation,
                run_name=run_name,
                submission_token=submission_token,
                now=self._now(),
            )
            if claimed is None:
                return SubmissionResult(job=self._reload(job), attached=False)
            claimed_job = self._reload(job)

            try:
                submitted = await spawn(claimed_job)
            except SubmissionOutcomeUnknownError:
                return SubmissionResult(
                    job=self.store.mark_state_unknown(
                        job.job_id,
                        reason=JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN,
                        now=self._now(),
                    ),
                    attached=False,
                )
            except Exception:
                self.store.release_operation(
                    job.job_id,
                    operation=operation,
                    submission_token=submission_token,
                    now=self._now(),
                )
                raise

            try:
                if submitted.run_name != run_name:
                    raise RuntimeError("Modal returned the wrong run name")
                if submitted.operation != operation:
                    raise RuntimeError("Modal returned the wrong Job operation")
                attached = self.store.attach_modal_call(
                    job.job_id,
                    operation=operation,
                    modal_call_id=submitted.modal_call_id,
                    submission_token=submission_token,
                    now=self._now(),
                )
            except Exception:
                LOGGER.exception(
                    "Could not attach Modal call for operation %s on Job %s",
                    operation,
                    job.job_id,
                )
                try:
                    await cancel(submitted.modal_call_id)
                except Exception:
                    LOGGER.exception(
                        "Could not cancel unattached Modal call %s",
                        submitted.modal_call_id,
                    )
                return SubmissionResult(
                    job=self.store.mark_state_unknown(
                        job.job_id,
                        reason=JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN,
                        now=self._now(),
                    ),
                    attached=False,
                )
            return SubmissionResult(job=attached, attached=True)

    def _reload(self, job: JobRecord) -> JobRecord:
        current = self.store.get_job(job.owner_user_id, job.job_id)
        if current is None:  # pragma: no cover - the admitted row cannot disappear
            raise RuntimeError(f"Admitted Job disappeared: {job.job_id}")
        return current
