"""Durable GROMACS operation reconciliation and Result finalization."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from typing import Protocol
from uuid import uuid4

import modal

from biomodals.service.artifacts import (
    ArtifactIntegrityError,
    ArtifactSourceMissingError,
)
from biomodals.service.gromacs.archive import GROMACS_ARCHIVE_SCHEMA_VERSION
from biomodals.service.gromacs.plan import (
    FINAL_OPERATION,
    all_operations_completed,
    ready_operations,
)
from biomodals.service.gromacs.provider import (
    DEFINITE_SUBMISSION_ERRORS,
    MODAL_SERVICE_ERRORS,
    PollOutcome,
    SubmittedModalCall,
)
from biomodals.service.gromacs.results import (
    PERMANENT_FINALIZATION_ERRORS,
    TRANSIENT_FINALIZATION_ERRORS,
    ArchiveNotReadyError,
    FinalArchive,
    GromacsResultInvalidError,
    ResultIdentityMismatchError,
)
from biomodals.service.gromacs.router import GromacsJobOptions
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.store import (
    JobOperationRecord,
    JobOperationState,
    JobRecord,
    JobState,
    JobStateUnknownReason,
    ServiceStore,
)
from biomodals.service.submission import ModalJobSubmitter

LOGGER = logging.getLogger(__name__)


class GromacsRuntime(Protocol):
    """Compute and Result capabilities required by reconciliation."""

    async def submit_operation(
        self,
        job: JobRecord,
        operation: str,
    ) -> SubmittedModalCall:
        """Spawn one dependency-ready operation."""
        ...

    async def poll(
        self,
        modal_call_id: str,
        *,
        operation: str | None = None,
    ) -> PollOutcome:
        """Poll one attached Modal call without blocking."""
        ...

    async def cancel(self, modal_call_id: str) -> None:
        """Request cancellation of one Modal call graph."""
        ...

    async def cleanup_intermediates(self, job: JobRecord) -> None:
        """Remove rebuildable remote intermediate files."""
        ...

    async def publish_archive(
        self,
        job: JobRecord,
        *,
        completed_at: int,
    ) -> FinalArchive:
        """Build and publish one immutable Result archive."""
        ...

    async def recover_archive(self, job: JobRecord) -> FinalArchive:
        """Recover an already published immutable Result."""
        ...


def _ready_operations(
    job: JobRecord,
    calls: Iterable[JobOperationRecord],
) -> list[str]:
    options = GromacsJobOptions.model_validate_json(job.parameters_json)
    return ready_operations(cpu_only=options.cpu_only, operations=calls)


def _all_operations_completed(
    job: JobRecord,
    calls: Iterable[JobOperationRecord],
) -> bool:
    options = GromacsJobOptions.model_validate_json(job.parameters_json)
    return all_operations_completed(cpu_only=options.cpu_only, operations=calls)


class GromacsReconciler:
    """Refresh locally persisted GROMACS jobs from Modal in one process."""

    def __init__(
        self,
        store: ServiceStore,
        adapter: GromacsRuntime,
        *,
        lifecycle_locks: JobLifecycleLocks | None = None,
        now: Callable[[], int] | None = None,
        intermediate_retention_days: int | None = None,
        max_concurrent_jobs: int = 4,
    ) -> None:
        """Bind durable state to the provider adapter."""
        self.store = store
        self.adapter = adapter
        self.lifecycle_locks = lifecycle_locks or JobLifecycleLocks()
        self._now = now or (lambda: int(time.time()))
        self.submitter = ModalJobSubmitter(
            store,
            self.lifecycle_locks,
            now=self._now,
        )
        if intermediate_retention_days is not None and intermediate_retention_days < 1:
            raise ValueError("intermediate_retention_days must be positive")
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be positive")
        self.max_concurrent_jobs = max_concurrent_jobs
        self.intermediate_retention_seconds = (
            intermediate_retention_days * 24 * 60 * 60
            if intermediate_retention_days is not None
            else None
        )

    async def reconcile(self) -> None:
        """Poll every active GROMACS call once."""
        jobs = iter(self.store.list_reconcilable_jobs("gromacs"))

        async def worker() -> None:
            for job in jobs:
                try:
                    await self._reconcile_job(job)
                except Exception:
                    LOGGER.exception(
                        "Could not reconcile GROMACS job %s",
                        job.job_id,
                    )

        await asyncio.gather(*(worker() for _ in range(self.max_concurrent_jobs)))
        await self._cleanup_intermediates()

    async def _reconcile_job(self, job: JobRecord) -> None:
        """Reconcile one durable Job without affecting independent Jobs."""
        now = self._now()
        if job.state in {JobState.FINALIZING, JobState.BLOCKED}:
            if job.next_retry_at is None or job.next_retry_at <= now:
                await self._finalize(job)
            return
        calls = self.store.list_operations(job.job_id)
        if not calls:
            if job.state == JobState.CANCEL_REQUESTED:
                self.store.set_job_state(
                    job.job_id,
                    JobState.CANCELLED,
                    now=now,
                )
            return
        submitting = [
            call for call in calls if call.state == JobOperationState.SUBMITTING
        ]
        if submitting:
            if any(
                call.submission_lease_until is None
                or call.submission_lease_until <= now
                for call in submitting
            ):
                self.store.mark_state_unknown(
                    job.job_id,
                    reason=JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN,
                    now=now,
                )
            return

        if job.state == JobState.CANCEL_REQUESTED:
            await self._reconcile_cancellation(job, calls)
            return

        await self._poll_running_calls(job, calls)
        current = self.store.get_job(job.owner_user_id, job.job_id)
        if current is None:  # pragma: no cover - reconciler loaded this row
            return
        calls = self.store.list_operations(job.job_id)
        if current.state == JobState.CANCEL_REQUESTED:
            await self._reconcile_cancellation(current, calls)
            return
        if any(call.state == JobOperationState.FAILED for call in calls):
            await self._settle_failed_job(current, calls)
            return
        if any(call.state == JobOperationState.CANCELLED for call in calls):
            await self._settle_cancelled_job(current, calls)
            return
        if _all_operations_completed(current, calls):
            await self._begin_finalization(current)
            return
        await self._submit_ready_operations(current)

    async def _poll_running_calls(
        self,
        job: JobRecord,
        calls: Iterable[JobOperationRecord],
    ) -> None:
        """Poll every directly running branch once and persist observations."""
        saw_running = False
        for call in calls:
            if call.state != JobOperationState.RUNNING or call.modal_call_id is None:
                continue
            try:
                outcome = await self.adapter.poll(
                    call.modal_call_id,
                    operation=call.operation,
                )
            except MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while polling stage %s for job %s",
                    call.operation,
                    job.job_id,
                )
                continue
            if outcome.kind == "running":
                saw_running = True
                continue
            if outcome.kind == "expired":
                terminal = (
                    JobOperationState.COMPLETED
                    if call.operation == FINAL_OPERATION
                    else JobOperationState.FAILED
                )
            else:
                terminal = JobOperationState(outcome.kind)
            self.store.record_operation_outcome(
                job.job_id,
                operation=call.operation,
                expected_modal_call_id=call.modal_call_id,
                outcome=terminal,
                now=self._now(),
            )
            if outcome.kind == "expired" and call.operation == FINAL_OPERATION:
                try:
                    archive = await self.adapter.recover_archive(job)
                except Exception:
                    LOGGER.info(
                        "Published Result is not yet recoverable for job %s; "
                        "rebuilding from raw outputs",
                        job.job_id,
                    )
                else:
                    self._complete(job, archive)
                    return
        if saw_running and job.state == JobState.QUEUED:
            self.store.set_job_state(job.job_id, JobState.RUNNING, now=self._now())

    async def _submit_ready_operations(self, job: JobRecord) -> None:
        """Attach every newly ready direct stage while excluding cancellation."""
        current = self.store.get_job(job.owner_user_id, job.job_id)
        if current is None:
            return
        if current.run_name is None:  # pragma: no cover - initial claim sets it
            raise RuntimeError(f"GROMACS Job has no run name: {job.job_id}")
        ready = _ready_operations(
            current,
            self.store.list_operations(job.job_id),
        )

        def is_retryable(error: Exception) -> bool:
            return isinstance(error, MODAL_SERVICE_ERRORS) and not isinstance(
                error,
                DEFINITE_SUBMISSION_ERRORS,
            )

        for operation in ready:
            submission_token = uuid4().hex

            def can_submit(
                candidate: JobRecord,
                selected_operation: str = operation,
            ) -> bool:
                return candidate.state in {
                    JobState.QUEUED,
                    JobState.RUNNING,
                } and selected_operation in _ready_operations(
                    candidate,
                    self.store.list_operations(job.job_id),
                )

            async def spawn(
                claimed: JobRecord,
                selected_operation: str = operation,
            ) -> SubmittedModalCall:
                return await self.adapter.submit_operation(
                    claimed,
                    selected_operation,
                )

            try:
                result = await self.submitter.submit(
                    current,
                    operation=operation,
                    run_name=current.run_name,
                    submission_token=submission_token,
                    spawn=spawn,
                    cancel=self.adapter.cancel,
                    can_submit=can_submit,
                    is_retryable_spawn_error=is_retryable,
                )
            except DEFINITE_SUBMISSION_ERRORS:
                LOGGER.exception(
                    "Modal rejected GROMACS stage %s for job %s",
                    operation,
                    job.job_id,
                )
                return
            except MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while submitting stage %s for job %s",
                    operation,
                    job.job_id,
                )
                return
            except Exception:
                LOGGER.exception(
                    "Could not submit GROMACS stage %s for job %s",
                    operation,
                    job.job_id,
                )
                return

            if result.job.state == JobState.STATE_UNKNOWN:
                return
            if result.job.state not in {JobState.QUEUED, JobState.RUNNING}:
                return
            if result.attached:
                LOGGER.info(
                    "event=stage_attached job_id=%s workload=gromacs "
                    "operation=%s function=%s",
                    result.job.job_id,
                    operation,
                    operation.partition(":")[0],
                )

    async def _stop_active_calls(
        self,
        job: JobRecord,
        calls: Iterable[JobOperationRecord],
    ) -> bool:
        """Request fail-fast cancellation and report whether every call stopped."""
        stopped = True
        status_unknown = False
        for call in calls:
            if call.state != JobOperationState.RUNNING or call.modal_call_id is None:
                continue
            try:
                await self.adapter.cancel(call.modal_call_id)
                outcome = await self.adapter.poll(
                    call.modal_call_id,
                    operation=call.operation,
                )
            except modal.exception.NotFoundError:
                outcome = PollOutcome("expired")
            except MODAL_SERVICE_ERRORS:
                stopped = False
                LOGGER.exception(
                    "Modal is unavailable while stopping stage %s for job %s",
                    call.operation,
                    job.job_id,
                )
                continue
            if outcome.kind == "running":
                stopped = False
                continue
            if outcome.kind == "expired":
                status_unknown = True
                continue
            self.store.record_operation_outcome(
                job.job_id,
                operation=call.operation,
                expected_modal_call_id=call.modal_call_id,
                outcome=JobOperationState(outcome.kind),
                now=self._now(),
            )
        if status_unknown:
            self.store.mark_state_unknown(
                job.job_id,
                reason=JobStateUnknownReason.CANCELLATION_OUTCOME_UNKNOWN,
                now=self._now(),
            )
            return False
        refreshed = self.store.list_operations(job.job_id)
        return stopped and not any(
            call.state in {JobOperationState.SUBMITTING, JobOperationState.RUNNING}
            for call in refreshed
        )

    async def _settle_failed_job(
        self,
        job: JobRecord,
        calls: Iterable[JobOperationRecord],
    ) -> None:
        if not await self._stop_active_calls(job, calls):
            return
        async with self.lifecycle_locks.for_job(job.job_id):
            current = self.store.get_job(job.owner_user_id, job.job_id)
            if current is None:
                return
            if current.state == JobState.CANCEL_REQUESTED:
                self.store.set_job_state(
                    job.job_id,
                    JobState.CANCELLED,
                    now=self._now(),
                )
                LOGGER.info(
                    "event=job_cancelled job_id=%s workload=gromacs",
                    job.job_id,
                )
                return
            self.store.fail_job(
                job.job_id,
                error_code="compute_failed",
                error_message="GROMACS could not complete the simulation.",
                now=self._now(),
            )
            LOGGER.info("event=job_failed job_id=%s workload=gromacs", job.job_id)

    async def _settle_cancelled_job(
        self,
        job: JobRecord,
        calls: Iterable[JobOperationRecord],
    ) -> None:
        if not await self._stop_active_calls(job, calls):
            return
        async with self.lifecycle_locks.for_job(job.job_id):
            self.store.set_job_state(job.job_id, JobState.CANCELLED, now=self._now())
        LOGGER.info("event=job_cancelled job_id=%s workload=gromacs", job.job_id)

    async def _reconcile_cancellation(
        self,
        job: JobRecord,
        calls: Iterable[JobOperationRecord],
    ) -> None:
        """Cancel every active branch without claiming an unknown outcome."""
        expired: list[JobOperationRecord] = []
        for call in calls:
            if call.state != JobOperationState.RUNNING or call.modal_call_id is None:
                continue
            try:
                await self.adapter.cancel(call.modal_call_id)
                outcome = await self.adapter.poll(
                    call.modal_call_id,
                    operation=call.operation,
                )
            except modal.exception.NotFoundError:
                outcome = PollOutcome("expired")
            except MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while cancelling stage %s for job %s",
                    call.operation,
                    job.job_id,
                )
                continue
            if outcome.kind == "running":
                continue
            if outcome.kind == "expired":
                expired.append(call)
                continue
            self.store.record_operation_outcome(
                job.job_id,
                operation=call.operation,
                expected_modal_call_id=call.modal_call_id,
                outcome=JobOperationState(outcome.kind),
                now=self._now(),
            )

        refreshed = self.store.list_operations(job.job_id)
        if expired:
            if (
                len(expired) == 1
                and expired[0].operation == FINAL_OPERATION
                and not any(
                    call.state
                    in {
                        JobOperationState.SUBMITTING,
                        JobOperationState.RUNNING,
                    }
                    and call.operation != FINAL_OPERATION
                    for call in refreshed
                )
            ):
                try:
                    archive = await self.adapter.recover_archive(job)
                except MODAL_SERVICE_ERRORS:
                    LOGGER.exception(
                        "Modal is unavailable while recovering cancelling job %s",
                        job.job_id,
                    )
                    return
                except Exception:
                    LOGGER.exception(
                        "Published Result is not recoverable for cancelling job %s",
                        job.job_id,
                    )
                else:
                    self.store.record_operation_outcome(
                        job.job_id,
                        operation=expired[0].operation,
                        expected_modal_call_id=expired[0].modal_call_id or "",
                        outcome=JobOperationState.COMPLETED,
                        now=self._now(),
                    )
                    self._complete(job, archive)
                    return
            self.store.mark_state_unknown(
                job.job_id,
                reason=JobStateUnknownReason.CANCELLATION_OUTCOME_UNKNOWN,
                now=self._now(),
            )
            return
        if any(
            call.state in {JobOperationState.SUBMITTING, JobOperationState.RUNNING}
            for call in refreshed
        ):
            return
        self.store.set_job_state(job.job_id, JobState.CANCELLED, now=self._now())
        LOGGER.info("event=job_cancelled job_id=%s workload=gromacs", job.job_id)

    async def _begin_finalization(self, job: JobRecord) -> None:
        async with self.lifecycle_locks.for_job(job.job_id):
            current = self.store.get_job(job.owner_user_id, job.job_id)
            if current is None or current.state not in {
                JobState.QUEUED,
                JobState.RUNNING,
            }:
                return
            calls = self.store.list_operations(job.job_id)
            if not _all_operations_completed(current, calls):
                return
            finalizing = self.store.set_job_state(
                job.job_id,
                JobState.FINALIZING,
                now=self._now(),
            )
        LOGGER.info(
            "event=finalization_started job_id=%s workload=gromacs "
            "stage=prepare_result",
            job.job_id,
        )
        await self._finalize(finalizing)

    async def _cleanup_intermediates(self) -> None:
        if self.intermediate_retention_seconds is None:
            return
        now = self._now()
        jobs = self.store.list_intermediate_cleanup_candidates(
            "gromacs",
            completed_before=now - self.intermediate_retention_seconds,
        )
        for job in jobs:
            try:
                await self.adapter.cleanup_intermediates(job)
            except MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while cleaning job %s", job.job_id
                )
                continue
            except Exception:
                LOGGER.exception("Could not clean intermediates for job %s", job.job_id)
                continue
            self.store.mark_intermediates_cleaned(job.job_id, now=now)

    async def _finalize(self, job: JobRecord) -> None:
        """Publish existing outputs with durable retry and blocking policy."""
        now = self._now()
        if job.state == JobState.BLOCKED:
            if job.result_previous_state in {JobState.SUCCEEDED, JobState.PARTIAL}:
                try:
                    recovered = await self.adapter.recover_archive(job)
                except Exception:
                    LOGGER.info(
                        "Exact published Result is not yet recoverable for job %s; "
                        "trying deterministic reconstruction",
                        job.job_id,
                    )
                else:
                    self._complete(job, recovered)
                    return
            job = self.store.schedule_finalization_retry(
                job.job_id,
                now=now,
                next_retry_at=now,
            )
        completed_at = job.finalization_started_at or now
        try:
            archive = await self.adapter.publish_archive(
                job,
                completed_at=completed_at,
            )
        except ResultIdentityMismatchError:
            LOGGER.warning(
                "event=result_recovery_blocked job_id=%s workload=gromacs "
                "blocking_category=result_integrity",
                job.job_id,
            )
            self.store.block_job(
                job.job_id,
                category="result_integrity",
                previous_state=job.result_previous_state,
                now=now,
                next_retry_at=now + 15 * 60,
            )
            return
        except PERMANENT_FINALIZATION_ERRORS:
            LOGGER.exception("GROMACS finalization is blocked for job %s", job.job_id)
            self.store.block_job(
                job.job_id,
                category="modal_configuration",
                now=now,
                next_retry_at=now + 15 * 60,
            )
            LOGGER.info(
                "event=finalization_blocked job_id=%s workload=gromacs "
                "blocking_category=modal_configuration",
                job.job_id,
            )
            return
        except (ArchiveNotReadyError, *TRANSIENT_FINALIZATION_ERRORS):
            LOGGER.exception("Modal is unavailable while publishing job %s", job.job_id)
            self._retry_finalization(job, now=now, category="modal_unavailable")
            return
        except ArtifactSourceMissingError:
            LOGGER.exception("GROMACS job %s is missing required output", job.job_id)
            self._mark_invalid_result(job, now=now)
            return
        except GromacsResultInvalidError:
            LOGGER.exception("GROMACS job %s returned invalid output", job.job_id)
            self._mark_invalid_result(job, now=now)
            return
        except (OSError, ArtifactIntegrityError):
            LOGGER.exception(
                "Local Result staging is unavailable for job %s",
                job.job_id,
            )
            self._retry_finalization(job, now=now, category="local_storage")
            return
        except Exception:
            LOGGER.exception("GROMACS finalization failed for job %s", job.job_id)
            self._retry_finalization(job, now=now, category="internal_service")
            return
        self._complete(job, archive)

    def _retry_finalization(
        self,
        job: JobRecord,
        *,
        now: int,
        category: str,
    ) -> None:
        """Persist bounded retries for recoverable publication dependencies."""
        retry_started_at = job.finalization_retry_started_at or now
        if now - retry_started_at >= 30 * 60:
            self.store.block_job(
                job.job_id,
                category=category,
                now=now,
                next_retry_at=now + 15 * 60,
            )
            LOGGER.info(
                "event=finalization_blocked job_id=%s workload=gromacs "
                "blocking_category=%s",
                job.job_id,
                category,
            )
            return
        delay = min(15 * 60, 5 * 2**job.finalization_retry_count)
        self.store.schedule_finalization_retry(
            job.job_id,
            now=now,
            next_retry_at=now + delay,
        )
        LOGGER.info(
            "event=finalization_retry_scheduled job_id=%s workload=gromacs "
            "blocking_category=%s delay_seconds=%s",
            job.job_id,
            category,
            delay,
        )

    def _mark_invalid_result(self, job: JobRecord, *, now: int) -> None:
        """Preserve an established Result identity or fail a first publication."""
        if job.result_previous_state in {JobState.SUCCEEDED, JobState.PARTIAL}:
            self.store.block_job(
                job.job_id,
                category="result_integrity",
                previous_state=job.result_previous_state,
                now=now,
                next_retry_at=now + 15 * 60,
            )
            return
        self.store.fail_job(
            job.job_id,
            error_code="result_invalid",
            error_message="GROMACS completed, but its result archive was invalid.",
            now=now,
        )

    def _complete(self, job: JobRecord, archive: FinalArchive) -> None:
        try:
            self.store.complete_job(
                job.job_id,
                state=archive.state,
                result_volume_name=archive.volume_name,
                result_volume_path=archive.path,
                result_filename=archive.filename,
                result_size_bytes=archive.size_bytes,
                result_sha256=archive.sha256,
                result_archive_schema_version=GROMACS_ARCHIVE_SCHEMA_VERSION,
                warnings_json=archive.warnings_json,
                result_cached=archive.cache_lease is not None,
                now=self._now(),
            )
            LOGGER.info(
                "event=result_published job_id=%s workload=gromacs state=%s "
                "size_bytes=%s",
                job.job_id,
                archive.state.value,
                archive.size_bytes,
            )
        finally:
            if archive.cache_lease is not None:
                archive.cache_lease.close()
