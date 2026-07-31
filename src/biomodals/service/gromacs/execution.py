"""Caller-owned GROMACS coordination over the shared execution kernel."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from uuid import UUID

from biomodals.execution import (
    AsyncExecutionRuntime,
    AvailabilityStatus,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    RunStatusReason,
    TaskStatus,
    result_probe_frontier,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalDefiniteSubmissionError,
)
from biomodals.execution.scheduler import (
    TaskDispatchDescriptor,
    form_fixed_batches,
    ready_node_keys,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.service.gromacs.archive import GROMACS_ARCHIVE_SCHEMA_VERSION
from biomodals.service.gromacs.contracts import GromacsJobOptions
from biomodals.service.gromacs.plan import (
    PREPARE_RESULT,
    modal_invocation,
    operation_provider_binding,
    operation_task_plan,
)
from biomodals.service.gromacs.results import (
    ArchiveNotReadyError,
    FinalArchive,
    GromacsResultInvalidError,
    ResultIdentityMismatchError,
)
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.store import JobRecord, ServiceStore

_RESULT_ENVELOPE_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)


class GromacsExecutionAdapter(Protocol):
    """Modal calls and result publication required by GROMACS coordination."""

    async def resolve(self, binding: Any) -> Any:
        """Resolve one exact deployed function."""

    async def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Spawn one detached function call."""

    async def observe(self, provider_call_handle_id: str) -> ModalCallObservation:
        """Observe one attached function call."""

    async def cancel(self, provider_call_handle_id: str) -> None:
        """Cancel one attached function call."""

    async def publish_archive(
        self,
        job: JobRecord,
        *,
        completed_at: int,
    ) -> FinalArchive:
        """Publish and validate the user-facing result archive."""

    async def recover_archive(self, job: JobRecord) -> FinalArchive:
        """Recover metadata for an already published immutable archive."""

    async def cleanup_intermediates(self, job: JobRecord) -> None:
        """Remove remote files that can be reconstructed from publications."""


class GromacsExecutionCoordinator:
    """Advance one service-owned GROMACS Execution Run by one provider wave."""

    def __init__(
        self,
        store: ServiceStore,
        adapter: GromacsExecutionAdapter,
        *,
        lifecycle_locks: JobLifecycleLocks | None = None,
        now: Callable[[], int] | None = None,
        intermediate_retention_days: int | None = None,
        max_concurrent_jobs: int = 4,
    ) -> None:
        """Bind service metadata, Modal calls, and result publication."""
        self.store = store
        self.adapter = adapter
        self.lifecycle_locks = lifecycle_locks or JobLifecycleLocks()
        self._now = now or (lambda: int(time.time()))
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
        """Advance every active GROMACS Execution Run once."""
        jobs = iter(self.store.list_reconcilable_jobs("gromacs"))

        async def worker() -> None:
            for job in jobs:
                try:
                    await self.advance(job.job_id)
                except Exception:
                    LOGGER.exception(
                        "Could not reconcile GROMACS job %s",
                        job.job_id,
                    )

        await asyncio.gather(*(worker() for _ in range(self.max_concurrent_jobs)))
        await self._cleanup_intermediates()

    async def cancel_job(self, job_id: UUID) -> None:
        """Durably cancel one Run and every conclusively attached call."""
        async with self.lifecycle_locks.for_job(job_id):
            job = self.store.get_job_by_id(job_id)
            if job is None:
                raise LookupError(f"Job not found: {job_id}")
            if job.execution_run_id is None:
                raise ValueError("Job is not linked to an Execution Run")
            with self.store.async_execution_runtime(self.adapter) as runtime:
                run = await runtime.cancel_run(
                    job.execution_run_id,
                    now=self._now(),
                )
                run = runtime.repository.finalize_run_from_results(
                    job.execution_run_id,
                    now=self._now(),
                )
                runtime.checkpoint()
            self._project_terminal_or_running_job(job, run.status)

    async def advance(self, job_id: UUID) -> None:
        """Advance one Job and suspend unexpected coordinator failures."""
        try:
            await self._advance(job_id)
        except ModalDefiniteSubmissionError:
            raise
        except Exception as exc:
            self._suspend_after_coordinator_error(job_id, exc)
            raise

    async def _advance(self, job_id: UUID) -> None:
        """Reconcile existing calls, admit one ready wave, and publish results."""
        archive: FinalArchive | None = None
        clear_staged_input = False
        async with self.lifecycle_locks.for_job(job_id):
            job = self.store.get_job_by_id(job_id)
            if job is None:
                raise LookupError(f"Job not found: {job_id}")
            if job.execution_run_id is None:
                raise ValueError("Job is not linked to an Execution Run")
            execution_run_id = job.execution_run_id

            with self.store.async_execution_runtime(self.adapter) as runtime:
                archive = await self._recover_node_publications(runtime, job)
                required = self._required_nodes(runtime, execution_run_id)
                if required is None:
                    runtime.checkpoint()
                    snapshot = runtime.repository.snapshot(execution_run_id)
                    self._project_terminal_or_running_job(job, snapshot.run.status)
                    return

                calls_to_cancel = runtime.repository.prune_unrequired_nodes(
                    execution_run_id,
                    required_node_keys=required,
                    now=self._now(),
                )
                runtime.checkpoint()
                for provider_call_id in calls_to_cancel:
                    await runtime.request_provider_call_cancellation(
                        provider_call_id,
                        now=self._now(),
                    )

                await runtime.reconcile_provider_calls(
                    execution_run_id,
                    required_node_keys=required,
                    encode_result=_result_envelope,
                    now=self._now(),
                )

                self._decode_completed_calls(
                    runtime,
                    execution_run_id,
                )
                self._reconcile_running_nodes(runtime, execution_run_id)
                runtime.repository.skip_unreachable_nodes(
                    execution_run_id,
                    now=self._now(),
                )

                run = runtime.repository.get_run(execution_run_id)
                if run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
                    self._start_ready_nodes(
                        runtime,
                        execution_run_id,
                        required=required,
                    )
                    runtime.checkpoint()
                    archive = archive or await self._run_result_publication(
                        runtime, job
                    )
                    try:
                        await self._submit_ready_remote_tasks(
                            runtime,
                            job,
                            required=required,
                        )
                    except ModalDefiniteSubmissionError:
                        self._reconcile_running_nodes(runtime, execution_run_id)
                        runtime.repository.skip_unreachable_nodes(
                            execution_run_id,
                            now=self._now(),
                        )
                        runtime.repository.finalize_run_from_results(
                            execution_run_id,
                            now=self._now(),
                        )
                        runtime.checkpoint()
                        raise
                    self._reconcile_running_nodes(runtime, execution_run_id)
                    runtime.repository.skip_unreachable_nodes(
                        execution_run_id,
                        now=self._now(),
                    )

                runtime.repository.finalize_run_from_results(
                    execution_run_id,
                    now=self._now(),
                )
                runtime.checkpoint()
                snapshot = runtime.repository.snapshot(execution_run_id)
                prepare = next(
                    (
                        node
                        for node in snapshot.nodes
                        if node.node_key.startswith("prepare_tpr_")
                    ),
                    None,
                )
                clear_staged_input = bool(
                    prepare is not None and prepare.status == NodeStatus.SUCCEEDED
                )

            if clear_staged_input:
                self.store.clear_job_input(job_id)
            if (
                archive is None
                and job.result_filename is None
                and snapshot.run.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}
            ):
                archive = await self.adapter.recover_archive(job)
            if archive is not None:
                self._complete_job(job, archive)
            else:
                self._project_terminal_or_running_job(job, snapshot.run.status)

    def _suspend_after_coordinator_error(
        self,
        job_id: UUID,
        error: Exception,
    ) -> None:
        """Best-effort persistence without masking the coordinator exception."""
        try:
            job = self.store.get_job_by_id(job_id)
            if job is None or job.execution_run_id is None:
                return
            with self.store.execution_repository() as repository:
                run = repository.get_run(job.execution_run_id)
                if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
                    return
                repository.transition_run(
                    job.execution_run_id,
                    RunStatus.SUSPENDED,
                    reason=RunStatusReason.COORDINATOR_ERROR,
                    message=str(error) or type(error).__name__,
                    now=self._now(),
                )
        except Exception:
            LOGGER.exception(
                "Could not persist coordinator suspension for GROMACS job %s",
                job_id,
            )

    async def _recover_node_publications(
        self,
        runtime: AsyncExecutionRuntime,
        job: JobRecord,
    ) -> FinalArchive | None:
        """Probe GROMACS results backward before discovering any Tasks."""
        execution_run_id = job.execution_run_id
        if execution_run_id is None:
            return None
        run = runtime.repository.get_run(execution_run_id)
        observations: dict[str, AvailabilityStatus | None] = {}
        for node in runtime.repository.list_nodes(execution_run_id):
            if node.status == NodeStatus.SUCCEEDED:
                observations[node.node_key] = AvailabilityStatus.AVAILABLE
            elif node.status.is_terminal:
                observations[node.node_key] = AvailabilityStatus.MISSING
            elif (
                node.result_observation == AvailabilityStatus.UNKNOWN
                and run.status in {RunStatus.PENDING, RunStatus.RUNNING}
            ):
                observations[node.node_key] = None
            elif self._local_result_needs_recovery(
                runtime,
                execution_run_id,
                node.node_key,
            ):
                observations[node.node_key] = None
            else:
                observations[node.node_key] = node.result_observation

        archive: FinalArchive | None = None
        while frontier := result_probe_frontier(run.plan, observations):
            observed: list[tuple[str, AvailabilityStatus]] = []
            for node_key in frontier:
                if node_key != PREPARE_RESULT:
                    observation = AvailabilityStatus.MISSING
                else:
                    recovering_local_result = self._local_result_needs_recovery(
                        runtime,
                        execution_run_id,
                        node_key,
                    )
                    try:
                        archive = await self.adapter.recover_archive(job)
                    except ArchiveNotReadyError:
                        observation = AvailabilityStatus.MISSING
                        if recovering_local_result:
                            runtime.repository.record_task_result_observation(
                                execution_run_id,
                                PREPARE_RESULT,
                                "operation",
                                observation,
                                now=self._now(),
                            )
                    except (
                        GromacsResultInvalidError,
                        ResultIdentityMismatchError,
                        ValueError,
                    ) as exc:
                        observation = AvailabilityStatus.MISSING
                        if recovering_local_result:
                            self._fail_result_task(
                                runtime,
                                execution_run_id,
                                str(exc),
                            )
                            observations[node_key] = observation
                            continue
                    except Exception:
                        observation = AvailabilityStatus.UNKNOWN
                    else:
                        observation = AvailabilityStatus.AVAILABLE
                        if recovering_local_result:
                            self._complete_result_task(runtime, execution_run_id)
                            observations[node_key] = observation
                            continue
                observed.append((node_key, observation))

            for node_key, observation in observed:
                runtime.repository.record_node_result_observation(
                    execution_run_id,
                    node_key,
                    observation,
                    now=self._now(),
                )
                observations[node_key] = observation
            runtime.checkpoint()
            if any(
                observation == AvailabilityStatus.UNKNOWN for _, observation in observed
            ):
                return None
        return archive

    def _local_result_needs_recovery(
        self,
        runtime: AsyncExecutionRuntime,
        execution_run_id: UUID,
        node_key: str,
    ) -> bool:
        """Return whether interrupted local publication must be revalidated."""
        if node_key != PREPARE_RESULT:
            return False
        node = runtime.repository.get_node(execution_run_id, node_key)
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            return False
        task = runtime.repository.get_task(
            execution_run_id,
            node_key,
            "operation",
        )
        return (
            task.status == TaskStatus.RUNNING
            and task.local_owned
            and task.provider_call_id is None
            and task.worker_provider_call_id is None
        )

    def _required_nodes(
        self,
        runtime: AsyncExecutionRuntime,
        execution_run_id: UUID,
    ) -> set[str] | None:
        """Return the result-driven GROMACS repair closure."""
        required = runtime.required_node_keys(execution_run_id)
        return None if required is None else set(required)

    def _decode_completed_calls(
        self,
        runtime: AsyncExecutionRuntime,
        execution_run_id: UUID,
    ) -> None:
        """Turn durable call envelopes into scientific Task outcomes."""
        for call in runtime.repository.list_provider_calls(execution_run_id):
            if call.status != ProviderCallStatus.SUCCEEDED:
                continue
            envelope = call.result_envelope
            valid = (
                isinstance(envelope, dict)
                and envelope.get("schema_version") == _RESULT_ENVELOPE_SCHEMA_VERSION
                and isinstance(envelope.get("remote_workdir"), str)
                and bool(envelope["remote_workdir"])
            )
            for task_key in call.task_keys:
                task = runtime.repository.get_task(
                    execution_run_id,
                    call.node_key,
                    task_key,
                )
                if task.status != TaskStatus.RUNNING:
                    continue
                if valid:
                    runtime.repository.record_task_result_observation(
                        execution_run_id,
                        call.node_key,
                        task_key,
                        AvailabilityStatus.AVAILABLE,
                        now=self._now(),
                    )
                else:
                    runtime.repository.fail_task(
                        execution_run_id,
                        call.node_key,
                        task_key,
                        message="GROMACS returned an invalid operation result",
                        now=self._now(),
                    )

    def _reconcile_running_nodes(
        self,
        runtime: AsyncExecutionRuntime,
        execution_run_id: UUID,
    ) -> None:
        for node in runtime.repository.list_nodes(execution_run_id):
            if node.status == NodeStatus.RUNNING and node.discovery_complete:
                runtime.repository.reconcile_node_tasks(
                    execution_run_id,
                    node.node_key,
                    now=self._now(),
                )

    def _start_ready_nodes(
        self,
        runtime: AsyncExecutionRuntime,
        execution_run_id: UUID,
        *,
        required: set[str],
    ) -> None:
        run = runtime.repository.get_run(execution_run_id)
        nodes = runtime.repository.list_nodes(execution_run_id)
        statuses = {node.node_key: node.status for node in nodes}
        for node_key in ready_node_keys(run.plan, statuses):
            if node_key not in required:
                continue
            runtime.repository.start_node(
                execution_run_id,
                node_key,
                now=self._now(),
            )
            runtime.repository.discover_tasks(
                execution_run_id,
                node_key,
                (operation_task_plan(node_key),),
                now=self._now(),
            )
            runtime.repository.record_task_result_observation(
                execution_run_id,
                node_key,
                "operation",
                AvailabilityStatus.MISSING,
                now=self._now(),
            )

    async def _run_result_publication(
        self,
        runtime: AsyncExecutionRuntime,
        job: JobRecord,
    ) -> FinalArchive | None:
        execution_run_id = job.execution_run_id
        if execution_run_id is None:
            return None
        node = runtime.repository.get_node(execution_run_id, PREPARE_RESULT)
        if node.status != NodeStatus.RUNNING:
            return None
        task = runtime.repository.get_task(
            execution_run_id,
            PREPARE_RESULT,
            "operation",
        )
        if task.status == TaskStatus.SUCCEEDED:
            return None
        if not runtime.repository.acquire_local_task(
            execution_run_id,
            PREPARE_RESULT,
            "operation",
            now=self._now(),
        ):
            return None
        runtime.checkpoint()
        try:
            archive = await self.adapter.publish_archive(
                job,
                completed_at=self._now(),
            )
        except (
            GromacsResultInvalidError,
            ResultIdentityMismatchError,
            ValueError,
        ) as exc:
            self._fail_result_task(runtime, execution_run_id, str(exc))
            return None
        except Exception as exc:
            runtime.repository.transition_run(
                execution_run_id,
                RunStatus.SUSPENDED,
                reason=RunStatusReason.COORDINATOR_ERROR,
                message=str(exc),
                now=self._now(),
            )
            runtime.checkpoint()
            raise
        self._complete_result_task(runtime, execution_run_id)
        return archive

    def _complete_result_task(
        self,
        runtime: AsyncExecutionRuntime,
        execution_run_id: UUID,
    ) -> None:
        """Record one successfully validated result publication."""
        runtime.repository.record_task_result_observation(
            execution_run_id,
            PREPARE_RESULT,
            "operation",
            AvailabilityStatus.AVAILABLE,
            now=self._now(),
        )
        runtime.repository.reconcile_node_tasks(
            execution_run_id,
            PREPARE_RESULT,
            now=self._now(),
        )
        runtime.checkpoint()

    def _fail_result_task(
        self,
        runtime: AsyncExecutionRuntime,
        execution_run_id: UUID,
        message: str,
    ) -> None:
        """Record one conclusive result-publication failure."""
        runtime.repository.fail_task(
            execution_run_id,
            PREPARE_RESULT,
            "operation",
            message=message,
            now=self._now(),
        )
        runtime.repository.reconcile_node_tasks(
            execution_run_id,
            PREPARE_RESULT,
            now=self._now(),
        )
        runtime.checkpoint()

    async def _submit_ready_remote_tasks(
        self,
        runtime: AsyncExecutionRuntime,
        job: JobRecord,
        *,
        required: set[str],
    ) -> None:
        execution_run_id = job.execution_run_id
        if execution_run_id is None:
            return
        run = runtime.repository.get_run(execution_run_id)
        nodes = runtime.repository.list_nodes(execution_run_id)
        unfinished = {node.node_key for node in nodes if not node.status.is_terminal}
        ranks = required_node_ranks(
            run.plan,
            required_node_keys=required,
            unfinished_node_keys=unfinished,
        )
        descriptors: list[TaskDispatchDescriptor] = []
        for node in nodes:
            if (
                node.node_key not in required
                or node.node_key == PREPARE_RESULT
                or node.status != NodeStatus.RUNNING
            ):
                continue
            for task in runtime.repository.list_tasks(
                execution_run_id,
                node.node_key,
            ):
                if (
                    task.status == TaskStatus.PENDING
                    and task.result_observation == AvailabilityStatus.MISSING
                ):
                    binding = operation_provider_binding(
                        node.node_key,
                        environment=job.modal_environment,
                        app_name=job.modal_app_name,
                        app_version=job.modal_app_version,
                    )
                    rank = ranks[node.node_key]
                    descriptors.append(
                        TaskDispatchDescriptor(
                            node_key=node.node_key,
                            node_ordinal=node.ordinal,
                            task_key=task.task_key,
                            task_ordinal=task.ordinal,
                            binding=binding,
                            compatibility_key=node.node_key,
                            max_tasks_per_call=1,
                            depth=rank.depth,
                            unblocking_span=rank.unblocking_span,
                        )
                    )
        descriptors = list(
            runtime.persist_fixed_dispatch_policy(
                execution_run_id,
                tuple(descriptors),
                now=self._now(),
            )
        )
        counts = runtime.repository.active_provider_call_counts(execution_run_id)
        selected = select_admissible_candidates(
            form_fixed_batches(tuple(descriptors)),
            available_total_slots=max(
                run.max_active_provider_calls - counts.total,
                0,
            ),
            available_gpu_slots=max(
                run.max_active_gpu_provider_calls - counts.gpu,
                0,
            ),
        )
        options = GromacsJobOptions.model_validate_json(job.parameters_json)
        for candidate in selected:
            submitted = await runtime.submit_fixed_batch(
                execution_run_id,
                candidate,
                submission_token=(f"{execution_run_id}:{candidate.node_key}:operation"),
                kwargs=self._operation_kwargs(
                    job,
                    candidate.node_key,
                    options,
                ),
                now=self._now(),
            )
            if submitted is None:
                return

    def _operation_kwargs(
        self,
        job: JobRecord,
        operation: str,
        options: GromacsJobOptions,
    ) -> dict[str, object]:
        if job.run_name is None:
            raise ValueError("GROMACS Job has no run name")
        if operation.startswith("prepare_tpr_"):
            pdb_content = self.store.load_job_input(job.job_id)
            if pdb_content is None:
                raise RuntimeError("Staged GROMACS input is unavailable")
            return {
                "pdb_content": pdb_content,
                "run_name": job.run_name,
                "simulation_time_ns": options.simulation_time_ns,
                "run_pdbfixer": options.run_pdbfixer,
            }
        return modal_invocation(
            operation,
            cpu_only=options.cpu_only,
            run_name=job.run_name,
            simulation_time_ns=options.simulation_time_ns,
        ).kwargs

    def _complete_job(self, job: JobRecord, archive: FinalArchive) -> None:
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
        finally:
            if archive.cache_lease is not None:
                archive.cache_lease.close()

    def _project_terminal_or_running_job(
        self,
        job: JobRecord,
        status: RunStatus,
    ) -> None:
        if status == RunStatus.FAILED:
            self.store.fail_job(
                job.job_id,
                error_code="compute_failed",
                error_message="GROMACS could not complete the simulation.",
                now=self._now(),
            )

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
            except Exception:
                LOGGER.exception("Could not clean intermediates for job %s", job.job_id)
                continue
            self.store.mark_intermediates_cleaned(job.job_id, now=now)


def _result_envelope(result: Any) -> dict[str, object]:
    """Normalize a small provider result without treating it as publication."""
    return {
        "schema_version": _RESULT_ENVELOPE_SCHEMA_VERSION,
        "remote_workdir": result if isinstance(result, str) else None,
    }
