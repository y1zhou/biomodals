"""Modal SDK adapter and lifecycle reconciliation for GROMACS."""

from __future__ import annotations

import hashlib
import io
import logging
import re
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from typing import BinaryIO, Literal, cast
from uuid import uuid4

import modal
import orjson

from biomodals.service.gromacs.archive import (
    validate_gromacs_archive,
    write_gromacs_archive,
)
from biomodals.service.gromacs.router import (
    GromacsJobOptions,
    SubmissionOutcomeUnknownError,
)
from biomodals.service.runtime_config import ModalConfigurationSnapshot
from biomodals.service.store import (
    JobRecord,
    JobState,
    JobSubmissionConflictError,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_API_RUN_NAME = re.compile(r"api-[0-9a-f]{32}")
_NVT_ANALYSIS = "collect_traj_stats:nvt_"
_NPT_ANALYSIS = "collect_traj_stats:npt_"
_PRODUCTION_ANALYSIS = "collect_traj_stats:production_"
_FINAL_OPERATION = _PRODUCTION_ANALYSIS
_MODAL_SERVICE_ERRORS = (
    modal.exception.AuthError,
    modal.exception.ConnectionError,
    modal.exception.InternalError,
    modal.exception.InvalidError,
    modal.exception.NotFoundError,
    modal.exception.PermissionDeniedError,
    modal.exception.ResourceExhaustedError,
    modal.exception.ServiceError,
)


@dataclass(frozen=True, slots=True)
class SubmittedModalCall:
    """Identifiers persisted after detached submission."""

    modal_call_id: str
    run_name: str
    provider_operation: str


@dataclass(frozen=True, slots=True)
class PollOutcome:
    """Sanitized state observed from one Modal call graph."""

    kind: Literal["running", "completed", "cancelled", "failed", "expired"]


@dataclass(frozen=True, slots=True)
class FinalArchive:
    """Validated immutable artifact returned by the compute contract."""

    state: JobState
    volume_name: str
    path: str
    filename: str
    size_bytes: int
    sha256: str
    warnings_json: str


@dataclass(frozen=True, slots=True)
class _ResultMarker:
    request_sha256: str
    archive_sha256: str
    size_bytes: int


class ArchiveNotReadyError(RuntimeError):
    """Raised when a stable run has not published its completion marker yet."""


def _call_nodes(
    roots: Iterable[modal.call_graph.InputInfo],
) -> list[modal.call_graph.InputInfo]:
    nodes: list[modal.call_graph.InputInfo] = []
    pending = list(roots)
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.children)
    return nodes


class ModalGromacsAdapter:
    """Invoke one separately deployed GROMACS App through the Modal SDK."""

    def __init__(
        self,
        *,
        app_name: str,
        environment_name: str,
        output_volume_name: str = "Gromacs-outputs",
        call_resolver: Callable[[str], modal.FunctionCall] = modal.FunctionCall.from_id,
        function_resolver: Callable[..., modal.Function] | None = None,
    ) -> None:
        """Bind one deployed App in an explicit Modal Environment."""
        self.environment_name = environment_name
        self.app_name = app_name
        self.output_volume_name = output_volume_name
        self._function_resolver = function_resolver or modal.Function.from_name
        self._call_resolver = call_resolver

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
        modal_configuration: ModalConfigurationSnapshot,
    ) -> SubmittedModalCall:
        """Spawn the first deployed compute stage without a remote coordinator."""
        function_name = "prepare_tpr_cpu" if options.cpu_only else "prepare_tpr_gpu"
        function = self._function_resolver(
            modal_configuration.app_name,
            function_name,
            environment_name=modal_configuration.environment,
        )
        try:
            call = await function.spawn.aio(
                pdb_content=pdb_content,
                run_name=run_name,
                simulation_time_ns=options.simulation_time_ns,
                run_pdbfixer=options.run_pdbfixer,
            )
            modal_call_id = call.object_id
        except Exception as exc:
            raise SubmissionOutcomeUnknownError(
                "Modal did not return a durable FunctionCall handle"
            ) from exc
        return SubmittedModalCall(
            modal_call_id=modal_call_id,
            run_name=run_name,
            provider_operation=function_name,
        )

    async def advance(self, job: JobRecord) -> SubmittedModalCall:
        """Spawn the deployed stage following one completed direct call."""
        if job.run_name is None or job.provider_operation is None:
            raise ValueError("GROMACS Job has no active provider operation")
        options = GromacsJobOptions.model_validate_json(job.parameters_json)
        if job.provider_operation in {"prepare_tpr_cpu", "prepare_tpr_gpu"}:
            operation = _NVT_ANALYSIS
        elif job.provider_operation == _NVT_ANALYSIS:
            operation = _NPT_ANALYSIS
        elif job.provider_operation == _NPT_ANALYSIS:
            operation = (
                "production_run_cpu" if options.cpu_only else "production_run_gpu"
            )
        elif job.provider_operation in {"production_run_cpu", "production_run_gpu"}:
            operation = _PRODUCTION_ANALYSIS
        else:
            raise ValueError(
                f"GROMACS operation cannot advance: {job.provider_operation}"
            )

        function_name, _, traj_prefix = operation.partition(":")
        if function_name.startswith("production_run_"):
            kwargs = {
                "run_name": job.run_name,
                "simulation_time_ns": options.simulation_time_ns,
            }
        elif operation == _PRODUCTION_ANALYSIS:
            kwargs = {
                "traj_prefix": traj_prefix,
                "run_name": job.run_name,
                "save_processed_traj": True,
            }
        else:
            kwargs = {
                "traj_prefix": traj_prefix,
                "run_name": job.run_name,
            }

        function = self._function_resolver(
            job.modal_app_name,
            function_name,
            environment_name=job.modal_environment,
        )
        try:
            call = await function.spawn.aio(**kwargs)
            modal_call_id = call.object_id
        except Exception as exc:
            raise SubmissionOutcomeUnknownError(
                "Modal did not return a durable FunctionCall handle"
            ) from exc
        return SubmittedModalCall(
            modal_call_id=modal_call_id,
            run_name=job.run_name,
            provider_operation=operation,
        )

    async def poll(
        self,
        modal_call_id: str,
        *,
        provider_operation: str | None = None,
    ) -> PollOutcome:
        """Poll without blocking and distinguish a poll timeout from failure."""
        call = self._call_resolver(modal_call_id)
        try:
            raw_result = await call.get.aio(timeout=0)
        except modal.exception.InputCancellation:
            graph = await call.get_call_graph.aio()
            nodes = _call_nodes(graph)
            if any(
                node.status == modal.call_graph.InputStatus.PENDING for node in nodes
            ):
                return PollOutcome("running")
            return PollOutcome("cancelled")
        except (modal.exception.OutputExpiredError, modal.exception.NotFoundError):
            return PollOutcome("expired")
        except TimeoutError:
            graph = await call.get_call_graph.aio()
            nodes = _call_nodes(graph)
            if not nodes or any(
                node.status == modal.call_graph.InputStatus.PENDING for node in nodes
            ):
                return PollOutcome("running")
            if any(
                node.status == modal.call_graph.InputStatus.TERMINATED for node in nodes
            ):
                return PollOutcome("cancelled")
            return PollOutcome("failed")
        except _MODAL_SERVICE_ERRORS:
            raise
        except Exception:
            return PollOutcome("failed")
        if not isinstance(raw_result, str):
            LOGGER.error(
                "GROMACS stage %s (%s) returned an invalid result",
                provider_operation,
                modal_call_id,
            )
            return PollOutcome("failed")
        return PollOutcome("completed")

    async def cancel(self, modal_call_id: str) -> None:
        """Cancel the root and every currently visible active descendant."""
        root = self._call_resolver(modal_call_id)
        graph = await root.get_call_graph.aio()
        nodes = _call_nodes(graph)
        if nodes and not any(
            node.status == modal.call_graph.InputStatus.PENDING for node in nodes
        ):
            return
        call_ids = {
            node.function_call_id
            for node in nodes
            if node.status == modal.call_graph.InputStatus.PENDING
            and node.function_call_id
        }
        call_ids.discard(modal_call_id)
        for call_id in sorted(call_ids):
            call = self._call_resolver(call_id)
            await call.cancel.aio(terminate_containers=False)
        await root.cancel.aio(terminate_containers=False)

    async def read_artifact(self, job: JobRecord) -> AsyncIterator[bytes]:
        """Stream the recorded final ZIP from its authoritative Modal Volume."""
        if (
            job.result_volume_name != self.output_volume_name
            or job.result_volume_path is None
            or job.run_name is None
            or _API_RUN_NAME.fullmatch(job.run_name) is None
            or job.result_volume_path != f"api-results/{job.run_name}/result.zip"
            or job.result_filename != f"{job.run_name}.zip"
            or type(job.result_size_bytes) is not int
            or job.result_size_bytes < 1
            or not isinstance(job.result_sha256, str)
            or _SHA256.fullmatch(job.result_sha256) is None
        ):
            raise ValueError("Job does not reference the configured GROMACS Volume")
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )
        async for chunk in volume.read_file.aio(job.result_volume_path):
            yield chunk

    async def cleanup_intermediates(self, job: JobRecord) -> None:
        """Remove one retained run directory without touching its final ZIP."""
        if job.run_name is None or _API_RUN_NAME.fullmatch(job.run_name) is None:
            raise ValueError("Job has no valid API run directory")
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )
        try:
            await volume.remove_file.aio(job.run_name, recursive=True)
        except modal.exception.NotFoundError:
            pass

    async def publish_archive(
        self,
        job: JobRecord,
        *,
        completed_at: int,
    ) -> FinalArchive:
        """Build and publish a ZIP from the established app's Volume files."""
        if job.run_name is None or _API_RUN_NAME.fullmatch(job.run_name) is None:
            raise ValueError("Job has no valid API run name")
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )

        async def read_file(path: str):
            async for chunk in volume.read_file.aio(path):
                yield chunk

        archive_path = f"api-results/{job.run_name}/result.zip"
        marker_path = f"api-results/{job.run_name}/result.json"
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as archive:
            archive_handle = cast("BinaryIO", archive)
            built = await write_gromacs_archive(
                archive_handle,
                run_name=job.run_name,
                parameters_json=job.parameters_json,
                modal_app_name=job.modal_app_name,
                started_at=job.created_at,
                completed_at=completed_at,
                read_file=read_file,
            )
            marker = orjson.dumps({
                "archive_schema_version": 1,
                "request_sha256": built.request_sha256,
                "archive_sha256": built.sha256,
                "size_bytes": built.size_bytes,
            })
            archive_handle.seek(0)
            async with volume.batch_upload.aio(force=True) as upload:
                upload.put_file(archive_handle, archive_path)
                upload.put_file(io.BytesIO(marker), marker_path)

        return FinalArchive(
            state=JobState.SUCCEEDED,
            volume_name=self.output_volume_name,
            path=archive_path,
            filename=f"{job.run_name}.zip",
            size_bytes=built.size_bytes,
            sha256=built.sha256,
            warnings_json="[]",
        )

    async def _result_marker(self, job: JobRecord) -> _ResultMarker:
        """Read and validate the completion marker for one stable run."""
        if job.run_name is None or _API_RUN_NAME.fullmatch(job.run_name) is None:
            raise ValueError("Job has no valid API run name")
        marker_path = f"api-results/{job.run_name}/result.json"
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )
        marker_bytes = bytearray()
        try:
            async for chunk in volume.read_file.aio(marker_path):
                marker_bytes.extend(chunk)
                if len(marker_bytes) > 64 * 1024:
                    raise ValueError("GROMACS result marker is too large")
        except FileNotFoundError as exc:
            raise ArchiveNotReadyError("GROMACS result marker is missing") from exc
        try:
            marker = orjson.loads(marker_bytes)
        except orjson.JSONDecodeError as exc:
            raise ValueError("GROMACS result marker is invalid") from exc
        size_bytes = marker.get("size_bytes") if isinstance(marker, dict) else None
        archive_sha256 = (
            marker.get("archive_sha256") if isinstance(marker, dict) else None
        )
        request_sha256 = (
            marker.get("request_sha256") if isinstance(marker, dict) else None
        )
        if (
            not isinstance(marker, dict)
            or marker.get("archive_schema_version") != 1
            or type(size_bytes) is not int
            or size_bytes < 1
            or not isinstance(archive_sha256, str)
            or _SHA256.fullmatch(archive_sha256) is None
            or not isinstance(request_sha256, str)
            or _SHA256.fullmatch(request_sha256) is None
        ):
            raise ValueError("GROMACS result marker is invalid")
        return _ResultMarker(
            request_sha256=request_sha256,
            archive_sha256=archive_sha256,
            size_bytes=size_bytes,
        )

    async def _verify_archive_bytes(
        self,
        job: JobRecord,
        marker: _ResultMarker,
    ) -> None:
        """Verify Volume bytes and the complete ZIP contract before success."""
        if job.run_name is None or _API_RUN_NAME.fullmatch(job.run_name) is None:
            raise ValueError("Job has no valid API run name")
        archive_path = f"api-results/{job.run_name}/result.zip"
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )
        digest = hashlib.sha256()
        size_bytes = 0
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as archive:
            try:
                async for chunk in volume.read_file.aio(archive_path):
                    size_bytes += len(chunk)
                    if size_bytes > marker.size_bytes:
                        raise ValueError(
                            "GROMACS result archive is larger than recorded"
                        )
                    digest.update(chunk)
                    archive.write(chunk)
            except FileNotFoundError as exc:
                raise ArchiveNotReadyError("GROMACS result archive is missing") from exc
            if (
                size_bytes != marker.size_bytes
                or digest.hexdigest() != marker.archive_sha256
            ):
                raise ValueError("GROMACS result archive does not match its marker")
            validated = validate_gromacs_archive(archive, run_name=job.run_name)
        if validated.request_sha256 != marker.request_sha256:
            raise ValueError("GROMACS result archive has the wrong request identity")

    async def recover_archive(self, job: JobRecord) -> FinalArchive:
        """Recover and verify durable output after Modal call output expiry."""
        marker = await self._result_marker(job)
        await self._verify_archive_bytes(job, marker)
        return FinalArchive(
            state=JobState.SUCCEEDED,
            volume_name=self.output_volume_name,
            path=f"api-results/{job.run_name}/result.zip",
            filename=f"{job.run_name}.zip",
            size_bytes=marker.size_bytes,
            sha256=marker.archive_sha256,
            warnings_json="[]",
        )


class GromacsReconciler:
    """Refresh locally persisted GROMACS jobs from Modal in one process."""

    def __init__(
        self,
        store: ServiceStore,
        adapter: ModalGromacsAdapter,
        *,
        now: Callable[[], int] | None = None,
        intermediate_retention_days: int | None = None,
    ) -> None:
        """Bind durable state to the provider adapter."""
        self.store = store
        self.adapter = adapter
        self._now = now or (lambda: int(time.time()))
        if intermediate_retention_days is not None and intermediate_retention_days < 1:
            raise ValueError("intermediate_retention_days must be positive")
        self.intermediate_retention_seconds = (
            intermediate_retention_days * 24 * 60 * 60
            if intermediate_retention_days is not None
            else None
        )

    async def reconcile(self) -> None:
        """Poll every active GROMACS call once."""
        for job in self.store.list_reconcilable_jobs("gromacs"):
            if job.modal_call_id is None:
                now = self._now()
                if job.submission_lease_until is not None:
                    if job.submission_lease_until <= now:
                        self.store.fail_job(
                            job.job_id,
                            error_code="compute_failed",
                            error_message=(
                                "GROMACS submission was interrupted before remote "
                                "compute could be tracked."
                            ),
                            now=now,
                        )
                    continue
                if job.state == JobState.CANCEL_REQUESTED:
                    self.store.set_job_state(
                        job.job_id,
                        JobState.CANCELLED,
                        now=now,
                    )
                continue
            if job.submission_lease_until is not None:
                now = self._now()
                if job.submission_lease_until <= now:
                    self.store.fail_job(
                        job.job_id,
                        error_code="compute_failed",
                        error_message=(
                            "GROMACS stage submission was interrupted before remote "
                            "compute could be tracked."
                        ),
                        now=now,
                    )
                continue
            try:
                if job.state == JobState.CANCEL_REQUESTED:
                    try:
                        await self.adapter.cancel(job.modal_call_id)
                    except modal.exception.NotFoundError:
                        outcome = PollOutcome("expired")
                    else:
                        outcome = await self.adapter.poll(
                            job.modal_call_id,
                            provider_operation=job.provider_operation,
                        )
                else:
                    outcome = await self.adapter.poll(
                        job.modal_call_id,
                        provider_operation=job.provider_operation,
                    )
            except _MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while reconciling job %s", job.job_id
                )
                continue
            if outcome.kind == "expired":
                if job.state == JobState.CANCEL_REQUESTED:
                    self.store.set_job_state(
                        job.job_id,
                        JobState.CANCELLED,
                        now=self._now(),
                    )
                    continue
                if job.provider_operation not in {None, _FINAL_OPERATION}:
                    self.store.fail_job(
                        job.job_id,
                        error_code="result_unavailable",
                        error_message=(
                            "GROMACS completed, but its stage result expired before "
                            "the Job could continue."
                        ),
                        now=self._now(),
                    )
                    continue
                try:
                    archive = await self.adapter.recover_archive(job)
                except ArchiveNotReadyError:
                    self.store.fail_job(
                        job.job_id,
                        error_code="result_unavailable",
                        error_message=(
                            "GROMACS completed, but its result could not be recovered."
                        ),
                        now=self._now(),
                    )
                    continue
                except _MODAL_SERVICE_ERRORS:
                    LOGGER.exception(
                        "Modal is unavailable while recovering job %s", job.job_id
                    )
                    continue
                except Exception:
                    LOGGER.exception("Could not recover expired job %s", job.job_id)
                    self.store.fail_job(
                        job.job_id,
                        error_code="result_unavailable",
                        error_message=(
                            "GROMACS completed, but its result could not be recovered."
                        ),
                        now=self._now(),
                    )
                    continue
                self._complete(job, archive)
                continue
            if outcome.kind == "completed" and job.provider_operation not in {
                None,
                _FINAL_OPERATION,
            }:
                if job.state == JobState.CANCEL_REQUESTED:
                    self.store.set_job_state(
                        job.job_id,
                        JobState.CANCELLED,
                        now=self._now(),
                    )
                else:
                    await self._advance(job)
                continue
            await self._apply(job, outcome)
        await self._cleanup_intermediates()

    async def _advance(self, job: JobRecord) -> None:
        """Attach exactly one next direct Modal stage to a durable Job."""
        submission_token = uuid4().hex
        now = self._now()
        claimed = self.store.claim_provider_advance(
            job.job_id,
            expected_modal_call_id=job.modal_call_id or "",
            submission_token=submission_token,
            now=now,
        )
        if claimed is None:
            return
        try:
            submitted = await self.adapter.advance(claimed)
        except SubmissionOutcomeUnknownError:
            LOGGER.exception(
                "Could not confirm the next GROMACS stage for job %s", job.job_id
            )
            return
        except _MODAL_SERVICE_ERRORS:
            self.store.release_provider_advance(
                job.job_id,
                expected_modal_call_id=job.modal_call_id or "",
                submission_token=submission_token,
                now=self._now(),
            )
            LOGGER.exception("Modal is unavailable while advancing job %s", job.job_id)
            return
        except Exception:
            self.store.release_provider_advance(
                job.job_id,
                expected_modal_call_id=job.modal_call_id or "",
                submission_token=submission_token,
                now=self._now(),
            )
            LOGGER.exception("Could not advance GROMACS job %s", job.job_id)
            self.store.fail_job(
                job.job_id,
                error_code="compute_failed",
                error_message="GROMACS could not continue the simulation.",
                now=self._now(),
            )
            return

        try:
            advanced = self.store.replace_provider_call(
                job.job_id,
                expected_modal_call_id=job.modal_call_id or "",
                modal_call_id=submitted.modal_call_id,
                provider_operation=submitted.provider_operation,
                submission_token=submission_token,
                now=self._now(),
            )
        except JobSubmissionConflictError:
            LOGGER.warning(
                "Discarding duplicate GROMACS stage %s for job %s",
                submitted.modal_call_id,
                job.job_id,
            )
            try:
                await self.adapter.cancel(submitted.modal_call_id)
            except _MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Could not cancel duplicate GROMACS stage %s",
                    submitted.modal_call_id,
                )
            return

        if advanced.state == JobState.CANCEL_REQUESTED:
            try:
                await self.adapter.cancel(submitted.modal_call_id)
            except _MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Could not cancel newly attached GROMACS stage %s",
                    submitted.modal_call_id,
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
            except _MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while cleaning job %s", job.job_id
                )
                continue
            except Exception:
                LOGGER.exception("Could not clean intermediates for job %s", job.job_id)
                continue
            self.store.mark_intermediates_cleaned(job.job_id, now=now)

    async def _apply(self, job: JobRecord, outcome: PollOutcome) -> None:
        now = self._now()
        if outcome.kind == "running":
            if job.state != JobState.CANCEL_REQUESTED:
                self.store.set_job_state(job.job_id, JobState.RUNNING, now=now)
            return
        if outcome.kind == "cancelled":
            self.store.set_job_state(job.job_id, JobState.CANCELLED, now=now)
            return
        if outcome.kind == "failed":
            self.store.fail_job(
                job.job_id,
                error_code="compute_failed",
                error_message="GROMACS could not complete the simulation.",
                now=now,
            )
            return

        self.store.set_job_state(job.job_id, JobState.FINALIZING, now=now)
        try:
            archive = await self.adapter.publish_archive(job, completed_at=now)
        except ArchiveNotReadyError:
            LOGGER.info("GROMACS archive is not visible yet for job %s", job.job_id)
            return
        except _MODAL_SERVICE_ERRORS:
            LOGGER.exception("Modal is unavailable while publishing job %s", job.job_id)
            return
        except Exception:
            LOGGER.exception("GROMACS job %s returned an invalid archive", job.job_id)
            self.store.fail_job(
                job.job_id,
                error_code="result_invalid",
                error_message="GROMACS completed, but its result archive was invalid.",
                now=now,
            )
            return
        self._complete(job, archive)

    def _complete(self, job: JobRecord, archive: FinalArchive) -> None:
        self.store.complete_job(
            job.job_id,
            state=archive.state,
            result_volume_name=archive.volume_name,
            result_volume_path=archive.path,
            result_filename=archive.filename,
            result_size_bytes=archive.size_bytes,
            result_sha256=archive.sha256,
            warnings_json=archive.warnings_json,
            now=self._now(),
        )
