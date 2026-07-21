"""Modal SDK adapter and lifecycle reconciliation for GROMACS."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast
from uuid import uuid4

import modal
import orjson

from biomodals.service.artifacts import (
    ArtifactCache,
    ArtifactIntegrityError,
    ArtifactLease,
    ArtifactSourceMissingError,
)
from biomodals.service.gromacs.archive import (
    GROMACS_ARCHIVE_SCHEMA_VERSION,
    BuiltGromacsArchive,
    validate_gromacs_archive,
    write_gromacs_archive,
)
from biomodals.service.gromacs.router import (
    GromacsJobOptions,
    SubmissionOutcomeUnknownError,
    is_gromacs_run_name,
)
from biomodals.service.jobs import JobLifecycleLocks, JobView
from biomodals.service.runtime_config import ModalConfigurationSnapshot
from biomodals.service.store import (
    JobProviderCallRecord,
    JobProviderCallState,
    JobRecord,
    JobState,
    JobStateUnknownReason,
    JobSubmissionConflictError,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NVT_ANALYSIS = "collect_traj_stats:nvt_"
_NPT_ANALYSIS = "collect_traj_stats:npt_"
_PRODUCTION_ANALYSIS = "collect_traj_stats:production_"
_FINAL_OPERATION = _PRODUCTION_ANALYSIS
_REQUIRED_FUNCTIONS = (
    "prepare_tpr_cpu",
    "prepare_tpr_gpu",
    "collect_traj_stats",
    "production_run_cpu",
    "production_run_gpu",
)
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
_MODAL_PUBLICATION_ERRORS = _MODAL_SERVICE_ERRORS + (
    modal.exception.ExecutionError,
    modal.exception.VolumeUploadTimeoutError,
)
_DEFINITE_SUBMISSION_ERRORS = (
    modal.exception.AuthError,
    modal.exception.InvalidError,
    modal.exception.NotFoundError,
    modal.exception.PermissionDeniedError,
)
_PERMANENT_FINALIZATION_ERRORS = _DEFINITE_SUBMISSION_ERRORS
_TRANSIENT_FINALIZATION_ERRORS = (
    modal.exception.ConnectionError,
    modal.exception.ExecutionError,
    modal.exception.InternalError,
    modal.exception.ResourceExhaustedError,
    modal.exception.ServiceError,
    modal.exception.VolumeUploadTimeoutError,
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
    cache_lease: ArtifactLease | None = None


@dataclass(frozen=True, slots=True)
class _ResultMarker:
    request_sha256: str
    archive_sha256: str
    size_bytes: int


class ArchiveNotReadyError(RuntimeError):
    """Raised when a stable run has not published its completion marker yet."""


class ResultIdentityMismatchError(RuntimeError):
    """Raised when reconstruction cannot reproduce a published Result."""


class GromacsResultInvalidError(RuntimeError):
    """Raised when established GROMACS outputs violate the Result contract."""


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
        artifact_cache: ArtifactCache | None = None,
        call_resolver: Callable[[str], modal.FunctionCall] = modal.FunctionCall.from_id,
        function_resolver: Callable[..., modal.Function] | None = None,
    ) -> None:
        """Bind one deployed App in an explicit Modal Environment."""
        self.environment_name = environment_name
        self.app_name = app_name
        self.output_volume_name = output_volume_name
        self.artifact_cache = artifact_cache
        self._function_resolver = function_resolver or modal.Function.from_name
        self._call_resolver = call_resolver

    async def preflight(
        self,
        app_name: str,
        environment_name: str,
        app_version: int,
    ) -> None:
        """Hydrate required deployed resources without invoking compute."""
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=environment_name,
        )
        await volume.hydrate.aio()
        for function_name in _REQUIRED_FUNCTIONS:
            function = self._function_resolver(
                app_name,
                function_name,
                environment_name=environment_name,
                version=app_version,
            )
            await function.hydrate.aio()

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
            version=modal_configuration.app_version,
        )
        try:
            call = await function.spawn.aio(
                pdb_content=pdb_content,
                run_name=run_name,
                simulation_time_ns=options.simulation_time_ns,
                run_pdbfixer=options.run_pdbfixer,
            )
            modal_call_id = call.object_id
        except _DEFINITE_SUBMISSION_ERRORS:
            raise
        except Exception as exc:
            raise SubmissionOutcomeUnknownError(
                "Modal did not return a durable FunctionCall handle"
            ) from exc
        return SubmittedModalCall(
            modal_call_id=modal_call_id,
            run_name=run_name,
            provider_operation=function_name,
        )

    async def submit_operation(
        self,
        job: JobRecord,
        provider_operation: str,
    ) -> SubmittedModalCall:
        """Spawn one explicitly selected deployed stage in the Job graph."""
        if job.run_name is None:
            raise ValueError("GROMACS Job has no run name")
        options = GromacsJobOptions.model_validate_json(job.parameters_json)
        production_operation = (
            "production_run_cpu" if options.cpu_only else "production_run_gpu"
        )
        if provider_operation not in {
            _NVT_ANALYSIS,
            _NPT_ANALYSIS,
            production_operation,
            _PRODUCTION_ANALYSIS,
        }:
            raise ValueError(f"Unsupported GROMACS operation: {provider_operation}")

        function_name, _, traj_prefix = provider_operation.partition(":")
        if function_name.startswith("production_run_"):
            kwargs = {
                "run_name": job.run_name,
                "simulation_time_ns": options.simulation_time_ns,
            }
        elif provider_operation == _PRODUCTION_ANALYSIS:
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
            version=job.modal_app_version,
        )
        try:
            call = await function.spawn.aio(**kwargs)
            modal_call_id = call.object_id
        except _DEFINITE_SUBMISSION_ERRORS:
            raise
        except Exception as exc:
            raise SubmissionOutcomeUnknownError(
                "Modal did not return a durable FunctionCall handle"
            ) from exc
        return SubmittedModalCall(
            modal_call_id=modal_call_id,
            run_name=job.run_name,
            provider_operation=provider_operation,
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
            if any(
                node.status
                in {
                    modal.call_graph.InputStatus.FAILURE,
                    modal.call_graph.InputStatus.INIT_FAILURE,
                    modal.call_graph.InputStatus.TIMEOUT,
                }
                for node in nodes
            ):
                return PollOutcome("failed")
            # A SUCCESS graph can become visible just before get() can return
            # retained output. Poll again rather than inventing a failure.
            return PollOutcome("running")
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
            or not is_gromacs_run_name(job.run_name)
            or job.result_volume_path != f"api-results/{job.run_name}/result.zip"
            or job.result_filename != f"{job.run_name}.zip"
            or type(job.result_size_bytes) is not int
            or job.result_size_bytes < 1
            or not isinstance(job.result_sha256, str)
            or _SHA256.fullmatch(job.result_sha256) is None
        ):
            raise ValueError("Job does not reference the configured GROMACS Volume")
        try:
            marker = await self._result_marker(job)
        except (ArchiveNotReadyError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "GROMACS result marker is missing or invalid"
            ) from exc
        if (
            marker.size_bytes != job.result_size_bytes
            or marker.archive_sha256 != job.result_sha256
        ):
            raise ArtifactIntegrityError(
                "GROMACS result marker does not match the recorded Result"
            )
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )
        try:
            async for chunk in volume.read_file.aio(job.result_volume_path):
                yield chunk
        except FileNotFoundError as exc:
            raise ArtifactSourceMissingError(
                "Published GROMACS result archive is missing"
            ) from exc

    async def cleanup_intermediates(self, job: JobRecord) -> None:
        """Remove one retained run directory without touching its final ZIP."""
        if job.run_name is None or not is_gromacs_run_name(job.run_name):
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
        if job.run_name is None or not is_gromacs_run_name(job.run_name):
            raise ValueError("Job has no valid API run name")
        if (
            job.result_previous_state in {JobState.SUCCEEDED, JobState.PARTIAL}
            and job.result_archive_schema_version != GROMACS_ARCHIVE_SCHEMA_VERSION
        ):
            raise GromacsResultInvalidError(
                "Job uses an unsupported Result archive schema"
            )
        archive_path = f"api-results/{job.run_name}/result.zip"
        marker_path = f"api-results/{job.run_name}/result.json"
        archive_file = tempfile.NamedTemporaryFile(
            dir=(self.artifact_cache.directory if self.artifact_cache else None),
            prefix=f".{job.job_id}.",
            suffix=".part",
            delete=False,
        )
        local_path = Path(archive_file.name)
        cache_lease: ArtifactLease | None = None
        try:
            with (
                archive_file as archive,
                tempfile.NamedTemporaryFile(
                    dir=(
                        self.artifact_cache.directory if self.artifact_cache else None
                    ),
                    prefix=f".{job.job_id}.marker.",
                    suffix=".part",
                ) as marker_file,
            ):
                archive_handle = cast("BinaryIO", archive)
                built = await self._build_archive(
                    archive_handle,
                    job,
                    completed_at=completed_at,
                )
                restored_state = job.result_previous_state
                if restored_state in {JobState.SUCCEEDED, JobState.PARTIAL}:
                    if (
                        built.size_bytes != job.result_size_bytes
                        or built.sha256 != job.result_sha256
                    ):
                        raise ResultIdentityMismatchError(
                            "Rebuilt Result does not match its published identity"
                        )
                marker = orjson.dumps({
                    "archive_schema_version": GROMACS_ARCHIVE_SCHEMA_VERSION,
                    "request_sha256": built.request_sha256,
                    "archive_sha256": built.sha256,
                    "size_bytes": built.size_bytes,
                })
                marker_file.write(marker)
                archive_handle.flush()
                marker_file.flush()
                volume = modal.Volume.from_name(
                    self.output_volume_name,
                    environment_name=job.modal_environment,
                )
                async with volume.batch_upload.aio(force=True) as upload:
                    upload.put_file(archive.name, archive_path)
                    upload.put_file(marker_file.name, marker_path)
            if self.artifact_cache is not None:
                cache_lease = await self.artifact_cache.publish_staged(
                    str(job.job_id),
                    local_path,
                    size_bytes=built.size_bytes,
                    sha256=built.sha256,
                )
        finally:
            local_path.unlink(missing_ok=True)

        return FinalArchive(
            state=(
                job.result_previous_state
                if job.result_previous_state in {JobState.SUCCEEDED, JobState.PARTIAL}
                else JobState.SUCCEEDED
            ),
            volume_name=self.output_volume_name,
            path=archive_path,
            filename=f"{job.run_name}.zip",
            size_bytes=built.size_bytes,
            sha256=built.sha256,
            warnings_json="[]",
            cache_lease=cache_lease,
        )

    async def _build_archive(
        self,
        handle: BinaryIO,
        job: JobRecord,
        *,
        completed_at: int,
    ) -> BuiltGromacsArchive:
        """Deterministically rebuild the immutable allowlist from raw outputs."""
        if job.run_name is None:
            raise ValueError("Job has no GROMACS run name")
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )

        async def read_file(path: str):
            async for chunk in volume.read_file.aio(path):
                yield chunk

        stages = [
            stage.model_dump(mode="json")
            for stage in JobView.from_record(job).stage_history
        ]
        if stages and stages[-1]["code"] == "prepare_result":
            stages[-1]["ended_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(completed_at),
            )
            stages[-1]["outcome"] = "completed"
        try:
            return await write_gromacs_archive(
                handle,
                run_name=job.run_name,
                parameters_json=job.parameters_json,
                modal_app_name=job.modal_app_name,
                modal_app_version=job.modal_app_version,
                job_id=str(job.job_id),
                stages_json=orjson.dumps(stages).decode(),
                started_at=job.created_at,
                completed_at=completed_at,
                read_file=read_file,
                run_bounded=(
                    self.artifact_cache.run_bounded
                    if self.artifact_cache is not None
                    else None
                ),
            )
        except ValueError as exc:
            raise GromacsResultInvalidError(str(exc)) from exc

    async def rebuild_artifact(self, job: JobRecord) -> AsyncIterator[bytes]:
        """Rebuild exact recorded bytes from raw outputs without compute."""
        if (
            job.finalization_started_at is None
            or job.result_size_bytes is None
            or job.result_sha256 is None
        ):
            raise ValueError("Job lacks immutable Result identity")
        if job.result_archive_schema_version != GROMACS_ARCHIVE_SCHEMA_VERSION:
            raise ValueError("Job uses an unsupported Result archive schema")
        archive_file = tempfile.NamedTemporaryFile(
            dir=(self.artifact_cache.directory if self.artifact_cache else None),
            prefix=f".{job.job_id}.rebuild.",
            suffix=".part",
            delete=False,
        )
        local_path = Path(archive_file.name)
        try:
            with archive_file as archive:
                handle = cast("BinaryIO", archive)
                built = await self._build_archive(
                    handle,
                    job,
                    completed_at=job.finalization_started_at,
                )
                if (
                    built.size_bytes != job.result_size_bytes
                    or built.sha256 != job.result_sha256
                ):
                    raise ValueError(
                        "Rebuilt Result does not match its published identity"
                    )
                handle.seek(0)
                while chunk := handle.read(1024 * 1024):
                    yield chunk
        finally:
            local_path.unlink(missing_ok=True)

    async def _result_marker(self, job: JobRecord) -> _ResultMarker:
        """Read and validate the completion marker for one stable run."""
        if job.run_name is None or not is_gromacs_run_name(job.run_name):
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
            or marker.get("archive_schema_version")
            != (job.result_archive_schema_version or GROMACS_ARCHIVE_SCHEMA_VERSION)
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
        if job.run_name is None or not is_gromacs_run_name(job.run_name):
            raise ValueError("Job has no valid API run name")
        archive_path = f"api-results/{job.run_name}/result.zip"
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
        )
        size_bytes = 0
        archive_file = tempfile.NamedTemporaryFile(
            dir=(self.artifact_cache.directory if self.artifact_cache else None),
            prefix=f".{job.job_id}.verify.",
            suffix=".part",
            delete=False,
        )
        local_path = Path(archive_file.name)
        try:
            with archive_file as archive:
                archive_handle = cast("BinaryIO", archive)
                try:
                    async for chunk in volume.read_file.aio(archive_path):
                        size_bytes += len(chunk)
                        if size_bytes > marker.size_bytes:
                            raise ValueError(
                                "GROMACS result archive is larger than recorded"
                            )
                        archive_handle.write(chunk)
                except FileNotFoundError as exc:
                    raise ArchiveNotReadyError(
                        "GROMACS result archive is missing"
                    ) from exc
                if size_bytes != marker.size_bytes:
                    raise ValueError("GROMACS result archive does not match its marker")
                if self.artifact_cache is None:
                    _validate_published_archive(archive_handle, marker, job.run_name)
                else:
                    await self.artifact_cache.run_bounded(
                        _validate_published_archive,
                        archive_handle,
                        marker,
                        job.run_name,
                    )
        finally:
            local_path.unlink(missing_ok=True)

    async def recover_archive(self, job: JobRecord) -> FinalArchive:
        """Recover and verify durable output after Modal call output expiry."""
        marker = await self._result_marker(job)
        if job.result_previous_state in {JobState.SUCCEEDED, JobState.PARTIAL} and (
            marker.size_bytes != job.result_size_bytes
            or marker.archive_sha256 != job.result_sha256
        ):
            raise ResultIdentityMismatchError(
                "Restored Result does not match its published identity"
            )
        await self._verify_archive_bytes(job, marker)
        return FinalArchive(
            state=(
                job.result_previous_state
                if job.result_previous_state in {JobState.SUCCEEDED, JobState.PARTIAL}
                else JobState.SUCCEEDED
            ),
            volume_name=self.output_volume_name,
            path=f"api-results/{job.run_name}/result.zip",
            filename=f"{job.run_name}.zip",
            size_bytes=marker.size_bytes,
            sha256=marker.archive_sha256,
            warnings_json="[]",
        )


def _operation_dependencies(job: JobRecord) -> dict[str, tuple[str, ...]]:
    """Return the fixed GROMACS graph for one CPU or GPU submission."""
    options = GromacsJobOptions.model_validate_json(job.parameters_json)
    prepare = "prepare_tpr_cpu" if options.cpu_only else "prepare_tpr_gpu"
    production = "production_run_cpu" if options.cpu_only else "production_run_gpu"
    return {
        prepare: (),
        _NVT_ANALYSIS: (prepare,),
        _NPT_ANALYSIS: (prepare,),
        production: (prepare,),
        _PRODUCTION_ANALYSIS: (production,),
    }


def _ready_operations(
    job: JobRecord,
    calls: Iterable[JobProviderCallRecord],
) -> list[str]:
    dependencies = _operation_dependencies(job)
    call_list = list(calls)
    known = {call.provider_operation for call in call_list}
    completed = {
        call.provider_operation
        for call in call_list
        if call.state == JobProviderCallState.COMPLETED
    }
    return [
        operation
        for operation, requirements in tuple(dependencies.items())[1:]
        if operation not in known and all(item in completed for item in requirements)
    ]


def _all_operations_completed(
    job: JobRecord,
    calls: Iterable[JobProviderCallRecord],
) -> bool:
    completed = {
        call.provider_operation
        for call in calls
        if call.state == JobProviderCallState.COMPLETED
    }
    return set(_operation_dependencies(job)).issubset(completed)


class GromacsReconciler:
    """Refresh locally persisted GROMACS jobs from Modal in one process."""

    def __init__(
        self,
        store: ServiceStore,
        adapter: ModalGromacsAdapter,
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
        if job.submission_lease_until is not None:
            if job.submission_lease_until <= now:
                self.store.mark_state_unknown(
                    job.job_id,
                    reason=JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN,
                    now=now,
                )
            return
        calls = self.store.list_provider_calls(job.job_id)
        if not calls:
            if job.state == JobState.CANCEL_REQUESTED:
                self.store.set_job_state(
                    job.job_id,
                    JobState.CANCELLED,
                    now=now,
                )
            return
        submitting = [
            call for call in calls if call.state == JobProviderCallState.SUBMITTING
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
        calls = self.store.list_provider_calls(job.job_id)
        if current.state == JobState.CANCEL_REQUESTED:
            await self._reconcile_cancellation(current, calls)
            return
        if any(call.state == JobProviderCallState.FAILED for call in calls):
            await self._settle_failed_job(current, calls)
            return
        if any(call.state == JobProviderCallState.CANCELLED for call in calls):
            await self._settle_cancelled_job(current, calls)
            return
        if _all_operations_completed(current, calls):
            await self._begin_finalization(current)
            return
        await self._submit_ready_operations(current)

    async def _poll_running_calls(
        self,
        job: JobRecord,
        calls: Iterable[JobProviderCallRecord],
    ) -> None:
        """Poll every directly running branch once and persist observations."""
        saw_running = False
        for call in calls:
            if call.state != JobProviderCallState.RUNNING or call.modal_call_id is None:
                continue
            try:
                outcome = await self.adapter.poll(
                    call.modal_call_id,
                    provider_operation=call.provider_operation,
                )
            except _MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while polling stage %s for job %s",
                    call.provider_operation,
                    job.job_id,
                )
                continue
            if outcome.kind == "running":
                saw_running = True
                continue
            if outcome.kind == "expired":
                terminal = (
                    JobProviderCallState.COMPLETED
                    if call.provider_operation == _FINAL_OPERATION
                    else JobProviderCallState.FAILED
                )
            else:
                terminal = JobProviderCallState(outcome.kind)
            self.store.record_provider_call_outcome(
                job.job_id,
                provider_operation=call.provider_operation,
                expected_modal_call_id=call.modal_call_id,
                outcome=terminal,
                now=self._now(),
            )
            if (
                outcome.kind == "expired"
                and call.provider_operation == _FINAL_OPERATION
            ):
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
        ready = _ready_operations(
            current,
            self.store.list_provider_calls(job.job_id),
        )
        for operation in ready:
            async with self.lifecycle_locks.for_job(job.job_id):
                current = self.store.get_job(job.owner_user_id, job.job_id)
                if current is None or current.state not in {
                    JobState.QUEUED,
                    JobState.RUNNING,
                }:
                    return
                if operation not in _ready_operations(
                    current,
                    self.store.list_provider_calls(job.job_id),
                ):
                    continue
                submission_token = uuid4().hex
                claimed = self.store.claim_provider_operation(
                    job.job_id,
                    provider_operation=operation,
                    submission_token=submission_token,
                    now=self._now(),
                )
                if claimed is None:
                    continue
                try:
                    submitted = await self.adapter.submit_operation(current, operation)
                except SubmissionOutcomeUnknownError:
                    LOGGER.exception(
                        "Could not confirm GROMACS stage %s for job %s",
                        operation,
                        job.job_id,
                    )
                    self.store.mark_state_unknown(
                        job.job_id,
                        reason=JobStateUnknownReason.SUBMISSION_OUTCOME_UNKNOWN,
                        now=self._now(),
                    )
                    return
                except _MODAL_SERVICE_ERRORS:
                    self.store.release_provider_operation(
                        job.job_id,
                        provider_operation=operation,
                        submission_token=submission_token,
                        now=self._now(),
                    )
                    LOGGER.exception(
                        "Modal is unavailable while submitting stage %s for job %s",
                        operation,
                        job.job_id,
                    )
                    return
                except Exception:
                    self.store.record_provider_submission_failure(
                        job.job_id,
                        provider_operation=operation,
                        submission_token=submission_token,
                        now=self._now(),
                    )
                    LOGGER.exception(
                        "Could not submit GROMACS stage %s for job %s",
                        operation,
                        job.job_id,
                    )
                    return

                try:
                    advanced = self.store.attach_provider_call(
                        job.job_id,
                        provider_operation=operation,
                        modal_call_id=submitted.modal_call_id,
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
                LOGGER.info(
                    "event=stage_attached job_id=%s workload=gromacs "
                    "operation=%s function=%s",
                    advanced.job_id,
                    operation,
                    operation.partition(":")[0],
                )

    async def _stop_active_calls(
        self,
        job: JobRecord,
        calls: Iterable[JobProviderCallRecord],
    ) -> bool:
        """Request fail-fast cancellation and report whether every call stopped."""
        stopped = True
        for call in calls:
            if call.state != JobProviderCallState.RUNNING or call.modal_call_id is None:
                continue
            try:
                await self.adapter.cancel(call.modal_call_id)
                outcome = await self.adapter.poll(
                    call.modal_call_id,
                    provider_operation=call.provider_operation,
                )
            except modal.exception.NotFoundError:
                outcome = PollOutcome("expired")
            except _MODAL_SERVICE_ERRORS:
                stopped = False
                LOGGER.exception(
                    "Modal is unavailable while stopping stage %s for job %s",
                    call.provider_operation,
                    job.job_id,
                )
                continue
            if outcome.kind == "running":
                stopped = False
                continue
            terminal = (
                JobProviderCallState.CANCELLED
                if outcome.kind == "expired"
                else JobProviderCallState(outcome.kind)
            )
            self.store.record_provider_call_outcome(
                job.job_id,
                provider_operation=call.provider_operation,
                expected_modal_call_id=call.modal_call_id,
                outcome=terminal,
                now=self._now(),
            )
        refreshed = self.store.list_provider_calls(job.job_id)
        return stopped and not any(
            call.state
            in {JobProviderCallState.SUBMITTING, JobProviderCallState.RUNNING}
            for call in refreshed
        )

    async def _settle_failed_job(
        self,
        job: JobRecord,
        calls: Iterable[JobProviderCallRecord],
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
        calls: Iterable[JobProviderCallRecord],
    ) -> None:
        if not await self._stop_active_calls(job, calls):
            return
        async with self.lifecycle_locks.for_job(job.job_id):
            self.store.set_job_state(job.job_id, JobState.CANCELLED, now=self._now())
        LOGGER.info("event=job_cancelled job_id=%s workload=gromacs", job.job_id)

    async def _reconcile_cancellation(
        self,
        job: JobRecord,
        calls: Iterable[JobProviderCallRecord],
    ) -> None:
        """Cancel every active branch without claiming an unknown outcome."""
        expired: list[JobProviderCallRecord] = []
        for call in calls:
            if call.state != JobProviderCallState.RUNNING or call.modal_call_id is None:
                continue
            try:
                await self.adapter.cancel(call.modal_call_id)
                outcome = await self.adapter.poll(
                    call.modal_call_id,
                    provider_operation=call.provider_operation,
                )
            except modal.exception.NotFoundError:
                outcome = PollOutcome("expired")
            except _MODAL_SERVICE_ERRORS:
                LOGGER.exception(
                    "Modal is unavailable while cancelling stage %s for job %s",
                    call.provider_operation,
                    job.job_id,
                )
                continue
            if outcome.kind == "running":
                continue
            if outcome.kind == "expired":
                expired.append(call)
                continue
            self.store.record_provider_call_outcome(
                job.job_id,
                provider_operation=call.provider_operation,
                expected_modal_call_id=call.modal_call_id,
                outcome=JobProviderCallState(outcome.kind),
                now=self._now(),
            )

        refreshed = self.store.list_provider_calls(job.job_id)
        if expired:
            if (
                len(expired) == 1
                and expired[0].provider_operation == _FINAL_OPERATION
                and not any(
                    call.state
                    in {
                        JobProviderCallState.SUBMITTING,
                        JobProviderCallState.RUNNING,
                    }
                    and call.provider_operation != _FINAL_OPERATION
                    for call in refreshed
                )
            ):
                try:
                    archive = await self.adapter.recover_archive(job)
                except _MODAL_SERVICE_ERRORS:
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
                    self.store.record_provider_call_outcome(
                        job.job_id,
                        provider_operation=expired[0].provider_operation,
                        expected_modal_call_id=expired[0].modal_call_id or "",
                        outcome=JobProviderCallState.COMPLETED,
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
            call.state
            in {JobProviderCallState.SUBMITTING, JobProviderCallState.RUNNING}
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
            calls = self.store.list_provider_calls(job.job_id)
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
            except _MODAL_SERVICE_ERRORS:
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
        except _PERMANENT_FINALIZATION_ERRORS:
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
        except (ArchiveNotReadyError, *_TRANSIENT_FINALIZATION_ERRORS):
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


def _validate_published_archive(
    handle: BinaryIO,
    marker: _ResultMarker,
    run_name: str,
) -> None:
    """Perform whole-file identity and ZIP checks off the event loop."""
    handle.seek(0)
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    if digest.hexdigest() != marker.archive_sha256:
        raise ValueError("GROMACS result archive does not match its marker")
    validated = validate_gromacs_archive(handle, run_name=run_name)
    if validated.request_sha256 != marker.request_sha256:
        raise ValueError("GROMACS result archive has the wrong request identity")
