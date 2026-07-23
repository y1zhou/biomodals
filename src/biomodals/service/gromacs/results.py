"""Authoritative GROMACS Result publication, recovery, and cache access."""

from __future__ import annotations

import hashlib
import re
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

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
from biomodals.service.gromacs.contracts import is_gromacs_run_name
from biomodals.service.gromacs.provider import (
    DEFINITE_SUBMISSION_ERRORS,
    MODAL_SERVICE_ERRORS,
)
from biomodals.service.jobs import job_stage_history
from biomodals.service.store import JobRecord, JobState

_SHA256 = re.compile(r"[0-9a-f]{64}")
MODAL_PUBLICATION_ERRORS = MODAL_SERVICE_ERRORS + (
    modal.exception.ExecutionError,
    modal.exception.VolumeUploadTimeoutError,
)
PERMANENT_FINALIZATION_ERRORS = DEFINITE_SUBMISSION_ERRORS
TRANSIENT_FINALIZATION_ERRORS = (
    modal.exception.ConnectionError,
    modal.exception.ExecutionError,
    modal.exception.InternalError,
    modal.exception.ResourceExhaustedError,
    modal.exception.ServiceError,
    modal.exception.VolumeUploadTimeoutError,
)


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


class ModalGromacsResults:
    """Publish and recover immutable Results from the established Modal Volume."""

    def __init__(
        self,
        *,
        output_volume_name: str = "Gromacs-outputs",
        artifact_cache: ArtifactCache | None = None,
    ) -> None:
        """Configure authoritative remote storage and optional local caching."""
        self.output_volume_name = output_volume_name
        self.artifact_cache = artifact_cache

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

        remote_mtimes: dict[str, int] = {}
        for entry in await volume.listdir.aio(job.run_name):
            path = PurePosixPath(entry.path).as_posix().lstrip("/")
            if path in remote_mtimes or type(entry.mtime) is not int:
                raise GromacsResultInvalidError("GROMACS output metadata is invalid")
            remote_mtimes[path] = entry.mtime

        stages = [
            orjson.loads(stage.model_dump_json())
            for stage in job_stage_history(job, job.stage_history)
        ]
        if stages and stages[-1]["code"] == "prepare_result":
            stages[-1]["ended_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(completed_at),
            )
            stages[-1]["outcome"] = "completed"
        try:
            built = await write_gromacs_archive(
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
                remote_mtimes=remote_mtimes,
                run_bounded=(
                    self.artifact_cache.run_bounded
                    if self.artifact_cache is not None
                    else None
                ),
            )
        except ValueError as exc:
            raise GromacsResultInvalidError(str(exc)) from exc
        if (
            job.artifact_request_sha256 is None
            or built.request_sha256 != job.artifact_request_sha256
        ):
            raise GromacsResultInvalidError(
                "GROMACS Result does not match the admitted request"
            )
        return built

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
        if (
            job.artifact_request_sha256 is None
            or request_sha256 != job.artifact_request_sha256
        ):
            raise ValueError("GROMACS Result does not match the admitted request")
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
