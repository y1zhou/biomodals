"""AlphaFold3 request staging and archive reuse for the API service."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile

import modal

from biomodals.app.fold.alphafold3.execution_request import (
    load_execution_request_from_volume,
    stage_execution_request,
)
from biomodals.app.fold.alphafold3.invocation_cache import (
    load_invocation_manifest,
)
from biomodals.app.fold.alphafold3.request_results import create_request_archive
from biomodals.execution.modal import stage_execution_launch
from biomodals.service.alphafold3.validation import ValidatedInputStore
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.store import JobRecord
from biomodals.service.tool_runtime import PreparedResult

ALPHAFOLD3_ARCHIVE_SCHEMA = "alphafold3-request/1"


class AlphaFold3ToolAdapter:
    """Reuse the app's validated request and published request-view archive."""

    def __init__(
        self,
        validations: ValidatedInputStore,
        *,
        output_volume_name: str = "AlphaFold3-outputs",
    ) -> None:
        """Bind retained inputs to the established AlphaFold3 output Volume."""
        self.validations = validations
        self.output_volume_name = output_volume_name

    async def stage(self, job: JobRecord) -> None:
        """Stage one retained validation with its admitted provider limits."""
        if job.pending_validation_id is None:
            raise FileNotFoundError("Pending AlphaFold3 validation is unavailable")
        validated = self.validations.get(
            job.pending_validation_id,
            owner_user_id=job.owner_user_id,
        )
        if validated is None:
            raise FileNotFoundError("Pending AlphaFold3 validation is unavailable")
        request = await asyncio.to_thread(
            validated.request,
            max_active_provider_calls=job.max_active_provider_calls,
            max_active_gpu_provider_calls=job.max_active_gpu_provider_calls,
        )
        volume = self._volume(job)
        await volume.hydrate.aio()
        await asyncio.to_thread(stage_execution_request, volume, job.job_id, request)
        await asyncio.to_thread(stage_execution_launch, volume, job.job_id, None)

    async def discard_pending(self, job: JobRecord) -> None:
        """Consume a validation only after request and launch staging succeed."""
        if job.pending_validation_id is not None:
            await asyncio.to_thread(
                self.validations.delete_claimed,
                job.pending_validation_id,
            )

    async def prepare_result(
        self,
        job: JobRecord,
        cache: ArtifactCache,
        *,
        completed_at: int,
    ) -> PreparedResult:
        """Use the app's content-addressed request manifest and archive builder."""
        del completed_at
        volume = self._volume(job)
        request = await asyncio.to_thread(
            load_execution_request_from_volume,
            volume,
            job.job_id,
        )
        manifest = await asyncio.to_thread(
            load_invocation_manifest,
            volume,
            request.invocation,
        )
        if manifest is None:
            raise FileNotFoundError("AlphaFold3 request publication is unavailable")
        staging = cache.staging_path(str(job.job_id))
        try:
            with tempfile.TemporaryDirectory(
                dir=cache.directory,
                prefix=f".{job.job_id}.archive-",
            ) as directory:
                archive = await asyncio.to_thread(
                    create_request_archive,
                    volume,
                    manifest,
                    output_dir=directory,
                    display_name=request.config.name,
                )
                os.replace(archive, staging)
            size_bytes, digest = await cache.run_bounded(_file_identity, staging)
            lease = await cache.publish_staged(
                str(job.job_id),
                staging,
                size_bytes=size_bytes,
                sha256=digest,
            )
            lease.close()
        finally:
            staging.unlink(missing_ok=True)
        return PreparedResult(
            filename=archive.name,
            media_type="application/zstd",
            size_bytes=size_bytes,
            sha256=digest,
            archive_schema=ALPHAFOLD3_ARCHIVE_SCHEMA,
        )

    def _volume(self, job: JobRecord) -> modal.Volume:
        return modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
            version=2,
        )


def _file_identity(path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()
