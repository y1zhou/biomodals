"""AlphaFold3 request staging and archive reuse for the API service."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import IO, Any, cast

import modal

from biomodals.app.fold.alphafold3.execution_request import (
    load_execution_request_from_volume,
    stage_execution_request,
)
from biomodals.app.fold.alphafold3.inference_inputs import serialize_af3_input
from biomodals.app.fold.alphafold3.invocation_cache import (
    load_invocation_manifest,
)
from biomodals.app.fold.alphafold3.request_results import create_request_archive
from biomodals.execution.modal import stage_execution_launch
from biomodals.helper.artifacts import file_size_sha256
from biomodals.service.alphafold3.validation import ValidatedInputStore
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.store import JobRecord, JobState, ServiceStore
from biomodals.service.tool_runtime import PreparedResult, SubmissionWait

ALPHAFOLD3_ARCHIVE_SCHEMA = "alphafold3-request/1"


class AlphaFold3ToolAdapter:
    """Reuse the app's validated request and published request-view archive."""

    def __init__(
        self,
        validations: ValidatedInputStore,
        store: ServiceStore,
        *,
        output_volume_name: str = "AlphaFold3-outputs",
        modal_download_concurrency: int = 4,
    ) -> None:
        """Bind retained inputs to the established AlphaFold3 output Volume."""
        if (
            isinstance(modal_download_concurrency, bool)
            or modal_download_concurrency < 1
        ):
            raise ValueError("modal_download_concurrency must be a positive integer")
        self.validations = validations
        self.store = store
        self.output_volume_name = output_volume_name
        self.modal_download_concurrency = modal_download_concurrency

    async def stage(self, job: JobRecord) -> SubmissionWait | None:
        """Stage one retained validation with its admitted provider limits."""
        predecessors = self.store.list_preceding_jobs_for_request(job.job_id)
        if any(_execution_may_be_active(item) for item in predecessors):
            return SubmissionWait(
                reason="waiting_for_shared_publication",
                message=("Waiting for an earlier identical AlphaFold3 Job to finish"),
            )
        repair_execution_run_ids = tuple(
            item.job_id for item in predecessors if _needs_incomplete_repair(item)
        )
        if job.pending_validation_id is None:
            return
        volume = self._volume(job)
        await volume.hydrate.aio()
        validated = self.validations.get_claimed(
            job.pending_validation_id,
            owner_user_id=job.owner_user_id,
        )
        if validated is None:
            await asyncio.to_thread(
                load_execution_request_from_volume,
                volume,
                job.job_id,
            )
            return
        request = await asyncio.to_thread(
            validated.request,
            max_active_provider_calls=job.max_active_provider_calls,
            max_active_gpu_provider_calls=job.max_active_gpu_provider_calls,
            repair_execution_run_ids=repair_execution_run_ids,
        )
        await asyncio.to_thread(stage_execution_request, volume, job.job_id, request)
        await asyncio.to_thread(stage_execution_launch, volume, job.job_id, None)
        return None

    async def discard_pending(self, job: JobRecord) -> None:
        """Consume a validation only after request and launch staging succeed."""
        if job.pending_validation_id is not None:
            await asyncio.to_thread(
                self.validations.delete_claimed,
                job.pending_validation_id,
            )

    async def input_document(self, job: JobRecord) -> bytes:
        """Return the normalized AlphaFold3 input retained for one Job."""
        if job.pending_validation_id is not None:
            validated = self.validations.get_claimed(
                job.pending_validation_id,
                owner_user_id=job.owner_user_id,
            )
            if validated is not None:
                return await asyncio.to_thread(validated.document_path.read_bytes)
        volume = self._volume(job)
        return await asyncio.to_thread(
            lambda: serialize_af3_input(
                load_execution_request_from_volume(volume, job.job_id).config
            )
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
                    download_file=lambda path, handle: self._download_file(
                        volume,
                        path,
                        handle,
                    ),
                )
                os.replace(archive, staging)
            size_bytes, digest = await cache.run_bounded(file_size_sha256, staging)
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

    def _download_file(
        self,
        volume: modal.Volume,
        path: str,
        handle: IO[bytes],
    ) -> int:
        """Download one artifact with the configured per-Job concurrency."""
        return cast(Any, volume)._read_file_into_fileobj(
            path,
            handle,
            concurrency=self.modal_download_concurrency,
        )

    def _volume(self, job: JobRecord) -> modal.Volume:
        return modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
            version=2,
        )


def _execution_may_be_active(job: JobRecord) -> bool:
    if job.state in {
        JobState.QUEUED,
        JobState.RUNNING,
        JobState.CANCEL_REQUESTED,
        JobState.STATE_UNKNOWN,
    }:
        return True
    return job.state == JobState.BLOCKED and job.result_state is None


def _needs_incomplete_repair(job: JobRecord) -> bool:
    return (
        job.state
        in {
            JobState.PARTIAL,
            JobState.FAILED,
            JobState.CANCELLED,
        }
        or job.result_state == JobState.PARTIAL.value
    )
