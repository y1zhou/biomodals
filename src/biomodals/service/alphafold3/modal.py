"""AlphaFold3 request staging and archive reuse for the API service."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile

import modal

from biomodals.app.fold.alphafold3.chemistry import ChemistryReceipt
from biomodals.app.fold.alphafold3.execution_request import (
    load_execution_request_from_volume,
    load_input_document_from_volume,
    stage_execution_request,
)
from biomodals.app.fold.alphafold3.invocation_cache import (
    load_invocation_manifest,
)
from biomodals.app.fold.alphafold3.request_results import create_request_archive
from biomodals.execution import DeploymentIdentity
from biomodals.execution.modal import stage_execution_launch
from biomodals.helper.artifacts import file_size_sha256
from biomodals.helper.modal_volume import download_modal_volume_files
from biomodals.service.alphafold3.validation import ValidatedInputStore
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.http_contract import CodedAPIError
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

    async def check_chemistry(
        self, content: bytes, deployment: DeploymentIdentity
    ) -> ChemistryReceipt:
        """Use the exact prediction deployment's isolated native CPU checker."""
        call = None
        try:
            async with asyncio.timeout(180):
                function = modal.Function.from_name(
                    deployment.deployment_name,
                    "check_input_chemistry",
                    environment_name=deployment.environment,
                    version=deployment.deployment_version,
                )
                await function.hydrate.aio()
                call = await function.spawn.aio(content)
                response = await call.get.aio()
        except modal.exception.NotFoundError as error:
            raise CodedAPIError(
                409,
                "deployment_incompatible",
                "Deploy and pin an AlphaFold3 version supporting chemistry checks",
            ) from error
        except TimeoutError as error:
            if call is not None:
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(5):
                        await call.cancel.aio()
            raise CodedAPIError(
                504,
                "chemistry_timeout",
                "Chemistry check timed out; retry Continue",
            ) from error
        except asyncio.CancelledError:
            if call is not None:
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(5):
                        await call.cancel.aio()
            raise
        except Exception as error:
            raise CodedAPIError(
                503,
                "chemistry_unavailable",
                "Chemistry check is unavailable; retry Continue",
            ) from error
        if response.get("error"):
            raise CodedAPIError(400, "chemistry_invalid", str(response["error"])[:1000])
        try:
            return ChemistryReceipt.model_validate(response["receipt"])
        except (KeyError, TypeError, ValueError) as error:
            raise CodedAPIError(
                503,
                "chemistry_unavailable",
                "Native chemistry evidence is invalid; retry Continue",
            ) from error

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
        """Read saved native JSON without requiring execution compatibility."""
        if job.pending_validation_id is not None:
            validated = self.validations.get_claimed(
                job.pending_validation_id,
                owner_user_id=job.owner_user_id,
            )
            if validated is not None:
                return await asyncio.to_thread(validated.document_path.read_bytes)
        return await asyncio.to_thread(
            load_input_document_from_volume, self._volume(job), job.job_id
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
                    download_files=lambda downloads: download_modal_volume_files(
                        volume,
                        downloads,
                        concurrency=self.modal_download_concurrency,
                    ),
                )
                os.replace(archive, staging)
            size_bytes, digest = await cache.run_bounded(file_size_sha256, staging)
            await cache.publish_staged(
                str(job.job_id),
                staging,
                size_bytes=size_bytes,
                sha256=digest,
            )
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
