"""Humanization request staging and verified scientific result presentation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast

import modal

from biomodals.execution.modal import stage_execution_launch
from biomodals.helper.modal_volume import (
    download_modal_volume_files,
    read_modal_volume_file,
)
from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.humanization.results import (
    HumanizationManifest,
    build_humanization_archive,
)
from biomodals.service.pending import PendingRequestStore
from biomodals.service.store import JobRecord
from biomodals.service.tool_runtime import PreparedResult
from biomodals.workflow.humanization.execution import (
    HumanizationExecutionRequest,
    load_execution_request_from_volume,
    result_directory,
    stage_execution_request,
)
from biomodals.workflow.humanization.workflow import OUT_VOLUME_NAME

HUMANIZATION_ARCHIVE_SCHEMA = "humanization/1"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class HumanizationToolAdapter:
    """Adapt the existing workflow without introducing another execution lifecycle."""

    def __init__(
        self,
        pending: PendingRequestStore,
        *,
        modal_download_concurrency: int = 4,
    ) -> None:
        """Bind retained requests to the workflow's established output Volume."""
        if (
            isinstance(modal_download_concurrency, bool)
            or modal_download_concurrency < 1
        ):
            raise ValueError("modal_download_concurrency must be a positive integer")
        self.pending = pending
        self.modal_download_concurrency = modal_download_concurrency

    async def stage(self, job: JobRecord) -> None:
        """Verify one immutable request and its initial launch before discarding input."""
        content = self.pending.get(job.job_id)
        volume = self._volume(job)
        await volume.hydrate.aio()
        if content is None:
            await asyncio.to_thread(
                load_execution_request_from_volume, volume, job.job_id
            )
            return
        request = HumanizationExecutionRequest.from_bytes(content)
        await asyncio.to_thread(stage_execution_request, volume, job.job_id, request)
        await asyncio.to_thread(stage_execution_launch, volume, job.job_id, None)

    async def discard_pending(self, job: JobRecord) -> None:
        """Discard the local copy only after successful remote staging."""
        self.pending.delete(job.job_id)

    async def prepare_result(
        self,
        job: JobRecord,
        cache: ArtifactCache,
        *,
        completed_at: int,
    ) -> PreparedResult:
        """Download declared evidence once and publish a reproducible result ZIP."""
        del completed_at
        volume = self._volume(job)
        request = await asyncio.to_thread(
            load_execution_request_from_volume, volume, job.job_id
        )
        remote = result_directory(job.job_id)
        try:
            content = await read_modal_volume_file(
                volume,
                (remote / "manifest.json").as_posix(),
                max_bytes=MAX_MANIFEST_BYTES,
            )
            manifest = HumanizationManifest.model_validate_json(content)
            if (
                manifest.execution_run_id != job.job_id
                or manifest.parameters != request.settings
                or manifest.scientific_versions != request.scientific_versions
            ):
                raise ValueError("Humanization publication does not match its request")
        except ValueError as exc:
            raise ArtifactIntegrityError(
                "Humanization publication manifest is invalid"
            ) from exc

        staging = cache.staging_path(str(job.job_id))
        try:
            with TemporaryDirectory(
                dir=cache.directory, prefix=f".{job.job_id}.archive-"
            ) as directory:
                root = Path(directory)
                await cache.run_bounded((root / "manifest.json").write_bytes, content)
                await cache.run_bounded(
                    download_modal_volume_files,
                    volume,
                    (
                        (str(remote / record.path), root / record.path)
                        for record in manifest.files
                    ),
                    concurrency=self.modal_download_concurrency,
                )
                try:
                    with staging.open("w+b") as destination:
                        built = await cache.run_bounded(
                            build_humanization_archive,
                            root,
                            cast("BinaryIO", destination),
                        )
                except ValueError as exc:
                    raise ArtifactIntegrityError(
                        "Humanization scientific files failed verification"
                    ) from exc
            await cache.publish_staged(
                str(job.job_id),
                staging,
                size_bytes=built.size_bytes,
                sha256=built.sha256,
            )
        finally:
            staging.unlink(missing_ok=True)
        return PreparedResult(
            filename=f"humanization-{job.job_id}.zip",
            media_type="application/zip",
            size_bytes=built.size_bytes,
            sha256=built.sha256,
            archive_schema=HUMANIZATION_ARCHIVE_SCHEMA,
        )

    def _volume(self, job: JobRecord) -> modal.Volume:
        return modal.Volume.from_name(
            OUT_VOLUME_NAME,
            environment_name=job.modal_environment,
            version=2,
        )
