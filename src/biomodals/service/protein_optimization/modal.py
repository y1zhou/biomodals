"""Stage reviewed requests and restore CSV Results through the shared cache."""

from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import modal

from biomodals.execution import DeploymentIdentity
from biomodals.execution.modal import stage_execution_launch
from biomodals.helper.modal_volume import (
    download_modal_volume_files,
    read_modal_volume_file,
)
from biomodals.service.artifacts import (
    ArtifactCache,
    ArtifactIntegrityError,
    run_blocking_io,
)
from biomodals.service.pending import PendingRequestStore
from biomodals.service.protein_optimization.results import build_projection
from biomodals.service.store import JobRecord
from biomodals.service.tool_runtime import PreparedResult
from biomodals.workflow.protein_optimization.execution import (
    REQUEST_FILE,
    OptimizationExecutionRequest,
)
from biomodals.workflow.protein_optimization.nodes import FEATURE_OPERATION
from biomodals.workflow.protein_optimization.publication import (
    RESULT_SCHEMA,
    OptimizationManifest,
)
from biomodals.workflow.protein_optimization.workflow import OUT_VOLUME_NAME


class ProteinOptimizationAdapter:
    """No second scheduler or service-owned model cache."""

    def __init__(self, pending: PendingRequestStore) -> None:
        """Bind original immutable request staging."""
        self.pending = pending

    def _volume(self, job: JobRecord) -> modal.Volume:
        return modal.Volume.from_name(
            OUT_VOLUME_NAME, environment_name=job.modal_environment, version=2
        )

    async def preflight(self, deployment: DeploymentIdentity) -> None:
        """Resolve the containing workflow capability without invoking compute."""
        function = modal.Function.from_name(
            deployment.deployment_name,
            FEATURE_OPERATION,
            environment_name=deployment.environment,
            version=deployment.deployment_version,
        )
        await function.hydrate.aio()

    async def stage(self, job: JobRecord) -> None:
        """Persist one request and launch marker; model preparation belongs to its DAG."""
        volume = self._volume(job)
        content = await asyncio.to_thread(self.pending.get, job.job_id)
        if content is None:
            await self.input_request(job)
        else:
            await asyncio.to_thread(OptimizationExecutionRequest.from_bytes, content)
            await asyncio.to_thread(REQUEST_FILE.stage, volume, job.job_id, content)
        await asyncio.to_thread(stage_execution_launch, volume, job.job_id, None)

    async def discard_pending(self, job: JobRecord) -> None:
        """Remove local staging after remote publication, including deletion cleanup."""
        await asyncio.to_thread(self.pending.delete, job.job_id)

    async def input_request(self, job: JobRecord) -> OptimizationExecutionRequest:
        """Return original complete inputs without scientific work."""
        content = await asyncio.to_thread(self.pending.get, job.job_id)
        if content is None:
            content = await asyncio.to_thread(
                REQUEST_FILE.load_from_volume, self._volume(job), job.job_id
            )
        return await asyncio.to_thread(OptimizationExecutionRequest.from_bytes, content)

    async def prepare_result(
        self, job: JobRecord, cache: ArtifactCache, *, completed_at: int
    ) -> PreparedResult:
        """Restore verified source bytes, then build the disposable local projection."""
        del completed_at
        request = await self.input_request(job)
        volume = self._volume(job)
        root = (
            PurePosixPath("workflow-runs")
            / str(job.job_id)
            / "nodes/publish/result/protein_optimization"
        )
        manifest = OptimizationManifest.model_validate_json(
            await read_modal_volume_file(
                volume, str(root / "manifest.json"), max_bytes=1024 * 1024
            )
        )
        if (
            manifest.execution_run_id != job.job_id
            or manifest.design_digest != request.design.digest()
            or manifest.scientific_versions != request.scientific_versions
            or manifest.mode != request.design.settings.mode
            or manifest.direction != request.design.settings.direction
        ):
            raise ArtifactIntegrityError(
                "Protein optimization publication does not match its admitted request"
            )
        dataset = await asyncio.to_thread(request.design.dataset)
        if manifest.chain_columns != {
            chain: f"sequence_{chain}" for chain in dataset.parents
        } or manifest.candidate_count != await asyncio.to_thread(
            request.design.candidate_count, dataset
        ):
            raise ArtifactIntegrityError(
                "Protein optimization publication changed the admitted chains or candidate count"
            )
        staged = cache.staging_path(str(job.job_id))
        index = cache.staging_path(str(job.job_id))
        try:
            with TemporaryDirectory(
                dir=cache.directory, prefix=f".{job.job_id}.archive-"
            ) as temporary:
                source = Path(temporary) / "candidates.csv"
                await run_blocking_io(
                    download_modal_volume_files,
                    volume,
                    ((str(root / "candidates.csv"), source),),
                    concurrency=1,
                )
                try:
                    size, digest = await cache.run_bounded(
                        build_projection, source, staged, index, manifest
                    )
                except ValueError as error:
                    raise ArtifactIntegrityError(
                        "Protein optimization CSV is invalid"
                    ) from error
            await cache.run_bounded(cache.publish_derived, str(job.job_id), index)
            await cache.publish_staged(
                str(job.job_id), staged, size_bytes=size, sha256=digest
            )
        finally:
            staged.unlink(missing_ok=True)
            index.unlink(missing_ok=True)
        return PreparedResult(
            filename=f"protein-optimization-{job.job_id}.csv",
            media_type="text/csv",
            size_bytes=size,
            sha256=digest,
            archive_schema=RESULT_SCHEMA,
        )
