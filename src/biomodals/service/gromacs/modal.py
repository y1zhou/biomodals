"""GROMACS request staging and Result presentation for the API service."""

from __future__ import annotations

import asyncio
from pathlib import PurePosixPath
from typing import BinaryIO, cast

import modal
import orjson

from biomodals.app.bioinfo.gromacs.continuation import (
    ContinuationInspection,
    read_continuation_source,
)
from biomodals.app.bioinfo.gromacs.execution import PREPARE_RESULT
from biomodals.app.bioinfo.gromacs.execution_runtime import (
    GromacsExecutionRequest,
    gromacs_publication_path,
    load_execution_request_from_volume,
    parse_gromacs_publication,
    stage_execution_request,
)
from biomodals.execution import DeploymentIdentity
from biomodals.execution.modal import stage_execution_launch
from biomodals.helper.modal_volume import read_modal_volume_file
from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.gromacs.archive import (
    GROMACS_ARCHIVE_SCHEMA_VERSION,
    write_gromacs_archive,
)
from biomodals.service.pending import PendingRequestStore
from biomodals.service.store import JobRecord
from biomodals.service.tool_runtime import PreparedResult


class GromacsToolAdapter:
    """Keep GROMACS request and ZIP behavior outside shared lifecycle."""

    def __init__(
        self,
        pending: PendingRequestStore,
        *,
        output_volume_name: str = "Gromacs-outputs",
    ) -> None:
        """Bind local staging to the established GROMACS output Volume."""
        self.pending = pending
        self.output_volume_name = output_volume_name

    async def stage(self, job: JobRecord) -> None:
        """Stage one immutable request and root launch identity."""
        content = self.pending.get(job.job_id)
        volume = self._volume(job)
        await volume.hydrate.aio()
        if content is None:
            await asyncio.to_thread(
                load_execution_request_from_volume,
                volume,
                job.job_id,
            )
            return
        request = GromacsExecutionRequest.from_bytes(content)
        await asyncio.to_thread(stage_execution_request, volume, job.job_id, request)
        await asyncio.to_thread(stage_execution_launch, volume, job.job_id, None)

    async def discard_pending(self, job: JobRecord) -> None:
        """Remove the local request after both remote files were verified."""
        self.pending.delete(job.job_id)

    async def continuation_source(
        self, job: JobRecord, deployment: DeploymentIdentity
    ) -> ContinuationInspection:
        """Inspect in Modal without transferring native files to the API host."""
        return await read_continuation_source(deployment, job.job_id)

    async def preflight(self, deployment: DeploymentIdentity) -> None:
        """Reject older targets before admitting a version-three scientific plan."""
        function = modal.Function.from_name(
            deployment.deployment_name,
            "prepare_continuation",
            environment_name=deployment.environment,
            version=deployment.deployment_version,
        )
        await function.hydrate.aio()

    async def prepare_result(
        self,
        job: JobRecord,
        cache: ArtifactCache,
        *,
        completed_at: int,
    ) -> PreparedResult:
        """Build the established deterministic end-user ZIP in local cache."""
        volume = self._volume(job)
        request = await asyncio.to_thread(
            load_execution_request_from_volume,
            volume,
            job.job_id,
        )
        marker_path = gromacs_publication_path(request, PREPARE_RESULT).as_posix()
        marker = await read_modal_volume_file(
            volume,
            marker_path,
            max_bytes=1024 * 1024,
        )
        published_files = parse_gromacs_publication(
            request,
            PREPARE_RESULT,
            marker,
        )
        if published_files is None:
            raise ArtifactIntegrityError("GROMACS final publication marker is invalid")
        path = cache.staging_path(str(job.job_id))
        try:
            with path.open("w+b") as raw:
                handle = cast("BinaryIO", raw)

                async def read_file(remote_path: str):
                    async for chunk in volume.read_file.aio(remote_path):
                        yield chunk

                remote_mtimes: dict[str, int] = {}
                for entry in await volume.listdir.aio(request.run_name):
                    remote_path = PurePosixPath(entry.path).as_posix().lstrip("/")
                    if type(entry.mtime) is not int:
                        raise ValueError("GROMACS output metadata is invalid")
                    remote_mtimes[remote_path] = entry.mtime
                # The per-submission interval bound is not a cumulative cap.
                parameters_json = orjson.dumps({
                    "simulation_time_ns": request.simulation_time_ns,
                    "run_pdbfixer": request.run_pdbfixer,
                    "cpu_only": request.cpu_only,
                }).decode()
                continuation = (
                    {
                        "source": request.continuation.model_dump(mode="json"),
                        "additional_time_ns": request.simulation_time_ns
                        - request.continuation.simulation_time_ns,
                        "target_time_ns": request.simulation_time_ns,
                        "trajectory_scope": "cumulative",
                        "equilibration_analysis": "inherited",
                        "production_mdp": "original input; extended TPR is authoritative",
                    }
                    if request.continuation
                    else None
                )
                built = await write_gromacs_archive(
                    handle,
                    run_name=request.file_stem,
                    remote_directory=request.run_name,
                    continuation=continuation,
                    parameters_json=parameters_json,
                    modal_app_name=job.modal_app_name,
                    modal_app_version=job.modal_app_version,
                    job_id=str(job.job_id),
                    stages_json=orjson.dumps(
                        job.projection.get("stages", []),
                        option=orjson.OPT_SORT_KEYS,
                    ).decode(),
                    started_at=job.created_at,
                    completed_at=completed_at,
                    read_file=read_file,
                    remote_mtimes=remote_mtimes,
                    expected_input_sha256=request.pdb_sha256,
                    published_files=published_files,
                    run_bounded=cache.run_bounded,
                )
            await cache.publish_staged(
                str(job.job_id),
                path,
                size_bytes=built.size_bytes,
                sha256=built.sha256,
            )
        finally:
            path.unlink(missing_ok=True)
        return PreparedResult(
            filename=f"{request.run_name}.zip",
            media_type="application/zip",
            size_bytes=built.size_bytes,
            sha256=built.sha256,
            archive_schema=f"gromacs/{GROMACS_ARCHIVE_SCHEMA_VERSION}",
        )

    def _volume(self, job: JobRecord) -> modal.Volume:
        return modal.Volume.from_name(
            self.output_volume_name,
            environment_name=job.modal_environment,
            version=2,
        )
