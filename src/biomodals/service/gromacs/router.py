"""Typed GROMACS service submission."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import zipfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any
from uuid import UUID, uuid4

import orjson
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import Response
from modal.exception import NotFoundError
from modal.exception import TimeoutError as ModalTimeoutError

from biomodals.app.bioinfo.gromacs.clustering import (
    ClusteringExecutionRequest,
    ClusteringSource,
)
from biomodals.app.bioinfo.gromacs.execution import concrete_gromacs_seed
from biomodals.app.bioinfo.gromacs.execution_runtime import GromacsExecutionRequest
from biomodals.app.bioinfo.gromacs.protein import NonProteinTemplateError
from biomodals.execution import DeploymentIdentity
from biomodals.helper.pdb import validate_pdb_content
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.gromacs.archive import (
    TrajectoryMetric,
    TrajectoryPlotTooLargeError,
    read_trajectory_plot,
)
from biomodals.service.gromacs.contracts import (
    MAX_SIMULATION_TIME_NS,
    GromacsClusteringInfo,
    GromacsClusteringSubmission,
    GromacsContinuationInfo,
    GromacsContinuationSubmission,
    GromacsJobOptions,
    gromacs_run_name,
)
from biomodals.service.gromacs.modal import GromacsToolAdapter
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    PrivateResultRoute,
    require_session,
    require_unsafe_session,
)
from biomodals.service.jobs import JobView
from biomodals.service.pending import PendingRequestStore
from biomodals.service.remote_execution import RemoteExecutionClient
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobOperation,
    JobRecord,
    JobState,
    ServiceStore,
    UserNotFoundError,
)

MAX_PDB_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
SOURCE_CHECK_TIMEOUT_SECONDS = 45
SOURCE_CHECK_TIMEOUT_DETAIL = (
    "Checking the source simulation timed out after 45 seconds. Please try again."
)


def _source_error_detail(error: Exception) -> str:
    if isinstance(error, FileNotFoundError) and error.filename:
        return (
            f"Required source file is unavailable: {PurePosixPath(error.filename).name}"
        )
    return "Retained checkpoint inputs are missing, invalid, or incompatible"


async def _read_pdb(upload: UploadFile, *, max_bytes: int) -> bytes:
    content = bytearray()
    try:
        while chunk := await upload.read(min(1024 * 1024, max_bytes + 1)):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise CodedAPIError(
                    413,
                    "payload_too_large",
                    f"PDB upload exceeds {max_bytes} bytes",
                )
    finally:
        await upload.close()
    try:
        validate_pdb_content(bytes(content), max_bytes=max_bytes)
    except ValueError as error:
        raise CodedAPIError(400, "pdb_invalid", str(error)) from error
    return bytes(content)


def create_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    pending: PendingRequestStore,
    remote: RemoteExecutionClient,
    cache: ArtifactCache,
    adapter: GromacsToolAdapter,
    max_pdb_bytes: int = MAX_PDB_BYTES,
) -> APIRouter:
    """Create the GROMACS endpoint around shared admission."""
    router = APIRouter(prefix="/api/v1/gromacs", tags=["gromacs"])
    previews = APIRouter(
        prefix="/jobs/{job_id}/trajectory", route_class=PrivateResultRoute
    )
    source_checks: dict[tuple[UUID, DeploymentIdentity, object], asyncio.Task[Any]] = {}

    def owner_job(job_id: UUID, session: AuthenticatedSession) -> JobRecord:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "gromacs":
            raise HTTPException(404, "Job not found")
        return job

    def admit(
        execution_request: GromacsExecutionRequest | ClusteringExecutionRequest,
        *,
        job_id: UUID,
        normalized_name: str,
        digest: str,
        session: AuthenticatedSession,
        idempotency_key: UUID,
        deployment: DeploymentIdentity,
        request: Request,
    ) -> JobView:
        effective = configuration.tool("gromacs")
        pending.put(job_id, execution_request.to_bytes())
        try:
            admission = store.admit_job(
                owner_user_id=session.principal.user_id,
                tool="gromacs",
                display_name=normalized_name,
                idempotency_key=str(idempotency_key),
                request_digest=digest,
                modal_environment=deployment.environment,
                modal_app_name=deployment.deployment_name,
                modal_app_version=deployment.deployment_version,
                tool_active_job_limit=effective.active_job_limit.value,
                global_active_job_limit=configuration.global_active_job_limit().value,
                max_active_provider_calls=execution_request.max_active_provider_calls,
                max_active_gpu_provider_calls=execution_request.max_active_gpu_provider_calls,
                now=int(time.time()),
                new_job_id=job_id,
                operation=(
                    JobOperation.TRAJECTORY_CLUSTERING
                    if isinstance(execution_request, ClusteringExecutionRequest)
                    else JobOperation.RUN
                ),
                source_job_id=(
                    execution_request.source.execution_run_id
                    if isinstance(execution_request, ClusteringExecutionRequest)
                    else None
                ),
            )
        except (IdempotencyConflictError, JobLimitExceededError) as error:
            pending.delete(job_id)
            raise CodedAPIError(409, "job_conflict", str(error)) from error
        except UserNotFoundError as error:
            pending.delete(job_id)
            raise CodedAPIError(403, "account_disabled", str(error)) from error
        if not admission.created:
            pending.delete(job_id)
        request.app.state.reconcile_wakeup.set()
        return _view(admission.job, session, configuration)

    async def preflight(deployment: DeploymentIdentity) -> None:
        await remote.preflight(deployment)
        try:
            await adapter.preflight(deployment)
        except NotFoundError as error:
            raise CodedAPIError(
                409,
                "deployment_incompatible",
                "Deploy and pin the updated GROMACS app before submitting jobs",
            ) from error

    async def inspect_source[T](
        job: JobRecord,
        deployment: DeploymentIdentity,
        *,
        read: Callable[[JobRecord, DeploymentIdentity], Awaitable[T]],
    ) -> T:
        """Share only in-flight checks, with one deadline independent of callers."""
        key = (job.job_id, deployment, read)
        task = source_checks.get(key)
        if task is None:

            async def inspect() -> T:
                try:
                    async with asyncio.timeout(SOURCE_CHECK_TIMEOUT_SECONDS):
                        return await read(job, deployment)
                finally:
                    source_checks.pop(key, None)

            task = asyncio.create_task(inspect())
            source_checks[key] = task
            # Observe failures even when every waiting HTTP request disconnects.
            task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
        try:
            return await asyncio.shield(task)
        except NotFoundError as error:
            raise CodedAPIError(
                409,
                "deployment_incompatible",
                "Deploy and pin the updated GROMACS app before reusing completed simulations",
            ) from error
        except ModalTimeoutError as error:
            raise CodedAPIError(
                504,
                "source_check_timeout",
                "Remote source check timed out. Please try again.",
            ) from error
        except TimeoutError as error:
            raise CodedAPIError(
                504, "source_check_timeout", SOURCE_CHECK_TIMEOUT_DETAIL
            ) from error

    @router.get(
        "/jobs/{job_id}/continuation",
        response_model=GromacsContinuationInfo,
        responses={
            409: {"model": CodedErrorResponse},
            504: {"model": CodedErrorResponse},
        },
    )
    async def continuation_info(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> GromacsContinuationInfo:
        job = owner_job(job_id, session)
        info = GromacsContinuationInfo(
            source_job_id=job_id,
            source_display_name=job.display_name,
            simulation_time_ns=None,
            cpu_only=None,
            parent_job_id=None,
            eligible=False,
            code="source_not_completed",
            detail="Only completed GROMACS jobs can be continued",
        )
        if job.state != JobState.SUCCEEDED or job.operation != JobOperation.RUN:
            return info
        if job.modal_environment != configuration.modal_environment().value:
            info.code = "source_environment_mismatch"
            info.detail = "Source and target must use the same Modal environment"
            return info
        effective = configuration.tool("gromacs")
        deployment = DeploymentIdentity(
            job.modal_environment,
            effective.modal_app_name,
            effective.modal_app_version.value,
        )
        try:
            saved = await inspect_source(
                job, deployment, read=adapter.continuation_source
            )
        except (FileNotFoundError, ValueError, TypeError) as error:
            info.code = "source_unavailable"
            info.detail = _source_error_detail(error)
            return info
        info.simulation_time_ns = saved.source.simulation_time_ns
        info.cpu_only = saved.cpu_only
        info.parent_job_id = saved.parent_job_id
        info.eligible = True
        info.code = None
        info.detail = "Native inputs are available; the new job verifies checksums and checkpoint completion before production"
        return info

    @router.post(
        "/jobs/{job_id}/continue",
        response_model=JobView,
        status_code=202,
        responses={
            409: {"model": CodedErrorResponse},
            504: {"model": CodedErrorResponse},
        },
    )
    async def continue_job(
        job_id: UUID,
        submission: GromacsContinuationSubmission,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
    ) -> JobView:
        job = owner_job(job_id, session)
        normalized_name = (
            submission.display_name.strip() if submission.display_name else ""
        )
        normalized_name = normalized_name or f"{job.display_name[:100]} (continued)"
        digest = hashlib.sha256(
            orjson.dumps(
                {
                    "source_job_id": str(job_id),
                    "display_name": normalized_name,
                    "additional_time_ns": submission.additional_time_ns,
                    "cpu_only": submission.cpu_only,
                },
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest()
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="gromacs",
            idempotency_key=str(idempotency_key),
        )
        if replay is not None:
            if replay.request_digest != digest:
                raise CodedAPIError(
                    409,
                    "idempotency_conflict",
                    "Idempotency key was already used for another request",
                )
            return _view(replay, session, configuration)
        if job.state != JobState.SUCCEEDED or job.operation != JobOperation.RUN:
            raise CodedAPIError(
                409, "source_not_completed", "Source job must be completed"
            )
        if job.modal_environment != configuration.modal_environment().value:
            raise CodedAPIError(
                409,
                "source_environment_mismatch",
                "Source and target must use the same Modal environment",
            )
        effective = configuration.tool("gromacs")
        deployment = DeploymentIdentity(
            job.modal_environment,
            effective.modal_app_name,
            effective.modal_app_version.value,
        )
        try:
            inspection = await inspect_source(
                job, deployment, read=adapter.continuation_source
            )
        except (FileNotFoundError, ValueError, TypeError) as error:
            raise CodedAPIError(
                409,
                "source_unavailable",
                _source_error_detail(error),
            ) from error
        await preflight(deployment)
        child_id = uuid4()
        child = inspection.request(
            run_name=gromacs_run_name(normalized_name, child_id),
            additional_time_ns=submission.additional_time_ns,
            cpu_only=submission.cpu_only,
            max_active_provider_calls=effective.max_active_provider_calls,
            max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
        )
        return admit(
            child,
            job_id=child_id,
            normalized_name=normalized_name,
            digest=digest,
            session=session,
            idempotency_key=idempotency_key,
            deployment=deployment,
            request=request,
        )

    def clustering_deployment(job: JobRecord) -> DeploymentIdentity:
        if job.state != JobState.SUCCEEDED or job.operation != JobOperation.RUN:
            raise CodedAPIError(
                409, "source_not_completed", "Select a completed GROMACS simulation"
            )
        environment = configuration.modal_environment().value
        if job.modal_environment != environment:
            raise CodedAPIError(
                409,
                "source_environment_mismatch",
                "Source simulation belongs to a different environment",
            )
        effective = configuration.tool("gromacs")
        return DeploymentIdentity(
            environment, effective.modal_app_name, effective.modal_app_version.value
        )

    async def cluster_source(
        job: JobRecord, deployment: DeploymentIdentity
    ) -> ClusteringSource:
        try:
            source = await inspect_source(
                job, deployment, read=adapter.clustering_source
            )
            if source.execution_run_id != job.job_id:
                raise ValueError("Clustering source identity does not match the Job")
            return source
        except NonProteinTemplateError as error:
            raise CodedAPIError(409, "source_unavailable", str(error)) from error
        except (FileNotFoundError, ValueError, TypeError, KeyError) as error:
            raise CodedAPIError(
                409,
                "source_unavailable",
                "Retained processed trajectory or matching protein template is unavailable or invalid",
            ) from error

    @router.get(
        "/jobs/{job_id}/clustering",
        response_model=GromacsClusteringInfo,
        responses={
            409: {"model": CodedErrorResponse},
            504: {"model": CodedErrorResponse},
        },
    )
    async def clustering_info(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> GromacsClusteringInfo:
        job = owner_job(job_id, session)
        try:
            deployment = clustering_deployment(job)
            source = await cluster_source(job, deployment)
        except CodedAPIError as error:
            if error.code not in {
                "source_not_completed",
                "source_unavailable",
                "source_environment_mismatch",
            }:
                raise
            return GromacsClusteringInfo(
                source_job_id=job_id,
                source_display_name=job.display_name,
                eligible=False,
                code=error.code,
                detail=error.detail,
            )
        return GromacsClusteringInfo(
            source_job_id=job_id,
            source_display_name=job.display_name,
            eligible=True,
            detail="Clusters the complete processed production trajectory without additional simulation.",
            frame_count=source.frame_count,
            protein_atoms=source.protein_atoms,
            ca_atoms=source.ca_atoms,
            estimated_memory_bytes=source.estimated_memory_bytes,
            warnings=source.warnings,
        )

    @router.post(
        "/jobs/{job_id}/clustering",
        response_model=JobView,
        status_code=202,
        responses={
            409: {"model": CodedErrorResponse},
            504: {"model": CodedErrorResponse},
        },
    )
    async def submit_clustering(
        job_id: UUID,
        body: GromacsClusteringSubmission,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
    ) -> JobView:
        job = owner_job(job_id, session)
        name = (
            body.display_name or ""
        ).strip() or f"{job.display_name[:100]} (clusters)"
        digest = hashlib.sha256(
            orjson.dumps(
                {
                    "operation": "trajectory_clustering",
                    "source_job_id": str(job_id),
                    "display_name": name,
                    "cutoff_angstrom": body.cutoff_angstrom,
                },
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest()
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="gromacs",
            idempotency_key=str(idempotency_key),
        )
        if replay is not None:
            if replay.request_digest != digest:
                raise CodedAPIError(
                    409,
                    "idempotency_conflict",
                    "Idempotency key was already used for another request",
                )
            return _view(replay, session, configuration)
        deployment = clustering_deployment(job)
        await remote.preflight(deployment)
        try:
            await adapter.clustering_preflight(deployment)
        except NotFoundError as error:
            raise CodedAPIError(
                409,
                "deployment_incompatible",
                "Deploy and pin a GROMACS version supporting trajectory clustering",
            ) from error
        source = await cluster_source(job, deployment)
        child_id = uuid4()
        effective = configuration.tool("gromacs")
        child = ClusteringExecutionRequest(
            run_name=gromacs_run_name(name, child_id),
            source=source,
            cutoff_angstrom=body.cutoff_angstrom,
            max_active_provider_calls=effective.max_active_provider_calls,
            max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
        )
        return admit(
            child,
            job_id=child_id,
            normalized_name=name,
            digest=digest,
            session=session,
            idempotency_key=idempotency_key,
            deployment=deployment,
            request=request,
        )

    @previews.get(
        "/{metric}.png",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "image/png": {"schema": {"type": "string", "format": "binary"}}
                }
            },
            409: {"model": CodedErrorResponse},
            413: {"model": CodedErrorResponse},
        },
    )
    async def trajectory_plot(
        job_id: UUID,
        metric: TrajectoryMetric,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> Response:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "gromacs":
            raise HTTPException(404, "Job not found")
        if (
            job.state != JobState.SUCCEEDED
            or job.operation != JobOperation.RUN
            or job.result_size_bytes is None
            or job.result_sha256 is None
        ):
            raise CodedAPIError(409, "result_not_ready", "Result is not ready")
        lease = await cache.acquire_async(
            str(job_id), size_bytes=job.result_size_bytes, sha256=job.result_sha256
        )
        if lease is None:
            raise CodedAPIError(
                409,
                "result_not_cached",
                "Prepare the result download before opening these plots",
            )
        try:
            content = await cache.run_bounded(read_trajectory_plot, lease, metric)
        except TrajectoryPlotTooLargeError as error:
            raise CodedAPIError(
                413,
                "trajectory_plot_too_large",
                "Plot exceeds the preview limit; use the archive download",
            ) from error
        except (KeyError, ValueError, TypeError, OSError, zipfile.BadZipFile) as error:
            raise CodedAPIError(
                409,
                "result_invalid",
                "Production trajectory plot is unavailable or invalid",
            ) from error
        finally:
            lease.close()
        return Response(
            content,
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="production-{metric}.png"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    router.include_router(previews)

    @router.post(
        "/jobs",
        response_model=JobView,
        response_description="Job durably admitted for asynchronous staging and launch",
        status_code=202,
    )
    async def submit_job(
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
        pdb: Annotated[UploadFile, File(description="Input PDB structure")],
        display_name: Annotated[str | None, Form(max_length=120)] = None,
        simulation_time_ns: Annotated[int, Form(ge=1, le=MAX_SIMULATION_TIME_NS)] = 5,
        run_pdbfixer: Annotated[bool, Form()] = False,
        cpu_only: Annotated[bool, Form()] = False,
    ) -> JobView:
        pdb_content = await _read_pdb(pdb, max_bytes=max_pdb_bytes)
        options = GromacsJobOptions(
            simulation_time_ns=simulation_time_ns,
            run_pdbfixer=run_pdbfixer,
            cpu_only=cpu_only,
        )
        normalized_name = _display_name(pdb.filename, display_name)
        digest_name = (
            f"supplied:{normalized_name}"
            if display_name is not None and display_name.strip()
            else f"generated:{_filename_stem(pdb.filename)}"
        )
        digest = _submission_digest(pdb_content, digest_name, options)
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="gromacs",
            idempotency_key=str(idempotency_key),
        )
        if replay is not None:
            if replay.request_digest != digest:
                raise CodedAPIError(
                    409,
                    "idempotency_conflict",
                    "Idempotency key was already used for another request",
                )
            return _view(replay, session, configuration)

        effective = configuration.tool("gromacs")
        deployment = DeploymentIdentity(
            configuration.modal_environment().value,
            effective.modal_app_name,
            effective.modal_app_version.value,
        )
        await preflight(deployment)
        job_id = uuid4()
        run_name = gromacs_run_name(normalized_name, job_id)
        execution_request = GromacsExecutionRequest(
            run_name=run_name,
            pdb_content=pdb_content,
            simulation_time_ns=options.simulation_time_ns,
            run_pdbfixer=options.run_pdbfixer,
            cpu_only=options.cpu_only,
            num_threads=16,
            use_openmp_threads=False,
            ld_seed=concrete_gromacs_seed(
                -1,
                run_identity=str(job_id),
                purpose="ld-seed",
            ),
            gen_seed=concrete_gromacs_seed(
                -1,
                run_identity=str(job_id),
                purpose="gen-seed",
            ),
            genion_seed=concrete_gromacs_seed(
                0,
                run_identity=str(job_id),
                purpose="genion-seed",
                random_sentinel=0,
            ),
            max_active_provider_calls=effective.max_active_provider_calls,
            max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
        )
        return admit(
            execution_request,
            job_id=job_id,
            normalized_name=normalized_name,
            digest=digest,
            session=session,
            idempotency_key=idempotency_key,
            deployment=deployment,
            request=request,
        )

    return router


def _submission_digest(
    content: bytes,
    display_name: str,
    options: GromacsJobOptions,
) -> str:
    digest = hashlib.sha256(content)
    digest.update(options.model_dump_json().encode())
    digest.update(orjson.dumps({"display_name": display_name}))
    return digest.hexdigest()


def _display_name(filename: str | None, supplied: str | None) -> str:
    if supplied is not None and supplied.strip():
        return re.sub(r"\s+", " ", supplied).strip()
    return f"{_filename_stem(filename)} · {datetime.now(UTC):%Y-%m-%d}"


def _filename_stem(filename: str | None) -> str:
    """Return the stable generated-name component used for request identity."""
    safe_filename = (filename or "gromacs").replace("\\", "/")
    stem = PurePosixPath(safe_filename).stem.strip() or "gromacs"
    return re.sub(r"\s+", " ", stem)[:100]


def _view(
    job,
    session: AuthenticatedSession,
    configuration: RuntimeConfiguration,
) -> JobView:
    return JobView.from_record(
        job,
        can_view_logs=(
            session.principal.is_admin
            or configuration.tool("gromacs").job_logs_visible_to_owner.value
        ),
    )
