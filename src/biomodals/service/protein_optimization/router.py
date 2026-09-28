"""Authenticated CSV-first local review, independent of scientific admission."""

import asyncio
import hashlib
import sqlite3
import time
from typing import Annotated, Literal, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from modal.exception import NotFoundError
from starlette.background import BackgroundTask

from biomodals.execution import DeploymentIdentity
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    PrivateResultRoute,
    require_session,
    require_unsafe_session,
)
from biomodals.service.jobs import JobView
from biomodals.service.pending import PendingRequestStore
from biomodals.service.protein_optimization.contracts import (
    OptimizationCandidatePage,
    OptimizationDownloadTicket,
    OptimizationReview,
    OptimizationReviewRequest,
    OptimizationSelectedCandidates,
    OptimizationSubmission,
    ProteinOptimizationOptions,
    RetainedOptimizationInputs,
)
from biomodals.service.protein_optimization.downloads import (
    load_selection,
    prepare_selection,
)
from biomodals.service.protein_optimization.modal import ProteinOptimizationAdapter
from biomodals.service.protein_optimization.results import (
    query_candidates,
    selected_csv,
)
from biomodals.service.protein_optimization.review import review_inputs
from biomodals.service.remote_execution import RemoteExecutionClient
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobRecord,
    JobState,
    ServiceStore,
    UserNotFoundError,
)
from biomodals.workflow.protein_optimization.design import OptimizationDesign
from biomodals.workflow.protein_optimization.execution import (
    OptimizationExecutionRequest,
)


def create_review_router() -> APIRouter:
    """Expose options and explicit review without a Job or provider invocation."""
    router = APIRouter(
        prefix="/api/v1/protein-optimization",
        tags=["protein-optimization"],
        route_class=PrivateResultRoute,
    )

    @router.get("/options", response_model=ProteinOptimizationOptions)
    def options(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> ProteinOptimizationOptions:
        return ProteinOptimizationOptions()

    @router.post(
        "/review",
        response_model=OptimizationReview,
        responses={422: {"model": CodedErrorResponse}},
    )
    def review(
        body: OptimizationReviewRequest,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> OptimizationReview:
        try:
            return review_inputs(body)
        except ValueError as exc:
            raise CodedAPIError(422, "invalid_measurements", str(exc)) from exc

    return router


def create_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    pending: PendingRequestStore,
    remote: RemoteExecutionClient,
    cache: ArtifactCache,
    adapter: ProteinOptimizationAdapter,
) -> APIRouter:
    """Add owner-scoped admission and CSV views to the existing Job lifecycle."""
    router = APIRouter(
        prefix="/api/v1/protein-optimization",
        tags=["protein-optimization"],
        route_class=PrivateResultRoute,
    )

    def owned(job_id: UUID, session: AuthenticatedSession) -> JobRecord:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "protein_optimization":
            raise HTTPException(404, "Job not found")
        return job

    def view(job: JobRecord, session: AuthenticatedSession) -> JobView:
        return JobView.from_record(
            job,
            can_view_logs=session.principal.is_admin
            or configuration.tool(
                "protein_optimization"
            ).job_logs_visible_to_owner.value,
        )

    @router.post(
        "/jobs",
        response_model=JobView,
        status_code=202,
        responses={
            409: {"model": CodedErrorResponse},
            422: {"model": CodedErrorResponse},
        },
    )
    async def submit(
        body: OptimizationSubmission,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
    ) -> JobView:
        digest = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="protein_optimization",
            idempotency_key=str(idempotency_key),
        )
        if replay is not None:
            if replay.request_digest != digest:
                raise CodedAPIError(
                    409,
                    "idempotency_conflict",
                    "Idempotency key was already used for another request",
                )
            return view(replay, session)
        reviewed = OptimizationReviewRequest(
            measurements_csv=body.measurements_csv,
            parental_fasta=body.parental_fasta,
            settings=body.settings,
        )
        try:
            review = await asyncio.to_thread(review_inputs, reviewed)
            design = await asyncio.to_thread(
                OptimizationDesign,
                measurements_csv=body.measurements_csv,
                parental_fasta=body.parental_fasta,
                settings=body.settings,
            )
        except ValueError as error:
            raise CodedAPIError(
                422,
                "invalid_design",
                "Review and correct the input measurements and design settings before submission",
            ) from error
        if review.errors:
            raise CodedAPIError(
                422,
                "invalid_design",
                "Review and correct the input measurements and design settings before submission",
            )
        if review.review_digest != body.review_digest:
            raise CodedAPIError(
                409,
                "review_changed",
                "Inputs or settings changed. Review this design again before submitting.",
            )
        effective = configuration.tool("protein_optimization")
        deployment = DeploymentIdentity(
            configuration.modal_environment().value,
            effective.modal_app_name,
            effective.modal_app_version.value,
        )
        await remote.preflight(deployment)
        try:
            await adapter.preflight(deployment)
        except NotFoundError as error:
            raise CodedAPIError(
                409,
                "deployment_incompatible",
                "Deploy and pin the protein optimization workflow before submitting jobs.",
            ) from error
        job_id = uuid4()
        execution = OptimizationExecutionRequest(
            run_name=f"protein-{job_id}",
            design=design,
            max_active_provider_calls=effective.max_active_provider_calls,
            max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
        )
        await asyncio.to_thread(pending.put, job_id, execution.to_bytes())
        try:
            admission = store.admit_job(
                owner_user_id=session.principal.user_id,
                tool="protein_optimization",
                display_name=body.display_name,
                idempotency_key=str(idempotency_key),
                request_digest=digest,
                modal_environment=deployment.environment,
                modal_app_name=deployment.deployment_name,
                modal_app_version=deployment.deployment_version,
                tool_active_job_limit=effective.active_job_limit.value,
                global_active_job_limit=configuration.global_active_job_limit().value,
                max_active_provider_calls=effective.max_active_provider_calls,
                max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
                now=int(time.time()),
                new_job_id=job_id,
            )
        except (IdempotencyConflictError, JobLimitExceededError) as error:
            pending.delete(job_id)
            raise CodedAPIError(409, "job_conflict", str(error)) from error
        except UserNotFoundError as error:
            pending.delete(job_id)
            raise CodedAPIError(403, "account_disabled", str(error)) from error
        except BaseException:
            pending.delete(job_id)
            raise
        if not admission.created:
            pending.delete(job_id)
        request.app.state.reconcile_wakeup.set()
        return view(admission.job, session)

    @router.get(
        "/jobs/{job_id}/inputs",
        response_model=RetainedOptimizationInputs,
        responses={404: {"model": CodedErrorResponse}},
    )
    async def inputs(
        job_id: UUID, session: Annotated[AuthenticatedSession, Depends(require_session)]
    ) -> RetainedOptimizationInputs:
        job = owned(job_id, session)
        try:
            retained = await adapter.input_request(job)
        except FileNotFoundError as error:
            raise CodedAPIError(
                404,
                "job_input_unavailable",
                "Original protein optimization inputs are unavailable",
            ) from error
        return RetainedOptimizationInputs(
            **retained.design.model_dump(), display_name=job.display_name
        )

    async def acquire(job: JobRecord):
        if (
            job.state not in {JobState.SUCCEEDED, JobState.PARTIAL}
            or job.result_size_bytes is None
            or job.result_sha256 is None
        ):
            raise CodedAPIError(409, "result_not_ready", "Result is not ready")
        lease = await cache.acquire_async(
            str(job.job_id), size_bytes=job.result_size_bytes, sha256=job.result_sha256
        )
        if lease is None:
            raise CodedAPIError(
                409,
                "result_not_cached",
                "Prepare the result download before opening candidates",
            )
        if not cache.derived_path(str(job.job_id)).is_file():
            lease.close()
            await cache.discard_async(str(job.job_id))
            raise CodedAPIError(
                409,
                "result_not_cached",
                "Prepare the result download to restore candidate queries",
            )
        return lease

    @router.get(
        "/jobs/{job_id}/candidates",
        response_model=OptimizationCandidatePage,
        responses={409: {"model": CodedErrorResponse}},
    )
    async def candidates(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        sort_by: Literal[
            "id", "mutations", "predicted_label", "n_mutations", "n_new_mutations"
        ]
        | None = None,
        descending: bool = False,
        mutations: Annotated[str | None, Query(max_length=200)] = None,
        n_mutations: Annotated[int | None, Query(ge=1)] = None,
        n_new_mutations: Annotated[int | None, Query(ge=0)] = None,
    ) -> OptimizationCandidatePage:
        job = owned(job_id, session)
        lease = await acquire(job)
        try:
            return await cache.run_bounded(
                query_candidates,
                cache.derived_path(str(job_id)),
                job.result_sha256,
                offset=offset,
                limit=limit,
                sort_by=sort_by,
                descending=descending,
                mutations=mutations,
                n_mutations=n_mutations,
                n_new_mutations=n_new_mutations,
            )
        except (ValueError, sqlite3.Error) as error:
            raise CodedAPIError(
                409, "result_invalid", "Candidate projection is unavailable or invalid"
            ) from error
        finally:
            lease.close()

    @router.post(
        "/jobs/{job_id}/prepare-selected-download",
        response_model=OptimizationDownloadTicket,
        responses={
            409: {"model": CodedErrorResponse},
            422: {"model": CodedErrorResponse},
        },
    )
    async def prepare_selected_download(
        job_id: UUID,
        body: OptimizationSelectedCandidates,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> OptimizationDownloadTicket:
        job = owned(job_id, session)
        lease = await acquire(job)
        try:
            return await cache.run_bounded(
                prepare_selection,
                cache.download_request_path(str(job_id)),
                cache.derived_path(str(job_id)),
                cast(str, job.result_sha256),
                body.ids,
                owner_user_id=session.principal.user_id,
                job_id=job_id,
                now=int(time.time()),
            )
        except ValueError as error:
            raise CodedAPIError(422, "selection_invalid", str(error)) from error
        finally:
            lease.close()

    @router.get(
        "/jobs/{job_id}/candidates.csv",
        response_class=StreamingResponse,
        responses={
            200: {
                "content": {
                    "text/csv": {"schema": {"type": "string", "format": "binary"}}
                }
            },
            409: {"model": CodedErrorResponse},
            410: {"model": CodedErrorResponse},
        },
    )
    async def download_selected(
        job_id: UUID,
        ticket: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> StreamingResponse:
        job = owned(job_id, session)
        lease = await acquire(job)
        try:
            ids = await cache.run_bounded(
                load_selection,
                cache.download_request_path(str(job_id)),
                token=ticket,
                owner_user_id=session.principal.user_id,
                job_id=job_id,
                csv_sha256=cast(str, job.result_sha256),
                now=int(time.time()),
            )
        except (FileNotFoundError, ValueError) as error:
            lease.close()
            raise CodedAPIError(
                410,
                "selection_expired",
                "Selected download expired or was replaced. Prepare it again.",
            ) from error
        stream = selected_csv(
            cache.derived_path(str(job_id)), cast(str, job.result_sha256), ids
        )

        def close():
            stream.close()
            lease.close()

        try:
            first = await cache.run_bounded(next, stream)
        except ValueError as error:
            await cache.run_bounded(close)
            raise CodedAPIError(422, "selection_invalid", str(error)) from error
        except BaseException:
            await cache.run_bounded(close)
            raise

        async def content():
            try:
                yield first
                while chunk := await cache.run_bounded(next, stream, b""):
                    yield chunk
            finally:
                await cache.run_bounded(close)

        return StreamingResponse(
            content(),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="selected-candidates.csv"'
            },
            background=BackgroundTask(cache.run_bounded, close),
        )

    return router
