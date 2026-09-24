"""Authenticated local preparation before explicit scientific Job submission."""

from __future__ import annotations

import asyncio
import hashlib
import time
import zipfile
from io import BytesIO
from typing import IO, Annotated
from uuid import UUID, uuid4

import polars as pl
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import Response
from modal.exception import NotFoundError
from pydantic import TypeAdapter

from biomodals.execution import DeploymentIdentity
from biomodals.helper.antibody import GermlineAssignment
from biomodals.helper.antibody_tables import (
    GERMLINE_ASSIGNMENT_SCHEMA,
    GERMLINE_TABLE_SCHEMA,
)
from biomodals.service.antibody_sequence_analysis.reference import present_germlines
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
from biomodals.service.nanobody_humanization.contracts import (
    NanobodyInputErrors,
    NanobodyOptions,
    NanobodyPreparation,
    NanobodyPreparationRequest,
    NanobodySubmission,
    PreparedVHPreview,
    RetainedNanobodyInputs,
)
from biomodals.service.nanobody_humanization.modal import NanobodyToolAdapter
from biomodals.service.nanobody_humanization.results import (
    SELECTION_SCHEMA,
    NanobodySelectionPage,
    query_selection,
)
from biomodals.service.pending import PendingRequestStore
from biomodals.service.remote_execution import RemoteExecutionClient
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobRecord,
    ServiceStore,
    UserNotFoundError,
)
from biomodals.service.table_archive import read_cached_selection
from biomodals.workflow.nanobody_humanization.execution import NanobodyExecutionRequest
from biomodals.workflow.nanobody_humanization.preparation import (
    VHInput,
    preparation_digest,
    prepare_batch,
)


def create_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    pending: PendingRequestStore,
    remote: RemoteExecutionClient,
    cache: ArtifactCache,
    adapter: NanobodyToolAdapter,
    max_parents: int = 100,
) -> APIRouter:
    """Register preparation, admission and owner reads around the shared Job lifecycle."""
    if not 1 <= max_parents <= 200:
        raise ValueError("Nanobody max_parents must be between 1 and 200")
    router = APIRouter(
        prefix="/api/v1/nanobody-humanization",
        tags=["nanobody-humanization"],
        route_class=PrivateResultRoute,
    )

    @router.get("/options", response_model=NanobodyOptions)
    async def options(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> NanobodyOptions:
        return NanobodyOptions(max_parents=max_parents)

    @router.post(
        "/prepare",
        response_model=NanobodyPreparation,
        responses={503: {"model": CodedErrorResponse}},
    )
    async def prepare(
        body: NanobodyPreparationRequest,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> NanobodyPreparation:
        if len(body.parents) > max_parents:
            raise CodedAPIError(
                422,
                "batch_too_large",
                f"At most {max_parents} parents are allowed per job",
            )
        parents, issues = await request.app.state.antibody_analysis.run(
            prepare_batch, body.parents
        )
        return NanobodyPreparation.from_prepared(body.parents, parents, issues)

    def owned_job(job_id: UUID, session: AuthenticatedSession) -> JobRecord:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "nanobody_humanization":
            raise HTTPException(404, "Job not found")
        return job

    def view(job: JobRecord, session: AuthenticatedSession) -> JobView:
        return JobView.from_record(
            job,
            can_view_logs=session.principal.is_admin
            or configuration.tool(
                "nanobody_humanization"
            ).job_logs_visible_to_owner.value,
        )

    @router.post(
        "/jobs",
        response_model=JobView,
        status_code=202,
        responses={
            422: {"model": NanobodyInputErrors | CodedErrorResponse},
            409: {"model": CodedErrorResponse},
            503: {"model": CodedErrorResponse},
        },
    )
    async def submit(
        body: NanobodySubmission,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
    ) -> JobView | Response:
        digest = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="nanobody_humanization",
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
        if len(body.parents) > max_parents:
            raise CodedAPIError(
                422,
                "batch_too_large",
                f"At most {max_parents} parents are allowed per job",
            )
        try:
            body.settings.validate_budget(len(body.parents))
        except ValueError as error:
            raise CodedAPIError(
                422, "exploration_budget_exceeded", str(error)
            ) from error
        parents, issues = await request.app.state.antibody_analysis.run(
            prepare_batch, body.parents
        )
        if issues:
            return Response(
                NanobodyInputErrors(errors=issues).model_dump_json(),
                status_code=422,
                media_type="application/json",
            )
        if preparation_digest(parents) != body.preparation_digest:
            raise CodedAPIError(
                409,
                "preparation_changed",
                "Inputs or preparation have changed. Prepare and review this batch again.",
            )
        effective = configuration.tool("nanobody_humanization")
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
                "Deploy and pin the updated nanobody workflow before submitting jobs.",
            ) from error
        job_id = uuid4()
        execution_request = NanobodyExecutionRequest(
            run_name=f"nanobody-{job_id}",
            parents=parents,
            settings=body.settings,
            max_active_provider_calls=effective.max_active_provider_calls,
            max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
        )
        await asyncio.to_thread(pending.put, job_id, execution_request.to_bytes())
        try:
            admission = store.admit_job(
                owner_user_id=session.principal.user_id,
                tool="nanobody_humanization",
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

    @router.get("/jobs/{job_id}/inputs", response_model=RetainedNanobodyInputs)
    async def inputs(
        job_id: UUID, session: Annotated[AuthenticatedSession, Depends(require_session)]
    ) -> RetainedNanobodyInputs:
        job = owned_job(job_id, session)
        try:
            retained = await adapter.input_request(job)
        except FileNotFoundError as error:
            raise CodedAPIError(
                404, "job_input_unavailable", "Nanobody input is no longer available"
            ) from error
        return RetainedNanobodyInputs(
            display_name=job.display_name,
            settings=retained.settings,
            parents=[
                VHInput(id=parent.id, vhh=parent.original_sequence)
                for parent in retained.parents
            ],
            prepared_parents=[
                PreparedVHPreview(row_index=index, id=parent.id, vh=parent.sequence)
                for index, parent in enumerate(retained.parents)
            ],
            preparation_version=retained.parents[0].preparation_version,
        )

    @router.get("/jobs/{job_id}/selection", response_model=NanobodySelectionPage)
    async def selection(
        job_id: UUID,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        parent_id: Annotated[str | None, Query(max_length=200)] = None,
        sort_by: str | None = None,
        descending: bool = False,
    ) -> NanobodySelectionPage:
        if sort_by is not None and sort_by not in SELECTION_SCHEMA:
            raise CodedAPIError(422, "sort_invalid", "Unknown selection column")

        def read_page(source: IO[bytes], archive: zipfile.ZipFile):
            page = query_selection(
                source,
                offset=offset,
                limit=limit,
                parent_id=parent_id,
                sort_by=sort_by,
                descending=descending,
            )
            member = archive.getinfo("germlines.parquet")
            if member.file_size > 64 * 1024 * 1024:
                raise ValueError("Germline evidence exceeds its byte limit")
            ids = [row["candidate_id"] for row in page.rows]
            evidence = (
                pl
                .scan_parquet(BytesIO(archive.read(member)))
                .filter(
                    pl.col("candidate_id").is_in(
                        pl.Series(ids, dtype=pl.String).implode()
                    )
                )
                .collect()
            )
            if evidence.schema != pl.Schema(GERMLINE_TABLE_SCHEMA):
                raise ValueError("Unexpected germline evidence schema")
            expected = {
                row["candidate_id"]: (
                    row["parent_id"],
                    hashlib.sha256(str(row["vh"]).encode()).hexdigest(),
                )
                for row in page.rows
            }
            assignments = {}
            assignment_type = TypeAdapter(GermlineAssignment)
            for row in evidence.iter_rows(named=True):
                if row["chain"] != "vh" or expected.pop(row["candidate_id"], None) != (
                    row["parent_id"],
                    row["sequence_sha256"],
                ):
                    raise ValueError("Germline evidence changed candidate identity")
                assignments[row["candidate_id"]] = assignment_type.validate_python({
                    name: row[name] for name in GERMLINE_ASSIGNMENT_SCHEMA
                })
            if any(
                row["candidate_id"] in expected and row["annotation_error"] is None
                for row in page.rows
            ):
                raise ValueError(
                    "Germline evidence does not cover annotated candidates"
                )
            return page, assignments

        page, assignments = await read_cached_selection(
            owned_job(job_id, session), cache, read_page
        )
        (
            snapshot,
            page.reference,
        ) = await request.app.state.antibody_analysis.reference_snapshot()
        page.germlines = {
            key: present_germlines(assignment, "vh", snapshot)
            for key, assignment in assignments.items()
        }
        return page

    @router.get(
        "/jobs/{job_id}/selection.csv",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "text/csv": {"schema": {"type": "string", "format": "binary"}}
                }
            }
        },
    )
    async def selection_download(
        job_id: UUID, session: Annotated[AuthenticatedSession, Depends(require_session)]
    ) -> Response:
        return Response(
            await read_cached_selection(
                owned_job(job_id, session), cache, lambda source, archive: source.read()
            ),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="selection.csv"'},
        )

    return router
