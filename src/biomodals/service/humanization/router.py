"""Humanization submission and owner-scoped, bounded selection-table reads."""

from __future__ import annotations

import asyncio
import hashlib
import time
import zipfile
from collections.abc import Callable
from functools import partial
from typing import IO, Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import Response

from biomodals.execution import DeploymentIdentity
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    require_session,
    require_unsafe_session,
)
from biomodals.service.humanization.contracts import (
    HumanizationOptions,
    HumanizationSubmission,
    InputErrors,
    InputIssue,
    validate_pairs,
)
from biomodals.service.humanization.modal import HumanizationToolAdapter
from biomodals.service.humanization.results import (
    SELECTION_SCHEMA,
    SelectionPage,
    query_selection,
)
from biomodals.service.jobs import JobView
from biomodals.service.pending import PendingRequestStore
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
from biomodals.workflow.humanization.settings import HumanizationSettings

MAX_SELECTION_BYTES = 32 * 1024 * 1024


def create_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    pending: PendingRequestStore,
    remote: RemoteExecutionClient,
    cache: ArtifactCache,
    adapter: HumanizationToolAdapter,
    max_pairs: int = 100,
) -> APIRouter:
    """Reuse durable admission and result cache without another job scheduler."""
    if not 1 <= max_pairs <= 1000:
        raise ValueError("Humanization max_pairs must be between 1 and 1000")
    router = APIRouter(prefix="/api/v1/humanization", tags=["humanization"])

    def view(job: JobRecord, session: AuthenticatedSession) -> JobView:
        return JobView.from_record(
            job,
            can_view_logs=(
                session.principal.is_admin
                or configuration.tool("humanization").job_logs_visible_to_owner.value
            ),
        )

    @router.get("/options", response_model=HumanizationOptions)
    async def options(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> HumanizationOptions:
        from biomodals.app.design.humatch.app import VH_FAMILIES, VL_FAMILIES
        from biomodals.app.design.sapiens.app import MAX_VH_LENGTH, MAX_VL_LENGTH

        schema = HumanizationSettings.model_json_schema()
        properties = schema["properties"]
        properties["humatch_vh_target_family"]["enum"] = ["auto", *VH_FAMILIES]
        properties["humatch_vl_target_family"]["enum"] = ["auto", *VL_FAMILIES]
        return HumanizationOptions(
            max_pairs=max_pairs,
            max_vh_length=MAX_VH_LENGTH,
            max_vl_length=MAX_VL_LENGTH,
            settings_schema=schema,
        )

    @router.get("/jobs/{job_id}/inputs", response_model=HumanizationSubmission)
    async def inputs(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        response: Response,
    ) -> HumanizationSubmission:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "humanization":
            raise HTTPException(404, "Job not found")
        try:
            retained = await adapter.input_request(job)
        except FileNotFoundError as error:
            raise CodedAPIError(
                404,
                "job_input_unavailable",
                "Humanization input is no longer available",
            ) from error
        response.headers["Cache-Control"] = "private, no-store"
        return HumanizationSubmission(
            display_name=job.display_name,
            pairs=[pair.model_dump() for pair in retained.pairs],
            settings=retained.settings,
        )

    @router.post(
        "/jobs",
        response_model=JobView,
        status_code=202,
        responses={422: {"model": InputErrors}},
    )
    async def submit(
        body: HumanizationSubmission,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
    ) -> JobView | Response:
        pairs, issues = validate_pairs(body, max_pairs=max_pairs)
        if not issues:
            from biomodals.workflow.humanization.workflow import (
                validate_humanization_settings,
            )

            try:
                await asyncio.to_thread(
                    validate_humanization_settings, body.settings, pair_count=len(pairs)
                )
            except ValueError as error:
                issues.append(
                    InputIssue(
                        row_index=None,
                        field="settings",
                        code="settings_invalid",
                        message=str(error),
                    )
                )
        if issues:
            return Response(
                InputErrors(errors=issues).model_dump_json(),
                status_code=422,
                media_type="application/json",
            )
        digest = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="humanization",
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
        effective = configuration.tool("humanization")
        deployment = DeploymentIdentity(
            configuration.modal_environment().value,
            effective.modal_app_name,
            effective.modal_app_version.value,
        )
        await remote.preflight(deployment)
        from biomodals.workflow.humanization.execution import (
            HumanizationExecutionRequest,
        )

        job_id = uuid4()
        execution_request = HumanizationExecutionRequest(
            pairs=pairs,
            settings=body.settings,
            run_name=f"humanization-{job_id}",
            max_active_provider_calls=effective.max_active_provider_calls,
            max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
        )
        await asyncio.to_thread(pending.put, job_id, execution_request.to_bytes())
        try:
            admission = store.admit_job(
                owner_user_id=session.principal.user_id,
                tool="humanization",
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
        if not admission.created:
            pending.delete(job_id)
        request.app.state.reconcile_wakeup.set()
        return view(admission.job, session)

    async def read_selection[T](
        job_id: UUID,
        session: AuthenticatedSession,
        reader: Callable[[IO[bytes]], T],
    ) -> T:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "humanization":
            raise HTTPException(404, "Job not found")
        if (
            job.state not in {JobState.SUCCEEDED, JobState.PARTIAL}
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
                "Prepare the result download before opening this table",
            )
        try:

            def read_member() -> T:
                with zipfile.ZipFile(lease) as archive:
                    member = archive.getinfo("selection.csv")
                    if member.file_size > MAX_SELECTION_BYTES:
                        raise ValueError(
                            "Selection table exceeds the bounded read limit"
                        )
                    with archive.open(member) as source:
                        return reader(source)

            return await cache.run_bounded(read_member)
        except (KeyError, ValueError, zipfile.BadZipFile) as error:
            raise CodedAPIError(
                409, "result_invalid", "Selection table is unavailable or invalid"
            ) from error
        finally:
            lease.close()

    @router.get("/jobs/{job_id}/selection", response_model=SelectionPage)
    async def selection(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        parent_id: Annotated[str | None, Query(max_length=200)] = None,
        sort_by: str | None = None,
        descending: bool = False,
    ) -> SelectionPage:
        if sort_by is not None and sort_by not in SELECTION_SCHEMA:
            raise CodedAPIError(422, "sort_invalid", "Unknown selection column")
        return await read_selection(
            job_id,
            session,
            partial(
                query_selection,
                offset=offset,
                limit=limit,
                parent_id=parent_id,
                sort_by=sort_by,
                descending=descending,
            ),
        )

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
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> Response:
        return Response(
            await read_selection(job_id, session, lambda source: source.read()),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="selection.csv"'},
        )

    return router
