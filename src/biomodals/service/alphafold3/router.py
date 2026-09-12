"""AlphaFold3 retained validation and typed Job submission routes."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import orjson
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict
from starlette.datastructures import MutableHeaders
from starlette.types import Message, Receive, Scope, Send

from biomodals.execution import DeploymentIdentity
from biomodals.service.alphafold3.modal import AlphaFold3ToolAdapter
from biomodals.service.alphafold3.results import (
    MAX_PAE_GRID,
    PaeWindow,
    PredictionData,
    PredictionReader,
    PredictionSummary,
    PreviewTooLargeError,
    pae_window,
)
from biomodals.service.alphafold3.validation import (
    MAX_VALIDATION_BYTES,
    ValidatedInput,
    ValidatedInputStore,
    ValidationLimitExceededError,
    ValidationSettings,
    ValidationStorageLowError,
)
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    require_session,
    require_unsafe_session,
)
from biomodals.service.jobs import JobView
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


class ValidationView(BaseModel):
    """Bounded confirmation data for one retained native document."""

    model_config = ConfigDict(frozen=True)

    validation_id: UUID
    request_digest: str
    created_at: datetime
    expires_at: datetime
    preview: dict[str, object]


class AlphaFold3JobRequest(BaseModel):
    """Job creation consumes an already validated server resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validation_id: UUID


class _PrivatePredictionRoute(APIRoute):
    """Protect preview responses, including authentication and validation errors."""

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_private(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "private, no-store"
            await send(message)

        await super().handle(scope, receive, send_private)


def create_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    validations: ValidatedInputStore,
    adapter: AlphaFold3ToolAdapter,
    remote: RemoteExecutionClient,
    cache: ArtifactCache,
) -> APIRouter:
    """Create AlphaFold3 validation and submission routes."""
    router = APIRouter(prefix="/api/v1/alphafold3", tags=["alphafold3"])
    preview_router = APIRouter(
        prefix="/jobs/{job_id}/prediction", route_class=_PrivatePredictionRoute
    )
    upload_slots = asyncio.Semaphore(2)
    validation_lock = asyncio.Lock()
    predictions = PredictionReader()

    async def read_result[T](
        job_id: UUID,
        session: AuthenticatedSession,
        operation: Callable[[PredictionData], T],
    ) -> T:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "alphafold3":
            raise HTTPException(404, "Job not found")
        if (
            job.state != JobState.SUCCEEDED
            or job.result_size_bytes is None
            or job.result_sha256 is None
        ):
            raise CodedAPIError(409, "result_not_ready", "Result is not ready")
        if job.result_archive_schema != "alphafold3-request/1":
            raise CodedAPIError(409, "result_invalid", "Unsupported result archive")
        digest = job.result_sha256
        lease = await cache.acquire_async(
            str(job_id), size_bytes=job.result_size_bytes, sha256=digest
        )
        if lease is None:
            raise CodedAPIError(
                409,
                "result_not_cached",
                "Prepare the result download before opening this prediction",
            )
        try:

            def read() -> T:
                data = predictions.read(
                    lease,
                    job_id=str(job_id),
                    digest=digest,
                    display_name=job.display_name,
                )
                return operation(data)

            return await cache.run_bounded(read)
        except PreviewTooLargeError as error:
            raise CodedAPIError(
                413,
                "preview_too_large",
                "Result exceeds the structure preview limit; use the archive download",
            ) from error
        except (KeyError, ValueError, TypeError, OSError) as error:
            raise CodedAPIError(
                409, "result_invalid", "Prediction is unavailable or invalid"
            ) from error
        finally:
            lease.close()

    @preview_router.get(
        "",
        response_model=PredictionSummary,
        responses={
            409: {"model": CodedErrorResponse},
            413: {"model": CodedErrorResponse},
        },
    )
    async def prediction(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> PredictionSummary:
        return await read_result(job_id, session, lambda data: data.summary)

    @preview_router.get(
        "/model.cif",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "chemical/x-mmcif": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                }
            },
            409: {"model": CodedErrorResponse},
            413: {"model": CodedErrorResponse},
        },
    )
    async def prediction_model(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> Response:
        return Response(
            await read_result(job_id, session, lambda data: data.cif),
            media_type="chemical/x-mmcif",
            headers={
                "Content-Disposition": 'inline; filename="model.cif"',
            },
        )

    @preview_router.get(
        "/pae",
        response_model=PaeWindow,
        responses={
            409: {"model": CodedErrorResponse},
            413: {"model": CodedErrorResponse},
            422: {"model": CodedErrorResponse},
        },
    )
    async def prediction_pae(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        x_start: Annotated[int, Query(ge=0)] = 0,
        x_end: Annotated[int | None, Query(ge=1)] = None,
        y_start: Annotated[int, Query(ge=0)] = 0,
        y_end: Annotated[int | None, Query(ge=1)] = None,
        max_size: Annotated[int, Query(ge=1, le=MAX_PAE_GRID)] = MAX_PAE_GRID,
    ) -> PaeWindow:
        def window(data: PredictionData) -> PaeWindow:
            if data.summary.pae_error is not None:
                raise CodedAPIError(
                    409,
                    data.summary.pae_error,
                    "PAE preview is unavailable; native confidence values remain in the archive",
                )
            try:
                return pae_window(
                    data,
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y_start,
                    y_end=y_end,
                    max_size=max_size,
                )
            except ValueError as error:
                raise CodedAPIError(
                    422, "pae_window_invalid", "PAE window is outside the token matrix"
                ) from error

        return await read_result(job_id, session, window)

    router.include_router(preview_router)

    @router.post(
        "/validations",
        response_model=ValidationView,
        status_code=201,
        responses={
            429: {"model": CodedErrorResponse},
            507: {"model": CodedErrorResponse},
        },
    )
    async def validate_document(
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        search_msa: Annotated[bool, Query()] = True,
        search_protein_templates: Annotated[bool, Query()] = True,
        recycle: Annotated[int, Query(ge=0)] = 10,
        sample: Annotated[int, Query(ge=1)] = 5,
    ) -> ValidationView:
        async with upload_slots:
            async with validation_lock:
                await asyncio.to_thread(
                    validations.cleanup_expired,
                    claimed=store.claimed_validation_ids(),
                    now=int(time.time()),
                )
                try:
                    validations.require_capacity(
                        session.principal.user_id,
                        additional_bytes=1,
                    )
                except ValidationLimitExceededError as error:
                    raise CodedAPIError(
                        429,
                        "validation_limit",
                        str(error),
                    ) from error
                except ValidationStorageLowError as error:
                    raise CodedAPIError(
                        507,
                        "validation_storage_low",
                        str(error),
                    ) from error
            descriptor, raw_path = tempfile.mkstemp(
                dir=validations.directory.parent,
                prefix=".alphafold3-upload-",
            )
            path = Path(raw_path)
            digest = hashlib.sha256()
            size = 0
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    validations.require_free_space()
                    next_space_check = 8 * 1024 * 1024
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_VALIDATION_BYTES:
                            raise CodedAPIError(
                                413,
                                "payload_too_large",
                                "AlphaFold3 JSON exceeds 256 MiB",
                            )
                        if size >= next_space_check:
                            validations.require_free_space(len(chunk))
                            next_space_check = size + 8 * 1024 * 1024
                        digest.update(chunk)
                        handle.write(chunk)
                if size == 0:
                    raise CodedAPIError(
                        400,
                        "document_invalid",
                        "JSON body is empty",
                    )
                settings = ValidationSettings(
                    search_msa=search_msa,
                    search_protein_templates=search_protein_templates,
                    recycle=recycle,
                    sample=sample,
                )
                try:
                    async with validation_lock:
                        validated = await asyncio.to_thread(
                            validations.validate_and_publish,
                            path,
                            owner_user_id=session.principal.user_id,
                            digest=digest.hexdigest(),
                            settings=settings,
                        )
                except ValidationLimitExceededError as error:
                    raise CodedAPIError(429, "validation_limit", str(error)) from error
                except ValidationStorageLowError as error:
                    raise CodedAPIError(
                        507,
                        "validation_storage_low",
                        str(error),
                    ) from error
                except (TypeError, ValueError, orjson.JSONDecodeError) as error:
                    raise CodedAPIError(400, "document_invalid", str(error)) from error
            except ValidationStorageLowError as error:
                raise CodedAPIError(
                    507,
                    "validation_storage_low",
                    str(error),
                ) from error
            finally:
                path.unlink(missing_ok=True)
        return _validation_view(validated)

    @router.get("/validations/{validation_id}", response_model=ValidationView)
    async def get_validation(
        validation_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> ValidationView:
        return _validation_view(_owned_validation(validations, validation_id, session))

    @router.get("/validations/{validation_id}/document")
    async def download_document(
        validation_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> FileResponse:
        validated = _owned_validation(validations, validation_id, session)
        return FileResponse(
            validated.document_path,
            media_type="application/json",
            filename="alphafold3-input.json",
            headers={
                "Cache-Control": "private, no-store",
            },
        )

    @router.delete("/validations/{validation_id}", status_code=204)
    async def delete_validation(
        validation_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> Response:
        async with validation_lock:
            if store.validation_is_claimed(validation_id):
                raise HTTPException(409, "Validation is already claimed by a Job")
            if not await asyncio.to_thread(
                validations.delete,
                validation_id,
                owner_user_id=session.principal.user_id,
            ):
                raise HTTPException(404, "Validation not found")
        return Response(status_code=204)

    @router.get("/jobs/{job_id}/document")
    async def download_job_document(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> Response:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None or job.tool != "alphafold3":
            raise HTTPException(404, "Job not found")
        try:
            document = await adapter.input_document(job)
        except FileNotFoundError as error:
            raise CodedAPIError(
                404,
                "job_input_unavailable",
                "AlphaFold3 input is no longer available",
            ) from error
        return Response(
            content=document,
            media_type="application/json",
            headers={
                "Cache-Control": "private, no-store",
                "Content-Disposition": ('attachment; filename="alphafold3-input.json"'),
            },
        )

    @router.post(
        "/jobs",
        response_model=JobView,
        response_description="Job durably admitted for asynchronous staging and launch",
        status_code=202,
    )
    async def submit_job(
        request: Request,
        body: AlphaFold3JobRequest,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
    ) -> JobView:
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="alphafold3",
            idempotency_key=str(idempotency_key),
        )
        if replay is not None:
            async with validation_lock:
                replay = _idempotent_replay(
                    store,
                    validations,
                    session,
                    idempotency_key=idempotency_key,
                    validation_id=body.validation_id,
                )
                if replay is not None:
                    return _job_view(replay, session, configuration)
        validated = _owned_validation(validations, body.validation_id, session)
        effective = configuration.tool("alphafold3")
        deployment = DeploymentIdentity(
            configuration.modal_environment().value,
            effective.modal_app_name,
            effective.modal_app_version.value,
        )
        await remote.preflight(deployment)
        async with validation_lock:
            replay = _idempotent_replay(
                store,
                validations,
                session,
                idempotency_key=idempotency_key,
                validation_id=body.validation_id,
            )
            if replay is not None:
                return _job_view(replay, session, configuration)
            validated = _owned_validation(validations, body.validation_id, session)
            digest = _request_digest(validated)
            try:
                admission = store.admit_job(
                    owner_user_id=session.principal.user_id,
                    tool="alphafold3",
                    display_name=str(validated.preview["name"]),
                    idempotency_key=str(idempotency_key),
                    request_digest=digest,
                    publication_scope_digest=(validated.publication_scope_digest),
                    modal_environment=deployment.environment,
                    modal_app_name=deployment.deployment_name,
                    modal_app_version=deployment.deployment_version,
                    tool_active_job_limit=effective.active_job_limit.value,
                    global_active_job_limit=(
                        configuration.global_active_job_limit().value
                    ),
                    max_active_provider_calls=(effective.max_active_provider_calls),
                    max_active_gpu_provider_calls=(
                        effective.max_active_gpu_provider_calls
                    ),
                    now=int(time.time()),
                    new_job_id=uuid4(),
                    pending_validation_id=validated.validation_id,
                )
            except sqlite3.IntegrityError as error:
                raise CodedAPIError(
                    409,
                    "validation_claimed",
                    "Validation is already claimed by another Job",
                ) from error
            except (IdempotencyConflictError, JobLimitExceededError) as error:
                raise CodedAPIError(409, "job_conflict", str(error)) from error
            except UserNotFoundError as error:
                raise CodedAPIError(403, "account_disabled", str(error)) from error
        request.app.state.reconcile_wakeup.set()
        return _job_view(admission.job, session, configuration)

    return router


def _idempotent_replay(
    store: ServiceStore,
    validations: ValidatedInputStore,
    session: AuthenticatedSession,
    *,
    idempotency_key: UUID,
    validation_id: UUID,
) -> JobRecord | None:
    """Return an existing Job; a consumed key is the remaining request identity."""
    replay = store.find_idempotent_job(
        session.principal.user_id,
        tool="alphafold3",
        idempotency_key=str(idempotency_key),
    )
    if replay is None:
        return None
    validated = validations.get(
        validation_id,
        owner_user_id=session.principal.user_id,
    )
    if validated is not None and replay.request_digest != _request_digest(validated):
        raise CodedAPIError(
            409,
            "idempotency_conflict",
            "Idempotency key was already used for another request",
        )
    return replay


def _owned_validation(
    validations: ValidatedInputStore,
    validation_id: UUID,
    session: AuthenticatedSession,
) -> ValidatedInput:
    validated = validations.get(
        validation_id,
        owner_user_id=session.principal.user_id,
    )
    if validated is None:
        raise HTTPException(404, "Validation not found")
    return validated


def _validation_view(validated: ValidatedInput) -> ValidationView:
    return ValidationView(
        validation_id=validated.validation_id,
        request_digest=validated.digest,
        created_at=datetime.fromtimestamp(validated.created_at, UTC),
        expires_at=datetime.fromtimestamp(validated.expires_at, UTC),
        preview=validated.preview,
    )


def _request_digest(validated: ValidatedInput) -> str:
    digest = hashlib.sha256(bytes.fromhex(validated.digest))
    digest.update(
        orjson.dumps(
            {
                "search_msa": validated.settings.search_msa,
                "search_protein_templates": validated.settings.search_protein_templates,
                "recycle": validated.settings.recycle,
                "sample": validated.settings.sample,
            },
            option=orjson.OPT_SORT_KEYS,
        )
    )
    return digest.hexdigest()


def _job_view(
    job,
    session: AuthenticatedSession,
    configuration: RuntimeConfiguration,
) -> JobView:
    return JobView.from_record(
        job,
        can_view_logs=(
            session.principal.is_admin
            or configuration.tool("alphafold3").job_logs_visible_to_owner.value
        ),
    )
