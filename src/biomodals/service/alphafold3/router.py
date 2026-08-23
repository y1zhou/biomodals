"""AlphaFold3 retained validation and typed Job submission routes."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import orjson
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from biomodals.execution import DeploymentIdentity
from biomodals.service.alphafold3.validation import (
    MAX_VALIDATION_BYTES,
    ValidatedInput,
    ValidatedInputStore,
    ValidationSettings,
)
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
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
    ServiceStore,
    UserNotFoundError,
)
from biomodals.service.tool_runtime import JobLifecycle


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


def create_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    validations: ValidatedInputStore,
    remote: RemoteExecutionClient,
    lifecycle: JobLifecycle,
) -> APIRouter:
    """Create AlphaFold3 validation and submission routes."""
    router = APIRouter(prefix="/api/v1/alphafold3", tags=["alphafold3"])
    validation_lock = asyncio.Lock()

    @router.post("/validations", response_model=ValidationView, status_code=201)
    async def validate_document(
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        search_msa: Annotated[bool, Query()] = True,
        search_protein_templates: Annotated[bool, Query()] = True,
        recycle: Annotated[int, Query(ge=0)] = 10,
        sample: Annotated[int, Query(ge=1)] = 5,
    ) -> ValidationView:
        descriptor, raw_path = tempfile.mkstemp(
            dir=validations.directory.parent,
            prefix=".alphafold3-upload-",
        )
        path = Path(raw_path)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_VALIDATION_BYTES:
                        raise CodedAPIError(
                            413,
                            "payload_too_large",
                            "AlphaFold3 JSON exceeds 256 MiB",
                        )
                    digest.update(chunk)
                    handle.write(chunk)
            if size == 0:
                raise CodedAPIError(400, "document_invalid", "JSON body is empty")
            settings = ValidationSettings(
                search_msa=search_msa,
                search_protein_templates=search_protein_templates,
                recycle=recycle,
                sample=sample,
            )
            try:
                async with validation_lock:
                    await asyncio.to_thread(
                        validations.cleanup_expired,
                        claimed=store.claimed_validation_ids(),
                        now=int(time.time()),
                    )
                    validated = await asyncio.to_thread(
                        validations.validate_and_publish,
                        path,
                        owner_user_id=session.principal.user_id,
                        digest=digest.hexdigest(),
                        settings=settings,
                    )
            except (TypeError, ValueError, orjson.JSONDecodeError) as error:
                raise CodedAPIError(400, "document_invalid", str(error)) from error
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

    @router.post("/jobs", response_model=JobView, status_code=202)
    async def submit_job(
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
            effective.modal_app_name.value,
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
                    modal_environment=deployment.environment,
                    modal_app_name=deployment.deployment_name,
                    modal_app_version=deployment.deployment_version,
                    tool_active_job_limit=effective.active_job_limit.value,
                    global_active_job_limit=(
                        configuration.global_active_job_limit().value
                    ),
                    max_active_provider_calls=(
                        effective.max_active_provider_calls.value
                    ),
                    max_active_gpu_provider_calls=(
                        effective.max_active_gpu_provider_calls.value
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
        try:
            job = await lifecycle.advance(admission.job.job_id)
        except Exception:
            job = admission.job
        return _job_view(job, session, configuration)

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
