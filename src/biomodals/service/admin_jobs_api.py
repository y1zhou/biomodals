"""Administrator-only inspection of active Job operation logs."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol, cast, runtime_checkable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.types import Receive, Scope, Send

from biomodals.service.admin_api import AdminForbiddenResponse, require_admin
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    request_id_from,
)
from biomodals.service.jobs import WorkloadRegistration
from biomodals.service.store import (
    JobOperationRecord,
    JobOperationState,
    JobRecord,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)
AdminJobLogTargetState = Literal["running", "state_unknown"]
_LOGGABLE_STATES = {
    JobOperationState.RUNNING,
    JobOperationState.STATE_UNKNOWN,
}


@runtime_checkable
class _AsyncClosable(Protocol):
    async def aclose(self) -> None:
        """Release resources owned by an asynchronous iterator."""


class _ClosingStreamingResponse(StreamingResponse):
    """Close a resource-owning body iterator after completion or disconnect."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if isinstance(self.body_iterator, _AsyncClosable):
                await self.body_iterator.aclose()


class AdminJobLogTargetView(BaseModel):
    """One active provider operation whose logs an Administrator may inspect."""

    model_config = ConfigDict(frozen=True)

    stage_code: str
    function_name: str
    state: AdminJobLogTargetState
    started_at: datetime


class AdminJobLogTargetsView(BaseModel):
    """Safe selectors for a Job's currently inspectable provider calls."""

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    targets: list[AdminJobLogTargetView]


class AdminJobLogTargetUnavailableResponse(CodedErrorResponse):
    """The selected stage no longer identifies an active provider call."""

    code: Literal["job_log_target_unavailable"]


class AdminJobLogsUnavailableResponse(CodedErrorResponse):
    """The workload cannot start a provider log stream."""

    code: Literal["job_logs_unavailable"]


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")


def _log_targets(
    job: JobRecord,
    registration: WorkloadRegistration | None,
) -> list[tuple[AdminJobLogTargetView, JobOperationRecord]]:
    if registration is None or registration.open_operation_logs is None:
        return []
    targets: list[tuple[AdminJobLogTargetView, JobOperationRecord]] = []
    for operation in job.operations:
        stage = registration.definition.stage(operation.operation)
        if (
            operation.state not in _LOGGABLE_STATES
            or not operation.modal_call_id
            or operation.started_at is None
            or stage is None
            or stage.function_name is None
        ):
            continue
        targets.append((
            AdminJobLogTargetView(
                stage_code=stage.code,
                function_name=stage.function_name,
                state=cast(AdminJobLogTargetState, operation.state.value),
                started_at=datetime.fromtimestamp(operation.started_at, UTC),
            ),
            operation,
        ))
    return targets


async def _redact_provider_call_id(
    stream: AsyncIterable[bytes],
    provider_call_id: str,
) -> AsyncIterator[bytes]:
    """Remove the selected provider identifier, including across chunk edges."""
    secret = provider_call_id.encode()
    if not secret:  # pragma: no cover - empty IDs are excluded from targets
        raise ValueError("Provider call ID must not be empty")
    replacement = b"[function-call-id-redacted]"
    pending = b""
    iterator = aiter(stream)
    try:
        async for chunk in iterator:
            pending += chunk
            output = bytearray()
            while True:
                index = pending.find(secret)
                if index >= 0:
                    output.extend(pending[:index])
                    output.extend(replacement)
                    pending = pending[index + len(secret) :]
                    continue
                safe_length = max(0, len(pending) - len(secret) + 1)
                output.extend(pending[:safe_length])
                pending = pending[safe_length:]
                break
            if output:
                yield bytes(output)
        if pending:
            yield pending.replace(secret, replacement)
    finally:
        if isinstance(iterator, _AsyncClosable):
            await iterator.aclose()


def create_admin_jobs_router() -> APIRouter:
    """Create Administrator-only Job diagnostics routes."""
    router = APIRouter(prefix="/api/v1/admin/jobs", tags=["admin"])
    read_responses: dict[int | str, dict[str, Any]] = {
        401: {"model": ErrorResponse},
        403: {"model": AdminForbiddenResponse},
        404: {"model": ErrorResponse},
    }

    @router.get(
        "/{job_id}/log-targets",
        response_model=AdminJobLogTargetsView,
        responses=read_responses,
    )
    async def job_log_targets(
        request: Request,
        job_id: UUID,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
    ) -> AdminJobLogTargetsView:
        store: ServiceStore = request.app.state.store
        job = store.get_job_by_id(job_id)
        if job is None:
            raise _not_found()
        registration: WorkloadRegistration | None = request.app.state.workloads.get(
            job.workload
        )
        return AdminJobLogTargetsView(
            job_id=job.job_id,
            targets=[target for target, _operation in _log_targets(job, registration)],
        )

    @router.get(
        "/{job_id}/logs",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Live logs for the selected active stage",
                "content": {"text/plain": {"schema": {"type": "string"}}},
            },
            **read_responses,
            409: {"model": AdminJobLogTargetUnavailableResponse},
            503: {"model": AdminJobLogsUnavailableResponse},
        },
    )
    async def stream_job_logs(
        request: Request,
        job_id: UUID,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
        stage: Annotated[str, Query(min_length=1, max_length=120)],
    ) -> StreamingResponse:
        store: ServiceStore = request.app.state.store
        job = store.get_job_by_id(job_id)
        if job is None:
            raise _not_found()
        registration: WorkloadRegistration | None = request.app.state.workloads.get(
            job.workload
        )
        selected = next(
            (
                operation
                for target, operation in _log_targets(job, registration)
                if target.stage_code == stage
            ),
            None,
        )
        if (
            registration is None
            or registration.open_operation_logs is None
            or selected is None
            or selected.modal_call_id is None
        ):
            raise CodedAPIError(
                409,
                "job_log_target_unavailable",
                "The selected Job stage is no longer active",
            )
        try:
            stream = await registration.open_operation_logs(
                job,
                selected.modal_call_id,
            )
        except OSError as exc:
            LOGGER.exception(
                "Could not start Job log stream job_id=%s stage=%s request_id=%s",
                job.job_id,
                stage,
                request_id_from(request),
            )
            raise CodedAPIError(
                503,
                "job_logs_unavailable",
                "Job logs are temporarily unavailable",
            ) from exc
        LOGGER.info(
            "event=job_log_stream_started job_id=%s stage=%s request_id=%s",
            job.job_id,
            stage,
            request_id_from(request),
        )
        return _ClosingStreamingResponse(
            _redact_provider_call_id(stream, selected.modal_call_id),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-store, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return router
