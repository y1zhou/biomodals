"""Administrator-only inspection of retained Job operation logs."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator
from datetime import UTC, datetime, timedelta
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
from biomodals.service.jobs import (
    OperationLogMode,
    OperationLogRequest,
    WorkloadRegistration,
    operation_log_mode,
)
from biomodals.service.store import (
    JobOperationRecord,
    JobRecord,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)
_MAX_LOG_WINDOW = timedelta(minutes=15)
AdminJobLogTargetState = Literal[
    "running",
    "state_unknown",
    "completed",
    "failed",
    "cancelled",
]
_LOG_STREAM_RESPONSE_HEADERS: dict[str, dict[str, object]] = {
    "Cache-Control": {
        "description": "Disables storage and response transformation for logs.",
        "schema": {"type": "string"},
    },
    "X-Accel-Buffering": {
        "description": "Requests that compatible reverse proxies stream immediately.",
        "schema": {"type": "string", "enum": ["no"]},
    },
    "X-BioModals-Log-Mode": {
        "description": "Whether the response is a live stream or historical window.",
        "schema": {"type": "string", "enum": ["live", "historical"]},
    },
    "X-BioModals-Log-Since": {
        "description": "Inclusive beginning of a selected historical window.",
        "schema": {"type": "string", "format": "date-time"},
    },
    "X-BioModals-Log-Until": {
        "description": "Exclusive end of a selected historical window.",
        "schema": {"type": "string", "format": "date-time"},
    },
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
    """One provider operation whose logs an Administrator may inspect."""

    model_config = ConfigDict(frozen=True)

    stage_code: str
    function_name: str
    state: AdminJobLogTargetState
    mode: OperationLogMode
    started_at: datetime
    ended_at: datetime | None


class AdminJobLogTargetsView(BaseModel):
    """Safe selectors for a Job's currently inspectable provider calls."""

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    targets: list[AdminJobLogTargetView]


class AdminJobLogTargetUnavailableResponse(CodedErrorResponse):
    """The selected stage does not identify a retained provider call."""

    code: Literal["job_log_target_unavailable"]


class AdminJobLogsUnavailableResponse(CodedErrorResponse):
    """The workload cannot start a provider log stream."""

    code: Literal["job_logs_unavailable"]


class AdminJobLogWindowInvalidResponse(CodedErrorResponse):
    """The requested historical window is incomplete or too large."""

    code: Literal["job_log_window_invalid"]


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
        mode = operation_log_mode(operation.state)
        if (
            mode is None
            or not operation.modal_call_id
            or operation.started_at is None
            or (mode == "historical" and operation.completed_at is None)
            or stage is None
            or stage.function_name is None
        ):
            continue
        targets.append((
            AdminJobLogTargetView(
                stage_code=stage.code,
                function_name=stage.function_name,
                state=cast(AdminJobLogTargetState, operation.state.value),
                mode=mode,
                started_at=datetime.fromtimestamp(operation.started_at, UTC),
                ended_at=(
                    datetime.fromtimestamp(operation.completed_at, UTC)
                    if operation.completed_at is not None
                    else None
                ),
            ),
            operation,
        ))
    return targets


def _operation_log_request(
    target: AdminJobLogTargetView,
    *,
    since: datetime | None,
    until: datetime | None,
    now: datetime,
) -> OperationLogRequest | None:
    """Validate and clamp an optional historical window to durable Stage bounds."""
    if since is None and until is None:
        return OperationLogRequest(mode=target.mode)
    if since is None or until is None:
        raise CodedAPIError(
            400,
            "job_log_window_invalid",
            "Historical Job log windows require both since and until",
        )
    if since.tzinfo is None or until.tzinfo is None or since >= until:
        raise CodedAPIError(
            400,
            "job_log_window_invalid",
            "Historical Job log windows require valid timezone-aware timestamps",
        )
    if until - since > _MAX_LOG_WINDOW:
        raise CodedAPIError(
            400,
            "job_log_window_invalid",
            "Historical Job log windows cannot exceed 15 minutes",
        )
    lower_bound = target.started_at - timedelta(seconds=1)
    upper_bound = (target.ended_at or now) + timedelta(seconds=1)
    bounded_since = max(since, lower_bound)
    bounded_until = min(until, upper_bound)
    if bounded_since >= bounded_until:
        return None
    return OperationLogRequest(
        mode="historical",
        since=bounded_since,
        until=bounded_until,
    )


async def _empty_log_stream() -> AsyncIterator[bytes]:
    """Represent a valid requested window outside the operation's lifetime."""
    if False:  # pragma: no cover - makes this an async iterator without output
        yield b""


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
                held_length = next(
                    (
                        length
                        for length in range(
                            min(len(pending), len(secret) - 1),
                            0,
                            -1,
                        )
                        if pending.endswith(secret[:length])
                    ),
                    0,
                )
                safe_length = len(pending) - held_length
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
                "description": "Logs for the selected remote stage",
                "headers": _LOG_STREAM_RESPONSE_HEADERS,
                "content": {"text/plain": {"schema": {"type": "string"}}},
            },
            **read_responses,
            409: {"model": AdminJobLogTargetUnavailableResponse},
            400: {"model": AdminJobLogWindowInvalidResponse},
            503: {"model": AdminJobLogsUnavailableResponse},
        },
    )
    async def stream_job_logs(
        request: Request,
        job_id: UUID,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
        stage: Annotated[str, Query(min_length=1, max_length=120)],
        since: Annotated[datetime | None, Query()] = None,
        until: Annotated[datetime | None, Query()] = None,
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
                (target, operation)
                for target, operation in _log_targets(job, registration)
                if target.stage_code == stage
            ),
            None,
        )
        if (
            registration is None
            or registration.open_operation_logs is None
            or selected is None
        ):
            raise CodedAPIError(
                409,
                "job_log_target_unavailable",
                "Logs are not available for the selected Job stage",
            )
        selected_target, selected_operation = selected
        modal_call_id = selected_operation.modal_call_id
        if modal_call_id is None:  # pragma: no cover - filtered by _log_targets
            raise CodedAPIError(
                409,
                "job_log_target_unavailable",
                "Logs are not available for the selected Job stage",
            )
        selection = _operation_log_request(
            selected_target,
            since=since,
            until=until,
            now=datetime.now(UTC),
        )
        try:
            stream = (
                _empty_log_stream()
                if selection is None
                else await registration.open_operation_logs(
                    job,
                    selected_operation,
                    selection,
                )
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
            "event=job_log_stream_started job_id=%s stage=%s mode=%s request_id=%s",
            job.job_id,
            stage,
            selection.mode if selection is not None else "historical",
            request_id_from(request),
        )
        response_mode: OperationLogMode = (
            selection.mode if selection is not None else "historical"
        )
        response_headers = {
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",
            "X-BioModals-Log-Mode": response_mode,
        }
        if selection is not None and selection.since is not None:
            response_headers["X-BioModals-Log-Since"] = selection.since.isoformat()
        if selection is not None and selection.until is not None:
            response_headers["X-BioModals-Log-Until"] = selection.until.isoformat()
        return _ClosingStreamingResponse(
            _redact_provider_call_id(stream, modal_call_id),
            media_type="text/plain",
            headers=response_headers,
        )

    return router
