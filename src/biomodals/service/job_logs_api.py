"""Authorized Modal SDK log access for remote Tool Provider Calls."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

import orjson
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from biomodals.execution import ProviderCallDiagnostic
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import require_session
from biomodals.service.remote_execution import ExecutionLocator, RemoteExecutionClient
from biomodals.service.store import JobRecord, ServiceStore
from biomodals.service.tool_runtime import ToolRegistration
from biomodals.service.tools import ToolDefinition

LogMode = Literal["live", "historical"]


class JobLogTargetView(BaseModel):
    """Opaque selector for one inspectable remote Provider Call."""

    model_config = ConfigDict(frozen=True)

    target_id: UUID
    stage_code: str
    function_name: str
    status: str
    mode: LogMode
    started_at: datetime | None
    ended_at: datetime | None


class JobLogTargetsView(BaseModel):
    """Provider Calls grouped by public semantic stage."""

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    targets: list[JobLogTargetView]
    next_cursor: UUID | None


def create_job_logs_router() -> APIRouter:
    """Create owner- and Administrator-authorized SDK log routes."""
    router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

    @router.get("/{job_id}/log-targets", response_model=JobLogTargetsView)
    async def targets(
        request: Request,
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        cursor: Annotated[UUID | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> JobLogTargetsView:
        job, registration = _authorized(request, job_id, session)
        remote: RemoteExecutionClient = request.app.state.remote_execution
        page = await remote.provider_calls(_locator(job), cursor=cursor, limit=limit)
        values = [
            _target(registration.definition, call)
            for call in page.calls
            if call.provider_call_handle_id is not None
        ]
        return JobLogTargetsView(
            job_id=job_id,
            targets=[value for value in values if value is not None],
            next_cursor=page.next_cursor,
        )

    @router.get("/{job_id}/logs")
    async def logs(
        request: Request,
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        target: Annotated[UUID, Query()],
        since: Annotated[datetime | None, Query()] = None,
        until: Annotated[datetime | None, Query()] = None,
    ) -> StreamingResponse:
        job, registration = _authorized(request, job_id, session)
        remote: RemoteExecutionClient = request.app.state.remote_execution
        call = await remote.provider_call(_locator(job), target)
        if (
            call is None
            or call.provider_call_handle_id is None
            or _target(registration.definition, call) is None
        ):
            raise HTTPException(409, "Job log target is unavailable")
        live = not call.status.is_terminal

        async def content():
            async for entry in remote.log_entries(
                call.provider_call_handle_id,
                live=live,
                since=since,
                until=until,
            ):
                yield orjson.dumps(
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "message": entry.message,
                        "source": entry.source,
                    },
                    option=orjson.OPT_APPEND_NEWLINE,
                )

        return StreamingResponse(
            content(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "private, no-store",
                "X-Accel-Buffering": "no",
                "X-BioModals-Log-Mode": "live" if live else "historical",
            },
        )

    return router


def _authorized(
    request: Request,
    job_id: UUID,
    session: AuthenticatedSession,
) -> tuple[JobRecord, ToolRegistration]:
    store: ServiceStore = request.app.state.store
    job = (
        store.get_job_by_id(job_id)
        if session.principal.is_admin
        else store.get_job(session.principal.user_id, job_id)
    )
    if job is None:
        raise HTTPException(404, "Job not found")
    registration = request.app.state.registrations[job.tool]
    visible = request.app.state.configuration.tool(
        job.tool
    ).job_logs_visible_to_owner.value
    if not session.principal.is_admin and not visible:
        raise HTTPException(403, "Job logs are Administrator-only for this Tool")
    return job, registration


def _target(
    definition: ToolDefinition,
    call: ProviderCallDiagnostic,
) -> JobLogTargetView | None:
    stage = next(
        (
            stage
            for stage in definition.stages
            if any(
                call.node_key == key or call.node_key.startswith(key)
                for key in stage.node_keys
            )
        ),
        None,
    )
    if stage is None:
        return None
    return JobLogTargetView(
        target_id=call.provider_call_id,
        stage_code=stage.code,
        function_name=call.function_name,
        status=call.status.value,
        mode="historical" if call.status.is_terminal else "live",
        started_at=_time(call.started_at),
        ended_at=_time(call.completed_at),
    )


def _locator(job: JobRecord) -> ExecutionLocator:
    from biomodals.execution import DeploymentIdentity

    return ExecutionLocator(
        job.job_id,
        DeploymentIdentity(
            job.modal_environment,
            job.modal_app_name,
            job.modal_app_version,
        ),
    )


def _time(value: int | None) -> datetime | None:
    return datetime.fromtimestamp(value, UTC) if value is not None else None
