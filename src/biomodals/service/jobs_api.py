"""Shared Job inspection, cancellation, refresh, and Result delivery."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from biomodals.service.artifacts import ArtifactCache, ArtifactLease, run_blocking_io
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    require_session,
    require_unsafe_session,
)
from biomodals.service.jobs import JobPageView, JobView
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    JobCursorError,
    JobNotCancellableError,
    JobRecord,
    JobState,
    ServiceStore,
)
from biomodals.service.tool_runtime import (
    JobLifecycle,
    PreparedResult,
    ResultIntegrityError,
)


def create_jobs_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    lifecycle: JobLifecycle,
    cache: ArtifactCache,
) -> APIRouter:
    """Create provider-neutral owner-scoped Job routes."""
    router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

    def view(job: JobRecord, session: AuthenticatedSession) -> JobView:
        visibility = configuration.tool(job.tool).job_logs_visible_to_owner.value
        return JobView.from_record(
            job,
            can_view_logs=session.principal.is_admin or visibility,
        )

    @router.get("")
    async def list_jobs(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        cursor: UUID | None = None,
    ) -> JobPageView:
        try:
            page = store.list_jobs_page(
                session.principal.user_id,
                limit=limit,
                cursor=cursor,
            )
        except JobCursorError as error:
            raise HTTPException(400, str(error)) from error
        return JobPageView(
            jobs=[view(job, session) for job in page.jobs],
            next_cursor=page.next_cursor,
        )

    @router.get("/{job_id}")
    async def get_job(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> JobView:
        job = _owned(store, session, job_id)
        if job.state in {
            JobState.RUNNING,
            JobState.CANCEL_REQUESTED,
            JobState.BLOCKED,
        }:
            job = await lifecycle.advance(job_id)
        return view(job, session)

    @router.post("/{job_id}/refresh")
    async def refresh_job(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> JobView:
        _owned(store, session, job_id)
        job = await lifecycle.advance(job_id, force_refresh=True)
        return view(job, session)

    @router.post("/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
    async def cancel_job(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> JobView:
        _owned(store, session, job_id)
        try:
            job = await lifecycle.cancel(job_id)
        except JobNotCancellableError as error:
            raise HTTPException(409, str(error)) from error
        return view(job, session)

    @router.post("/{job_id}/prepare-download")
    async def prepare_download(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> JobView:
        job = _owned(store, session, job_id)
        result = _require_result(job)
        lease = await cache.acquire_async(
            str(job.job_id),
            size_bytes=result.size_bytes,
            sha256=result.sha256,
        )
        if lease is None:
            try:
                await lifecycle.restore_result(job)
            except ResultIntegrityError as error:
                raise CodedAPIError(
                    status.HTTP_409_CONFLICT,
                    "result_invalid",
                    "The published Result could not be restored exactly",
                ) from error
        else:
            lease.close()
        cache.protect_prepared(str(job.job_id))
        return view(_owned(store, session, job_id), session)

    @router.get("/{job_id}/download")
    async def download(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
        range_header: Annotated[str | None, Header(alias="Range")] = None,
    ) -> StreamingResponse:
        job = _owned(store, session, job_id)
        result = _require_result(job)
        lease = await cache.acquire_async(
            str(job.job_id),
            size_bytes=result.size_bytes,
            sha256=result.sha256,
        )
        if lease is None:
            raise HTTPException(409, "Result must be prepared before download")
        return _response(
            lease,
            filename=result.filename,
            media_type=result.media_type,
            size_bytes=result.size_bytes,
            sha256=result.sha256,
            range_header=range_header,
        )

    return router


def _owned(
    store: ServiceStore,
    session: AuthenticatedSession,
    job_id: UUID,
) -> JobRecord:
    job = store.get_job(session.principal.user_id, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _require_result(job: JobRecord) -> PreparedResult:
    if (
        job.state not in {JobState.SUCCEEDED, JobState.PARTIAL}
        or job.result_filename is None
        or job.result_media_type is None
        or job.result_size_bytes is None
        or job.result_sha256 is None
        or job.result_archive_schema is None
    ):
        raise HTTPException(409, "Result is not ready")
    return PreparedResult(
        filename=job.result_filename,
        media_type=job.result_media_type,
        size_bytes=job.result_size_bytes,
        sha256=job.result_sha256,
        archive_schema=job.result_archive_schema,
    )


def _response(
    lease: ArtifactLease,
    *,
    filename: str,
    media_type: str,
    size_bytes: int,
    sha256: str,
    range_header: str | None,
) -> StreamingResponse:
    first, last, response_status = 0, size_bytes - 1, status.HTTP_200_OK
    if range_header is not None:
        try:
            unit, raw = range_header.split("=", 1)
            start, end = raw.split("-", 1)
            if unit != "bytes" or "," in raw:
                raise ValueError
            if start:
                first = int(start)
                last = int(end) if end else last
            elif end:
                first = max(size_bytes - int(end), 0)
            else:
                raise ValueError
            if first < 0 or first >= size_bytes or last < first:
                raise ValueError
            last = min(last, size_bytes - 1)
        except ValueError as error:
            lease.close()
            raise HTTPException(
                416,
                headers={"Content-Range": f"bytes */{size_bytes}"},
            ) from error
        response_status = status.HTTP_206_PARTIAL_CONTENT
    length = last - first + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Length": str(length),
        "ETag": f'"{sha256}"',
    }
    if response_status == status.HTTP_206_PARTIAL_CONTENT:
        headers["Content-Range"] = f"bytes {first}-{last}/{size_bytes}"

    async def content():
        remaining = length
        try:
            await run_blocking_io(lease.seek, first)
            while remaining:
                chunk = await run_blocking_io(lease.read, min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("Cached Result ended unexpectedly")
                remaining -= len(chunk)
                yield chunk
        finally:
            await run_blocking_io(lease.close)

    return StreamingResponse(
        content(),
        status_code=response_status,
        media_type=media_type,
        headers=headers,
    )
