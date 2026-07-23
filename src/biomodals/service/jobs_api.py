"""Provider-neutral Job inspection, cancellation, and Result routes."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from collections.abc import AsyncIterable, Mapping
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse

from biomodals.service.artifacts import (
    ArtifactCache,
    ArtifactIntegrityError,
    ArtifactLease,
    ArtifactSourceMissingError,
)
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    MutationForbiddenResponse,
    request_id_from,
    require_session,
    require_unsafe_session,
)
from biomodals.service.jobs import (
    JobPageView,
    JobView,
    WorkloadRegistration,
    can_view_job_logs,
)
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    JobCursorError,
    JobNotCancellableError,
    JobNotFoundError,
    JobRecord,
    JobState,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DOWNLOAD_NAME_SEPARATOR = re.compile(r"[^a-z0-9]+")
_DOWNLOAD_RESPONSE_HEADERS: dict[str, dict[str, object]] = {
    "Accept-Ranges": {
        "description": "Supported range unit for resumable downloads.",
        "schema": {"type": "string", "enum": ["bytes"]},
    },
    "Cache-Control": {
        "description": "Prevents shared or browser caching of private Results.",
        "schema": {"type": "string"},
    },
    "Content-Disposition": {
        "description": "Browser attachment using the server-provided result filename.",
        "schema": {"type": "string"},
    },
    "Content-Length": {
        "description": "Number of archive bytes in this response.",
        "schema": {"type": "integer", "minimum": 0},
    },
    "ETag": {
        "description": "Immutable archive identity derived from its SHA-256 digest.",
        "schema": {"type": "string"},
    },
}


class JobNotCancellableResponse(CodedErrorResponse):
    """Cancellation raced with a state that no longer accepts it."""

    code: Literal["job_not_cancellable"]


class ResultPrepareConflictResponse(CodedErrorResponse):
    """A Result cannot currently be prepared."""

    code: Literal["result_invalid", "result_not_ready"]


class ResultDownloadConflictResponse(CodedErrorResponse):
    """A prepared Result is not currently downloadable."""

    code: Literal["result_not_prepared", "result_not_ready"]


class ResultInvalidResponse(CodedErrorResponse):
    """A Result failed its immutable identity check."""

    code: Literal["result_invalid"]


class ResultStorageUnavailableResponse(CodedErrorResponse):
    """Local or authoritative Result storage is temporarily unavailable."""

    code: Literal["result_storage_unavailable"]


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")


def _download_headers(job_sha256: str, size_bytes: int) -> dict[str, str]:
    return {
        "Cache-Control": "private, no-store",
        "ETag": f'"{job_sha256}"',
        "Content-Length": str(size_bytes),
    }


def _download_filename(display_name: str) -> str:
    """Build a friendly filename without provider paths or Job identity."""
    ascii_name = (
        unicodedata
        .normalize("NFKD", display_name)
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    slug = _DOWNLOAD_NAME_SEPARATOR.sub("-", ascii_name).strip("-")
    slug = slug[:120].rstrip("-")
    return f"{slug or 'gromacs'}-results.zip"


def _cached_archive_response(
    request: Request,
    *,
    lease: ArtifactLease,
    filename: str,
    sha256: str,
    size_bytes: int,
) -> StreamingResponse:
    """Stream a held local archive, including one standard byte range."""
    first = 0
    last = size_bytes - 1
    response_status = status.HTTP_200_OK
    range_header = request.headers.get("Range")
    if range_header is not None:
        unit, separator, raw_range = range_header.partition("=")
        start_text, dash, end_text = raw_range.partition("-")
        if unit != "bytes" or not separator or not dash or "," in raw_range:
            lease.close()
            raise HTTPException(
                status.HTTP_416_RANGE_NOT_SATISFIABLE,
                headers={"Content-Range": f"bytes */{size_bytes}"},
            )
        try:
            if start_text:
                first = int(start_text)
                last = int(end_text) if end_text else last
            elif end_text:
                suffix = int(end_text)
                first = max(size_bytes - suffix, 0)
            else:
                raise ValueError
        except ValueError as exc:
            lease.close()
            raise HTTPException(
                status.HTTP_416_RANGE_NOT_SATISFIABLE,
                headers={"Content-Range": f"bytes */{size_bytes}"},
            ) from exc
        if first < 0 or first >= size_bytes or last < first:
            lease.close()
            raise HTTPException(
                status.HTTP_416_RANGE_NOT_SATISFIABLE,
                headers={"Content-Range": f"bytes */{size_bytes}"},
            )
        last = min(last, size_bytes - 1)
        response_status = status.HTTP_206_PARTIAL_CONTENT

    length = last - first + 1
    headers = {
        **_download_headers(sha256, length),
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    if response_status == status.HTTP_206_PARTIAL_CONTENT:
        headers["Content-Range"] = f"bytes {first}-{last}/{size_bytes}"

    async def content():
        remaining = length
        try:
            lease.seek(first)
            while remaining:
                chunk = lease.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("Cached archive ended unexpectedly")
                remaining -= len(chunk)
                yield chunk
        finally:
            lease.close()

    return StreamingResponse(
        content(),
        status_code=response_status,
        media_type="application/zip",
        headers=headers,
    )


def create_jobs_router(
    *,
    store: ServiceStore,
    workloads: Mapping[str, WorkloadRegistration],
    configuration: RuntimeConfiguration,
    cache: ArtifactCache | None,
) -> APIRouter:
    """Create shared Job routes over registered workload capabilities."""
    router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

    def caller_can_view_logs(
        workload: str,
        session: AuthenticatedSession,
    ) -> bool:
        registration = workloads.get(workload)
        logs_supported = bool(
            registration is not None and registration.open_operation_logs is not None
        )
        return can_view_job_logs(
            is_admin=session.principal.is_admin,
            owner_visibility_enabled=bool(
                logs_supported
                and configuration.workload(workload).job_logs_visible_to_owner.value
            ),
            logs_supported=logs_supported,
        )

    def job_view(
        job: JobRecord,
        session: AuthenticatedSession,
        *,
        can_view_logs: bool | None = None,
    ) -> JobView:
        if can_view_logs is None:
            can_view_logs = caller_can_view_logs(job.workload, session)
        return JobView.from_record(job, can_view_logs=can_view_logs)

    @router.get(
        "",
        response_model=JobPageView,
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        },
    )
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
        except JobCursorError as exc:
            raise HTTPException(400, str(exc)) from exc
        log_visibility = {
            workload: caller_can_view_logs(workload, session)
            for workload in dict.fromkeys(job.workload for job in page.jobs)
        }
        return JobPageView(
            jobs=[
                job_view(
                    job,
                    session,
                    can_view_logs=log_visibility[job.workload],
                )
                for job in page.jobs
            ],
            next_cursor=page.next_cursor,
        )

    @router.get(
        "/{job_id}",
        response_model=JobView,
        response_model_exclude_none=True,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        },
    )
    async def inspect_job(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> JobView:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None:
            raise _not_found()
        return job_view(job, session)

    @router.post(
        "/{job_id}/cancel",
        response_model=JobView,
        response_model_exclude_none=True,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": MutationForbiddenResponse},
            status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
            status.HTTP_409_CONFLICT: {"model": JobNotCancellableResponse},
        },
    )
    async def cancel_job(
        request: Request,
        job_id: UUID,
        session: Annotated[
            AuthenticatedSession,
            Depends(require_unsafe_session),
        ],
    ) -> JobView:
        existing = store.get_job(session.principal.user_id, job_id)
        if existing is None:
            raise _not_found()
        registration = workloads.get(existing.workload)
        lifecycle_lock = (
            registration.lifecycle_locks.for_job(job_id)
            if registration is not None
            else asyncio.Lock()
        )
        async with lifecycle_lock:
            try:
                job = store.request_cancel(
                    session.principal.user_id,
                    job_id,
                    now=int(time.time()),
                )
            except JobNotFoundError as exc:
                raise _not_found() from exc
            except JobNotCancellableError as exc:
                raise CodedAPIError(
                    status.HTTP_409_CONFLICT,
                    "job_not_cancellable",
                    str(exc),
                ) from exc
        stage = JobView.from_record(job).stage
        LOGGER.info(
            "event=cancellation_requested job_id=%s workload=%s stage=%s request_id=%s",
            job.job_id,
            job.workload,
            stage.code if stage is not None else "none",
            request_id_from(request),
        )
        if registration is not None and registration.cancel is not None:
            try:
                await registration.cancel(store, job)
            except Exception:
                LOGGER.exception(
                    "Could not yet cancel job %s request_id=%s",
                    job.job_id,
                    request_id_from(request),
                )
        return job_view(job, session)

    async def prepare_cached_artifact(job: JobRecord, *, request_id: str) -> None:
        if (
            job.result_size_bytes is None
            or type(job.result_size_bytes) is not int
            or job.result_size_bytes < 1
            or job.result_sha256 is None
            or _SHA256.fullmatch(job.result_sha256) is None
        ):
            raise CodedAPIError(
                409,
                "result_invalid",
                "Result archive metadata is invalid",
            )
        if cache is None:
            raise CodedAPIError(
                503,
                "result_storage_unavailable",
                "Result storage is temporarily unavailable",
            )
        try:
            existing = await cache.acquire_async(
                str(job.job_id),
                size_bytes=job.result_size_bytes,
                sha256=job.result_sha256,
            )
        except ArtifactIntegrityError as exc:
            raise CodedAPIError(
                409,
                "result_invalid",
                "Result archive metadata is invalid",
            ) from exc
        except OSError as exc:
            raise CodedAPIError(
                503,
                "result_storage_unavailable",
                "Result storage is temporarily unavailable",
            ) from exc
        if existing is not None:
            try:
                store.set_result_cached(job.job_id, cached=True)
                cache.protect_prepared(str(job.job_id))
            finally:
                existing.close()
            return
        store.set_result_cached(job.job_id, cached=False)
        registration = workloads.get(job.workload)
        if registration is None or registration.read_artifact is None:
            raise CodedAPIError(
                503,
                "result_storage_unavailable",
                "Result storage is temporarily unavailable",
            )
        result_size_bytes = job.result_size_bytes
        result_sha256 = job.result_sha256

        async def fill(chunks: AsyncIterable[bytes]) -> ArtifactLease:
            return await cache.store(
                str(job.job_id),
                size_bytes=result_size_bytes,
                sha256=result_sha256,
                chunks=chunks,
            )

        try:
            try:
                cached = await fill(registration.read_artifact(job))
            except (ArtifactIntegrityError, ArtifactSourceMissingError):
                if registration.rebuild_artifact is None:
                    raise
                cached = await fill(registration.rebuild_artifact(job))
        except (ArtifactIntegrityError, ArtifactSourceMissingError, ValueError) as exc:
            LOGGER.exception(
                "Artifact integrity failure for job %s request_id=%s",
                job.job_id,
                request_id,
            )
            store.block_job(
                job.job_id,
                category="result_integrity",
                previous_state=job.state,
                now=int(time.time()),
                next_retry_at=int(time.time()) + 15 * 60,
            )
            raise CodedAPIError(
                409,
                "result_invalid",
                "Result archive failed verification",
            ) from exc
        except Exception as exc:
            LOGGER.exception(
                "Could not restore artifact for job %s request_id=%s",
                job.job_id,
                request_id,
            )
            raise CodedAPIError(
                503,
                "result_storage_unavailable",
                "Result storage is temporarily unavailable",
            ) from exc
        try:
            store.set_result_cached(job.job_id, cached=True)
            cache.protect_prepared(str(job.job_id))
        finally:
            cached.close()

    @router.post(
        "/{job_id}/prepare-download",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": MutationForbiddenResponse},
            404: {"model": ErrorResponse},
            409: {"model": ResultPrepareConflictResponse},
            503: {"model": ResultStorageUnavailableResponse},
        },
    )
    async def prepare_download(
        request: Request,
        job_id: UUID,
        session: Annotated[
            AuthenticatedSession,
            Depends(require_unsafe_session),
        ],
    ) -> Response:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None:
            raise _not_found()
        if job.state not in {JobState.SUCCEEDED, JobState.PARTIAL}:
            raise CodedAPIError(
                409,
                "result_not_ready",
                f"Job is {job.state.value}",
            )
        await prepare_cached_artifact(job, request_id=request_id_from(request))
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get(
        "/{job_id}/download",
        response_class=StreamingResponse,
        responses={
            status.HTTP_200_OK: {
                "description": "Complete ZIP result archive.",
                "headers": _DOWNLOAD_RESPONSE_HEADERS,
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            status.HTTP_206_PARTIAL_CONTENT: {
                "description": "Requested byte range of the ZIP result archive.",
                "headers": {
                    **_DOWNLOAD_RESPONSE_HEADERS,
                    "Content-Range": {
                        "description": "Byte range returned from the complete archive.",
                        "schema": {"type": "string"},
                    },
                },
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
            status.HTTP_409_CONFLICT: {"model": ResultDownloadConflictResponse},
            status.HTTP_416_RANGE_NOT_SATISFIABLE: {
                "model": ErrorResponse,
                "description": "The requested byte range is invalid.",
                "headers": {
                    "Content-Range": {
                        "description": "Unsatisfied range and complete archive size.",
                        "schema": {"type": "string"},
                    }
                },
            },
            status.HTTP_502_BAD_GATEWAY: {"model": ResultInvalidResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ResultStorageUnavailableResponse
            },
        },
    )
    async def download_job(
        request: Request,
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> Response:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None:
            raise _not_found()
        if job.state not in {JobState.SUCCEEDED, JobState.PARTIAL}:
            raise CodedAPIError(
                409,
                "result_not_ready",
                f"Job is {job.state.value}",
            )
        if (
            job.result_size_bytes is None
            or type(job.result_size_bytes) is not int
            or job.result_size_bytes < 1
            or job.result_sha256 is None
            or _SHA256.fullmatch(job.result_sha256) is None
        ):
            raise CodedAPIError(
                502,
                "result_invalid",
                "Result archive is unavailable",
            )
        if cache is not None:
            try:
                cached = await cache.acquire_async(
                    str(job.job_id),
                    size_bytes=job.result_size_bytes,
                    sha256=job.result_sha256,
                )
            except ArtifactIntegrityError as exc:
                raise CodedAPIError(
                    502,
                    "result_invalid",
                    "Result archive metadata is invalid",
                ) from exc
            except OSError as exc:
                raise CodedAPIError(
                    503,
                    "result_storage_unavailable",
                    "Result storage is temporarily unavailable",
                ) from exc
            if cached is not None:
                return _cached_archive_response(
                    request,
                    lease=cached,
                    filename=_download_filename(job.display_name),
                    sha256=job.result_sha256,
                    size_bytes=job.result_size_bytes,
                )
        store.set_result_cached(job.job_id, cached=False)
        raise CodedAPIError(
            409,
            "result_not_prepared",
            "Prepare the result download before fetching it",
        )

    return router
