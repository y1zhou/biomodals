"""Provider-neutral HTTP contract for asynchronous GROMACS jobs."""

from __future__ import annotations

import sys
from secrets import compare_digest
from typing import Annotated, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from biomodals.helper.pdb import validate_pdb_content

if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    from backports.strenum import StrEnum  # noqa: UP035,I001

MAX_PDB_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class RequestGuardMiddleware:
    """Authenticate and reject oversized bodies before multipart parsing."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        api_key: str | None,
    ) -> None:
        """Configure the wrapped app, body limit, and optional bearer key."""
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.api_key = api_key

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Guard one ASGI request before passing it to FastAPI."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        if self.api_key is not None:
            scheme, separator, token = headers.get(b"authorization", b"").partition(
                b" "
            )
            if (
                separator != b" "
                or scheme.lower() != b"bearer"
                or not compare_digest(token, self.api_key.encode())
            ):
                response = _model_response(
                    ErrorResponse(detail="Invalid or missing bearer token"),
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )
                response.headers["WWW-Authenticate"] = "Bearer"
                await response(scope, receive, send)
                return

        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = 0
            if declared_bytes > self.max_body_bytes:
                await _model_response(
                    ErrorResponse(detail="Request body is too large"),
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )(scope, receive, send)
                return

        received_bytes = 0
        messages: list[Message] = []
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break
            received_bytes += len(message.get("body", b""))
            if received_bytes > self.max_body_bytes:
                await _model_response(
                    ErrorResponse(detail="Request body is too large"),
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)


class JobNotFoundError(LookupError):
    """Raised when a backend cannot resolve a public job id."""

    def __init__(self, job_id: str) -> None:
        """Remember the missing id for the HTTP error response."""
        self.job_id = job_id
        super().__init__(f"Job '{job_id}' was not found")


class JobBackendUnavailableError(RuntimeError):
    """Raised when the compute backend cannot currently answer a request."""


class JobState(StrEnum):
    """Provider-neutral states exposed by the job API."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class GromacsJobOptions(BaseModel):
    """Bounded user settings accepted by the public API."""

    model_config = ConfigDict(frozen=True)

    simulation_time_ns: int = Field(default=5, ge=1, le=100)
    run_pdbfixer: bool = False
    cpu_only: bool = False


class JobStatus(BaseModel):
    """Current state of one submitted job."""

    job_id: str
    status: JobState
    run_name: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Stable JSON error response."""

    detail: str


class JobArtifactFile(BaseModel):
    """One provider-neutral file in an output artifact."""

    path: str
    role: str | None = None
    media_type: str | None = None


class JobArtifact(BaseModel):
    """One named output and its file manifest."""

    name: str
    kind: str
    files: list[JobArtifactFile] = Field(default_factory=list)


class JobResult(BaseModel):
    """Portable result returned after a GROMACS job completes."""

    run_name: str
    artifacts: list[JobArtifact] = Field(default_factory=list)


class JobSnapshot(JobStatus):
    """Backend snapshot used by status and result routes."""

    result: JobResult | None = None


class JobBackend(Protocol):
    """Compute boundary used by the HTTP layer."""

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
    ) -> JobStatus:
        """Submit a GROMACS job without waiting for its result."""
        ...

    async def inspect(self, job_id: str) -> JobSnapshot:
        """Poll one job without waiting for completion."""
        ...

    async def cancel(self, job_id: str) -> JobStatus:
        """Request cancellation of one job."""
        ...


def _model_response(model: BaseModel, *, status_code: int) -> Response:
    return Response(
        content=model.model_dump_json(exclude_none=True),
        status_code=status_code,
        media_type="application/json",
    )


async def _read_pdb(upload: UploadFile, *, max_bytes: int) -> bytes:
    content = bytearray()
    while chunk := await upload.read(min(1024 * 1024, max_bytes + 1)):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"PDB upload exceeds the {max_bytes}-byte limit",
            )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="PDB upload is empty",
        )
    try:
        validate_pdb_content(bytes(content), max_bytes=max_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return bytes(content)


def create_app(
    backend: JobBackend,
    *,
    max_pdb_bytes: int = MAX_PDB_BYTES,
    api_key: str | None = None,
    trusted_proxy_auth: bool = False,
) -> FastAPI:
    """Create a FastAPI app around a pluggable GROMACS job backend."""
    if not api_key and not trusted_proxy_auth:
        raise ValueError("Configure api_key or explicitly trust upstream proxy auth")

    web_app = FastAPI(
        title="Biomodals GROMACS API",
        version="1.0.0",
    )
    web_app.add_middleware(
        RequestGuardMiddleware,
        max_body_bytes=max_pdb_bytes + MAX_MULTIPART_OVERHEAD_BYTES,
        api_key=api_key,
    )

    @web_app.exception_handler(JobNotFoundError)
    async def job_not_found(
        _request: Request,
        exc: JobNotFoundError,
    ) -> Response:
        return _model_response(
            ErrorResponse(detail=str(exc)),
            status_code=status.HTTP_404_NOT_FOUND,
        )

    @web_app.exception_handler(JobBackendUnavailableError)
    async def backend_unavailable(
        _request: Request,
        _exc: JobBackendUnavailableError,
    ) -> Response:
        return _model_response(
            ErrorResponse(detail="Job backend is temporarily unavailable"),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @web_app.post(
        "/jobs",
        response_model=JobStatus,
        response_model_exclude_none=True,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_job(
        pdb: Annotated[UploadFile, File(description="Input structure in PDB format")],
        simulation_time_ns: Annotated[int, Form(ge=1, le=100)] = 5,
        run_pdbfixer: Annotated[bool, Form()] = False,
        cpu_only: Annotated[bool, Form()] = False,
    ) -> JobStatus:
        try:
            pdb_content = await _read_pdb(pdb, max_bytes=max_pdb_bytes)
        finally:
            await pdb.close()
        return await backend.submit(
            pdb_content,
            GromacsJobOptions(
                simulation_time_ns=simulation_time_ns,
                run_pdbfixer=run_pdbfixer,
                cpu_only=cpu_only,
            ),
        )

    @web_app.get(
        "/jobs/{job_id}",
        response_model=JobStatus,
        response_model_exclude_none=True,
    )
    async def inspect_job(job_id: str) -> JobStatus:
        return await backend.inspect(job_id)

    @web_app.get(
        "/jobs/{job_id}/result",
        response_model=JobResult,
        response_model_exclude_none=True,
        responses={status.HTTP_202_ACCEPTED: {"model": JobStatus}},
    )
    async def get_job_result(job_id: str) -> JobResult | Response:
        snapshot = await backend.inspect(job_id)
        if snapshot.status == JobState.PENDING:
            return _model_response(
                JobStatus.model_validate(snapshot),
                status_code=status.HTTP_202_ACCEPTED,
            )
        if snapshot.result is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is {snapshot.status}",
            )
        return snapshot.result

    @web_app.delete(
        "/jobs/{job_id}",
        response_model=JobStatus,
        response_model_exclude_none=True,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_job(job_id: str) -> JobStatus:
        return await backend.cancel(job_id)

    return web_app
