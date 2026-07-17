"""Unified FastAPI control plane for Biomodals apps and workflows."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from biomodals.service.artifacts import (
    ArtifactCache,
    ArtifactIntegrityError,
    ArtifactLease,
)
from biomodals.service.auth import (
    MIN_PASSWORD_CHARACTERS,
    SESSION_ABSOLUTE_LIFETIME_SECONDS,
    AuthenticatedSession,
    AuthService,
    InvalidCredentialsError,
    InvalidPasswordTokenError,
    PasswordExecutor,
    PasswordPolicyError,
    Principal,
)
from biomodals.service.config import ServiceSettings
from biomodals.service.jobs import (
    JobView,
    WorkloadRegistration,
    reconciliation_loop,
)
from biomodals.service.store import (
    JobNotCancellableError,
    JobNotFoundError,
    JobState,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)
SESSION_COOKIE = "biomodals-session"
SECURE_SESSION_COOKIE = "__Host-biomodals-session"
CSRF_COOKIE = "biomodals-csrf"
_CSRF_HEADER_DESCRIPTION = (
    f"Required for authenticated mutations. Copy the value of the `{CSRF_COOKIE}` "
    "cookie set by a successful login."
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.zip")


class ErrorResponse(BaseModel):
    """Stable JSON error response."""

    detail: str


class PrincipalView(BaseModel):
    """The small identity document needed by the separate frontend."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    email: str
    display_name: str

    @classmethod
    def from_principal(cls, principal: Principal) -> PrincipalView:
        """Convert an internal principal to its browser representation."""
        return cls(
            user_id=principal.user_id,
            email=principal.email,
            display_name=principal.display_name,
        )


class LoginRequest(BaseModel):
    """Credentials submitted only by the browser login form."""

    email: str = Field(max_length=320)
    password: str = Field(max_length=128)


class SetPasswordRequest(BaseModel):
    """One-time password setup or reset submission."""

    token: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=MIN_PASSWORD_CHARACTERS, max_length=128)


def model_response(model: BaseModel, *, status_code: int) -> Response:
    """Serialize a Pydantic model into an explicit JSON response."""
    return Response(
        content=model.model_dump_json(exclude_none=True),
        status_code=status_code,
        media_type="application/json",
    )


class _RequestSizeMiddleware:
    """Reject oversized bodies before FastAPI parses multipart content."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > self.max_body_bytes
            except ValueError:
                too_large = False
            if too_large:
                await model_response(
                    ErrorResponse(detail="Request body is too large"),
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )(scope, receive, send)
                return

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await model_response(
                ErrorResponse(detail="Request body is too large"),
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )(scope, receive, send)


class _RequestBodyTooLarge(Exception):
    """Internal signal raised before an oversized body reaches a route."""


async def require_origin(request: Request) -> None:
    """Require the exact configured browser Origin on unsafe requests."""
    if request.headers.get("Origin") != request.app.state.allowed_origin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Request origin is not allowed")


async def require_session(request: Request) -> AuthenticatedSession:
    """Authenticate the host-only opaque browser session cookie."""
    token = request.cookies.get(request.app.state.session_cookie_name)
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    auth: AuthService = request.app.state.auth
    session = auth.authenticate(token)
    if session is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return session


async def require_unsafe_session(
    request: Request,
    csrf_token: Annotated[
        str | None,
        Header(alias="X-CSRF-Token", description=_CSRF_HEADER_DESCRIPTION),
    ] = None,
) -> AuthenticatedSession:
    """Authenticate a state-changing browser request and its CSRF token."""
    await require_origin(request)
    session = await require_session(request)
    auth: AuthService = request.app.state.auth
    if csrf_token is None or not auth.verify_csrf(session, csrf_token):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF validation failed")
    return session


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")


def _document_required_csrf_header(app: FastAPI) -> None:
    """Align OpenAPI with the custom 403 behavior for a missing CSRF header."""
    schema = app.openapi()
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            for parameter in operation.get("parameters", []):
                if (
                    parameter.get("in") == "header"
                    and parameter.get("name") == "X-CSRF-Token"
                ):
                    parameter["required"] = True
                    parameter["schema"] = {"type": "string"}


def _download_headers(job_sha256: str, size_bytes: int) -> dict[str, str]:
    return {
        "Cache-Control": "private, no-store",
        "ETag": f'"{job_sha256}"',
        "Content-Length": str(size_bytes),
    }


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
                status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
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
                status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                headers={"Content-Range": f"bytes */{size_bytes}"},
            ) from exc
        if first < 0 or first >= size_bytes or last < first:
            lease.close()
            raise HTTPException(
                status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
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


def create_app(
    *,
    store: ServiceStore,
    auth: AuthService,
    workloads: Sequence[WorkloadRegistration],
    allowed_origin: str,
    secure_cookies: bool,
    cache: ArtifactCache | None = None,
    reconcile_interval_seconds: float = 10,
) -> FastAPI:
    """Assemble one control plane from explicitly registered workloads."""
    registrations = {workload.name: workload for workload in workloads}
    if len(registrations) != len(workloads):
        raise ValueError("Workload names must be unique")
    if reconcile_interval_seconds <= 0:
        raise ValueError("reconcile_interval_seconds must be positive")
    if any(workload.max_body_bytes < 1 for workload in workloads):
        raise ValueError("Workload body limits must be positive")
    if cache is None and any(
        workload.read_artifact is not None for workload in workloads
    ):
        raise ValueError("A verified artifact cache is required for downloads")
    if not allowed_origin or allowed_origin.endswith("/"):
        raise ValueError("allowed_origin must be an exact origin without a slash")
    session_cookie_name = SECURE_SESSION_COOKIE if secure_cookies else SESSION_COOKIE
    password_executor = PasswordExecutor()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop = asyncio.Event()
        task: asyncio.Task[None] | None = None
        if any(workload.reconciler is not None for workload in workloads):
            task = asyncio.create_task(
                reconciliation_loop(
                    workloads,
                    interval_seconds=reconcile_interval_seconds,
                    stop=stop,
                ),
                name="biomodals-job-reconciler",
            )
        try:
            yield
        finally:
            stop.set()
            try:
                if task is not None:
                    await task
            finally:
                await password_executor.shutdown()

    app = FastAPI(
        title="Biomodals API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.auth = auth
    app.state.allowed_origin = allowed_origin
    app.state.session_cookie_name = session_cookie_name
    app.state.workloads = registrations
    app.state.cache = cache
    app.add_middleware(
        _RequestSizeMiddleware,
        max_body_bytes=max(
            (workload.max_body_bytes for workload in workloads),
            default=1024 * 1024,
        ),
    )

    @app.get("/api/v1/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/api/v1/auth/login",
        response_model=PrincipalView,
        responses={
            status.HTTP_200_OK: {
                "headers": {
                    "Set-Cookie": {
                        "description": (
                            "Sets the HttpOnly session cookie and the readable "
                            f"`{CSRF_COOKIE}` cookie used as `X-CSRF-Token`."
                        ),
                        "schema": {"type": "string"},
                    }
                }
            }
        },
    )
    async def login(request: Request, credentials: LoginRequest) -> Response:
        await require_origin(request)
        try:
            issued = await password_executor.run(
                auth.login,
                credentials.email,
                credentials.password,
            )
        except InvalidCredentialsError as exc:
            LOGGER.warning("Rejected login attempt")
            raise HTTPException(401, "Invalid email or password") from exc
        response = model_response(
            PrincipalView.from_principal(issued.principal),
            status_code=status.HTTP_200_OK,
        )
        response.set_cookie(
            session_cookie_name,
            issued.session_token,
            max_age=SESSION_ABSOLUTE_LIFETIME_SECONDS,
            secure=secure_cookies,
            httponly=True,
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            CSRF_COOKIE,
            issued.csrf_token,
            max_age=SESSION_ABSOLUTE_LIFETIME_SECONDS,
            secure=secure_cookies,
            httponly=False,
            samesite="lax",
            path="/",
        )
        return response

    @app.post("/api/v1/auth/set-password", response_model=PrincipalView)
    async def set_password(
        request: Request,
        submission: SetPasswordRequest,
    ) -> PrincipalView:
        await require_origin(request)
        try:
            principal = await password_executor.run(
                auth.set_password,
                submission.token,
                submission.password,
            )
        except PasswordPolicyError as exc:
            raise HTTPException(400, str(exc)) from exc
        except InvalidPasswordTokenError as exc:
            raise HTTPException(400, "Password link is invalid or expired") from exc
        return PrincipalView.from_principal(principal)

    @app.get("/api/v1/auth/me", response_model=PrincipalView)
    async def me(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> PrincipalView:
        return PrincipalView.from_principal(session.principal)

    @app.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    async def logout(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> Response:
        token = request.cookies[session_cookie_name]
        auth.logout(token)
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(session_cookie_name, path="/")
        response.delete_cookie(CSRF_COOKIE, path="/")
        return response

    @app.get("/api/v1/jobs", response_model=list[JobView])
    async def list_jobs(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> list[JobView]:
        return [
            JobView.from_record(job)
            for job in store.list_jobs(session.principal.user_id)
        ]

    @app.get("/api/v1/jobs/{job_id}", response_model=JobView)
    async def inspect_job(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> JobView:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None:
            raise _not_found()
        return JobView.from_record(job)

    @app.post(
        "/api/v1/jobs/{job_id}/cancel",
        response_model=JobView,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_job(
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> JobView:
        existing = store.get_job(session.principal.user_id, job_id)
        if existing is None:
            raise _not_found()
        try:
            job = store.request_cancel(
                session.principal.user_id,
                job_id,
                now=int(time.time()),
            )
        except JobNotFoundError as exc:
            raise _not_found() from exc
        except JobNotCancellableError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        registration = registrations.get(job.workload)
        if (
            job.modal_call_id is not None
            and registration is not None
            and registration.cancel is not None
        ):
            try:
                await registration.cancel(job)
            except Exception:
                LOGGER.exception("Could not yet cancel job %s", job.job_id)
        return JobView.from_record(job)

    @app.get("/api/v1/jobs/{job_id}/download")
    async def download_job(
        request: Request,
        job_id: UUID,
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> Response:
        job = store.get_job(session.principal.user_id, job_id)
        if job is None:
            raise _not_found()
        if job.state not in {JobState.SUCCEEDED, JobState.PARTIAL}:
            raise HTTPException(409, f"Job is {job.state.value}")
        if (
            job.result_filename is None
            or _ARCHIVE_FILENAME.fullmatch(job.result_filename) is None
            or job.result_size_bytes is None
            or type(job.result_size_bytes) is not int
            or job.result_size_bytes < 1
            or job.result_sha256 is None
            or _SHA256.fullmatch(job.result_sha256) is None
        ):
            raise HTTPException(502, "Result archive is unavailable")
        if cache is not None:
            try:
                cached = cache.acquire(
                    str(job.job_id),
                    size_bytes=job.result_size_bytes,
                    sha256=job.result_sha256,
                )
            except ArtifactIntegrityError as exc:
                raise HTTPException(502, "Result archive metadata is invalid") from exc
            if cached is not None:
                return _cached_archive_response(
                    request,
                    lease=cached,
                    filename=job.result_filename,
                    sha256=job.result_sha256,
                    size_bytes=job.result_size_bytes,
                )

        registration = registrations.get(job.workload)
        if registration is None or registration.read_artifact is None:
            raise HTTPException(503, "Result storage is temporarily unavailable")
        if cache is None:  # guarded during app construction for registered workloads
            raise HTTPException(503, "Result storage is temporarily unavailable")
        chunks = registration.read_artifact(job)
        try:
            cached = await cache.store(
                str(job.job_id),
                size_bytes=job.result_size_bytes,
                sha256=job.result_sha256,
                chunks=chunks,
            )
        except ArtifactIntegrityError as exc:
            LOGGER.exception("Artifact integrity failure for job %s", job.job_id)
            raise HTTPException(502, "Result archive failed verification") from exc
        except Exception as exc:
            LOGGER.exception("Could not restore artifact for job %s", job.job_id)
            raise HTTPException(
                503, "Result storage is temporarily unavailable"
            ) from exc
        return _cached_archive_response(
            request,
            lease=cached,
            filename=job.result_filename,
            sha256=job.result_sha256,
            size_bytes=job.result_size_bytes,
        )

    for workload in workloads:
        app.include_router(workload.router)
    _document_required_csrf_header(app)
    return app


def create_deployed_app() -> FastAPI:
    """Create the local Linux service backed by deployed Modal compute Apps."""
    from biomodals.service.gromacs import (
        GromacsReconciler,
        ModalGromacsAdapter,
        create_registration,
    )

    settings = ServiceSettings.from_environment()
    store = ServiceStore(settings.database_path)
    store.initialize()
    auth = AuthService(store, frontend_url=settings.frontend_url)
    adapter = ModalGromacsAdapter(
        app_name=settings.gromacs_app_name,
        environment_name=settings.modal_environment,
    )
    registration = create_registration(
        adapter,
        active_limit=settings.gromacs_active_limit,
        reconciler=GromacsReconciler(
            store,
            adapter,
            intermediate_retention_days=settings.intermediate_retention_days,
        ),
    )
    cache = ArtifactCache(
        settings.cache_dir / "results", max_bytes=settings.cache_max_bytes
    )
    return create_app(
        store=store,
        auth=auth,
        workloads=[registration],
        allowed_origin=settings.allowed_origin,
        secure_cookies=settings.secure_cookies,
        cache=cache,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
    )
