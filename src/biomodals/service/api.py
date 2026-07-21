"""Unified FastAPI control plane for Biomodals apps and workflows."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from biomodals.service.artifacts import (
    ArtifactCache,
    ArtifactIntegrityError,
    ArtifactLease,
    ArtifactSourceMissingError,
)
from biomodals.service.auth import (
    MIN_PASSWORD_CHARACTERS,
    SESSION_ABSOLUTE_LIFETIME_SECONDS,
    AuthenticatedSession,
    AuthService,
    InvalidCredentialsError,
    InvalidPasswordTokenError,
    IssuedSession,
    PasswordExecutor,
    PasswordExecutorBusyError,
    PasswordPolicyError,
    Principal,
)
from biomodals.service.config import ServiceSettings
from biomodals.service.jobs import (
    JobView,
    WorkloadRegistration,
    reconciliation_loop,
)
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    JobNotCancellableError,
    JobNotFoundError,
    JobRecord,
    JobState,
    ServiceStore,
)

LOGGER = logging.getLogger(__name__)
SESSION_COOKIE = "biomodals-session"
SECURE_SESSION_COOKIE = "__Host-biomodals-session"
CSRF_COOKIE = "biomodals-csrf"
_CSRF_HEADER_DESCRIPTION = (
    f"Required for authenticated mutations. Copy the value of the `{CSRF_COOKIE}` "
    "cookie set by a successful login or Password Setup."
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DOWNLOAD_NAME_SEPARATOR = re.compile(r"[^a-z0-9]+")


class ErrorResponse(BaseModel):
    """Stable JSON error response."""

    detail: str


class CodedErrorResponse(ErrorResponse):
    """Recoverable error with a stable machine-readable code."""

    code: str


class PayloadTooLargeResponse(CodedErrorResponse):
    """Request body exceeded the service-wide parsing limit."""

    code: Literal["payload_too_large"]


class OriginErrorResponse(CodedErrorResponse):
    """Browser origin rejected before an unsafe request."""

    code: Literal["origin_not_allowed"]


class MutationForbiddenResponse(CodedErrorResponse):
    """Unsafe request rejected for origin or CSRF state."""

    code: Literal["csrf_invalid", "origin_not_allowed"]


class PasswordErrorResponse(CodedErrorResponse):
    """Password Setup errors with distinct recovery behavior."""

    code: Literal[
        "password_link_invalid",
        "password_policy_rejected",
    ]


class AuthenticationBusyResponse(CodedErrorResponse):
    """Bounded Argon2 capacity is temporarily exhausted."""

    code: Literal["authentication_busy"]


class HealthView(BaseModel):
    """Minimal local service probe."""

    status: Literal["ok"]


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


class CodedAPIError(Exception):
    """Signal one flat coded response from a route or dependency."""

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Capture the response status, stable code, and safe detail."""
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.headers = headers or {}


class PrincipalView(BaseModel):
    """The small identity document needed by the separate frontend."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    email: str
    display_name: str
    is_admin: bool

    @classmethod
    def from_principal(cls, principal: Principal) -> PrincipalView:
        """Convert an internal principal to its browser representation."""
        return cls(
            user_id=principal.user_id,
            email=principal.email,
            display_name=principal.display_name,
            is_admin=principal.is_admin,
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


def request_id_from(request: Request) -> str:
    """Return the correlation ID installed before route handling."""
    return str(getattr(request.state, "request_id", "unavailable"))


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
                    CodedErrorResponse(
                        code="payload_too_large",
                        detail="Request body is too large",
                    ),
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
                CodedErrorResponse(
                    code="payload_too_large",
                    detail="Request body is too large",
                ),
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )(scope, receive, send)


class _RequestBodyTooLarge(Exception):
    """Internal signal raised before an oversized body reaches a route."""


class _RequestIdMiddleware:
    """Attach one server-generated correlation identifier to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(name.lower() == b"x-request-id" for name, _ in headers):
                    headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
                if not scope["state"].get("completion_logged"):
                    LOGGER.info(
                        "request_complete event=http_request request_id=%s "
                        "method=%s path=%s status=%s",
                        request_id,
                        scope.get("method"),
                        scope.get("path"),
                        message.get("status"),
                    )
                    scope["state"]["completion_logged"] = True
            await send(message)

        await self.app(scope, receive, send_with_request_id)


async def require_origin(request: Request) -> None:
    """Require the exact configured browser Origin on unsafe requests."""
    if request.headers.get("Origin") != request.app.state.allowed_origin:
        raise CodedAPIError(
            status.HTTP_403_FORBIDDEN,
            "origin_not_allowed",
            "Request origin is not allowed",
        )


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
        raise CodedAPIError(
            status.HTTP_403_FORBIDDEN,
            "csrf_invalid",
            "CSRF validation failed",
        )
    return session


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")


def _document_contract_headers(app: FastAPI, *, session_cookie_name: str) -> None:
    """Align generated OpenAPI with middleware and custom CSRF behavior."""
    schema = app.openapi()
    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["SessionCookie"] = {
        "type": "apiKey",
        "in": "cookie",
        "name": session_cookie_name,
        "description": "Opaque BioModals browser session cookie.",
    }
    public_operations = {
        ("/api/v1/health", "get"),
        ("/api/v1/ready", "get"),
        ("/api/v1/auth/login", "post"),
        ("/api/v1/auth/set-password", "post"),
    }
    for path, path_item in schema["paths"].items():
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
            for response in operation.get("responses", {}).values():
                response.setdefault("headers", {})["X-Request-ID"] = {
                    "description": "Server-generated request correlation identifier.",
                    "schema": {"type": "string", "format": "uuid"},
                }
        for method, operation in path_item.items():
            if (
                isinstance(operation, dict)
                and method
                in {"get", "post", "put", "patch", "delete", "options", "head"}
                and path.startswith("/api/v1/")
                and (path, method) not in public_operations
            ):
                operation["security"] = [{"SessionCookie": []}]


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


def create_app(
    *,
    store: ServiceStore,
    auth: AuthService,
    configuration: RuntimeConfiguration,
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
        for workload in workloads:
            if workload.preflight is None:
                continue
            effective = configuration.workload(workload.name)
            await workload.preflight(
                effective.modal_app_name.value,
                configuration.modal_environment().value,
                effective.modal_app_version.value,
            )
        if cache is not None:
            store.reconcile_result_cache(await cache.cached_job_ids_async())
        if any(workload.reconciler is not None for workload in workloads):
            task = asyncio.create_task(
                reconciliation_loop(
                    workloads,
                    interval_seconds=reconcile_interval_seconds,
                    stop=stop,
                ),
                name="biomodals-job-reconciler",
            )
        _app.state.reconciler_task = task
        _app.state.ready = True
        LOGGER.info("event=readiness_changed ready=true")
        try:
            yield
        finally:
            _app.state.ready = False
            LOGGER.info("event=readiness_changed ready=false")
            stop.set()
            try:
                if task is not None:
                    await task
            finally:
                try:
                    await password_executor.shutdown()
                finally:
                    if cache is not None:
                        await cache.shutdown()

    app = FastAPI(
        title="Biomodals API",
        version="1.0.0",
        lifespan=lifespan,
        responses={
            status.HTTP_413_CONTENT_TOO_LARGE: {"model": PayloadTooLargeResponse},
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        },
    )
    app.state.store = store
    app.state.auth = auth
    app.state.configuration = configuration
    app.state.allowed_origin = allowed_origin
    app.state.session_cookie_name = session_cookie_name
    app.state.workloads = registrations
    app.state.cache = cache
    app.state.ready = False
    app.state.reconciler_task = None

    @app.exception_handler(CodedAPIError)
    async def coded_api_error(_request: Request, exc: CodedAPIError) -> Response:
        response = model_response(
            CodedErrorResponse(code=exc.code, detail=exc.detail),
            status_code=exc.status_code,
        )
        response.headers.update(exc.headers)
        return response

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> Response:
        request_id = request_id_from(request)
        LOGGER.exception(
            "event=unhandled_exception request_id=%s method=%s path=%s status=500",
            request_id,
            request.method,
            request.url.path,
            exc_info=exc,
        )
        LOGGER.info(
            "request_complete event=http_request request_id=%s method=%s "
            "path=%s status=500",
            request_id,
            request.method,
            request.url.path,
        )
        request.scope["state"]["completion_logged"] = True
        response = model_response(
            ErrorResponse(detail="Internal Server Error"),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    app.add_middleware(
        _RequestSizeMiddleware,
        max_body_bytes=max(
            (workload.max_body_bytes for workload in workloads),
            default=1024 * 1024,
        ),
    )
    app.add_middleware(_RequestIdMiddleware)

    def session_response(issued: IssuedSession) -> Response:
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

    @app.get("/api/v1/health", response_model=HealthView, tags=["operations"])
    async def health() -> HealthView:
        return HealthView(status="ok")

    @app.get(
        "/api/v1/ready",
        response_model=HealthView,
        tags=["operations"],
        responses={503: {"model": ErrorResponse}},
    )
    async def ready(request: Request) -> HealthView:
        task: asyncio.Task[None] | None = request.app.state.reconciler_task
        try:
            if not request.app.state.ready or (task is not None and task.done()):
                raise RuntimeError("Background service is unavailable")
            store.check_ready()
            if cache is not None:
                await cache.check_ready_async()
        except Exception as exc:
            raise HTTPException(503, "Service is not ready") from exc
        return HealthView(status="ok")

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
            },
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": OriginErrorResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": AuthenticationBusyResponse,
                "headers": {
                    "Retry-After": {
                        "description": "Seconds before retrying authentication.",
                        "schema": {"type": "integer"},
                    }
                },
            },
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
            LOGGER.warning(
                "Rejected login attempt request_id=%s",
                request_id_from(request),
            )
            raise HTTPException(401, "Invalid email or password") from exc
        except PasswordExecutorBusyError as exc:
            raise CodedAPIError(
                503,
                "authentication_busy",
                "Authentication is temporarily busy; try again shortly",
                headers={"Retry-After": "1"},
            ) from exc
        return session_response(issued)

    @app.post(
        "/api/v1/auth/set-password",
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
            },
            status.HTTP_400_BAD_REQUEST: {"model": PasswordErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": OriginErrorResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": AuthenticationBusyResponse,
                "headers": {
                    "Retry-After": {
                        "description": "Seconds before retrying authentication.",
                        "schema": {"type": "integer"},
                    }
                },
            },
        },
    )
    async def set_password(
        request: Request,
        submission: SetPasswordRequest,
    ) -> Response:
        await require_origin(request)
        try:
            issued = await password_executor.run(
                auth.set_password,
                submission.token,
                submission.password,
            )
        except PasswordPolicyError as exc:
            raise CodedAPIError(
                status.HTTP_400_BAD_REQUEST,
                "password_policy_rejected",
                str(exc),
            ) from exc
        except InvalidPasswordTokenError as exc:
            raise CodedAPIError(
                status.HTTP_400_BAD_REQUEST,
                "password_link_invalid",
                "Password link is invalid or expired",
            ) from exc
        except PasswordExecutorBusyError as exc:
            raise CodedAPIError(
                503,
                "authentication_busy",
                "Authentication is temporarily busy; try again shortly",
                headers={"Retry-After": "1"},
            ) from exc
        return session_response(issued)

    @app.get(
        "/api/v1/auth/me",
        response_model=PrincipalView,
        responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
    )
    async def me(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> PrincipalView:
        return PrincipalView.from_principal(session.principal)

    @app.post(
        "/api/v1/auth/logout",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": MutationForbiddenResponse},
        },
    )
    async def logout(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> Response:
        token = request.cookies[session_cookie_name]
        auth.logout(token)
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(
            session_cookie_name,
            path="/",
            secure=secure_cookies,
            httponly=True,
            samesite="lax",
        )
        response.delete_cookie(
            CSRF_COOKIE,
            path="/",
            secure=secure_cookies,
            httponly=False,
            samesite="lax",
        )
        return response

    @app.get(
        "/api/v1/jobs",
        response_model=list[JobView],
        response_model_exclude_none=True,
        responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
    )
    async def list_jobs(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> list[JobView]:
        return [
            JobView.from_record(job)
            for job in store.list_jobs(session.principal.user_id)
        ]

    @app.get(
        "/api/v1/jobs/{job_id}",
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
        return JobView.from_record(job)

    @app.post(
        "/api/v1/jobs/{job_id}/cancel",
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
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> JobView:
        existing = store.get_job(session.principal.user_id, job_id)
        if existing is None:
            raise _not_found()
        registration = registrations.get(existing.workload)
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
        if (
            job.modal_call_id is not None
            and registration is not None
            and registration.cancel is not None
        ):
            try:
                await registration.cancel(job)
            except Exception:
                LOGGER.exception(
                    "Could not yet cancel job %s request_id=%s",
                    job.job_id,
                    request_id_from(request),
                )
        return JobView.from_record(job)

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
        registration = registrations.get(job.workload)
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

    @app.post(
        "/api/v1/jobs/{job_id}/prepare-download",
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
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
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

    @app.get(
        "/api/v1/jobs/{job_id}/download",
        response_class=StreamingResponse,
        responses={
            status.HTTP_200_OK: {
                "description": "Complete ZIP result archive.",
                "headers": {
                    "Content-Disposition": {
                        "description": (
                            "Browser attachment using the server-provided result "
                            "filename."
                        ),
                        "schema": {"type": "string"},
                    }
                },
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            status.HTTP_206_PARTIAL_CONTENT: {
                "description": "Requested byte range of the ZIP result archive.",
                "headers": {
                    "Content-Disposition": {
                        "description": (
                            "Browser attachment using the server-provided result "
                            "filename."
                        ),
                        "schema": {"type": "string"},
                    },
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

    for workload in workloads:
        app.include_router(workload.router)
    from biomodals.service.admin_api import create_admin_router

    app.include_router(create_admin_router())
    _document_contract_headers(app, session_cookie_name=session_cookie_name)
    return app


def create_deployed_app() -> FastAPI:
    """Create the local Linux service backed by deployed Modal compute Apps."""
    settings = ServiceSettings.from_environment()
    settings.install_modal_credentials()

    from biomodals.service.gromacs import (
        GromacsReconciler,
        ModalGromacsAdapter,
        create_registration,
    )
    from biomodals.service.jobs import JobLifecycleLocks

    store = ServiceStore(settings.database_path)
    store.initialize()
    configuration = RuntimeConfiguration(store, settings)
    auth = AuthService(store, frontend_url=settings.public_url)
    gromacs_configuration = configuration.workload("gromacs")
    cache = ArtifactCache(settings.cache_dir / "results")
    adapter = ModalGromacsAdapter(
        app_name=gromacs_configuration.modal_app_name.value,
        environment_name=configuration.modal_environment().value,
        artifact_cache=cache,
    )
    lifecycle_locks = JobLifecycleLocks()
    registration = create_registration(
        adapter,
        reconciler=GromacsReconciler(
            store,
            adapter,
            lifecycle_locks=lifecycle_locks,
            intermediate_retention_days=settings.intermediate_retention_days,
        ),
        lifecycle_locks=lifecycle_locks,
    )
    return create_app(
        store=store,
        auth=auth,
        configuration=configuration,
        workloads=[registration],
        allowed_origin=settings.public_url,
        secure_cookies=settings.secure_cookies,
        cache=cache,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
    )
