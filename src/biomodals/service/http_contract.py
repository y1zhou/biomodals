"""Shared HTTP behavior for Biomodals API route modules."""

from __future__ import annotations

import logging
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from biomodals.service.auth import AuthenticatedSession, AuthService

LOGGER = logging.getLogger(__name__)
SESSION_COOKIE = "biomodals-session"
SECURE_SESSION_COOKIE = "__Host-biomodals-session"
CSRF_COOKIE = "biomodals-csrf"
_CSRF_HEADER_DESCRIPTION = (
    f"Required for authenticated mutations. Copy the value of the `{CSRF_COOKIE}` "
    "cookie set by a successful login or Password Setup."
)


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

        async def limited_receive() -> Message:
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


def install_http_contract(app: FastAPI, *, max_body_bytes: int) -> None:
    """Install common errors, request limits, IDs, and safe failure logging."""

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

    app.add_middleware(_RequestSizeMiddleware, max_body_bytes=max_body_bytes)
    app.add_middleware(_RequestIdMiddleware)


def document_contract_headers(app: FastAPI, *, session_cookie_name: str) -> None:
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
