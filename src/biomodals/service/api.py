"""Shared FastAPI construction for Biomodals app and workflow services."""

from __future__ import annotations

from secrets import compare_digest

from fastapi import FastAPI, status
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class ErrorResponse(BaseModel):
    """Stable JSON error response."""

    detail: str


def model_response(model: BaseModel, *, status_code: int) -> Response:
    """Serialize a Pydantic model into an explicit JSON response."""
    return Response(
        content=model.model_dump_json(exclude_none=True),
        status_code=status_code,
        media_type="application/json",
    )


class _RequestGuardMiddleware:
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
        self.api_key = api_key.encode() if api_key is not None else None

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
                or not compare_digest(token, self.api_key)
            ):
                response = model_response(
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
                await model_response(
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
                await model_response(
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


def create_api(
    *,
    title: str,
    version: str,
    max_body_bytes: int,
    api_key: str | None = None,
    trusted_proxy_auth: bool = False,
) -> FastAPI:
    """Create a guarded FastAPI app for one Biomodals service module."""
    if not api_key and not trusted_proxy_auth:
        raise ValueError("Configure api_key or explicitly trust upstream proxy auth")

    web_app = FastAPI(title=title, version=version)
    web_app.add_middleware(
        _RequestGuardMiddleware,
        max_body_bytes=max_body_bytes,
        api_key=api_key,
    )
    return web_app
