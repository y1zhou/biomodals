"""Browser authentication routes for the Biomodals API."""

from __future__ import annotations

import logging
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

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
from biomodals.service.http_contract import (
    CSRF_COOKIE,
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    MutationForbiddenResponse,
    OriginErrorResponse,
    model_response,
    request_id_from,
    require_origin,
    require_session,
    require_unsafe_session,
)

LOGGER = logging.getLogger(__name__)


class PasswordErrorResponse(CodedErrorResponse):
    """Password Setup errors with distinct recovery behavior."""

    code: Literal[
        "password_link_invalid",
        "password_policy_rejected",
    ]


class AuthenticationBusyResponse(CodedErrorResponse):
    """Bounded Argon2 capacity is temporarily exhausted."""

    code: Literal["authentication_busy"]


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


def create_auth_router(
    *,
    auth: AuthService,
    password_executor: PasswordExecutor,
    secure_cookies: bool,
    session_cookie_name: str,
) -> APIRouter:
    """Create browser authentication routes with explicit runtime services."""
    router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

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

    @router.post(
        "/login",
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

    @router.post(
        "/set-password",
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

    @router.get(
        "/me",
        response_model=PrincipalView,
        responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
    )
    async def me(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> PrincipalView:
        return PrincipalView.from_principal(session.principal)

    @router.post(
        "/logout",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": MutationForbiddenResponse},
        },
    )
    async def logout(
        request: Request,
        _session: Annotated[
            AuthenticatedSession,
            Depends(require_unsafe_session),
        ],
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

    return router
