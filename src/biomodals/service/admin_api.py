"""Administrator-only HTTP contract for Users and live Modal configuration."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from biomodals.service.api import (
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    require_session,
    require_unsafe_session,
)
from biomodals.service.auth import AuthenticatedSession, AuthService
from biomodals.service.runtime_config import (
    EffectiveSetting,
    RuntimeConfiguration,
    SettingOverrideError,
    SettingSource,
)
from biomodals.service.store import (
    LastActiveAdminError,
    ServiceStore,
    UserAlreadyExistsError,
    UserNotFoundError,
    UserRecord,
)


class AdminForbiddenResponse(CodedErrorResponse):
    """Authenticated User lacks administrator access."""

    code: Literal["admin_required"]


class AdminMutationForbiddenResponse(CodedErrorResponse):
    """Admin mutation rejected for authorization, origin, or CSRF state."""

    code: Literal["admin_required", "csrf_invalid", "origin_not_allowed"]


class AdminUserAlreadyExistsResponse(CodedErrorResponse):
    """Provisioning reused an existing normalized email address."""

    code: Literal["user_already_exists"]


class LastActiveAdminResponse(CodedErrorResponse):
    """A User update would remove the final active administrator."""

    code: Literal["last_active_admin"]


class InactiveUserResponse(CodedErrorResponse):
    """A Password Link was requested for a disabled User."""

    code: Literal["user_inactive"]


class AdminUserInvalidResponse(CodedErrorResponse):
    """Structurally valid provisioning fields failed identity validation."""

    code: Literal["user_invalid"]


class AdminSettingConflictResponse(CodedErrorResponse):
    """Database edit is shadowed by a process environment variable."""

    code: Literal["setting_overridden"]


class AdminSettingInvalidResponse(CodedErrorResponse):
    """Structurally valid Runtime Setting content failed validation."""

    code: Literal["setting_invalid"]


class AdminUserView(BaseModel):
    """Administrator-visible identity and admission policy."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    email: str
    display_name: str
    active: bool
    is_admin: bool
    active_job_limit: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, user: UserRecord) -> AdminUserView:
        """Convert a persisted User without exposing password state."""
        return cls(
            user_id=user.user_id,
            email=user.email,
            display_name=user.display_name,
            active=user.active,
            is_admin=user.is_admin,
            active_job_limit=user.active_job_limit,
            created_at=datetime.fromtimestamp(user.created_at, UTC),
            updated_at=datetime.fromtimestamp(user.updated_at, UTC),
        )


class CreateAdminUserRequest(BaseModel):
    """Administrator-provisioned User fields."""

    email: str = Field(min_length=3, max_length=320)
    display_name: str = Field(min_length=1, max_length=120)
    is_admin: bool = False
    active_job_limit: int | None = Field(default=None, ge=1)


class CreatedAdminUserView(BaseModel):
    """New User plus the one-time Password Link shown exactly once."""

    user: AdminUserView
    password_link: str


class UpdateAdminUserRequest(BaseModel):
    """Editable User status and admission policy fields."""

    active: bool | None = None
    is_admin: bool | None = None
    active_job_limit: int | None = Field(default=None, ge=1)


class PasswordLinkView(BaseModel):
    """One newly issued one-time Password Link."""

    password_link: str


class TextSettingView(BaseModel):
    """One effective text setting and its controlling source."""

    value: str
    source: SettingSource
    editable: bool


class IntegerSettingView(BaseModel):
    """One effective integer setting and its controlling source."""

    value: int
    source: SettingSource
    editable: bool


class AdminModalEnvironmentView(BaseModel):
    """Service-user identity and cross-Tool Modal settings."""

    service_token_id: str
    modal_environment: TextSettingView
    global_active_job_limit: IntegerSettingView


class AdminModalToolView(BaseModel):
    """One fixed workload's deployment and admission state."""

    workload: str
    modal_app_name: TextSettingView
    running_jobs: int
    active_job_limit: IntegerSettingView


class AdminModalView(BaseModel):
    """Complete Modal Admin page document."""

    environment: AdminModalEnvironmentView
    tools: list[AdminModalToolView]


class UpdateAdminModalEnvironmentRequest(BaseModel):
    """Editable cross-Tool Modal configuration."""

    modal_environment: str | None = Field(default=None, min_length=1, max_length=120)
    global_active_job_limit: int | None = Field(default=None, ge=1)


class UpdateAdminModalToolRequest(BaseModel):
    """Editable per-Tool Modal configuration."""

    modal_app_name: str | None = Field(default=None, min_length=1, max_length=120)
    active_job_limit: int | None = Field(default=None, ge=1)


async def require_admin(
    session: Annotated[AuthenticatedSession, Depends(require_session)],
) -> AuthenticatedSession:
    """Require an authenticated administrator for a read."""
    if not session.principal.is_admin:
        raise CodedAPIError(403, "admin_required", "Administrator access required")
    return session


async def require_unsafe_admin(
    session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
) -> AuthenticatedSession:
    """Require administrator identity, exact Origin, and CSRF for a mutation."""
    if not session.principal.is_admin:
        raise CodedAPIError(403, "admin_required", "Administrator access required")
    return session


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "User not found")


def _text_view(setting: EffectiveSetting[str]) -> TextSettingView:
    return TextSettingView(
        value=setting.value,
        source=setting.source,
        editable=setting.editable,
    )


def _integer_view(setting: EffectiveSetting[int]) -> IntegerSettingView:
    return IntegerSettingView(
        value=setting.value,
        source=setting.source,
        editable=setting.editable,
    )


def _modal_view(
    configuration: RuntimeConfiguration,
    store: ServiceStore,
) -> AdminModalView:
    workload = configuration.workload("gromacs")
    return AdminModalView(
        environment=AdminModalEnvironmentView(
            service_token_id=configuration.modal_token_id,
            modal_environment=_text_view(configuration.modal_environment()),
            global_active_job_limit=_integer_view(
                configuration.global_active_job_limit()
            ),
        ),
        tools=[
            AdminModalToolView(
                workload=workload.workload,
                modal_app_name=_text_view(workload.modal_app_name),
                running_jobs=store.count_running_jobs(workload.workload),
                active_job_limit=_integer_view(workload.active_job_limit),
            )
        ],
    )


def create_admin_router() -> APIRouter:
    """Create the administrator-only API router."""
    router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
    read_responses: dict[int | str, dict[str, Any]] = {
        401: {"model": ErrorResponse},
        403: {"model": AdminForbiddenResponse},
    }
    mutation_responses: dict[int | str, dict[str, Any]] = {
        401: {"model": ErrorResponse},
        403: {"model": AdminMutationForbiddenResponse},
    }

    @router.get(
        "/users",
        response_model=list[AdminUserView],
        responses=read_responses,
    )
    async def list_users(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
    ) -> list[AdminUserView]:
        store: ServiceStore = request.app.state.store
        return [AdminUserView.from_record(user) for user in store.list_users()]

    @router.post(
        "/users",
        response_model=CreatedAdminUserView,
        status_code=status.HTTP_201_CREATED,
        responses={
            **mutation_responses,
            400: {"model": AdminUserInvalidResponse},
            409: {"model": AdminUserAlreadyExistsResponse},
        },
    )
    async def create_user(
        request: Request,
        submission: CreateAdminUserRequest,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> CreatedAdminUserView:
        auth: AuthService = request.app.state.auth
        configuration: RuntimeConfiguration = request.app.state.configuration
        try:
            link = auth.create_user(
                submission.email,
                display_name=submission.display_name,
                is_admin=submission.is_admin,
                active_job_limit=(
                    configuration.default_user_active_job_limit
                    if submission.active_job_limit is None
                    else submission.active_job_limit
                ),
            )
        except UserAlreadyExistsError as exc:
            raise CodedAPIError(409, "user_already_exists", str(exc)) from exc
        except ValueError as exc:
            raise CodedAPIError(400, "user_invalid", str(exc)) from exc
        store: ServiceStore = request.app.state.store
        user = store.get_user_by_email(submission.email.strip().casefold())
        if user is None:  # pragma: no cover - committed creation guarantees this
            raise RuntimeError("Created User could not be loaded")
        return CreatedAdminUserView(
            user=AdminUserView.from_record(user),
            password_link=link,
        )

    @router.patch(
        "/users/{user_id}",
        response_model=AdminUserView,
        responses={
            **mutation_responses,
            404: {"model": ErrorResponse},
            409: {"model": LastActiveAdminResponse},
        },
    )
    async def update_user(
        request: Request,
        user_id: UUID,
        submission: UpdateAdminUserRequest,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> AdminUserView:
        store: ServiceStore = request.app.state.store
        try:
            user = store.update_user(
                user_id,
                active=submission.active,
                is_admin=submission.is_admin,
                active_job_limit=submission.active_job_limit,
                now=int(time.time()),
            )
        except UserNotFoundError as exc:
            raise _not_found() from exc
        except LastActiveAdminError as exc:
            raise CodedAPIError(409, "last_active_admin", str(exc)) from exc
        return AdminUserView.from_record(user)

    @router.post(
        "/users/{user_id}/password-link",
        response_model=PasswordLinkView,
        responses={
            **mutation_responses,
            404: {"model": ErrorResponse},
            409: {"model": InactiveUserResponse},
        },
    )
    async def create_password_link(
        request: Request,
        user_id: UUID,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> PasswordLinkView:
        store: ServiceStore = request.app.state.store
        user = store.get_user(user_id)
        if user is None:
            raise _not_found()
        if not user.active:
            raise CodedAPIError(409, "user_inactive", "User is disabled")
        auth: AuthService = request.app.state.auth
        try:
            link = auth.create_password_reset(user.email)
        except UserNotFoundError as exc:
            raise CodedAPIError(409, "user_inactive", "User is disabled") from exc
        return PasswordLinkView(password_link=link)

    @router.get(
        "/modal",
        response_model=AdminModalView,
        responses=read_responses,
    )
    async def modal_configuration(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
    ) -> AdminModalView:
        configuration: RuntimeConfiguration = request.app.state.configuration
        store: ServiceStore = request.app.state.store
        return _modal_view(configuration, store)

    @router.patch(
        "/modal/environment",
        response_model=AdminModalEnvironmentView,
        responses={
            **mutation_responses,
            400: {"model": AdminSettingInvalidResponse},
            409: {"model": AdminSettingConflictResponse},
        },
    )
    async def update_modal_environment(
        request: Request,
        submission: UpdateAdminModalEnvironmentRequest,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> AdminModalEnvironmentView:
        configuration: RuntimeConfiguration = request.app.state.configuration
        try:
            configuration.update_environment(
                modal_environment=submission.modal_environment,
                global_active_job_limit=submission.global_active_job_limit,
            )
        except SettingOverrideError as exc:
            raise CodedAPIError(409, "setting_overridden", str(exc)) from exc
        except ValueError as exc:
            raise CodedAPIError(400, "setting_invalid", str(exc)) from exc
        return _modal_view(configuration, request.app.state.store).environment

    @router.patch(
        "/modal/tools/{workload}",
        response_model=AdminModalToolView,
        responses={
            **mutation_responses,
            400: {"model": AdminSettingInvalidResponse},
            404: {"model": ErrorResponse},
            409: {"model": AdminSettingConflictResponse},
        },
    )
    async def update_modal_tool(
        request: Request,
        workload: str,
        submission: UpdateAdminModalToolRequest,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> AdminModalToolView:
        configuration: RuntimeConfiguration = request.app.state.configuration
        try:
            configuration.set_workload(
                workload,
                modal_app_name=submission.modal_app_name,
                active_job_limit=submission.active_job_limit,
            )
        except SettingOverrideError as exc:
            raise CodedAPIError(409, "setting_overridden", str(exc)) from exc
        except ValueError as exc:
            if str(exc).startswith("Unknown workload"):
                raise HTTPException(404, "Tool not found") from exc
            raise CodedAPIError(400, "setting_invalid", str(exc)) from exc
        view = _modal_view(configuration, request.app.state.store)
        return next(tool for tool in view.tools if tool.workload == workload)

    return router
