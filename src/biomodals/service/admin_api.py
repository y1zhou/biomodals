"""Administrator-only HTTP contract for Users and live Modal configuration."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from biomodals.service.auth import (
    MAX_DISPLAY_NAME_CHARACTERS,
    AuthenticatedSession,
    AuthService,
)
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    request_id_from,
    require_session,
    require_unsafe_session,
)
from biomodals.service.runtime_config import (
    EffectiveSetting,
    RuntimeConfiguration,
    SettingOverrideError,
    SettingSource,
)
from biomodals.service.store import (
    JobNotFoundError,
    JobStateResolutionError,
    LastActiveAdminError,
    ServiceStore,
    UserAlreadyExistsError,
    UserCursorError,
    UserNotFoundError,
    UserRecord,
    UserStatus,
)

LOGGER = logging.getLogger(__name__)
BlockingCategory = Literal[
    "internal_service",
    "local_storage",
    "modal_configuration",
    "modal_unavailable",
    "result_integrity",
]


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

    code: Literal["modal_preflight_failed", "setting_invalid"]


class AdminJobStateConflictResponse(CodedErrorResponse):
    """A reviewed Job no longer has unknown remote state."""

    code: Literal["job_state_changed"]


class AdminUserView(BaseModel):
    """Administrator-visible identity and admission policy."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    email: str
    display_name: str
    status: UserStatus
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
            status=user.status,
            is_admin=user.is_admin,
            active_job_limit=user.active_job_limit,
            created_at=datetime.fromtimestamp(user.created_at, UTC),
            updated_at=datetime.fromtimestamp(user.updated_at, UTC),
        )


class AdminUserPageView(BaseModel):
    """One bounded page of Administrator-visible Users."""

    users: list[AdminUserView]
    next_cursor: UUID | None = None


class CreateAdminUserRequest(BaseModel):
    """Administrator-provisioned User fields."""

    email: str = Field(min_length=3, max_length=320)
    display_name: str = Field(
        min_length=1,
        max_length=MAX_DISPLAY_NAME_CHARACTERS,
    )
    is_admin: bool = False
    active_job_limit: int | None = Field(default=None, ge=0)


class CreatedAdminUserView(BaseModel):
    """New User plus the one-time Password Link shown exactly once."""

    user: AdminUserView
    password_link: str
    expires_at: datetime


class UpdateAdminUserRequest(BaseModel):
    """Editable User presentation, status, role, and admission fields."""

    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_DISPLAY_NAME_CHARACTERS,
    )
    status: Literal["enabled", "disabled"] | None = None
    is_admin: bool | None = None
    active_job_limit: int | None = Field(default=None, ge=0)


class PasswordLinkView(BaseModel):
    """One newly issued one-time Password Link."""

    password_link: str
    expires_at: datetime


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


class BooleanSettingView(BaseModel):
    """One effective boolean setting and its controlling source."""

    value: bool
    source: SettingSource
    editable: bool


class AdminModalEnvironmentView(BaseModel):
    """Service-user identity and cross-Tool Modal settings."""

    service_token_id: str
    modal_environment: TextSettingView
    global_active_job_limit: IntegerSettingView


class AdminModalToolView(BaseModel):
    """One fixed Tool's deployment, admission, and execution limits."""

    tool: str
    display_name: str
    modal_app_name: TextSettingView
    modal_app_version: IntegerSettingView
    active_jobs: int
    active_job_limit: IntegerSettingView
    max_active_provider_calls: IntegerSettingView
    max_active_gpu_provider_calls: IntegerSettingView
    job_logs_visible_to_owner: BooleanSettingView


class AdminStateUnknownJobView(BaseModel):
    """Safe Job identity needed for manual Modal review."""

    job_id: UUID
    tool: str
    display_name: str
    reason: str
    state_unknown_at: datetime


class AdminModalView(BaseModel):
    """Complete Modal Admin page document."""

    environment: AdminModalEnvironmentView
    tools: list[AdminModalToolView]
    state_unknown_jobs: list[AdminStateUnknownJobView]
    blocked_jobs: list[AdminBlockedJobsView]


class AdminCostGroupView(BaseModel):
    """One Modal billing attribution group."""

    name: str
    cost: Decimal


class AdminCostsView(BaseModel):
    """One explicit workspace billing interval."""

    start: datetime
    end: datetime
    resolution: Literal["h", "d"]
    total: Decimal
    environments: list[AdminCostGroupView]
    tools: list[AdminCostGroupView]
    other_workspace_usage: Decimal


class AdminBlockedJobsView(BaseModel):
    """Safe aggregate of recoverable finalization failures."""

    category: BlockingCategory
    count: int
    oldest_blocked_at: datetime


class AdminStorageView(BaseModel):
    """Durable Result and rebuildable local cache accounting."""

    published_result_entries: int
    published_result_bytes: int
    local_cache_entries: int
    local_cache_bytes: int
    staging_entries: int
    staging_bytes: int
    pending_request_entries: int
    pending_request_bytes: int
    validation_entries: int
    validation_bytes: int
    free_bytes: int
    warning_threshold_bytes: int
    over_warning_threshold: bool
    reclaimable_entries: int
    reclaimable_bytes: int


class AdminCacheCleanupView(BaseModel):
    """Actual cache files and bytes removed by one explicit cleanup."""

    removed_entries: int
    removed_bytes: int


class UpdateAdminModalEnvironmentRequest(BaseModel):
    """Editable cross-Tool Modal configuration."""

    modal_environment: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Omit to keep unchanged; null restores the configured default.",
    )
    global_active_job_limit: int | None = Field(
        default=None,
        ge=0,
        description="Omit to keep unchanged; null restores the configured default.",
    )


class UpdateAdminModalToolRequest(BaseModel):
    """Editable per-Tool Modal configuration."""

    modal_app_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Omit to keep unchanged; null restores the configured default.",
    )
    modal_app_version: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Exact Modal deployment version used for new Jobs. Omit to keep "
            "unchanged; null restores the configured default."
        ),
    )
    active_job_limit: int | None = Field(
        default=None,
        ge=0,
        description="Omit to keep unchanged; null restores the configured default.",
    )
    max_active_provider_calls: int | None = Field(
        default=None,
        ge=1,
        description="Maximum active workload containers for newly admitted Runs.",
    )
    max_active_gpu_provider_calls: int | None = Field(
        default=None,
        ge=1,
        description="Maximum active GPU containers within the total limit.",
    )
    job_logs_visible_to_owner: bool | None = Field(
        default=None,
        description=(
            "Whether authenticated Job owners may inspect provider logs. "
            "Administrators always retain access. Omit to keep unchanged; "
            "null restores the Tool default."
        ),
    )


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


def _boolean_view(setting: EffectiveSetting[bool]) -> BooleanSettingView:
    return BooleanSettingView(
        value=setting.value,
        source=setting.source,
        editable=setting.editable,
    )


def _modal_view(
    configuration: RuntimeConfiguration,
    store: ServiceStore,
) -> AdminModalView:
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
                tool=tool.tool,
                display_name=configuration.tool_definition(tool.tool).display_name,
                modal_app_name=_text_view(tool.modal_app_name),
                modal_app_version=_integer_view(tool.modal_app_version),
                active_jobs=store.count_active_jobs(tool.tool),
                active_job_limit=_integer_view(tool.active_job_limit),
                max_active_provider_calls=_integer_view(tool.max_active_provider_calls),
                max_active_gpu_provider_calls=_integer_view(
                    tool.max_active_gpu_provider_calls
                ),
                job_logs_visible_to_owner=_boolean_view(tool.job_logs_visible_to_owner),
            )
            for tool in (
                configuration.tool(name) for name in configuration.tool_names()
            )
        ],
        state_unknown_jobs=[
            AdminStateUnknownJobView(
                job_id=job.job_id,
                tool=job.tool,
                display_name=job.display_name,
                reason=job.state_reason,
                state_unknown_at=datetime.fromtimestamp(job.updated_at, UTC),
            )
            for job in store.list_state_unknown_jobs()
            if job.state_reason is not None
        ],
        blocked_jobs=[
            AdminBlockedJobsView(
                category=cast(BlockingCategory, summary.category),
                count=summary.count,
                oldest_blocked_at=datetime.fromtimestamp(
                    summary.oldest_blocked_at,
                    UTC,
                ),
            )
            for summary in store.blocked_job_summaries()
        ],
    )


async def _storage_view(request: Request) -> AdminStorageView:
    cache = request.app.state.cache
    if cache is None:
        raise HTTPException(503, "Result cache is unavailable")
    usage = await cache.usage_async()
    store: ServiceStore = request.app.state.store
    published = store.published_result_usage()
    pending = (
        await asyncio.to_thread(request.app.state.pending_requests.usage)
        if request.app.state.pending_requests is not None
        else (0, 0)
    )
    validations = (
        await asyncio.to_thread(request.app.state.validated_inputs.usage)
        if request.app.state.validated_inputs is not None
        else (0, 0)
    )
    threshold = request.app.state.configuration.settings.cache_warning_bytes
    return AdminStorageView(
        published_result_entries=published.entries,
        published_result_bytes=published.bytes,
        local_cache_entries=usage.cached_entries,
        local_cache_bytes=usage.cached_bytes,
        staging_entries=usage.staging_entries,
        staging_bytes=usage.staging_bytes,
        pending_request_entries=pending[0],
        pending_request_bytes=pending[1],
        validation_entries=validations[0],
        validation_bytes=validations[1],
        free_bytes=usage.free_bytes,
        warning_threshold_bytes=threshold,
        over_warning_threshold=(
            usage.cached_bytes + usage.staging_bytes + pending[1] + validations[1]
            > threshold
        ),
        reclaimable_entries=usage.reclaimable_entries,
        reclaimable_bytes=usage.reclaimable_bytes,
    )


def create_admin_router() -> APIRouter:
    """Create the administrator-only API router."""
    router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
    runtime_configuration_lock = asyncio.Lock()

    async def serialize_runtime_configuration_mutation() -> AsyncIterator[None]:
        """Keep provider preflight and its corresponding commit indivisible."""
        async with runtime_configuration_lock:
            yield

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
        response_model=AdminUserPageView,
        responses={**read_responses, 400: {"model": ErrorResponse}},
    )
    async def list_users(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        cursor: UUID | None = None,
    ) -> AdminUserPageView:
        store: ServiceStore = request.app.state.store
        try:
            page = store.list_users_page(limit=limit, cursor=cursor)
        except UserCursorError as exc:
            raise HTTPException(400, str(exc)) from exc
        return AdminUserPageView(
            users=[AdminUserView.from_record(user) for user in page.users],
            next_cursor=page.next_cursor,
        )

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
            password_link=link.url,
            expires_at=datetime.fromtimestamp(link.expires_at, UTC),
        )

    @router.patch(
        "/users/{user_id}",
        response_model=AdminUserView,
        responses={
            **mutation_responses,
            400: {"model": AdminUserInvalidResponse},
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
                display_name=submission.display_name,
                active=(
                    submission.status == "enabled"
                    if submission.status is not None
                    else None
                ),
                is_admin=submission.is_admin,
                active_job_limit=submission.active_job_limit,
                now=int(time.time()),
            )
        except ValueError as exc:
            raise CodedAPIError(400, "user_invalid", str(exc)) from exc
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
        if user.status == UserStatus.DISABLED:
            raise CodedAPIError(409, "user_inactive", "User is disabled")
        auth: AuthService = request.app.state.auth
        try:
            link = auth.create_password_reset(user.email)
        except UserNotFoundError as exc:
            raise CodedAPIError(409, "user_inactive", "User is disabled") from exc
        return PasswordLinkView(
            password_link=link.url,
            expires_at=datetime.fromtimestamp(link.expires_at, UTC),
        )

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

    @router.get(
        "/storage",
        response_model=AdminStorageView,
        responses={**read_responses, 503: {"model": ErrorResponse}},
    )
    async def storage_configuration(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
    ) -> AdminStorageView:
        return await _storage_view(request)

    @router.get("/modal/costs", response_model=AdminCostsView)
    async def modal_costs(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_admin)],
        start: datetime,
        end: datetime,
        refresh: bool = False,
    ) -> AdminCostsView:
        """Return optional Modal billing without affecting service readiness."""
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise HTTPException(400, "Billing interval must be timezone-aware")
        resolution: Literal["h", "d"] = (
            "h" if (end - start).total_seconds() <= 2 * 86400 else "d"
        )
        try:
            summary = await request.app.state.billing.report(
                start=start,
                end=end,
                resolution=resolution,
                refresh=refresh,
            )
        except Exception as error:
            LOGGER.exception("Modal billing report is unavailable")
            raise CodedAPIError(
                503,
                "modal_billing_unavailable",
                "Modal billing is unavailable for this workspace or interval",
            ) from error
        return AdminCostsView(
            start=start,
            end=end,
            resolution=resolution,
            total=summary.total,
            environments=[
                AdminCostGroupView(name=item.name, cost=item.cost)
                for item in summary.environments
            ],
            tools=[
                AdminCostGroupView(name=item.name, cost=item.cost)
                for item in summary.tools
            ],
            other_workspace_usage=summary.other,
        )

    @router.post(
        "/storage/cache/clear",
        response_model=AdminCacheCleanupView,
        responses={**mutation_responses, 503: {"model": ErrorResponse}},
    )
    async def clear_result_cache(
        request: Request,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> AdminCacheCleanupView:
        cache = request.app.state.cache
        if cache is None:
            raise HTTPException(503, "Result cache is unavailable")
        result = await cache.clear_async()
        store: ServiceStore = request.app.state.store
        store.mark_result_cache_cleared(result.job_ids)
        LOGGER.info(
            "event=result_cache_cleared entries=%s bytes=%s request_id=%s",
            result.entries,
            result.bytes,
            request_id_from(request),
        )
        return AdminCacheCleanupView(
            removed_entries=result.entries,
            removed_bytes=result.bytes,
        )

    @router.post(
        "/modal/state-unknown-jobs/{job_id}/resolve",
        response_model=AdminModalView,
        responses={
            **mutation_responses,
            404: {"model": ErrorResponse},
            409: {"model": AdminJobStateConflictResponse},
        },
    )
    async def resolve_state_unknown_job(
        request: Request,
        job_id: UUID,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> AdminModalView:
        store: ServiceStore = request.app.state.store
        try:
            store.resolve_state_unknown(job_id, now=int(time.time()))
        except JobNotFoundError as exc:
            raise HTTPException(404, "Job not found") from exc
        except JobStateResolutionError as exc:
            raise CodedAPIError(
                409,
                "job_state_changed",
                "This Job no longer has unknown remote state",
            ) from exc
        LOGGER.info(
            "event=state_unknown_resolved job_id=%s resolution=reconcile request_id=%s",
            job_id,
            request_id_from(request),
        )
        return _modal_view(request.app.state.configuration, store)

    @router.patch(
        "/modal/environment",
        response_model=AdminModalEnvironmentView,
        dependencies=[Depends(serialize_runtime_configuration_mutation)],
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
        if "modal_environment" in submission.model_fields_set:
            if not configuration.modal_environment().editable:
                raise CodedAPIError(
                    409,
                    "setting_overridden",
                    "BIOMODALS_MODAL_ENVIRONMENT is controlled by an environment "
                    "variable",
                )
            candidate_environment = (
                configuration.settings.modal_environment
                if submission.modal_environment is None
                else submission.modal_environment.strip()
            )
            if not candidate_environment:
                raise CodedAPIError(
                    400,
                    "setting_invalid",
                    "Modal environment must not be empty",
                )
            try:
                from biomodals.execution import DeploymentIdentity

                for tool_name in request.app.state.registrations:
                    effective = configuration.tool(tool_name)
                    await request.app.state.remote_execution.preflight(
                        DeploymentIdentity(
                            candidate_environment,
                            effective.modal_app_name.value,
                            effective.modal_app_version.value,
                        )
                    )
            except Exception as exc:
                LOGGER.exception(
                    "Modal Environment preflight failed request_id=%s",
                    request_id_from(request),
                )
                raise CodedAPIError(
                    400,
                    "modal_preflight_failed",
                    "The configured Modal resources could not be validated",
                ) from exc
        try:
            configuration.update_environment(
                **submission.model_dump(exclude_unset=True)
            )
        except SettingOverrideError as exc:
            raise CodedAPIError(409, "setting_overridden", str(exc)) from exc
        except ValueError as exc:
            raise CodedAPIError(400, "setting_invalid", str(exc)) from exc
        LOGGER.info(
            "event=runtime_setting_changed scope=modal_environment fields=%s "
            "request_id=%s",
            ",".join(sorted(submission.model_fields_set)),
            request_id_from(request),
        )
        return _modal_view(configuration, request.app.state.store).environment

    @router.patch(
        "/modal/tools/{tool}",
        response_model=AdminModalToolView,
        dependencies=[Depends(serialize_runtime_configuration_mutation)],
        responses={
            **mutation_responses,
            400: {"model": AdminSettingInvalidResponse},
            404: {"model": ErrorResponse},
            409: {"model": AdminSettingConflictResponse},
        },
    )
    async def update_modal_tool(
        request: Request,
        tool: str,
        submission: UpdateAdminModalToolRequest,
        _session: Annotated[AuthenticatedSession, Depends(require_unsafe_admin)],
    ) -> AdminModalToolView:
        configuration: RuntimeConfiguration = request.app.state.configuration
        registration = request.app.state.registrations.get(tool)
        if registration is None:
            raise HTTPException(404, "Tool not found")
        provider_fields = {"modal_app_name", "modal_app_version"}
        if submission.model_fields_set & provider_fields:
            effective = configuration.tool(tool)
            definition = configuration.tool_definition(tool)
            if (
                "modal_app_name" in submission.model_fields_set
                and not effective.modal_app_name.editable
            ):
                raise CodedAPIError(
                    409,
                    "setting_overridden",
                    f"{definition.modal_app_name_environment} is controlled by "
                    "an environment variable",
                )
            if (
                "modal_app_version" in submission.model_fields_set
                and not effective.modal_app_version.editable
            ):
                raise CodedAPIError(
                    409,
                    "setting_overridden",
                    f"{definition.modal_app_version_environment} is controlled "
                    "by an environment variable",
                )
            candidate_app_name = effective.modal_app_name.value
            if "modal_app_name" in submission.model_fields_set:
                candidate_app_name = (
                    configuration.modal_app_name_fallback(tool)
                    if submission.modal_app_name is None
                    else submission.modal_app_name.strip()
                )
            if not candidate_app_name:
                raise CodedAPIError(
                    400,
                    "setting_invalid",
                    "Modal app name must not be empty",
                )
            candidate_app_version = effective.modal_app_version.value
            if "modal_app_version" in submission.model_fields_set:
                candidate_app_version = (
                    configuration.modal_app_version_fallback(tool)
                    if submission.modal_app_version is None
                    else submission.modal_app_version
                )
            try:
                from biomodals.execution import DeploymentIdentity

                await request.app.state.remote_execution.preflight(
                    DeploymentIdentity(
                        configuration.modal_environment().value,
                        candidate_app_name,
                        candidate_app_version,
                    )
                )
            except Exception as exc:
                LOGGER.exception(
                    "Modal App preflight failed for %s request_id=%s",
                    tool,
                    request_id_from(request),
                )
                raise CodedAPIError(
                    400,
                    "modal_preflight_failed",
                    "The exact deployed ExecutionCoordinator is unavailable or incompatible",
                ) from exc
        try:
            configuration.set_tool(tool, **submission.model_dump(exclude_unset=True))
        except SettingOverrideError as exc:
            raise CodedAPIError(409, "setting_overridden", str(exc)) from exc
        except ValueError as exc:
            raise CodedAPIError(400, "setting_invalid", str(exc)) from exc
        LOGGER.info(
            "event=runtime_setting_changed scope=tool tool=%s fields=%s request_id=%s",
            tool,
            ",".join(sorted(submission.model_fields_set)),
            request_id_from(request),
        )
        view = _modal_view(configuration, request.app.state.store)
        return next(item for item in view.tools if item.tool == tool)

    return router
