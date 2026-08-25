"""FastAPI assembly for the Biomodals service control plane."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, status

from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthService, PasswordExecutor
from biomodals.service.auth_api import create_auth_router
from biomodals.service.billing import BillingService
from biomodals.service.http_contract import (
    SECURE_SESSION_COOKIE,
    SESSION_COOKIE,
    ErrorResponse,
    PayloadTooLargeResponse,
    document_contract_headers,
    install_http_contract,
)
from biomodals.service.jobs_api import create_jobs_router
from biomodals.service.operations_api import create_operations_router
from biomodals.service.remote_execution import RemoteExecutionClient
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import ServiceStore
from biomodals.service.tool_runtime import (
    JobLifecycle,
    ToolRegistration,
    reconciliation_loop,
)

LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    store: ServiceStore,
    auth: AuthService,
    configuration: RuntimeConfiguration,
    registrations: Sequence[ToolRegistration],
    tool_routers: Sequence[APIRouter],
    remote: RemoteExecutionClient,
    lifecycle: JobLifecycle,
    cache: ArtifactCache,
    allowed_origin: str,
    secure_cookies: bool,
    reconcile_interval_seconds: float = 60,
) -> FastAPI:
    """Assemble explicitly registered Tools around one shared lifecycle."""
    if len(registrations) != len(tool_routers):
        raise ValueError("Every Tool registration requires one typed router")
    if (
        tuple(item.definition.key for item in registrations)
        != configuration.tool_names()
    ):
        raise ValueError("Runtime Tool definitions must match registrations")
    if reconcile_interval_seconds <= 0:
        raise ValueError("reconcile_interval_seconds must be positive")
    if not allowed_origin or allowed_origin.endswith("/"):
        raise ValueError("allowed_origin must be an exact origin without a slash")
    session_cookie_name = SECURE_SESSION_COOKIE if secure_cookies else SESSION_COOKIE
    password_executor = PasswordExecutor()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stop = asyncio.Event()
        for registration in registrations:
            effective = configuration.tool(registration.definition.key)
            from biomodals.execution import DeploymentIdentity

            await remote.preflight(
                DeploymentIdentity(
                    configuration.modal_environment().value,
                    effective.modal_app_name.value,
                    effective.modal_app_version.value,
                )
            )
        await cache.check_ready_async()
        store.reconcile_result_cache(await cache.cached_job_ids_async())
        task = asyncio.create_task(
            reconciliation_loop(
                lifecycle,
                interval_seconds=reconcile_interval_seconds,
                stop=stop,
            ),
            name="biomodals-job-reconciler",
        )
        app.state.reconciler_task = task
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            stop.set()
            await task
            await password_executor.shutdown()
            await cache.shutdown()

    app = FastAPI(
        title="Biomodals API",
        version="1.0.0",
        lifespan=lifespan,
        responses={
            status.HTTP_413_CONTENT_TOO_LARGE: {
                "model": PayloadTooLargeResponse,
                "description": "Request Entity Too Large",
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        },
    )
    app.state.store = store
    app.state.auth = auth
    app.state.configuration = configuration
    app.state.registrations = {
        registration.definition.key: registration for registration in registrations
    }
    app.state.remote_execution = remote
    app.state.lifecycle = lifecycle
    app.state.cache = cache
    app.state.billing = BillingService()
    app.state.pending_requests = None
    app.state.validated_inputs = None
    app.state.allowed_origin = allowed_origin
    app.state.session_cookie_name = session_cookie_name
    app.state.ready = False
    install_http_contract(app, max_body_bytes=256 * 1024 * 1024)
    app.include_router(create_operations_router(store=store, cache=cache))
    app.include_router(
        create_auth_router(
            auth=auth,
            password_executor=password_executor,
            secure_cookies=secure_cookies,
            session_cookie_name=session_cookie_name,
        )
    )
    app.include_router(
        create_jobs_router(
            store=store,
            configuration=configuration,
            lifecycle=lifecycle,
            cache=cache,
        )
    )
    for router in tool_routers:
        app.include_router(router)

    from biomodals.service.admin_api import create_admin_router
    from biomodals.service.job_logs_api import create_job_logs_router

    app.include_router(create_admin_router())
    app.include_router(create_job_logs_router())
    document_contract_headers(app, session_cookie_name=session_cookie_name)
    return app


def create_deployed_app() -> FastAPI:
    """Create the Linux service backed by exact deployed Modal coordinators."""
    from biomodals.service.alphafold3.modal import AlphaFold3ToolAdapter
    from biomodals.service.alphafold3.router import create_router as af3_router
    from biomodals.service.alphafold3.validation import ValidatedInputStore
    from biomodals.service.config import ServiceSettings
    from biomodals.service.gromacs.modal import GromacsToolAdapter
    from biomodals.service.gromacs.router import create_router as gromacs_router
    from biomodals.service.pending import PendingRequestStore
    from biomodals.service.tools import ALPHAFOLD3_TOOL, GROMACS_TOOL, TOOLS

    settings = ServiceSettings.from_environment()
    settings.install_modal_credentials()
    store = ServiceStore(settings.database_path)
    store.initialize()
    pending = PendingRequestStore(settings.state_dir)
    pending.initialize()
    validations = ValidatedInputStore(settings.state_dir)
    validations.initialize()
    validations.cleanup_expired(
        claimed=store.claimed_validation_ids(),
        now=int(time.time()),
    )
    pending.cleanup_orphans(retained=store.unstaged_job_ids())
    cache = ArtifactCache(settings.cache_dir / "results")
    remote = RemoteExecutionClient()
    configuration = RuntimeConfiguration(store, settings, tool_definitions=TOOLS)
    gromacs = ToolRegistration(GROMACS_TOOL, GromacsToolAdapter(pending))
    alphafold3 = ToolRegistration(
        ALPHAFOLD3_TOOL,
        AlphaFold3ToolAdapter(
            validations,
            store,
            modal_download_concurrency=settings.modal_download_concurrency,
        ),
    )
    registrations = (gromacs, alphafold3)
    lifecycle = JobLifecycle(store, remote, registrations, cache)
    routers = (
        gromacs_router(
            store=store,
            configuration=configuration,
            pending=pending,
            remote=remote,
            lifecycle=lifecycle,
        ),
        af3_router(
            store=store,
            configuration=configuration,
            validations=validations,
            remote=remote,
            lifecycle=lifecycle,
        ),
    )
    auth = AuthService(store, frontend_url=settings.public_url)
    app = create_app(
        store=store,
        auth=auth,
        configuration=configuration,
        registrations=registrations,
        tool_routers=routers,
        remote=remote,
        lifecycle=lifecycle,
        cache=cache,
        allowed_origin=settings.public_url,
        secure_cookies=settings.secure_cookies,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
    )
    app.state.pending_requests = pending
    app.state.validated_inputs = validations
    return app
