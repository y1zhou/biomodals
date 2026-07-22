"""FastAPI application assembly for the Biomodals control plane."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI, status

from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthService, PasswordExecutor
from biomodals.service.auth_api import create_auth_router
from biomodals.service.config import ServiceSettings
from biomodals.service.http_contract import (
    SECURE_SESSION_COOKIE,
    SESSION_COOKIE,
    ErrorResponse,
    PayloadTooLargeResponse,
    document_contract_headers,
    install_http_contract,
)
from biomodals.service.jobs import WorkloadRegistration, reconciliation_loop
from biomodals.service.jobs_api import create_jobs_router
from biomodals.service.operations_api import create_operations_router
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import ServiceStore

LOGGER = logging.getLogger(__name__)


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

    install_http_contract(
        app,
        max_body_bytes=max(
            (workload.max_body_bytes for workload in workloads),
            default=1024 * 1024,
        ),
    )
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
            workloads=registrations,
            cache=cache,
        )
    )
    for workload in workloads:
        app.include_router(workload.router)

    from biomodals.service.admin_api import create_admin_router
    from biomodals.service.admin_jobs_api import create_admin_jobs_router

    app.include_router(create_admin_router())
    app.include_router(create_admin_jobs_router())
    document_contract_headers(app, session_cookie_name=session_cookie_name)
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
    cache = ArtifactCache(settings.cache_dir / "results")
    adapter = ModalGromacsAdapter(
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
        read_artifact=adapter.read_artifact,
        rebuild_artifact=adapter.rebuild_artifact,
        open_operation_logs=adapter.open_operation_logs,
        preflight=adapter.preflight,
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
