"""Current browser API route and authorization contracts."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from biomodals.service.alphafold3.router import create_router as af3_router
from biomodals.service.alphafold3.validation import ValidatedInputStore
from biomodals.service.api import create_app
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import (
    AuthenticatedSession,
    AuthService,
    Principal,
)
from biomodals.service.config import ServiceSettings
from biomodals.service.gromacs.router import create_router as gromacs_router
from biomodals.service.http_contract import require_session, require_unsafe_session
from biomodals.service.pending import PendingRequestStore
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import ServiceStore
from biomodals.service.tool_runtime import JobLifecycle, ToolRegistration
from biomodals.service.tools import ALPHAFOLD3_TOOL, GROMACS_TOOL, TOOLS

ORIGIN = "https://biomodals.internal"


class Remote:
    def __init__(self):
        self.preflights = 0

    async def preflight(self, _deployment):
        self.preflights += 1


class Adapter:
    async def stage(self, _job):
        return None

    async def discard_pending(self, _job):
        return None

    async def prepare_result(self, _job, _cache, *, completed_at):
        raise AssertionError("No result is prepared in contract tests")


def _app(tmp_path: Path):
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.initialize()
    settings = ServiceSettings.from_environment({
        "MODAL_TOKEN_ID": "test-id",
        "MODAL_TOKEN_SECRET": "test-secret",
        "BIOMODALS_PUBLIC_URL": ORIGIN,
        "BIOMODALS_SECURE_COOKIES": "true",
    })
    configuration = RuntimeConfiguration(
        store,
        settings,
        tool_definitions=TOOLS,
    )
    remote = Remote()
    pending = PendingRequestStore(tmp_path / "pending")
    pending.initialize()
    validations = ValidatedInputStore(tmp_path / "validations")
    validations.initialize()
    cache = ArtifactCache(tmp_path / "cache")
    registrations = (
        ToolRegistration(GROMACS_TOOL, Adapter()),
        ToolRegistration(ALPHAFOLD3_TOOL, Adapter()),
    )
    lifecycle = JobLifecycle(store, remote, registrations, cache)
    app = create_app(
        store=store,
        auth=AuthService(store, frontend_url=ORIGIN),
        configuration=configuration,
        registrations=registrations,
        tool_routers=(
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
        ),
        remote=remote,
        lifecycle=lifecycle,
        cache=cache,
        allowed_origin=ORIGIN,
        secure_cookies=True,
    )
    return app


def _session(user_id: UUID | None = None) -> AuthenticatedSession:
    now = 1_700_000_000
    return AuthenticatedSession(
        principal=Principal(
            user_id=user_id or uuid4(),
            email="scientist@example.com",
            display_name="Scientist",
            is_admin=False,
        ),
        csrf_digest=b"digest",
        created_at=now,
        last_seen_at=now,
        absolute_expires_at=now + 3600,
    )


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=ORIGIN,
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_openapi_exposes_typed_tool_and_shared_job_routes(tmp_path: Path) -> None:
    document = _app(tmp_path).openapi()
    paths = document["paths"]

    assert "/api/v1/gromacs/jobs" in paths
    assert "/api/v1/alphafold3/validations" in paths
    assert "/api/v1/alphafold3/validations/{validation_id}/document" in paths
    assert "/api/v1/alphafold3/jobs" in paths
    assert "/api/v1/jobs/{job_id}/refresh" in paths
    assert "/api/v1/jobs/{job_id}/download" in paths
    alpha_request = document["components"]["schemas"]["AlphaFold3JobRequest"]
    assert alpha_request["required"] == ["validation_id"]
    assert alpha_request["additionalProperties"] is False
    job_view = document["components"]["schemas"]["JobView"]
    assert {"state_reason", "state_message"} <= set(job_view["properties"])
    unknown_job = document["components"]["schemas"]["AdminStateUnknownJobView"]
    assert {
        "diagnostic_message",
        "modal_environment",
        "modal_app_name",
        "modal_app_version",
        "root_function_call_id",
    } <= set(unknown_job["required"])
    recycle = next(
        parameter
        for parameter in paths["/api/v1/alphafold3/validations"]["post"]["parameters"]
        if parameter["name"] == "recycle"
    )
    assert recycle["name"] == "recycle"
    assert recycle["schema"]["minimum"] == 0


def test_private_routes_require_a_session(tmp_path: Path) -> None:
    app = _app(tmp_path)

    assert _request(app, "GET", "/api/v1/auth/me").status_code == 401
    assert _request(app, "GET", "/api/v1/jobs").status_code == 401
    assert (
        _request(app, "GET", f"/api/v1/alphafold3/validations/{uuid4()}").status_code
        == 401
    )


def test_authenticated_routes_receive_the_principal(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session = _session()

    async def authenticated() -> AuthenticatedSession:
        return session

    app.dependency_overrides[require_session] = authenticated
    app.dependency_overrides[require_unsafe_session] = authenticated

    principal = _request(app, "GET", "/api/v1/auth/me")
    jobs = _request(app, "GET", "/api/v1/jobs")

    assert principal.status_code == 200
    assert principal.json()["email"] == "scientist@example.com"
    assert jobs.status_code == 200
    assert jobs.json() == {"jobs": [], "next_cursor": None}


def test_alphafold3_lost_response_replays_after_validation_consumption(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    store = app.state.store
    user = store.create_user(
        email="scientist@example.com",
        display_name="Scientist",
        token_digest=b"setup",
        token_expires_at=100,
        now=1,
        is_admin=True,
        active_job_limit=10,
    )
    store.set_password_from_token(
        b"setup",
        password_hash="test",  # noqa: S106
        session_token_digest=b"session",
        csrf_digest=b"csrf",
        now=2,
        absolute_expires_at=1000,
    )
    session = _session(user.user_id)
    idempotency_key = uuid4()
    job_id = uuid4()
    tool = app.state.configuration.tool("alphafold3")
    store.admit_job(
        owner_user_id=user.user_id,
        tool="alphafold3",
        display_name="prediction",
        idempotency_key=str(idempotency_key),
        request_digest="a" * 64,
        modal_environment=app.state.configuration.modal_environment().value,
        modal_app_name=tool.modal_app_name.value,
        modal_app_version=tool.modal_app_version.value,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=1,
        now=10,
        new_job_id=job_id,
    )

    async def authenticated() -> AuthenticatedSession:
        return session

    app.dependency_overrides[require_unsafe_session] = authenticated
    response = _request(
        app,
        "POST",
        "/api/v1/alphafold3/jobs",
        headers={"Idempotency-Key": str(idempotency_key), "Origin": ORIGIN},
        json={"validation_id": str(uuid4())},
    )

    assert response.status_code == 202
    assert response.json()["job_id"] == str(job_id)
    assert app.state.remote_execution.preflights == 0
