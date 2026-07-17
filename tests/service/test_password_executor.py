"""Bounded password-operation execution at the HTTP boundary."""

# ruff: noqa: D101,D103

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from biomodals.service.api import create_app
from biomodals.service.auth import AuthService, IssuedSession, Principal
from biomodals.service.config import ServiceSettings
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import ServiceStore

ORIGIN = "https://biomodals.internal"


@pytest.mark.parametrize(
    ("auth_method", "path", "payload"),
    [
        (
            "login",
            "/api/v1/auth/login",
            {"email": "alice@example.com", "password": "not-used"},
        ),
        (
            "set_password",
            "/api/v1/auth/set-password",
            {"token": "not-used", "password": "not-used-password"},
        ),
    ],
)
def test_password_work_is_bounded_and_workers_stop_with_the_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auth_method: str,
    path: str,
    payload: dict[str, str],
) -> None:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    auth = AuthService(store, frontend_url=ORIGIN)
    principal = Principal(
        user_id=uuid4(),
        email="alice@example.com",
        display_name="Alice",
        is_admin=False,
    )
    release = threading.Event()
    two_started = threading.Event()
    lock = threading.Lock()
    worker_threads: set[threading.Thread] = set()
    active = 0
    maximum_active = 0

    def slow_password_operation(*_args: object):
        nonlocal active, maximum_active
        with lock:
            worker_threads.add(threading.current_thread())
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_started.set()
        try:
            if not release.wait(timeout=5):
                raise RuntimeError("test password worker timed out")
        finally:
            with lock:
                active -= 1
        return IssuedSession("session", "csrf", principal)

    monkeypatch.setattr(auth, auth_method, slow_password_operation)
    app = create_app(
        store=store,
        auth=auth,
        configuration=RuntimeConfiguration(
            store,
            ServiceSettings.from_environment({}),
        ),
        workloads=[],
        allowed_origin=ORIGIN,
        secure_cookies=True,
    )

    async def exercise() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport,
                base_url=ORIGIN,
            ) as client:
                requests = [
                    asyncio.create_task(
                        client.post(path, headers={"Origin": ORIGIN}, json=payload)
                    )
                    for _ in range(3)
                ]
                try:
                    deadline = asyncio.get_running_loop().time() + 2
                    while not two_started.is_set():
                        if asyncio.get_running_loop().time() >= deadline:
                            raise AssertionError("two password workers did not start")
                        await asyncio.sleep(0.01)
                    await asyncio.sleep(0.05)
                    assert maximum_active == 2
                    health = await asyncio.wait_for(
                        client.get("/api/v1/health"),
                        timeout=0.5,
                    )
                    assert health.status_code == 200
                finally:
                    release.set()
                responses = await asyncio.gather(*requests)
                return responses

    responses = asyncio.run(exercise())

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert maximum_active == 2
    assert worker_threads
    assert all(not thread.is_alive() for thread in worker_threads)
