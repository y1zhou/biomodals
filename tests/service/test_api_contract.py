"""Unified browser API contracts for authentication and private jobs."""

# ruff: noqa: D101,D102,D103,D107,S105

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI

from biomodals.service.api import create_app
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthService, IssuedPasswordLink
from biomodals.service.config import ServiceSettings
from biomodals.service.gromacs import GromacsJobOptions, create_registration
from biomodals.service.gromacs.modal import GromacsReconciler
from biomodals.service.jobs import JobLifecycleLocks, WorkloadRegistration
from biomodals.service.jobs_api import _download_filename
from biomodals.service.runtime_config import (
    ModalConfigurationSnapshot,
    RuntimeConfiguration,
)
from biomodals.service.store import JobOperationState, JobState, ServiceStore
from biomodals.service.submission import SubmissionOutcomeUnknownError

ORIGIN = "https://biomodals.internal"
PASSWORD = "correct horse battery staple"  # noqa: S105 - test credential
SESSION_COOKIE = "__Host-biomodals-session"
CSRF_COOKIE = "biomodals-csrf"
VALID_PDB = (
    b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
    b"END\n"
)


@dataclass(frozen=True)
class SubmittedCall:
    modal_call_id: str
    run_name: str
    operation: str


class FakeGromacsAdapter:
    """Small fake for the Modal boundary; no Modal object reaches HTTP tests."""

    def __init__(self) -> None:
        self.submissions: list[tuple[bytes, str, GromacsJobOptions]] = []
        self.submission_configurations: list[tuple[str, str, int]] = []
        self.cancellations: list[str] = []
        self.recovery_attempts: list[UUID] = []
        self.downloads = 0
        self.artifact_content = b"PK\x03\x04verified archive"
        self.failures_remaining = 0
        self.unknown_failures_remaining = 0
        self.preflights: list[tuple[str, str, int]] = []
        self.preflight_failures_remaining = 0

    async def preflight(
        self,
        app_name: str,
        environment_name: str,
        app_version: int,
    ) -> None:
        self.preflights.append((app_name, environment_name, app_version))
        if self.preflight_failures_remaining:
            self.preflight_failures_remaining -= 1
            raise RuntimeError("configured resource is unavailable")

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
        modal_configuration: ModalConfigurationSnapshot,
    ) -> SubmittedCall:
        self.submissions.append((pdb_content, run_name, options))
        self.submission_configurations.append((
            modal_configuration.app_name,
            modal_configuration.environment,
            modal_configuration.app_version,
        ))
        if self.unknown_failures_remaining:
            self.unknown_failures_remaining -= 1
            raise SubmissionOutcomeUnknownError("provider outcome unknown")
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary Modal failure")
        return SubmittedCall(
            modal_call_id=f"fc-{len(self.submissions)}",
            run_name=run_name,
            operation=("prepare_tpr_cpu" if options.cpu_only else "prepare_tpr_gpu"),
        )

    async def cancel(self, modal_call_id: str) -> None:
        self.cancellations.append(modal_call_id)

    async def recover_archive(self, job) -> None:
        self.recovery_attempts.append(job.job_id)
        raise AssertionError("an unsubmitted cancellation must not touch Modal")

    async def read_artifact(self, _job):
        self.downloads += 1
        yield self.artifact_content


class APIClient:
    """Synchronous facade over httpx's supported async ASGI transport."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.cookies = httpx.Cookies()

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=ORIGIN,
                cookies=self.cookies,
            ) as client:
                response = await client.request(method, url, **kwargs)
                self.cookies.update(client.cookies)
                return response

        return asyncio.run(send())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def patch(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


def _password_token(link: str | IssuedPasswordLink) -> str:
    url = link.url if isinstance(link, IssuedPasswordLink) else link
    return parse_qs(urlparse(url).fragment)["token"][0]


def _activate(auth: AuthService, email: str, *, is_admin: bool = False) -> None:
    link = auth.create_user(
        email,
        display_name=email.partition("@")[0].title(),
        is_admin=is_admin or not auth.store.list_users(),
    )
    auth.set_password(_password_token(link.url), PASSWORD)


def _service(
    tmp_path: Path,
    *,
    max_pdb_bytes: int = 10 * 1024 * 1024,
) -> tuple[APIClient, AuthService, ServiceStore, FakeGromacsAdapter]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    settings = ServiceSettings.from_environment({
        "MODAL_TOKEN_ID": "test-token-id",
        "MODAL_TOKEN_SECRET": "test-token-secret",
    })
    configuration = RuntimeConfiguration(store, settings)
    auth = AuthService(store, frontend_url=ORIGIN)
    adapter = FakeGromacsAdapter()
    registration = create_registration(
        adapter,
        read_artifact=adapter.read_artifact,
        preflight=adapter.preflight,
        max_pdb_bytes=max_pdb_bytes,
    )
    assert isinstance(registration, WorkloadRegistration)
    app = create_app(
        store=store,
        auth=auth,
        configuration=configuration,
        workloads=[registration],
        allowed_origin=ORIGIN,
        secure_cookies=True,
        cache=ArtifactCache(tmp_path / "cache"),
    )
    return APIClient(app), auth, store, adapter


def _login(client: APIClient, email: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"email": email, "password": PASSWORD},
    )
    assert response.status_code == 200
    return client.cookies[CSRF_COOKIE]


def _unsafe_headers(csrf_token: str, **extra: str) -> dict[str, str]:
    return {"Origin": ORIGIN, "X-CSRF-Token": csrf_token, **extra}


def _submit(
    client: APIClient,
    csrf_token: str,
    *,
    idempotency_key: str | None = None,
    simulation_time_ns: int = 3,
    display_name: str | None = "First simulation",
    filename: str = "protein.pdb",
):
    headers = _unsafe_headers(csrf_token)
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    data = {
        "simulation_time_ns": str(simulation_time_ns),
        "cpu_only": "true",
    }
    if display_name is not None:
        data["display_name"] = display_name
    return client.post(
        "/api/v1/gromacs/jobs",
        headers=headers,
        files={"pdb": (filename, VALID_PDB, "chemical/x-pdb")},
        data=data,
    )


def _jobs(response) -> list[dict[str, object]]:
    body = response.json()
    return body if isinstance(body, list) else body["jobs"]


def _response_codes(schema: dict, path: str, status_code: int) -> list[str]:
    response = schema["paths"][path]["post"]["responses"][str(status_code)]
    reference = response["content"]["application/json"]["schema"]["$ref"]
    code = schema["components"]["schemas"][reference.rsplit("/", 1)[1]]["properties"][
        "code"
    ]
    return code["enum"] if "enum" in code else [code["const"]]


def test_result_filename_falls_back_without_exposing_job_identity() -> None:
    assert _download_filename("日本語") == "gromacs-results.zip"


def test_login_requires_the_frontend_origin_and_sets_hardened_cookies(
    tmp_path: Path,
) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")

    missing_origin = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": PASSWORD},
    )
    response = client.post(
        "/api/v1/auth/login",
        headers={"Origin": ORIGIN},
        json={"email": "alice@example.com", "password": PASSWORD},
    )

    assert missing_origin.status_code == 403
    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"
    set_cookies = response.headers.get_list("set-cookie")
    session_cookie = next(
        item for item in set_cookies if item.startswith(f"{SESSION_COOKIE}=")
    )
    csrf_cookie = next(
        item for item in set_cookies if item.startswith(f"{CSRF_COOKIE}=")
    )
    assert "HttpOnly" in session_cookie
    assert "Secure" in session_cookie
    assert "SameSite=lax" in session_cookie
    assert "Path=/" in session_cookie
    assert "Domain=" not in session_cookie
    assert "HttpOnly" not in csrf_cookie
    assert "Secure" in csrf_cookie
    assert client.get("/api/v1/auth/me").json()["email"] == "alice@example.com"


def test_one_time_password_link_and_logout_complete_browser_flow(
    tmp_path: Path,
) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    link = auth.create_user(
        "alice@example.com",
        display_name="Alice",
        is_admin=True,
    )
    token = _password_token(link)

    missing_origin = client.post(
        "/api/v1/auth/set-password",
        json={"token": token, "password": PASSWORD},
    )
    activated = client.post(
        "/api/v1/auth/set-password",
        headers={"Origin": ORIGIN},
        json={"token": token, "password": PASSWORD},
    )
    replay = client.post(
        "/api/v1/auth/set-password",
        headers={"Origin": ORIGIN},
        json={"token": token, "password": PASSWORD},
    )
    csrf_token = _login(client, "alice@example.com")
    logout = client.post(
        "/api/v1/auth/logout",
        headers=_unsafe_headers(csrf_token),
    )

    assert missing_origin.status_code == 403
    assert activated.status_code == 200
    assert replay.status_code == 400
    assert logout.status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401


def test_password_reset_establishes_fresh_session_and_revokes_old_one(
    tmp_path: Path,
) -> None:
    old_client, auth, _store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    _login(old_client, "alice@example.com")
    link = auth.create_password_reset("alice@example.com")
    reset_client = APIClient(old_client.app)

    response = reset_client.post(
        "/api/v1/auth/set-password",
        headers={"Origin": ORIGIN},
        json={"token": _password_token(link), "password": PASSWORD},
    )

    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"
    assert reset_client.get("/api/v1/auth/me").status_code == 200
    assert old_client.get("/api/v1/auth/me").status_code == 401
    set_cookies = response.headers.get_list("set-cookie")
    assert any(item.startswith(f"{SESSION_COOKIE}=") for item in set_cookies)
    assert any(item.startswith(f"{CSRF_COOKIE}=") for item in set_cookies)


def test_openapi_exposes_the_set_password_minimum(tmp_path: Path) -> None:
    client, _auth, _store, _adapter = _service(tmp_path)

    schema = client.get("/openapi.json").json()
    password_schema = schema["components"]["schemas"]["SetPasswordRequest"][
        "properties"
    ]["password"]

    assert password_schema["minLength"] == 15


def test_admin_user_management_requires_admin_and_preserves_last_admin(
    tmp_path: Path,
) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com", is_admin=True)
    _activate(auth, "ordinary@example.com")
    csrf_token = _login(client, "ordinary@example.com")

    assert client.get("/api/v1/admin/users").status_code == 403
    client.cookies.clear()
    csrf_token = _login(client, "alice@example.com")
    created = client.post(
        "/api/v1/admin/users",
        headers=_unsafe_headers(csrf_token),
        json={
            "email": "new@example.com",
            "display_name": "New User",
            "active_job_limit": 4,
        },
    )
    assert created.status_code == 201
    assert created.json()["user"]["active_job_limit"] == 4
    assert created.json()["user"]["status"] == "pending_setup"
    assert created.json()["password_link"].startswith(f"{ORIGIN}/set-password#token=")
    assert created.json()["expires_at"]
    new_user_id = created.json()["user"]["user_id"]

    invalid = client.post(
        "/api/v1/admin/users",
        headers=_unsafe_headers(csrf_token),
        json={"email": "invalid", "display_name": "   "},
    )
    assert invalid.status_code == 400
    assert invalid.json()["code"] == "user_invalid"

    users = client.get("/api/v1/admin/users")
    assert users.status_code == 200
    user_page = users.json()
    assert {user["email"] for user in user_page["users"]} == {
        "alice@example.com",
        "new@example.com",
        "ordinary@example.com",
    }
    alice_id = next(
        user["user_id"]
        for user in user_page["users"]
        if user["email"] == "alice@example.com"
    )
    first_user_page = client.get("/api/v1/admin/users?limit=2").json()
    second_user_page = client.get(
        f"/api/v1/admin/users?limit=2&cursor={first_user_page['next_cursor']}"
    ).json()
    assert len(first_user_page["users"]) == 2
    assert len(second_user_page["users"]) == 1
    assert second_user_page["next_cursor"] is None

    last_admin = client.patch(
        f"/api/v1/admin/users/{alice_id}",
        headers=_unsafe_headers(csrf_token),
        json={"is_admin": False},
    )
    assert last_admin.status_code == 409
    assert last_admin.json()["code"] == "last_active_admin"

    promoted = client.patch(
        f"/api/v1/admin/users/{new_user_id}",
        headers=_unsafe_headers(csrf_token),
        json={
            "display_name": "  New Researcher  ",
            "is_admin": True,
            "active_job_limit": 6,
        },
    )
    assert promoted.status_code == 200
    assert promoted.json()["display_name"] == "New Researcher"
    assert promoted.json()["is_admin"] is True
    assert promoted.json()["active_job_limit"] == 6

    invalid_name = client.patch(
        f"/api/v1/admin/users/{new_user_id}",
        headers=_unsafe_headers(csrf_token),
        json={"display_name": "   "},
    )
    assert invalid_name.status_code == 400
    assert invalid_name.json()["code"] == "user_invalid"

    disabled = client.patch(
        f"/api/v1/admin/users/{new_user_id}",
        headers=_unsafe_headers(csrf_token),
        json={"status": "disabled"},
    )
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "disabled"
    unavailable_link = client.post(
        f"/api/v1/admin/users/{new_user_id}/password-link",
        headers=_unsafe_headers(csrf_token),
    )
    assert unavailable_link.status_code == 409
    assert unavailable_link.json()["code"] == "user_inactive"

    enabled = client.patch(
        f"/api/v1/admin/users/{new_user_id}",
        headers=_unsafe_headers(csrf_token),
        json={"status": "enabled"},
    )
    assert enabled.status_code == 200
    assert enabled.json()["status"] == "pending_setup"
    password_link = client.post(
        f"/api/v1/admin/users/{new_user_id}/password-link",
        headers=_unsafe_headers(csrf_token),
    )
    assert password_link.status_code == 200
    assert password_link.json()["password_link"].startswith(
        f"{ORIGIN}/set-password#token="
    )
    assert password_link.json()["expires_at"]


def test_modal_admin_configuration_is_live_and_job_configuration_is_pinned(
    tmp_path: Path,
) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com", is_admin=True)
    csrf_token = _login(client, "alice@example.com")

    initial = client.get("/api/v1/admin/modal")
    assert initial.status_code == 200
    assert initial.json()["environment"]["service_token_id"] == "test-token-id"
    assert "test-token-secret" not in initial.text
    assert initial.json()["tools"][0]["workload"] == "gromacs"
    assert initial.json()["tools"][0]["display_name"] == "GROMACS MD simulation"
    assert initial.json()["state_unknown_jobs"] == []

    environment = client.patch(
        "/api/v1/admin/modal/environment",
        headers=_unsafe_headers(csrf_token),
        json={
            "modal_environment": "department-a",
            "global_active_job_limit": 9,
        },
    )
    tool = client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=_unsafe_headers(csrf_token),
        json={
            "modal_app_name": "GromacsA",
            "modal_app_version": 17,
            "active_job_limit": 3,
        },
    )
    assert environment.json()["modal_environment"] == {
        "value": "department-a",
        "source": "database",
        "editable": True,
    }
    assert tool.json()["modal_app_name"]["value"] == "GromacsA"

    first = _submit(client, csrf_token, idempotency_key=str(uuid4()))
    assert first.status_code == 202
    first_job_id = UUID(first.json()["job_id"])
    store.set_job_state(first_job_id, JobState.RUNNING, now=1_800_000_001)
    assert adapter.submission_configurations == [("GromacsA", "department-a", 17)]

    client.patch(
        "/api/v1/admin/modal/environment",
        headers=_unsafe_headers(csrf_token),
        json={"modal_environment": "department-b"},
    )
    client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=_unsafe_headers(csrf_token),
        json={"modal_app_name": "GromacsB", "modal_app_version": 23},
    )
    second = _submit(client, csrf_token, idempotency_key=str(uuid4()))

    assert second.status_code == 202
    assert adapter.submission_configurations == [
        ("GromacsA", "department-a", 17),
        ("GromacsB", "department-b", 23),
    ]
    first_job = store.get_job(
        UUID(client.get("/api/v1/auth/me").json()["user_id"]),
        first_job_id,
    )
    assert first_job is not None
    assert first_job.modal_environment == "department-a"
    assert first_job.modal_app_name == "GromacsA"
    assert first_job.modal_app_version == 17
    current = client.get("/api/v1/admin/modal").json()
    assert current["tools"][0]["active_jobs"] == 2


def test_modal_admin_updates_and_resets_each_field_independently(
    tmp_path: Path,
) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com", is_admin=True)
    csrf_token = _login(client, "alice@example.com")
    headers = _unsafe_headers(csrf_token)

    tool = client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=headers,
        json={"active_job_limit": 3},
    ).json()
    assert tool["active_job_limit"] == {
        "value": 3,
        "source": "database",
        "editable": True,
    }
    assert tool["modal_app_name"] == {
        "value": "Gromacs",
        "source": "default",
        "editable": True,
    }
    assert tool["modal_app_version"] == {
        "value": 1,
        "source": "default",
        "editable": True,
    }

    restored_tool = client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=headers,
        json={"active_job_limit": None},
    ).json()
    assert restored_tool["active_job_limit"] == {
        "value": 2,
        "source": "default",
        "editable": True,
    }
    assert restored_tool["modal_app_name"]["source"] == "default"

    environment = client.patch(
        "/api/v1/admin/modal/environment",
        headers=headers,
        json={"global_active_job_limit": 12},
    ).json()
    assert environment["global_active_job_limit"]["source"] == "database"
    assert environment["modal_environment"]["source"] == "default"

    restored_environment = client.patch(
        "/api/v1/admin/modal/environment",
        headers=headers,
        json={"global_active_job_limit": None},
    ).json()
    assert restored_environment["global_active_job_limit"] == {
        "value": 10,
        "source": "default",
        "editable": True,
    }
    assert restored_environment["modal_environment"]["source"] == "default"


def test_modal_admin_rejects_blank_runtime_settings(tmp_path: Path) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com", is_admin=True)
    csrf_token = _login(client, "alice@example.com")

    environment = client.patch(
        "/api/v1/admin/modal/environment",
        headers=_unsafe_headers(csrf_token),
        json={"modal_environment": "   "},
    )
    tool = client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=_unsafe_headers(csrf_token),
        json={"modal_app_name": "   "},
    )

    assert environment.status_code == 400
    assert environment.json()["code"] == "setting_invalid"
    assert tool.status_code == 400
    assert tool.json()["code"] == "setting_invalid"


def test_openapi_includes_admin_contract_and_admin_principal(tmp_path: Path) -> None:
    client, _auth, _store, _adapter = _service(tmp_path)

    schema = client.get("/openapi.json").json()

    assert (
        schema["components"]["schemas"]["PrincipalView"]["properties"]["is_admin"][
            "type"
        ]
        == "boolean"
    )
    for path in (
        "/api/v1/admin/users",
        "/api/v1/admin/users/{user_id}",
        "/api/v1/admin/users/{user_id}/password-link",
        "/api/v1/admin/modal",
        "/api/v1/admin/modal/environment",
        "/api/v1/admin/modal/tools/{workload}",
        "/api/v1/admin/modal/state-unknown-jobs/{job_id}/mark-failed",
    ):
        assert path in schema["paths"]

    expected_conflicts = {
        ("/api/v1/admin/users", "post"): "user_already_exists",
        ("/api/v1/admin/users/{user_id}", "patch"): "last_active_admin",
        (
            "/api/v1/admin/users/{user_id}/password-link",
            "post",
        ): "user_inactive",
        (
            "/api/v1/admin/modal/state-unknown-jobs/{job_id}/mark-failed",
            "post",
        ): "job_state_changed",
    }
    for (path, method), expected_code in expected_conflicts.items():
        response = schema["paths"][path][method]["responses"]["409"]
        response_schema = response["content"]["application/json"]["schema"]
        model_name = response_schema["$ref"].rsplit("/", maxsplit=1)[-1]
        code_schema = schema["components"]["schemas"][model_name]["properties"]["code"]
        assert code_schema["const"] == expected_code

    update_user = schema["paths"]["/api/v1/admin/users/{user_id}"]["patch"]
    invalid_response = update_user["responses"]["400"]["content"]["application/json"][
        "schema"
    ]
    invalid_model = invalid_response["$ref"].rsplit("/", maxsplit=1)[-1]
    assert (
        schema["components"]["schemas"][invalid_model]["properties"]["code"]["const"]
        == "user_invalid"
    )
    display_name = schema["components"]["schemas"]["UpdateAdminUserRequest"][
        "properties"
    ]["display_name"]
    assert display_name["anyOf"][0]["maxLength"] == 120


def test_modal_preflight_runs_only_for_changed_provider_fields(
    tmp_path: Path,
) -> None:
    client, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com", is_admin=True)
    csrf_token = _login(client, "alice@example.com")
    headers = _unsafe_headers(csrf_token)

    limit_only = client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=headers,
        json={"active_job_limit": 0},
    )
    adapter.preflight_failures_remaining = 1
    rejected = client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=headers,
        json={"modal_app_name": "UnavailableApp", "modal_app_version": 17},
    )
    unchanged = client.get("/api/v1/admin/modal")
    accepted = client.patch(
        "/api/v1/admin/modal/tools/gromacs",
        headers=headers,
        json={"modal_app_name": "AvailableApp", "modal_app_version": 23},
    )

    assert limit_only.status_code == 200
    assert adapter.preflights == [
        ("UnavailableApp", "production", 17),
        ("AvailableApp", "production", 23),
    ]
    assert rejected.status_code == 400
    assert rejected.json()["code"] == "modal_preflight_failed"
    assert unchanged.json()["tools"][0]["modal_app_name"]["value"] == "Gromacs"
    assert accepted.status_code == 200
    assert accepted.json()["modal_app_name"]["value"] == "AvailableApp"


def test_storage_metrics_and_explicit_cleanup_report_actual_reclamation(
    tmp_path: Path,
) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com", is_admin=True)
    csrf_token = _login(client, "alice@example.com")
    submitted = _submit(client, csrf_token, idempotency_key=str(uuid4()))
    job_id = UUID(submitted.json()["job_id"])
    content = adapter.artifact_content
    store.complete_job(
        job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{job_id}/result.zip",
        result_filename="result.zip",
        result_size_bytes=len(content),
        result_sha256=hashlib.sha256(content).hexdigest(),
        result_archive_schema_version=1,
        now=1_800_000_001,
    )
    cache: ArtifactCache = client.app.state.cache

    async def fill_cache() -> None:
        async def chunks():
            yield content

        lease = await cache.store(
            str(job_id),
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            chunks=chunks(),
        )
        try:
            store.set_result_cached(job_id, cached=True)
        finally:
            lease.close()

    asyncio.run(fill_cache())
    metrics = client.get("/api/v1/admin/storage")
    cleared = client.post(
        "/api/v1/admin/storage/cache/clear",
        headers=_unsafe_headers(csrf_token),
    )
    refreshed = client.get("/api/v1/admin/storage")

    assert metrics.status_code == 200
    assert metrics.json()["published_result_entries"] == 1
    assert metrics.json()["published_result_bytes"] == len(content)
    assert metrics.json()["local_cache_entries"] == 1
    assert metrics.json()["local_cache_bytes"] == len(content)
    assert metrics.json()["staging_entries"] == 0
    assert metrics.json()["reclaimable_entries"] == 1
    assert cleared.json() == {
        "removed_entries": 1,
        "removed_bytes": len(content),
    }
    assert refreshed.json()["local_cache_entries"] == 0
    stored = store.get_job(
        UUID(client.get("/api/v1/auth/me").json()["user_id"]),
        job_id,
    )
    assert stored is not None and stored.result_cached is False


def test_every_response_has_request_id_and_openapi_declares_session_cookie(
    tmp_path: Path,
) -> None:
    client, _auth, _store, _adapter = _service(tmp_path)

    health = client.get("/api/v1/health")
    unauthorized = client.get("/api/v1/jobs")
    schema = client.get("/openapi.json").json()

    assert UUID(health.headers["x-request-id"])
    assert UUID(unauthorized.headers["x-request-id"])
    assert schema["components"]["securitySchemes"]["SessionCookie"] == {
        "type": "apiKey",
        "in": "cookie",
        "name": SESSION_COOKIE,
        "description": "Opaque BioModals browser session cookie.",
    }
    assert schema["paths"]["/api/v1/jobs"]["get"]["security"] == [{"SessionCookie": []}]
    assert "security" not in schema["paths"]["/api/v1/health"]["get"]
    for response in schema["paths"]["/api/v1/jobs"]["get"]["responses"].values():
        assert response["headers"]["X-Request-ID"]["schema"]["format"] == "uuid"


def test_route_lifecycle_logs_share_the_response_request_id(
    tmp_path: Path,
    caplog,
) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com", is_admin=True)
    csrf_token = _login(client, "alice@example.com")

    with caplog.at_level(logging.INFO):
        response = client.patch(
            "/api/v1/admin/modal/tools/gromacs",
            headers=_unsafe_headers(csrf_token),
            json={"active_job_limit": 3},
        )

    request_id = response.headers["x-request-id"]
    lifecycle_records = [
        record.getMessage()
        for record in caplog.records
        if "event=runtime_setting_changed" in record.getMessage()
    ]

    assert UUID(request_id)
    assert lifecycle_records == [
        "event=runtime_setting_changed scope=tool workload=gromacs "
        f"fields=active_job_limit request_id={request_id}"
    ]


def test_unhandled_errors_keep_the_request_id_for_support_correlation(
    tmp_path: Path,
    caplog,
) -> None:
    client, _auth, _store, _adapter = _service(tmp_path)

    async def crash() -> None:
        raise RuntimeError("private diagnostic detail")

    client.app.add_api_route("/api/v1/test-crash", crash)

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(
            app=client.app,
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url=ORIGIN,
        ) as browser:
            return await browser.get("/api/v1/test-crash")

    with caplog.at_level(logging.INFO):
        response = asyncio.run(scenario())

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert UUID(response.headers["x-request-id"])
    assert "private diagnostic detail" not in response.text
    messages = [record.getMessage() for record in caplog.records]
    assert sum("event=unhandled_exception" in message for message in messages) == 1
    assert (
        sum("request_complete event=http_request" in message for message in messages)
        == 1
    )


def test_health_is_live_before_startup_and_ready_is_local_after_preflight(
    tmp_path: Path,
) -> None:
    client, _auth, _store, adapter = _service(tmp_path)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=ORIGIN,
        ) as browser:
            assert (await browser.get("/api/v1/health")).status_code == 200
            assert (await browser.get("/api/v1/ready")).status_code == 503
            async with client.app.router.lifespan_context(client.app):
                assert adapter.preflights == [("Gromacs", "production", 1)]
                ready = await browser.get("/api/v1/ready")
                assert ready.status_code == 200
                assert ready.json() == {"status": "ok"}
                assert adapter.preflights == [("Gromacs", "production", 1)]
            assert (await browser.get("/api/v1/ready")).status_code == 503

    asyncio.run(scenario())


def test_cache_metadata_is_reconciled_before_job_reconciliation_starts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    settings = ServiceSettings.from_environment({
        "MODAL_TOKEN_ID": "test-token-id",
        "MODAL_TOKEN_SECRET": "test-token-secret",
    })
    configuration = RuntimeConfiguration(store, settings)
    adapter = FakeGromacsAdapter()

    class RecordingReconciler:
        started = False

        async def reconcile(self) -> None:
            self.started = True

    reconciler = RecordingReconciler()
    registration = create_registration(
        adapter,
        reconciler=reconciler,
        lifecycle_locks=JobLifecycleLocks(),
        read_artifact=adapter.read_artifact,
        preflight=adapter.preflight,
    )
    cache = ArtifactCache(tmp_path / "cache")

    async def cached_job_ids() -> set[str]:
        await asyncio.sleep(0)
        assert not reconciler.started
        return set()

    monkeypatch.setattr(cache, "cached_job_ids_async", cached_job_ids)
    app = create_app(
        store=store,
        auth=AuthService(store, frontend_url=ORIGIN),
        configuration=configuration,
        workloads=[registration],
        allowed_origin=ORIGIN,
        secure_cookies=True,
        cache=cache,
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)
            assert reconciler.started

    asyncio.run(scenario())


def test_password_setup_exposes_coded_errors_and_cookie_contract(
    tmp_path: Path,
) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    link = auth.create_user(
        "alice@example.com",
        display_name="Alice",
        is_admin=True,
    )
    token = _password_token(link)

    wrong_origin = client.post(
        "/api/v1/auth/set-password",
        json={"token": token, "password": PASSWORD},
    )
    rejected_password = client.post(
        "/api/v1/auth/set-password",
        headers={"Origin": ORIGIN},
        json={"token": token, "password": "passwordpassword"},
    )
    invalid_link = client.post(
        "/api/v1/auth/set-password",
        headers={"Origin": ORIGIN},
        json={"token": "not-valid", "password": PASSWORD},
    )
    schema = client.get("/openapi.json").json()
    responses = schema["paths"]["/api/v1/auth/set-password"]["post"]["responses"]

    assert wrong_origin.json() == {
        "code": "origin_not_allowed",
        "detail": "Request origin is not allowed",
    }
    assert rejected_password.json() == {
        "code": "password_policy_rejected",
        "detail": "Choose a less common password",
    }
    assert invalid_link.json() == {
        "code": "password_link_invalid",
        "detail": "Password link is invalid or expired",
    }
    password_error_ref = responses["400"]["content"]["application/json"]["schema"][
        "$ref"
    ]
    password_error = schema["components"]["schemas"][
        password_error_ref.rsplit("/", 1)[1]
    ]
    assert password_error["properties"]["code"]["enum"] == [
        "password_link_invalid",
        "password_policy_rejected",
    ]
    assert "Set-Cookie" in responses["200"]["headers"]


def test_openapi_documents_csrf_cookie_and_required_header(tmp_path: Path) -> None:
    client, _auth, _store, _adapter = _service(tmp_path)

    schema = client.get("/openapi.json").json()
    login_headers = schema["paths"]["/api/v1/auth/login"]["post"]["responses"]["200"][
        "headers"
    ]
    assert "biomodals-csrf" in login_headers["Set-Cookie"]["description"]

    mutations = (
        "/api/v1/auth/logout",
        "/api/v1/jobs/{job_id}/cancel",
        "/api/v1/gromacs/jobs",
    )
    for path in mutations:
        operation = schema["paths"][path]["post"]
        parameters = operation["parameters"]
        csrf = next(item for item in parameters if item["name"] == "X-CSRF-Token")
        assert csrf["required"] is True
        assert csrf["schema"] == {"type": "string"}
        assert "biomodals-csrf" in csrf["description"]
        forbidden_ref = operation["responses"]["403"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        forbidden = schema["components"]["schemas"][forbidden_ref.rsplit("/", 1)[1]]
        expected = ["csrf_invalid", "origin_not_allowed"]
        if path == "/api/v1/gromacs/jobs":
            expected.insert(0, "account_disabled")
        assert forbidden["properties"]["code"]["enum"] == expected


def test_openapi_documents_binary_job_download(tmp_path: Path) -> None:
    client, _auth, _store, _adapter = _service(tmp_path)

    schema = client.get("/openapi.json").json()
    responses = schema["paths"]["/api/v1/jobs/{job_id}/download"]["get"]["responses"]
    binary_zip = {"application/zip": {"schema": {"type": "string", "format": "binary"}}}

    assert responses["200"]["content"] == binary_zip
    assert "Content-Disposition" in responses["200"]["headers"]
    assert responses["206"]["content"] == binary_zip
    assert "Content-Disposition" in responses["206"]["headers"]
    assert "Content-Range" in responses["206"]["headers"]


def test_openapi_documents_frontend_handled_error_statuses(tmp_path: Path) -> None:
    client, _auth, _store, _adapter = _service(tmp_path)
    schema = client.get("/openapi.json").json()
    expected = {
        ("/api/v1/auth/login", "post"): {"401", "403", "413", "422"},
        ("/api/v1/auth/set-password", "post"): {"400", "403", "422"},
        ("/api/v1/auth/me", "get"): {"401"},
        ("/api/v1/auth/logout", "post"): {"401", "403"},
        ("/api/v1/jobs", "get"): {"400", "401"},
        ("/api/v1/jobs/{job_id}", "get"): {"401", "404", "422"},
        ("/api/v1/jobs/{job_id}/cancel", "post"): {
            "401",
            "403",
            "404",
            "409",
            "422",
        },
        ("/api/v1/jobs/{job_id}/prepare-download", "post"): {
            "401",
            "403",
            "404",
            "409",
            "422",
            "503",
        },
        ("/api/v1/jobs/{job_id}/download", "get"): {
            "401",
            "404",
            "409",
            "422",
            "502",
        },
        ("/api/v1/gromacs/jobs", "post"): {
            "400",
            "401",
            "403",
            "409",
            "413",
            "422",
            "503",
        },
    }

    for (path, method), statuses in expected.items():
        documented = set(schema["paths"][path][method]["responses"])
        assert statuses <= documented, (path, statuses - documented)


def test_unsafe_cookie_requests_require_exact_origin_and_session_csrf(
    tmp_path: Path,
) -> None:
    client, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    key = str(uuid4())

    no_origin = client.post(
        "/api/v1/gromacs/jobs",
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": key},
        files={"pdb": ("protein.pdb", VALID_PDB, "chemical/x-pdb")},
    )
    no_csrf = client.post(
        "/api/v1/gromacs/jobs",
        headers={"Origin": ORIGIN, "Idempotency-Key": key},
        files={"pdb": ("protein.pdb", VALID_PDB, "chemical/x-pdb")},
    )
    wrong_origin = client.post(
        "/api/v1/gromacs/jobs",
        headers={
            "Origin": f"{ORIGIN}.attacker.example",
            "X-CSRF-Token": csrf_token,
            "Idempotency-Key": key,
        },
        files={"pdb": ("protein.pdb", VALID_PDB, "chemical/x-pdb")},
    )

    assert no_origin.status_code == 403
    assert no_origin.json()["code"] == "origin_not_allowed"
    assert no_csrf.status_code == 403
    assert no_csrf.json()["code"] == "csrf_invalid"
    assert wrong_origin.status_code == 403
    assert wrong_origin.json()["code"] == "origin_not_allowed"
    assert adapter.submissions == []


def test_upload_and_total_request_size_are_bounded_before_submission(
    tmp_path: Path,
) -> None:
    client, auth, _store, adapter = _service(tmp_path, max_pdb_bytes=8)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    headers = _unsafe_headers(
        csrf_token,
        **{"Idempotency-Key": str(uuid4())},
    )

    oversized_pdb = client.post(
        "/api/v1/gromacs/jobs",
        headers=headers,
        files={"pdb": ("large.pdb", b"ATOM      1\n", "chemical/x-pdb")},
    )
    oversized_body = client.post(
        "/api/v1/gromacs/jobs",
        headers={**headers, "Content-Type": "application/octet-stream"},
        content=b"x" * (64 * 1024 + 9),
    )

    assert oversized_pdb.status_code == 413
    assert oversized_pdb.json()["code"] == "payload_too_large"
    assert oversized_body.status_code == 413
    assert oversized_body.json()["code"] == "payload_too_large"
    assert adapter.submissions == []


def test_semantic_pdb_and_active_limit_errors_match_openapi(tmp_path: Path) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    headers = _unsafe_headers(
        csrf_token,
        **{"Idempotency-Key": str(uuid4())},
    )

    invalid_pdb = client.post(
        "/api/v1/gromacs/jobs",
        headers=headers,
        files={"pdb": ("invalid.pdb", b"not a PDB", "chemical/x-pdb")},
    )
    _submit(client, csrf_token, idempotency_key=str(uuid4()))
    _submit(client, csrf_token, idempotency_key=str(uuid4()))
    over_limit = _submit(client, csrf_token, idempotency_key=str(uuid4()))
    schema = client.get("/openapi.json").json()
    path = "/api/v1/gromacs/jobs"

    assert invalid_pdb.status_code == 400
    assert invalid_pdb.json()["code"] == "pdb_invalid"
    assert over_limit.status_code == 409
    assert over_limit.json()["code"] == "active_job_limit_reached"
    assert _response_codes(schema, path, 400) == ["pdb_invalid"]
    assert _response_codes(schema, path, 409) == [
        "idempotency_conflict",
        "active_job_limit_reached",
    ]
    assert _response_codes(schema, path, 413) == ["payload_too_large"]
    assert _response_codes(schema, path, 503) == ["compute_unavailable"]


def test_gromacs_submission_is_idempotent_for_one_owner_and_payload(
    tmp_path: Path,
) -> None:
    client, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    key = str(uuid4())

    first = _submit(client, csrf_token, idempotency_key=key)
    replay = _submit(client, csrf_token, idempotency_key=key)
    conflict = _submit(
        client,
        csrf_token,
        idempotency_key=key,
        simulation_time_ns=4,
    )
    missing_key = _submit(client, csrf_token)
    malformed_key = _submit(client, csrf_token, idempotency_key="not-a-uuid")

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json()["job_id"] == first.json()["job_id"]
    assert first.json()["workload"] == "gromacs"
    assert first.json()["display_name"] == "First simulation"
    assert first.json()["state"] == "queued"
    assert first.json()["stage"]["code"] == "prepare_simulation"
    assert first.json()["stage"]["function_name"] == "prepare_tpr_cpu"
    assert first.json()["stage"]["started_at"]
    assert first.json()["active_stages"] == [first.json()["stage"]]
    assert first.json()["stage_history"] == [first.json()["stage"]]
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"
    assert missing_key.status_code == 422
    assert malformed_key.status_code == 422
    assert len(adapter.submissions) == 1
    pdb_content, run_name, options = adapter.submissions[0]
    assert pdb_content == VALID_PDB
    assert run_name == f"first-simulation-{UUID(first.json()['job_id']).hex}"
    assert options == GromacsJobOptions(simulation_time_ns=3, cpu_only=True)


def test_filename_derived_display_name_is_part_of_submission_identity(
    tmp_path: Path,
) -> None:
    client, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    key = str(uuid4())

    first = _submit(
        client,
        csrf_token,
        idempotency_key=key,
        display_name=None,
        filename="kinase.pdb",
    )
    conflict = _submit(
        client,
        csrf_token,
        idempotency_key=key,
        display_name=None,
        filename="receptor.pdb",
    )

    assert first.status_code == 202
    assert first.json()["display_name"].startswith("kinase ")
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"
    assert len(adapter.submissions) == 1


def test_job_stage_contract_supports_parallel_deployed_functions(
    tmp_path: Path,
) -> None:
    client, auth, store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    submitted = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
    )
    job_id = UUID(submitted.json()["job_id"])
    store.record_operation_outcome(
        job_id,
        operation="prepare_tpr_cpu",
        expected_modal_call_id="fc-1",
        outcome=JobOperationState.COMPLETED,
        now=1_800_000_001,
    )
    operations = (
        ("collect_traj_stats:nvt_", "fc-nvt"),
        ("collect_traj_stats:npt_", "fc-npt"),
        ("production_run_cpu", "fc-production"),
    )
    for operation, modal_call_id in operations:
        token = uuid4().hex
        claimed = store.claim_modal_operation(
            job_id,
            operation=operation,
            submission_token=token,
            now=1_800_000_001,
        )
        assert claimed is not None
        store.attach_modal_call(
            job_id,
            operation=operation,
            modal_call_id=modal_call_id,
            submission_token=token,
            now=1_800_000_001,
        )
    analyzing = client.get(f"/api/v1/jobs/{job_id}")

    assert analyzing.json()["stage"] == {
        "code": "run_production",
        "function_name": "production_run_cpu",
        "started_at": "2027-01-15T08:00:01Z",
    }
    preparation, nvt_analysis, npt_analysis, production = analyzing.json()[
        "stage_history"
    ]
    assert preparation["code"] == "prepare_simulation"
    assert preparation["function_name"] == "prepare_tpr_cpu"
    assert preparation["started_at"]
    assert preparation["ended_at"] == "2027-01-15T08:00:01Z"
    assert preparation["outcome"] == "completed"
    assert [stage["code"] for stage in analyzing.json()["active_stages"]] == [
        "analyze_nvt",
        "analyze_npt",
        "run_production",
    ]
    assert nvt_analysis == analyzing.json()["active_stages"][0]
    assert npt_analysis == analyzing.json()["active_stages"][1]
    assert production == analyzing.json()["active_stages"][2]
    assert "modal_call_id" not in analyzing.json()
    assert "operation" not in analyzing.json()

    store.fail_job(
        job_id,
        error_code="compute_failed",
        error_message="The simulation failed.",
        now=1_800_000_002,
    )
    unavailable = client.get(f"/api/v1/jobs/{job_id}")

    assert unavailable.json()["active_stages"] == []
    assert unavailable.json()["stage"] == {
        "code": "run_production",
        "function_name": "production_run_cpu",
        "started_at": "2027-01-15T08:00:01Z",
        "ended_at": "2027-01-15T08:00:02Z",
        "outcome": "failed",
    }

    submitted_packaging = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
    )
    packaging_job_id = UUID(submitted_packaging.json()["job_id"])
    store.record_operation_outcome(
        packaging_job_id,
        operation="prepare_tpr_cpu",
        expected_modal_call_id="fc-2",
        outcome=JobOperationState.COMPLETED,
        now=1_800_000_003,
    )
    store.set_job_state(
        packaging_job_id,
        JobState.FINALIZING,
        now=1_800_000_003,
    )
    packaging = client.get(f"/api/v1/jobs/{packaging_job_id}")

    assert packaging.json()["stage"] == {
        "code": "prepare_result",
        "started_at": "2027-01-15T08:00:03Z",
    }
    assert packaging.json()["active_stages"] == [packaging.json()["stage"]]
    assert packaging.json()["stage_history"][-2]["ended_at"] == ("2027-01-15T08:00:03Z")
    assert packaging.json()["stage_history"][-2]["outcome"] == "completed"
    assert packaging.json()["stage_history"][-1] == packaging.json()["stage"]

    store.block_job(
        packaging_job_id,
        category="modal_unavailable",
        now=1_800_000_004,
        next_retry_at=1_800_000_904,
    )
    blocked = client.get(f"/api/v1/jobs/{packaging_job_id}")
    assert blocked.json()["state"] == "blocked"
    assert blocked.json()["stage"] == {
        "code": "prepare_result",
        "started_at": "2027-01-15T08:00:03Z",
    }

    schema = client.get("/openapi.json").json()
    stage_schema = schema["components"]["schemas"]["JobStageView"]
    assert stage_schema["properties"]["code"] == {
        "title": "Code",
        "type": "string",
    }
    assert stage_schema["properties"]["function_name"]["anyOf"][0] == {"type": "string"}
    assert stage_schema["properties"]["started_at"]["format"] == "date-time"
    assert stage_schema["properties"]["ended_at"]["anyOf"][0]["format"] == ("date-time")
    assert schema["components"]["schemas"]["JobView"]["properties"]["stage_history"][
        "items"
    ]["$ref"].endswith("/JobStageView")
    assert schema["components"]["schemas"]["JobView"]["properties"]["active_stages"][
        "items"
    ]["$ref"].endswith("/JobStageView")


def test_gromacs_run_name_is_readable_sanitized_and_unique(tmp_path: Path) -> None:
    client, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")

    response = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
        display_name="  Kinase / alpha: trial #1  ",
    )

    assert response.status_code == 202
    assert response.json()["display_name"] == "Kinase / alpha: trial #1"
    assert adapter.submissions[0][1] == (
        f"kinase-alpha-trial-1-{UUID(response.json()['job_id']).hex}"
    )


def test_gromacs_simulation_time_accepts_200_ns_and_rejects_201(
    tmp_path: Path,
) -> None:
    client, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")

    accepted = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
        simulation_time_ns=200,
    )
    rejected = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
        simulation_time_ns=201,
    )

    assert accepted.status_code == 202
    assert rejected.status_code == 422
    assert adapter.submissions[0][2].simulation_time_ns == 200


def test_failed_spawn_can_retry_the_same_stable_run(
    tmp_path: Path,
) -> None:
    client, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    key = str(uuid4())
    adapter.failures_remaining = 1

    failed = _submit(client, csrf_token, idempotency_key=key)
    retried = _submit(client, csrf_token, idempotency_key=key)

    assert failed.status_code == 503
    assert failed.json()["code"] == "compute_unavailable"
    assert retried.status_code == 202
    assert retried.json()["state"] == "queued"
    assert len(adapter.submissions) == 2
    assert adapter.submissions[0][1] == adapter.submissions[1][1]
    assert adapter.submissions[0][1] == (
        f"first-simulation-{UUID(retried.json()['job_id']).hex}"
    )


def test_unknown_spawn_outcome_is_not_retried(
    tmp_path: Path,
) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    key = str(uuid4())
    adapter.unknown_failures_remaining = 1

    uncertain = _submit(client, csrf_token, idempotency_key=key)
    replayed = _submit(client, csrf_token, idempotency_key=key)

    assert uncertain.status_code == 202
    assert uncertain.json()["state"] == "state_unknown"
    assert uncertain.json()["state_unknown_at"]
    assert replayed.status_code == 202
    assert replayed.json() == uncertain.json()
    assert len(adapter.submissions) == 1
    owner_id = UUID(client.get("/api/v1/auth/me").json()["user_id"])
    job = store.get_job(owner_id, UUID(replayed.json()["job_id"]))
    assert job is not None
    assert job.state == JobState.STATE_UNKNOWN
    assert job.operations[0].submission_lease_until is None


def test_admin_can_resolve_state_unknown_after_manual_provider_review(
    tmp_path: Path,
) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "admin@example.com", is_admin=True)
    csrf_token = _login(client, "admin@example.com")
    adapter.unknown_failures_remaining = 1

    submitted = _submit(client, csrf_token, idempotency_key=str(uuid4()))
    job_id = submitted.json()["job_id"]
    modal_view = client.get("/api/v1/admin/modal")

    assert submitted.json()["state"] == "state_unknown"
    assert modal_view.status_code == 200
    assert modal_view.json()["state_unknown_jobs"] == [
        {
            "job_id": job_id,
            "workload": "gromacs",
            "display_name": "First simulation",
            "run_name": f"first-simulation-{UUID(job_id).hex}",
            "reason": "submission_outcome_unknown",
            "state_unknown_at": submitted.json()["state_unknown_at"],
        }
    ]

    resolved = client.post(
        f"/api/v1/admin/modal/state-unknown-jobs/{job_id}/mark-failed",
        headers=_unsafe_headers(csrf_token),
    )

    assert resolved.status_code == 200
    assert resolved.json()["state_unknown_jobs"] == []
    owner_id = UUID(client.get("/api/v1/auth/me").json()["user_id"])
    job = store.get_job(owner_id, UUID(job_id))
    assert job is not None
    assert job.state == JobState.FAILED
    assert job.error_message == (
        "An administrator could not confirm the remote compute state."
    )
    repeated = client.post(
        f"/api/v1/admin/modal/state-unknown-jobs/{job_id}/mark-failed",
        headers=_unsafe_headers(csrf_token),
    )
    assert repeated.status_code == 409
    assert repeated.json()["code"] == "job_state_changed"


def test_cancel_after_failed_spawn_finishes_without_modal_access(
    tmp_path: Path,
) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    adapter.failures_remaining = 1

    failed = _submit(client, csrf_token, idempotency_key=str(uuid4()))
    assert failed.status_code == 503

    [job] = _jobs(client.get("/api/v1/jobs"))
    job_id = UUID(str(job["job_id"]))
    cancelling = client.post(
        f"/api/v1/jobs/{job_id}/cancel",
        headers=_unsafe_headers(csrf_token),
    )
    assert cancelling.status_code == 202
    assert cancelling.json()["state"] == "cancel_requested"

    asyncio.run(
        GromacsReconciler(
            store,
            cast(Any, adapter),
            now=lambda: 1_800_000_001,
        ).reconcile()
    )

    cancelled = client.get(f"/api/v1/jobs/{job_id}")
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert cancelled.json()["completed_at"] == "2027-01-15T08:00:01Z"
    assert adapter.cancellations == []
    assert adapter.recovery_attempts == []


def test_my_jobs_and_job_lookup_are_private_to_the_cookie_owner(
    tmp_path: Path,
) -> None:
    alice, auth, _store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    _activate(auth, "bob@example.com")
    alice_csrf = _login(alice, "alice@example.com")
    submitted = _submit(
        alice,
        alice_csrf,
        idempotency_key=str(uuid4()),
    )
    job_id = submitted.json()["job_id"]
    bob = APIClient(alice.app)
    bob_csrf = _login(bob, "bob@example.com")
    anonymous = APIClient(alice.app)

    alice_list = alice.get("/api/v1/jobs")
    bob_list = bob.get("/api/v1/jobs")
    anonymous_list = anonymous.get("/api/v1/jobs")
    other_owner = bob.get(f"/api/v1/jobs/{job_id}")
    missing = bob.get(f"/api/v1/jobs/{uuid4()}")
    forbidden_cancel = bob.post(
        f"/api/v1/jobs/{job_id}/cancel",
        headers=_unsafe_headers(bob_csrf),
    )

    assert alice_list.status_code == 200
    assert [job["job_id"] for job in _jobs(alice_list)] == [job_id]
    assert bob_list.status_code == 200
    assert _jobs(bob_list) == []
    assert anonymous_list.status_code == 401
    assert other_owner.status_code == 404
    assert (other_owner.status_code, other_owner.content) == (
        missing.status_code,
        missing.content,
    )
    assert forbidden_cancel.status_code == 404
    assert adapter.cancellations == []


def test_job_history_uses_an_owner_scoped_cursor(tmp_path: Path) -> None:
    client, auth, store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    alice = store.get_user_by_email("alice@example.com")
    assert alice is not None
    store.update_user(alice.user_id, active_job_limit=10, now=1_800_000_000)
    for index in range(3):
        submitted = _submit(
            client,
            csrf_token,
            idempotency_key=str(uuid4()),
            display_name=f"Simulation {index}",
        )
        assert submitted.status_code == 202
        store.fail_job(
            UUID(submitted.json()["job_id"]),
            error_code="compute_failed",
            error_message="Completed test fixture",
            now=1_800_000_001 + index,
        )

    first = client.get("/api/v1/jobs?limit=2")
    first_body = first.json()
    second = client.get(f"/api/v1/jobs?limit=2&cursor={first_body['next_cursor']}")
    second_body = second.json()
    unknown = client.get(f"/api/v1/jobs?cursor={uuid4()}")
    schema = client.get("/openapi.json").json()

    assert first.status_code == 200
    assert len(first_body["jobs"]) == 2
    assert first_body["next_cursor"] is not None
    assert second.status_code == 200
    assert len(second_body["jobs"]) == 1
    assert second_body["next_cursor"] is None
    assert len({job["job_id"] for job in first_body["jobs"] + second_body["jobs"]}) == 3
    assert unknown.status_code == 400
    response_schema = schema["paths"]["/api/v1/jobs"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("/JobPageView")


def test_failed_jobs_expose_safe_typed_errors_only(tmp_path: Path) -> None:
    client, auth, store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    submitted = _submit(client, csrf_token, idempotency_key=str(uuid4()))
    job_id = UUID(submitted.json()["job_id"])

    assert "detail" not in submitted.json()
    assert "error_code" not in submitted.json()
    assert "error_message" not in submitted.json()

    store.fail_job(
        job_id,
        error_code="compute_failed",
        error_message="GROMACS could not complete the simulation.",
        now=1_800_000_001,
    )
    failed = client.get(f"/api/v1/jobs/{job_id}")
    schema = client.get("/openapi.json").json()
    error_code = schema["components"]["schemas"]["JobView"]["properties"]["error_code"]
    allowed_codes = next(item["enum"] for item in error_code["anyOf"] if "enum" in item)

    assert failed.status_code == 200
    assert failed.json()["error_code"] == "compute_failed"
    assert failed.json()["error_message"] == (
        "GROMACS could not complete the simulation."
    )
    assert "detail" not in failed.json()
    assert allowed_codes == [
        "compute_failed",
        "result_invalid",
    ]


def test_cancel_is_a_posted_idempotent_state_transition(tmp_path: Path) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    submitted = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
    )
    job_id = submitted.json()["job_id"]
    parsed_job_id = UUID(job_id)
    store.record_operation_outcome(
        parsed_job_id,
        operation="prepare_tpr_cpu",
        expected_modal_call_id="fc-1",
        outcome=JobOperationState.COMPLETED,
        now=1_799_999_999,
    )
    for operation, modal_call_id in (
        ("collect_traj_stats:nvt_", "fc-nvt"),
        ("collect_traj_stats:npt_", "fc-npt"),
        ("production_run_cpu", "fc-production"),
    ):
        token = uuid4().hex
        store.claim_modal_operation(
            parsed_job_id,
            operation=operation,
            submission_token=token,
            now=1_799_999_999,
        )
        store.attach_modal_call(
            parsed_job_id,
            operation=operation,
            modal_call_id=modal_call_id,
            submission_token=token,
            now=1_799_999_999,
        )

    first = client.post(
        f"/api/v1/jobs/{job_id}/cancel",
        headers=_unsafe_headers(csrf_token),
    )
    replay = client.post(
        f"/api/v1/jobs/{job_id}/cancel",
        headers=_unsafe_headers(csrf_token),
    )
    old_delete_route = client.delete(
        f"/api/v1/jobs/{job_id}",
        headers=_unsafe_headers(csrf_token),
    )

    assert first.status_code == 202
    assert first.json()["state"] == "cancel_requested"
    assert replay.status_code == 202
    assert replay.json()["state"] == "cancel_requested"
    assert adapter.cancellations == [
        "fc-nvt",
        "fc-npt",
        "fc-production",
        "fc-nvt",
        "fc-npt",
        "fc-production",
    ]
    assert old_delete_route.status_code == 405

    store.set_job_state(UUID(job_id), JobState.CANCELLED, now=1_800_000_000)
    terminal = client.post(
        f"/api/v1/jobs/{job_id}/cancel",
        headers=_unsafe_headers(csrf_token),
    )
    assert terminal.status_code == 409
    assert terminal.json()["code"] == "job_not_cancellable"

    second = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
    )
    second_id = UUID(second.json()["job_id"])
    store.set_job_state(second_id, JobState.FINALIZING, now=1_800_000_001)
    finalizing = client.post(
        f"/api/v1/jobs/{second_id}/cancel",
        headers=_unsafe_headers(csrf_token),
    )
    schema = client.get("/openapi.json").json()

    assert finalizing.status_code == 409
    assert finalizing.json()["code"] == "job_not_cancellable"
    assert _response_codes(schema, "/api/v1/jobs/{job_id}/cancel", 409) == [
        "job_not_cancellable"
    ]


def test_completed_archive_download_is_private_and_cached(tmp_path: Path) -> None:
    alice, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    _activate(auth, "bob@example.com")
    alice_csrf = _login(alice, "alice@example.com")
    submitted = _submit(alice, alice_csrf, idempotency_key=str(uuid4()))
    job_id = UUID(submitted.json()["job_id"])
    content = adapter.artifact_content
    store.complete_job(
        job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{job_id}/result.zip",
        result_filename="simulation.zip",
        result_size_bytes=len(content),
        result_sha256=hashlib.sha256(content).hexdigest(),
        result_archive_schema_version=1,
        now=1_800_000_001,
    )
    bob = APIClient(alice.app)
    _login(bob, "bob@example.com")

    hidden = bob.get(f"/api/v1/jobs/{job_id}/download")
    unprepared = alice.get(f"/api/v1/jobs/{job_id}/download")
    prepared = alice.post(
        f"/api/v1/jobs/{job_id}/prepare-download",
        headers=_unsafe_headers(alice_csrf),
    )
    cache: ArtifactCache = alice.app.state.cache
    cleanup_between_prepare_and_get = cache.clear()
    first = alice.get(f"/api/v1/jobs/{job_id}/download")
    second = alice.get(f"/api/v1/jobs/{job_id}/download")
    ranged = alice.get(
        f"/api/v1/jobs/{job_id}/download",
        headers={"Range": "bytes=2-5"},
    )
    invalid_range = alice.get(
        f"/api/v1/jobs/{job_id}/download",
        headers={"Range": "bytes=999999-"},
    )
    schema = alice.get("/openapi.json").json()

    assert hidden.status_code == 404
    assert unprepared.status_code == 409
    assert unprepared.json()["code"] == "result_not_prepared"
    assert prepared.status_code == 204
    assert cleanup_between_prepare_and_get.entries == 0
    assert first.status_code == 200
    assert first.content == content
    assert first.headers["content-type"] == "application/zip"
    assert first.headers["content-disposition"] == (
        'attachment; filename="first-simulation-results.zip"'
    )
    assert first.headers["cache-control"] == "private, no-store"
    assert first.headers["etag"] == f'"{hashlib.sha256(content).hexdigest()}"'
    assert second.content == content
    assert ranged.status_code == 206
    assert ranged.content == content[2:6]
    assert ranged.headers["content-range"] == f"bytes 2-5/{len(content)}"
    assert invalid_range.status_code == 416
    assert invalid_range.headers["content-range"] == f"bytes */{len(content)}"
    range_contract = schema["paths"]["/api/v1/jobs/{job_id}/download"]["get"][
        "responses"
    ]["416"]
    assert range_contract["headers"]["Content-Range"]["schema"] == {"type": "string"}
    assert adapter.downloads == 1
    assert [path.name for path in (tmp_path / "cache").iterdir()] == [f"{job_id}.zip"]


def test_local_result_storage_failures_are_503_and_preserve_the_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    submitted = _submit(client, csrf_token, idempotency_key=str(uuid4()))
    job_id = UUID(submitted.json()["job_id"])
    content = adapter.artifact_content
    store.complete_job(
        job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{job_id}/result.zip",
        result_filename="simulation.zip",
        result_size_bytes=len(content),
        result_sha256=hashlib.sha256(content).hexdigest(),
        result_archive_schema_version=1,
        now=1_800_000_001,
    )
    (tmp_path / "cache").rmdir()

    unavailable_prepare = client.post(
        f"/api/v1/jobs/{job_id}/prepare-download",
        headers=_unsafe_headers(csrf_token),
    )
    preserved = store.get_job(
        UUID(client.get("/api/v1/auth/me").json()["user_id"]), job_id
    )

    assert unavailable_prepare.status_code == 503
    assert unavailable_prepare.json()["code"] == "result_storage_unavailable"
    assert preserved is not None
    assert preserved.state == JobState.SUCCEEDED
    assert preserved.blocking_category is None

    (tmp_path / "cache").mkdir()
    assert (
        client.post(
            f"/api/v1/jobs/{job_id}/prepare-download",
            headers=_unsafe_headers(csrf_token),
        ).status_code
        == 204
    )
    cache: ArtifactCache = client.app.state.cache

    async def permission_denied(*_args, **_kwargs):
        raise PermissionError("cache is not readable")

    monkeypatch.setattr(cache, "acquire_async", permission_denied)
    unavailable_download = client.get(f"/api/v1/jobs/{job_id}/download")
    schema = client.get("/openapi.json").json()

    assert unavailable_download.status_code == 503
    assert unavailable_download.json()["code"] == "result_storage_unavailable"
    assert (
        "503" in schema["paths"]["/api/v1/jobs/{job_id}/download"]["get"]["responses"]
    )


def test_large_result_validation_keeps_core_requests_and_cache_hits_responsive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, auth, store, adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    cached_submission = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
        display_name="Cached result",
    )
    cached_job_id = UUID(cached_submission.json()["job_id"])
    cached_archive = adapter.artifact_content
    store.complete_job(
        cached_job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{cached_job_id}/result.zip",
        result_filename="cached.zip",
        result_size_bytes=len(cached_archive),
        result_sha256=hashlib.sha256(cached_archive).hexdigest(),
        result_archive_schema_version=1,
        now=1_800_000_001,
    )
    assert (
        client.post(
            f"/api/v1/jobs/{cached_job_id}/prepare-download",
            headers=_unsafe_headers(csrf_token),
        ).status_code
        == 204
    )
    completed_submission = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
    )
    completed_job_id = UUID(completed_submission.json()["job_id"])
    large_archive = b"PK" + b"x" * (32 * 1024 * 1024)
    adapter.artifact_content = large_archive
    store.complete_job(
        completed_job_id,
        state=JobState.SUCCEEDED,
        result_volume_name="Gromacs-outputs",
        result_volume_path=f"api-results/{completed_job_id}/result.zip",
        result_filename="simulation.zip",
        result_size_bytes=len(large_archive),
        result_sha256=hashlib.sha256(large_archive).hexdigest(),
        result_archive_schema_version=1,
        now=1_800_000_001,
    )
    cancellable = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
        display_name="Cancellation target",
    )
    cancellable_job_id = cancellable.json()["job_id"]
    cache: ArtifactCache = client.app.state.cache
    original_matches = cache._matches
    validation_started = Event()
    release_validation = Event()

    def delayed_match(*args, **kwargs):
        validation_started.set()
        matches = original_matches(*args, **kwargs)
        release_validation.wait(timeout=5)
        return matches

    monkeypatch.setattr(cache, "_matches", delayed_match)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=ORIGIN,
            cookies=client.cookies,
        ) as browser:
            prepare = asyncio.create_task(
                browser.post(
                    f"/api/v1/jobs/{completed_job_id}/prepare-download",
                    headers=_unsafe_headers(csrf_token),
                )
            )
            assert await asyncio.to_thread(validation_started.wait, 5)
            try:
                health, login, inspected, cached, cancelled = await asyncio.wait_for(
                    asyncio.gather(
                        browser.get("/api/v1/health"),
                        browser.post(
                            "/api/v1/auth/login",
                            headers={"Origin": ORIGIN},
                            json={
                                "email": "alice@example.com",
                                "password": PASSWORD,
                            },
                        ),
                        browser.get(f"/api/v1/jobs/{completed_job_id}"),
                        browser.get(f"/api/v1/jobs/{cached_job_id}/download"),
                        browser.post(
                            f"/api/v1/jobs/{cancellable_job_id}/cancel",
                            headers=_unsafe_headers(csrf_token),
                        ),
                    ),
                    timeout=5,
                )
                assert health.status_code == 200
                assert login.status_code == 200
                assert inspected.status_code == 200
                assert cached.status_code == 200
                assert cached.content == cached_archive
                assert cancelled.status_code == 202
            finally:
                release_validation.set()
            assert (await prepare).status_code == 204

    asyncio.run(scenario())
