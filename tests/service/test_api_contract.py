"""Unified browser API contracts for authentication and private jobs."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI

from biomodals.service.api import create_app
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthService
from biomodals.service.gromacs import GromacsJobOptions, create_registration
from biomodals.service.jobs import WorkloadRegistration
from biomodals.service.store import JobState, ServiceStore

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


class FakeGromacsAdapter:
    """Small fake for the Modal boundary; no Modal object reaches HTTP tests."""

    def __init__(self) -> None:
        self.submissions: list[tuple[bytes, str, GromacsJobOptions]] = []
        self.cancellations: list[str] = []
        self.downloads = 0
        self.artifact_content = b"PK\x03\x04verified archive"
        self.failures_remaining = 0

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
    ) -> SubmittedCall:
        self.submissions.append((pdb_content, run_name, options))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary Modal failure")
        return SubmittedCall(
            modal_call_id=f"fc-{len(self.submissions)}",
            run_name=run_name,
        )

    async def cancel(self, modal_call_id: str) -> None:
        self.cancellations.append(modal_call_id)

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

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


def _password_token(link: str) -> str:
    return parse_qs(urlparse(link).fragment)["token"][0]


def _activate(auth: AuthService, email: str) -> None:
    link = auth.create_user(email, display_name=email.partition("@")[0].title())
    auth.set_password(_password_token(link), PASSWORD)


def _service(
    tmp_path: Path,
    *,
    max_pdb_bytes: int = 10 * 1024 * 1024,
) -> tuple[APIClient, AuthService, ServiceStore, FakeGromacsAdapter]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    auth = AuthService(store, frontend_url=ORIGIN)
    adapter = FakeGromacsAdapter()
    registration = create_registration(
        adapter,
        active_limit=2,
        max_pdb_bytes=max_pdb_bytes,
    )
    assert isinstance(registration, WorkloadRegistration)
    app = create_app(
        store=store,
        auth=auth,
        workloads=[registration],
        allowed_origin=ORIGIN,
        secure_cookies=True,
        cache=ArtifactCache(tmp_path / "cache", max_bytes=1024),
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
):
    headers = _unsafe_headers(csrf_token)
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return client.post(
        "/api/v1/gromacs/jobs",
        headers=headers,
        files={"pdb": ("protein.pdb", VALID_PDB, "chemical/x-pdb")},
        data={
            "display_name": "First simulation",
            "simulation_time_ns": str(simulation_time_ns),
            "cpu_only": "true",
        },
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
    link = auth.create_user("alice@example.com", display_name="Alice")
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


def test_password_setup_exposes_coded_errors_and_cookie_contract(
    tmp_path: Path,
) -> None:
    client, auth, _store, _adapter = _service(tmp_path)
    link = auth.create_user("alice@example.com", display_name="Alice")
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
        assert forbidden["properties"]["code"]["enum"] == [
            "csrf_invalid",
            "origin_not_allowed",
        ]


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
        ("/api/v1/auth/login", "post"): {"401", "403", "422"},
        ("/api/v1/auth/set-password", "post"): {"400", "403", "422"},
        ("/api/v1/auth/me", "get"): {"401"},
        ("/api/v1/auth/logout", "post"): {"401", "403"},
        ("/api/v1/jobs", "get"): {"401"},
        ("/api/v1/jobs/{job_id}", "get"): {"401", "404", "422"},
        ("/api/v1/jobs/{job_id}/cancel", "post"): {
            "401",
            "403",
            "404",
            "409",
            "422",
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
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"
    assert missing_key.status_code == 422
    assert malformed_key.status_code == 422
    assert len(adapter.submissions) == 1
    pdb_content, run_name, options = adapter.submissions[0]
    assert pdb_content == VALID_PDB
    assert run_name
    assert options == GromacsJobOptions(simulation_time_ns=3, cpu_only=True)


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
    assert adapter.submissions[0][1] == f"api-{UUID(retried.json()['job_id']).hex}"


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
        "result_unavailable",
    ]


def test_cancel_is_a_posted_idempotent_state_transition(tmp_path: Path) -> None:
    client, auth, store, _adapter = _service(tmp_path)
    _activate(auth, "alice@example.com")
    csrf_token = _login(client, "alice@example.com")
    submitted = _submit(
        client,
        csrf_token,
        idempotency_key=str(uuid4()),
    )
    job_id = submitted.json()["job_id"]

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
        now=1_800_000_001,
    )
    bob = APIClient(alice.app)
    _login(bob, "bob@example.com")

    hidden = bob.get(f"/api/v1/jobs/{job_id}/download")
    first = alice.get(f"/api/v1/jobs/{job_id}/download")
    second = alice.get(f"/api/v1/jobs/{job_id}/download")
    ranged = alice.get(
        f"/api/v1/jobs/{job_id}/download",
        headers={"Range": "bytes=2-5"},
    )

    assert hidden.status_code == 404
    assert first.status_code == 200
    assert first.content == content
    assert first.headers["content-type"] == "application/zip"
    assert first.headers["content-disposition"] == (
        'attachment; filename="simulation.zip"'
    )
    assert first.headers["cache-control"] == "private, no-store"
    assert first.headers["etag"] == f'"{hashlib.sha256(content).hexdigest()}"'
    assert second.content == content
    assert ranged.status_code == 206
    assert ranged.content == content[2:6]
    assert ranged.headers["content-range"] == f"bytes 2-5/{len(content)}"
    assert adapter.downloads == 1
    assert [path.name for path in (tmp_path / "cache").iterdir()] == [f"{job_id}.zip"]
