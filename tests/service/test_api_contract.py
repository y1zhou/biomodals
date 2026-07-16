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

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
    ) -> SubmittedCall:
        self.submissions.append((pdb_content, run_name, options))
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
    assert no_csrf.status_code == 403
    assert wrong_origin.status_code == 403
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
    assert oversized_body.status_code == 413
    assert adapter.submissions == []


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
    assert missing_key.status_code == 422
    assert malformed_key.status_code == 422
    assert len(adapter.submissions) == 1
    pdb_content, run_name, options = adapter.submissions[0]
    assert pdb_content == VALID_PDB
    assert run_name
    assert options == GromacsJobOptions(simulation_time_ns=3, cpu_only=True)


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
    assert first.headers["cache-control"] == "private, no-store"
    assert first.headers["etag"] == f'"{hashlib.sha256(content).hexdigest()}"'
    assert second.content == content
    assert ranged.status_code == 206
    assert ranged.content == content[2:6]
    assert ranged.headers["content-range"] == f"bytes 2-5/{len(content)}"
    assert adapter.downloads == 1
    assert [path.name for path in (tmp_path / "cache").iterdir()] == [f"{job_id}.zip"]
