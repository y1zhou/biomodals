"""HTTP contract tests for the GROMACS job API."""

# ruff: noqa: D101,D102,D103,D107

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from biomodals.service.gromacs_api import (
    ErrorResponse,
    GromacsJobOptions,
    JobArtifact,
    JobArtifactFile,
    JobNotFoundError,
    JobResult,
    JobSnapshot,
    JobState,
    JobStatus,
    create_app,
)

VALID_PDB = (
    b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
    b"END\n"
)


def request(
    app: FastAPI,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(send())


class FakeJobBackend:
    def __init__(self) -> None:
        self.submissions: list[tuple[bytes, GromacsJobOptions]] = []
        self.jobs = {
            "job-123": JobSnapshot(
                job_id="job-123",
                status=JobState.PENDING,
                run_name="api-123",
            )
        }

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
    ) -> JobStatus:
        self.submissions.append((pdb_content, options))
        return self.jobs["job-123"]

    async def inspect(self, job_id: str) -> JobSnapshot:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    async def cancel(self, job_id: str) -> JobStatus:
        self.jobs[job_id] = JobSnapshot(
            job_id=job_id,
            status=JobState.CANCELLED,
        )
        return self.jobs[job_id]


def test_app_requires_explicit_authentication_boundary() -> None:
    with pytest.raises(ValueError, match="Configure api_key"):
        create_app(FakeJobBackend())


def test_uploading_pdb_submits_job_and_returns_202() -> None:
    backend = FakeJobBackend()
    app = create_app(backend, trusted_proxy_auth=True)

    response = request(
        app,
        "POST",
        "/jobs",
        files={"pdb": ("protein.pdb", VALID_PDB, "chemical/x-pdb")},
        data={"simulation_time_ns": "3", "cpu_only": "true"},
    )

    assert response.status_code == 202
    assert JobStatus.model_validate_json(response.content) == JobStatus(
        job_id="job-123",
        status=JobState.PENDING,
        run_name="api-123",
    )
    assert backend.submissions == [
        (
            VALID_PDB,
            GromacsJobOptions(simulation_time_ns=3, cpu_only=True),
        )
    ]


def test_job_status_is_available_without_waiting() -> None:
    app = create_app(FakeJobBackend(), trusted_proxy_auth=True)

    response = request(app, "GET", "/jobs/job-123")

    assert response.status_code == 200
    assert JobStatus.model_validate_json(response.content) == JobStatus(
        job_id="job-123",
        status=JobState.PENDING,
        run_name="api-123",
    )


def test_completed_job_returns_provider_neutral_artifact_manifest() -> None:
    backend = FakeJobBackend()
    backend.jobs["job-123"] = JobSnapshot(
        job_id="job-123",
        status=JobState.SUCCEEDED,
        result=JobResult(
            run_name="api-deadbeef",
            artifacts=[
                JobArtifact(
                    name="gromacs_run",
                    kind="directory",
                    files=[
                        JobArtifactFile(
                            path="production_api-deadbeef.xtc",
                            role="trajectory",
                        )
                    ],
                )
            ],
        ),
    )
    app = create_app(backend, trusted_proxy_auth=True)

    response = request(app, "GET", "/jobs/job-123/result")

    assert response.status_code == 200
    assert (
        JobResult.model_validate_json(response.content)
        == backend.jobs["job-123"].result
    )


def test_job_can_be_cancelled() -> None:
    app = create_app(FakeJobBackend(), trusted_proxy_auth=True)

    response = request(app, "DELETE", "/jobs/job-123")

    assert response.status_code == 202
    assert JobStatus.model_validate_json(response.content) == JobStatus(
        job_id="job-123",
        status=JobState.CANCELLED,
    )


def test_unknown_job_returns_404() -> None:
    app = create_app(FakeJobBackend(), trusted_proxy_auth=True)

    response = request(app, "GET", "/jobs/missing")

    assert response.status_code == 404
    assert ErrorResponse.model_validate_json(response.content) == ErrorResponse(
        detail="Job 'missing' was not found"
    )


def test_oversized_upload_is_rejected_before_submission() -> None:
    backend = FakeJobBackend()
    app = create_app(backend, max_pdb_bytes=8, trusted_proxy_auth=True)

    response = request(
        app,
        "POST",
        "/jobs",
        files={"pdb": ("large.pdb", b"ATOM      1\n", "chemical/x-pdb")},
    )

    assert response.status_code == 413
    assert backend.submissions == []


def test_non_utf8_pdb_is_rejected_before_submission() -> None:
    backend = FakeJobBackend()
    app = create_app(backend, trusted_proxy_auth=True)

    response = request(
        app,
        "POST",
        "/jobs",
        files={"pdb": ("binary.pdb", b"ATOM  \xff\n", "chemical/x-pdb")},
    )

    assert response.status_code == 422
    assert backend.submissions == []


def test_malformed_atom_record_is_rejected_before_submission() -> None:
    backend = FakeJobBackend()
    app = create_app(backend, trusted_proxy_auth=True)

    response = request(
        app,
        "POST",
        "/jobs",
        files={"pdb": ("invalid.pdb", b"ATOM  nonsense\n", "chemical/x-pdb")},
    )

    assert response.status_code == 422
    assert backend.submissions == []


def test_total_request_body_is_limited_before_multipart_parsing() -> None:
    backend = FakeJobBackend()
    app = create_app(backend, max_pdb_bytes=8, trusted_proxy_auth=True)

    response = request(
        app,
        "POST",
        "/jobs",
        content=b"x" * (64 * 1024 + 9),
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 413
    assert ErrorResponse.model_validate_json(response.content) == ErrorResponse(
        detail="Request body is too large"
    )
    assert backend.submissions == []


def test_chunked_request_body_cannot_bypass_total_limit() -> None:
    backend = FakeJobBackend()
    app = create_app(backend, max_pdb_bytes=8, trusted_proxy_auth=True)
    boundary = "chunked-test"

    async def chunks():
        yield (
            b"--chunked-test\r\n"
            b'Content-Disposition: form-data; name="pdb"; filename="large.pdb"\r\n'
            b"Content-Type: chemical/x-pdb\r\n\r\n"
        )
        yield b"x" * (64 * 1024)
        yield b"\r\n--chunked-test--\r\n"

    response = request(
        app,
        "POST",
        "/jobs",
        content=chunks(),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )

    assert response.status_code == 413
    assert backend.submissions == []


def test_optional_bearer_authentication_fails_closed() -> None:
    app = create_app(FakeJobBackend(), api_key="secret")

    unauthorized = request(
        app,
        "POST",
        "/jobs",
        content=b"x" * (11 * 1024 * 1024),
    )
    authorized = request(
        app,
        "GET",
        "/jobs/job-123",
        headers={"authorization": "Bearer secret"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_pending_result_returns_202() -> None:
    app = create_app(FakeJobBackend(), trusted_proxy_auth=True)

    response = request(app, "GET", "/jobs/job-123/result")

    assert response.status_code == 202
    assert JobStatus.model_validate_json(response.content) == JobStatus(
        job_id="job-123",
        status=JobState.PENDING,
        run_name="api-123",
    )
