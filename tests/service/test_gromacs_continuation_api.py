"""Owner-scoped continuation admission reuses durable Jobs and retained inputs."""

# ruff: noqa: D103

import asyncio
import errno
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from modal.exception import NotFoundError
from test_api_contract import (
    ORIGIN,
    GromacsAdapter,
    _app,
    _humanization_session,
    _request,
    _session,
)
from test_gromacs_archive import REQUEST

from biomodals.app.bioinfo.gromacs.continuation import (
    ContinuationInspection,
    ContinuationSource,
)
from biomodals.app.bioinfo.gromacs.execution_runtime import GromacsExecutionRequest
from biomodals.service.gromacs import router
from biomodals.service.gromacs.modal import GromacsToolAdapter
from biomodals.service.http_contract import require_session, require_unsafe_session
from biomodals.service.pending import PendingRequestStore
from biomodals.service.store import JobState


def _completed(app, session, *, state=JobState.SUCCEEDED):
    job = app.state.store.admit_job(
        owner_user_id=session.principal.user_id,
        tool="gromacs",
        display_name="Source",
        idempotency_key=str(uuid4()),
        request_digest="a" * 64,
        modal_environment=app.state.configuration.modal_environment().value,
        modal_app_name="Gromacs",
        modal_app_version=1,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=8,
        max_active_gpu_provider_calls=1,
        now=3,
    ).job
    app.state.store.complete_job(
        job.job_id,
        result_state=state,
        result_filename="source.zip",
        result_media_type="application/zip",
        result_size_bytes=10,
        result_sha256="b" * 64,
        result_archive_schema="gromacs/5",
        now=4,
    )
    return app.state.store.get_job_by_id(job.job_id)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = _app(tmp_path)
    session = _humanization_session(app)
    parent = _completed(app, session)
    pending = PendingRequestStore(tmp_path / "pending")
    original = replace(
        REQUEST, run_name="source", simulation_time_ns=250, execution_plan_version="2"
    )
    pending.put(parent.job_id, original.to_bytes())
    reads = []

    async def source(self, job, deployment):
        reads.append(job.job_id)
        content = pending.get(job.job_id)
        if content is None:
            raise FileNotFoundError("Source inputs unavailable")
        saved = GromacsExecutionRequest.from_bytes(content)
        source = ContinuationSource(
            execution_run_id=job.job_id,
            run_name=saved.run_name,
            file_stem=saved.file_stem,
            simulation_time_ns=saved.simulation_time_ns,
            request_sha256=sha256(saved.to_bytes()).hexdigest(),
            publication_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
        )
        return ContinuationInspection.from_request(source, saved)

    monkeypatch.setattr(GromacsAdapter, "continuation_source", source)
    return app, session, parent, pending, reads


def _submit(app, parent_id, *, key=None, **changes):
    return _request(
        app,
        "POST",
        f"/api/v1/gromacs/jobs/{parent_id}/continue",
        json={"additional_time_ns": 250, "cpu_only": True, **changes},
        headers={"Origin": ORIGIN, "Idempotency-Key": str(key or uuid4())},
    )


def test_source_metadata_owner_scope_and_read_only_validation(setup):
    app, session, parent, pending, reads = setup
    path = f"/api/v1/gromacs/jobs/{parent.job_id}/continuation"
    metadata = _request(app, "GET", path)
    assert metadata.status_code == 200
    assert metadata.json() == {
        "source_job_id": str(parent.job_id),
        "source_display_name": "Source",
        "simulation_time_ns": 250,
        "cpu_only": False,
        "parent_job_id": None,
        "eligible": True,
        "code": None,
        "detail": "Native inputs are available; the new job verifies checksums and checkpoint completion before production",
        "min_additional_time_ns": 1,
        "max_additional_time_ns": 250,
    }
    assert app.state.store.get_job_by_id(parent.job_id) == parent
    app.dependency_overrides[require_session] = _session
    assert _request(app, "GET", path).status_code == 404
    assert reads == [parent.job_id]
    app.dependency_overrides[require_unsafe_session] = _session
    assert _submit(app, parent.job_id).status_code == 404


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_source_check_deadline_cancels_reads_and_allows_manual_retry(
    setup, monkeypatch, method
):
    app, _, parent, pending, _ = setup
    original = GromacsAdapter.continuation_source
    original_input = pending.get(parent.job_id)
    cancelled = []

    async def slow(self, job, deployment):
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancelled.append(job.job_id)
            raise
        return await original(self, job, deployment)

    monkeypatch.setattr(router, "SOURCE_CHECK_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(GromacsAdapter, "continuation_source", slow)
    path = f"/api/v1/gromacs/jobs/{parent.job_id}/continuation"
    response = (
        _request(app, "GET", path) if method == "GET" else _submit(app, parent.job_id)
    )
    assert response.status_code == 504
    assert response.json() == {
        "code": "source_check_timeout",
        "detail": "Checking the source simulation timed out after 45 seconds. Please try again.",
    }
    assert cancelled == [parent.job_id]
    assert app.state.store.get_job_by_id(parent.job_id) == parent
    assert pending.get(parent.job_id) == original_input
    monkeypatch.setattr(GromacsAdapter, "continuation_source", original)
    monkeypatch.setattr(router, "SOURCE_CHECK_TIMEOUT_SECONDS", 45)
    assert _request(app, "GET", path).json()["eligible"] is True


def test_source_deadline_stops_waiting_for_metadata_rpc(setup, monkeypatch):
    app, _, parent, pending, _ = setup
    pending.delete(parent.job_id)
    cancelled = []

    async def inspect(run_id):
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancelled.append(run_id)
            raise

    import modal

    monkeypatch.setattr(
        modal.Function,
        "from_name",
        lambda *args, **kwargs: SimpleNamespace(remote=SimpleNamespace(aio=inspect)),
    )
    monkeypatch.setattr(
        GromacsAdapter, "continuation_source", GromacsToolAdapter.continuation_source
    )
    monkeypatch.setattr(router, "SOURCE_CHECK_TIMEOUT_SECONDS", 0.005)
    response = _request(
        app, "GET", f"/api/v1/gromacs/jobs/{parent.job_id}/continuation"
    )
    assert response.status_code == 504
    assert cancelled == [str(parent.job_id)]


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    "failure,code,status",
    [("old", "deployment_incompatible", 409), ("timeout", "source_check_timeout", 504)],
)
def test_remote_inspector_failures_are_actionable_without_admission(
    setup, monkeypatch, method, failure, code, status
):
    from modal.exception import FunctionTimeoutError

    app, _, parent, pending, _ = setup
    original = pending.get(parent.job_id)

    async def fail(self, job, deployment):
        assert job.job_id == parent.job_id
        assert (
            deployment.deployment_version
            == app.state.configuration.tool("gromacs").modal_app_version.value
        )
        if failure == "old":
            raise NotFoundError("Missing inspector")
        raise FunctionTimeoutError("Read-only inspection timed out")

    monkeypatch.setattr(GromacsAdapter, "continuation_source", fail)
    response = (
        _request(app, "GET", f"/api/v1/gromacs/jobs/{parent.job_id}/continuation")
        if method == "GET"
        else _submit(app, parent.job_id)
    )
    assert response.status_code == status
    assert response.json()["code"] == code
    assert pending.get(parent.job_id) == original


def test_new_roots_inherit_physics_and_allow_chain_branch_and_replay(setup):
    app, session, parent, pending, reads = setup
    original = pending.get(parent.job_id)
    key = uuid4()
    first = _submit(app, parent.job_id, key=key)
    assert first.status_code == 202, first.text
    child_id = UUID(first.json()["job_id"])
    child = GromacsExecutionRequest.from_bytes(pending.get(child_id))
    assert child.simulation_time_ns == 500
    assert child.cpu_only
    assert child.continuation.execution_run_id == parent.job_id
    assert child.ld_seed == REQUEST.ld_seed
    assert child.gen_seed == REQUEST.gen_seed
    assert child.genion_seed == REQUEST.genion_seed
    assert child.run_pdbfixer == REQUEST.run_pdbfixer
    assert child.max_active_provider_calls == 16
    assert child.max_active_gpu_provider_calls == 2
    assert _submit(app, parent.job_id, key=key).json()["job_id"] == str(child_id)
    assert reads == [parent.job_id]
    assert _submit(app, parent.job_id, key=key, additional_time_ns=1).status_code == 409
    # Complete the child without running science, then extend it.
    app.state.store.complete_job(
        child_id,
        result_state=JobState.SUCCEEDED,
        result_filename="child.zip",
        result_media_type="application/zip",
        result_size_bytes=10,
        result_sha256="e" * 64,
        result_archive_schema="gromacs/5",
        now=5,
    )
    info = _request(app, "GET", f"/api/v1/gromacs/jobs/{child_id}/continuation").json()
    assert info["parent_job_id"] == str(parent.job_id)
    chain = _submit(app, child_id, additional_time_ns=1)
    assert chain.status_code == 202
    chained = GromacsExecutionRequest.from_bytes(
        pending.get(UUID(chain.json()["job_id"]))
    )
    assert chained.simulation_time_ns == 501
    assert chained.continuation.execution_run_id == child_id
    sibling = _submit(app, parent.job_id, additional_time_ns=1)
    assert sibling.status_code == 202
    assert sibling.json()["job_id"] != str(child_id)
    assert pending.get(parent.job_id) == original
    assert app.state.store.get_job_by_id(parent.job_id) == parent
    # Exact replay stays possible after source data disappears and admission pauses.
    pending.delete(parent.job_id)
    app.state.configuration.set_tool("gromacs", active_job_limit=0)
    count = len(reads)
    assert _submit(app, parent.job_id, key=key).json()["job_id"] == str(child_id)
    assert len(reads) == count


@pytest.mark.parametrize("duration", [0, 251, 1.5, True])
def test_invalid_added_intervals_do_not_read_source(setup, duration):
    app, _, parent, _, reads = setup
    assert _submit(app, parent.job_id, additional_time_ns=duration).status_code == 422
    assert reads == []


def test_missing_source_and_incompatible_target_never_admit(setup, monkeypatch):
    app, _, parent, pending, _ = setup

    async def unsupported(self, _deployment):
        raise NotFoundError("Old deployment")

    monkeypatch.setattr(GromacsAdapter, "preflight", unsupported)
    result = _submit(app, parent.job_id)
    assert result.status_code == 409
    assert result.json()["code"] == "deployment_incompatible"
    pending.delete(parent.job_id)
    result = _submit(app, parent.job_id)
    assert result.status_code == 409
    assert result.json()["code"] == "source_unavailable"


def test_missing_native_file_is_explained_on_form_and_submission(setup, monkeypatch):
    app, _, parent, _, _ = setup

    async def missing(self, _job, _deployment):
        raise FileNotFoundError(
            errno.ENOENT, "Missing restart file", "private/run/production_source.edr"
        )

    monkeypatch.setattr(GromacsAdapter, "continuation_source", missing)
    metadata = _request(
        app, "GET", f"/api/v1/gromacs/jobs/{parent.job_id}/continuation"
    )
    assert metadata.json()["eligible"] is False
    assert (
        metadata.json()["detail"]
        == "Required source file is unavailable: production_source.edr"
    )
    submission = _submit(app, parent.job_id)
    assert submission.status_code == 409
    assert submission.json() == {
        "code": "source_unavailable",
        "detail": metadata.json()["detail"],
    }


def test_fresh_schema_uses_same_250ns_limit(setup):
    app, *_ = setup
    schema = app.openapi()
    reference = schema["paths"]["/api/v1/gromacs/jobs"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    assert (
        schema["components"]["schemas"][reference]["properties"]["simulation_time_ns"][
            "maximum"
        ]
        == 250
    )
