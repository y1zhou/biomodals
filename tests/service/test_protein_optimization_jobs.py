"""Immutable optimization admission, ownership and native CSV result delivery."""

import asyncio
from io import BytesIO
from threading import Event
from uuid import UUID, uuid4

import httpx
import polars as pl
import pytest
from test_api_contract import ORIGIN, _app, _humanization_session, _request, _session
from test_protein_optimization_results import _result

from biomodals.service.http_contract import require_session
from biomodals.service.protein_optimization.results import build_projection
from biomodals.service.store import JobState
from biomodals.workflow.protein_optimization.execution import (
    OptimizationExecutionRequest,
)

ROOT = "/api/v1/protein-optimization"


def _body(app):
    inputs = {
        "measurements_csv": "mutations,label\n,0\nA:A1V,1\nB:G1A,2\n",
        "parental_fasta": ">A\nAG\n>B\nGA\n",
    }
    review = _request(app, "POST", ROOT + "/review", json=inputs)
    assert review.status_code == 200, review.text
    return {**inputs, "review_digest": review.json()["review_digest"]}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", "Protein optimization"),
        (" \t\n", "Protein optimization"),
        ("  Round  two  ", "Round two"),
    ],
)
def test_optional_name_normalization_survives_admission_and_replay(
    tmp_path, name, expected
):
    """Explicit blank names use the default in saved Jobs and retained inputs."""
    app = _app(tmp_path)
    _humanization_session(app)
    body = {**_body(app), "display_name": name}
    headers = {"Origin": ORIGIN, "Idempotency-Key": str(uuid4())}
    try:
        response = _request(app, "POST", ROOT + "/jobs", json=body, headers=headers)
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["display_name"] == expected
        retained = _request(app, "GET", ROOT + f"/jobs/{job['job_id']}/inputs")
        assert retained.json()["display_name"] == expected
        replay = _request(app, "POST", ROOT + "/jobs", json=body, headers=headers)
        assert replay.status_code == 202, replay.text
        assert replay.json()["job_id"] == job["job_id"]
    finally:
        asyncio.run(app.state.cache.shutdown())
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_review_binding_replay_retained_input_and_known_rejections(tmp_path):
    """Only an exact reviewed request admits; replay does not repeat preflight."""
    app = _app(tmp_path)
    _humanization_session(app)
    body = _body(app)
    headers = {"Origin": ORIGIN, "Idempotency-Key": str(uuid4())}
    changed = {**body, "settings": {"max_mutations": 1}}
    rejected = _request(app, "POST", ROOT + "/jobs", json=changed, headers=headers)
    assert (rejected.status_code, rejected.json()["code"]) == (409, "review_changed")
    response = _request(app, "POST", ROOT + "/jobs", json=body, headers=headers)
    assert response.status_code == 202, response.text
    job_id = UUID(response.json()["job_id"])
    preflights = app.state.lifecycle.remote.preflights
    assert _request(app, "POST", ROOT + "/jobs", json=body, headers=headers).json()[
        "job_id"
    ] == str(job_id)
    assert app.state.lifecycle.remote.preflights == preflights
    conflict = _request(
        app,
        "POST",
        ROOT + "/jobs",
        json={**body, "display_name": "Different"},
        headers=headers,
    )
    assert (conflict.status_code, conflict.json()["code"]) == (
        409,
        "idempotency_conflict",
    )
    inputs = _request(app, "GET", ROOT + f"/jobs/{job_id}/inputs")
    assert inputs.status_code == 200, inputs.text
    assert inputs.json()["measurements_csv"] == body["measurements_csv"]
    assert inputs.json()["settings"]["mode"] == "combination"
    pending = (
        tmp_path / "pending" / "pending-inputs" / str(job_id) / "request.bin"
    ).read_bytes()
    retained = OptimizationExecutionRequest.from_bytes(pending)
    assert retained.design.settings.direction == "maximize"
    assert retained.max_active_provider_calls == 16
    assert retained.max_active_gpu_provider_calls == 2
    assert (
        _request(app, "GET", ROOT + f"/jobs/{job_id}/candidates").json()["code"]
        == "result_not_ready"
    )
    app.dependency_overrides[require_session] = _session
    assert _request(app, "GET", ROOT + f"/jobs/{job_id}/inputs").status_code == 404
    assert _request(app, "GET", ROOT + f"/jobs/{job_id}/candidates").status_code == 404
    asyncio.run(app.state.cache.shutdown())
    asyncio.run(app.state.antibody_analysis.shutdown())


def test_candidate_routes_export_scientific_order_and_deleted_intent(tmp_path):
    """Selected IDs are checked against this owner result before streaming CSV."""
    app = _app(tmp_path)
    session = _humanization_session(app)
    body = _body(app)
    headers = {"Origin": ORIGIN, "Idempotency-Key": str(uuid4())}
    submitted = _request(app, "POST", ROOT + "/jobs", json=body, headers=headers)
    job_id = UUID(submitted.json()["job_id"])
    cache = app.state.cache
    _, source, manifest = _result(tmp_path)
    csv, database = cache.staging_path(str(job_id)), cache.staging_path(str(job_id))
    size, digest = build_projection(source, csv, database, manifest)
    cache.publish_derived(str(job_id), database)
    asyncio.run(cache.publish_staged(str(job_id), csv, size_bytes=size, sha256=digest))
    app.state.store.complete_job(
        job_id,
        result_state=JobState.SUCCEEDED,
        result_filename="candidates.csv",
        result_media_type="text/csv",
        result_size_bytes=size,
        result_sha256=digest,
        result_archive_schema="protein_optimization/1",
        now=10,
    )
    path = ROOT + f"/jobs/{job_id}/candidates"
    page = _request(
        app,
        "GET",
        path,
        params={"sort_by": "predicted_label", "descending": "true", "limit": 2},
    )
    assert page.status_code == 200, page.text
    assert page.json()["rows"][0]["id"] == "candidate_000000251"
    assert page.headers["cache-control"] == "private, no-store"
    prepare_path = ROOT + f"/jobs/{job_id}/prepare-selected-download"
    prepared = _request(
        app,
        "POST",
        prepare_path,
        json={"ids": ["candidate_000000251", "candidate_000000001"]},
    )
    assert prepared.status_code == 200, prepared.text
    selected = _request(app, "GET", prepared.json()["download_url"])
    assert selected.status_code == 200, selected.text
    table = pl.read_csv(BytesIO(selected.content))
    assert table["id"].to_list() == ["candidate_000000001", "candidate_000000251"]
    assert table["warnings"][0] == "'\t =formula"
    missing = _request(app, "POST", prepare_path, json={"ids": ["candidate_000000999"]})
    assert (missing.status_code, missing.json()["code"]) == (422, "selection_invalid")
    complete = _request(app, "GET", f"/api/v1/jobs/{job_id}/download")
    assert complete.status_code == 200
    assert pl.read_csv(BytesIO(complete.content)).height == 251
    app.dependency_overrides[require_session] = _session
    assert _request(app, "GET", prepared.json()["download_url"]).status_code == 404
    app.dependency_overrides[require_session] = lambda: session
    replaced = _request(
        app, "POST", prepare_path, json={"ids": ["candidate_000000001"]}
    )
    assert _request(app, "GET", prepared.json()["download_url"]).status_code == 410
    assert _request(app, "GET", replaced.json()["download_url"]).status_code == 200
    app.state.store.delete_job(session.principal.user_id, job_id, now=20)
    replay = _request(app, "POST", ROOT + "/jobs", json=body, headers=headers)
    assert (replay.status_code, replay.json()["code"]) == (409, "job_deleted")
    assert _request(app, "GET", path).status_code == 404
    assert _request(app, "GET", replaced.json()["download_url"]).status_code == 404
    asyncio.run(cache.shutdown())
    asyncio.run(app.state.antibody_analysis.shutdown())


@pytest.mark.parametrize("cancel", [False, True])
def test_selected_ticket_failure_releases_result_lease(tmp_path, monkeypatch, cancel):
    """Pre-stream I/O failures and task cancellation must not prevent deletion."""
    app = _app(tmp_path)
    _humanization_session(app)
    submitted = _request(
        app,
        "POST",
        ROOT + "/jobs",
        json=_body(app),
        headers={"Origin": ORIGIN, "Idempotency-Key": str(uuid4())},
    )
    job_id = UUID(submitted.json()["job_id"])
    cache = app.state.cache
    _, source, manifest = _result(tmp_path)
    csv, database = cache.staging_path(str(job_id)), cache.staging_path(str(job_id))
    size, digest = build_projection(source, csv, database, manifest)
    cache.publish_derived(str(job_id), database)
    asyncio.run(cache.publish_staged(str(job_id), csv, size_bytes=size, sha256=digest))
    app.state.store.complete_job(
        job_id,
        result_state=JobState.SUCCEEDED,
        result_filename="candidates.csv",
        result_media_type="text/csv",
        result_size_bytes=size,
        result_sha256=digest,
        result_archive_schema="protein_optimization/1",
        now=10,
    )
    entered, release = Event(), Event()

    def failing_ticket(*args, **kwargs):
        entered.set()
        if cancel:
            assert release.wait(5)
            return ["candidate_000000001"]
        raise OSError("Ticket read failed")

    monkeypatch.setattr(
        "biomodals.service.protein_optimization.router.load_selection", failing_ticket
    )

    async def download():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=ORIGIN
        ) as client:
            return await client.get(
                ROOT + f"/jobs/{job_id}/candidates.csv", params={"ticket": str(uuid4())}
            )

    async def scenario():
        task = asyncio.create_task(download())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            if cancel:
                task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError if cancel else OSError):
                await task
            assert cache.remove_job_files(str(job_id))
        finally:
            release.set()
            await cache.shutdown()
            await app.state.antibody_analysis.shutdown()

    asyncio.run(scenario())
