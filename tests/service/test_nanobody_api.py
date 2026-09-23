"""Authenticated reviewed admission, frozen parents and bounded native results."""

# ruff: noqa: D103

import asyncio
from io import BytesIO
from uuid import UUID, uuid4

import polars as pl
from antibody_fixture import reference_csv
from modal.exception import NotFoundError
from nanobody_fixture import VHH, result_directory
from test_api_contract import (
    ORIGIN,
    NanobodyAdapter,
    _app,
    _humanization_session,
    _request,
    _session,
)

from biomodals.service.antibody_sequence_analysis.reference import TherapeuticReference
from biomodals.service.http_contract import require_session
from biomodals.service.nanobody_humanization import router
from biomodals.service.nanobody_humanization.results import (
    SELECTION_SCHEMA,
    build_nanobody_archive,
    query_selection,
)
from biomodals.service.pending import PendingRequestStore
from biomodals.service.store import JobState
from biomodals.workflow.nanobody_humanization.execution import NanobodyExecutionRequest

ROOT = "/api/v1/nanobody-humanization"


def test_parent_only_result_has_zero_new_designs_on_any_page():
    table = pl.DataFrame(
        [
            {"parent_id": key, "candidate_id": key, "is_parent": True}
            for key in ("a", "b")
        ],
        schema=SELECTION_SCHEMA,
    )
    for query in ({}, {"offset": 1, "limit": 1}, {"parent_id": "missing"}):
        page = query_selection(BytesIO(table.write_csv().encode()), **query)
        assert page.nonparent_count == 0


def test_review_digest_exact_replay_and_frozen_parent(tmp_path, monkeypatch):
    app = _app(tmp_path)
    session = _humanization_session(app)
    originals = [{"id": "one", "vhh": "HHHHHH" + VHH + "GGG"}]
    headers = {"Origin": ORIGIN, "Idempotency-Key": str(uuid4())}
    try:
        preview = _request(
            app, "POST", ROOT + "/prepare", json={"parents": originals}
        ).json()
        payload = {"parents": originals, "preparation_digest": "0" * 64}
        stale = _request(app, "POST", ROOT + "/jobs", json=payload, headers=headers)
        assert stale.status_code == 409
        assert stale.json()["code"] == "preparation_changed"
        assert app.state.remote_execution.preflights == 0
        payload["preparation_digest"] = preview["preparation_digest"]
        response = _request(app, "POST", ROOT + "/jobs", json=payload, headers=headers)
        assert response.status_code == 202, response.text
        job_id = UUID(response.json()["job_id"])
        job = app.state.store.get_job(session.principal.user_id, job_id)
        assert (job.max_active_provider_calls, job.max_active_gpu_provider_calls) == (
            16,
            4,
        )
        assert job.tool == "nanobody_humanization"
        monkeypatch.setattr(
            router,
            "prepare_batch",
            lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected preparation")),
        )
        replay = _request(app, "POST", ROOT + "/jobs", json=payload, headers=headers)
        assert replay.json()["job_id"] == str(job_id)
        assert app.state.remote_execution.preflights == 1
        conflict = _request(
            app,
            "POST",
            ROOT + "/jobs",
            json={**payload, "display_name": "Changed"},
            headers=headers,
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "idempotency_conflict"
        retained = _request(app, "GET", f"{ROOT}/jobs/{job_id}/inputs")
        assert retained.status_code == 200, retained.text
        assert retained.headers["cache-control"] == "private, no-store"
        assert retained.json()["parents"] == originals
        assert retained.json()["prepared_parents"] == [
            {"row_index": 0, "id": "one", "vh": VHH}
        ]
        assert retained.json()["settings"]["hudiff_nb_candidate_count"] == 10
        app.dependency_overrides[require_session] = _session
        assert _request(app, "GET", f"{ROOT}/jobs/{job_id}/inputs").status_code == 404
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_result_page_frozen_germlines_download_and_cache_restore_boundary(tmp_path):
    app = _app(tmp_path)
    session = _humanization_session(app)
    app.state.antibody_analysis.reference = TherapeuticReference(
        tmp_path / "reference.json", download=reference_csv
    )
    try:
        parents = [{"id": "001", "vhh": VHH}, {"id": "002", "vhh": VHH}]
        preview = _request(
            app, "POST", ROOT + "/prepare", json={"parents": parents}
        ).json()
        response = _request(
            app,
            "POST",
            ROOT + "/jobs",
            json={
                "parents": parents,
                "preparation_digest": preview["preparation_digest"],
            },
            headers={"Origin": ORIGIN, "Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 202, response.text
        job_id = UUID(response.json()["job_id"])
        pending = PendingRequestStore(tmp_path / "pending")
        request = NanobodyExecutionRequest.from_bytes(pending.get(job_id))
        root = result_directory(tmp_path / "volume", job_id, request)
        cache = app.state.cache
        staging = cache.staging_path(str(job_id))
        with staging.open("w+b") as destination:
            built = build_nanobody_archive(root, destination)
        asyncio.run(
            cache.publish_staged(
                str(job_id), staging, size_bytes=built.size_bytes, sha256=built.sha256
            )
        )
        app.state.store.complete_job(
            job_id,
            result_state=JobState.SUCCEEDED,
            result_filename="nanobody.zip",
            result_media_type="application/zip",
            result_size_bytes=built.size_bytes,
            result_sha256=built.sha256,
            result_archive_schema="nanobody_humanization/1",
            now=3,
        )
        path = f"{ROOT}/jobs/{job_id}/selection"
        response = _request(app, "GET", path, params={"limit": 1, "parent_id": "002"})
        assert response.status_code == 200, response.text
        page = response.json()
        assert page["parent_ids"] == ["001", "002"]
        assert page["total_rows"] == 3 and len(page["rows"]) == 1
        assert page["nonparent_count"] == 4
        assert page["rows"][0]["parent_id"] == "002"
        assert page["rows"][0]["is_parent"] is True
        assert page["nativeness_ranges"]["abnativ2_vh_nativeness"] == {
            "min": 0.8,
            "max": 0.85,
        }
        assert page["nativeness_ranges"]["abnativ2_vhh_nativeness"] == {
            "min": 0.8,
            "max": 0.9,
        }
        assert page["default_hidden_columns"] == [
            "abnativ2_vh_nativeness_error",
            "abnativ2_vhh_nativeness_error",
            "annotation_error",
            "evaluation_complete",
        ]
        assert list(page["germlines"]) == [page["rows"][0]["candidate_id"]]
        assert page["germlines"][page["rows"][0]["candidate_id"]]["assignment"]["v"]
        assert page["reference"] is not None
        empty = _request(app, "GET", path, params={"parent_id": "missing"}).json()
        assert empty["total_rows"] == 0 and empty["nonparent_count"] == 4
        for descending in (True, False):
            sorted_page = _request(
                app,
                "GET",
                path,
                params={
                    "parent_id": "002",
                    "sort_by": "panel_order",
                    "descending": descending,
                },
            ).json()
            assert sorted_page["rows"][-1]["is_parent"] is True
        assert (
            _request(app, "GET", path + ".csv").content
            == (root / "selection.csv").read_bytes()
        )
        assert (
            _request(app, "GET", path, params={"sort_by": "invalid"}).status_code == 422
        )
        assert _request(app, "GET", path, params={"limit": 201}).status_code == 422
        app.dependency_overrides[require_session] = _session
        assert _request(app, "GET", path).status_code == 404
        app.dependency_overrides[require_session] = lambda: session
        cache.discard(str(job_id))
        assert _request(app, "GET", path).json()["code"] == "result_not_cached"
        assert _request(app, "GET", path + ".csv").json()["code"] == "result_not_cached"
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_invalid_rows_and_oversized_body_never_admit_jobs(tmp_path):
    app = _app(tmp_path)
    _humanization_session(app)
    try:
        response = _request(
            app,
            "POST",
            ROOT + "/jobs",
            json={
                "parents": [{"id": "bad", "vhh": "ACDE"}],
                "preparation_digest": "0" * 64,
            },
            headers={"Origin": ORIGIN, "Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 422
        assert response.json()["errors"][0]["row_index"] == 0
        assert response.json()["errors"][0]["code"] == "domain_invalid"
        assert app.state.remote_execution.preflights == 0
        for path in ("/jobs", "/prepare"):
            assert (
                _request(
                    app,
                    "POST",
                    ROOT + path,
                    content=b"{}",
                    headers={"Content-Length": str(1024 * 1024 + 1)},
                ).status_code
                == 413
            )
        assert _request(app, "GET", "/api/v1/jobs").json()["jobs"] == []
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_exploration_options_and_over_budget_rejection_precede_preparation(
    tmp_path, monkeypatch
):
    app = _app(tmp_path)
    _humanization_session(app)
    monkeypatch.setattr(
        router,
        "prepare_batch",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("Preparation should not start")
        ),
    )
    try:
        options = _request(app, "GET", ROOT + "/options").json()
        assert options["defaults"]["abnativ2_explore"] is False
        assert options["defaults"]["abnativ2_candidate_budget"] == 1000
        assert options["max_exploration_candidates_per_parent"] == 5000
        assert options["max_exploration_candidates_per_job"] == 10000
        response = _request(
            app,
            "POST",
            ROOT + "/jobs",
            json={
                "parents": [{"id": f"p{i}", "vhh": VHH} for i in range(11)],
                "preparation_digest": "0" * 64,
                "settings": {
                    "abnativ2_explore": True,
                    "abnativ2_candidate_budget": 1000,
                },
            },
            headers={"Origin": ORIGIN, "Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "exploration_budget_exceeded"
        assert app.state.remote_execution.preflights == 0
        assert _request(app, "GET", "/api/v1/jobs").json()["jobs"] == []
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_unsupported_pinned_workflow_rejects_before_admission(tmp_path, monkeypatch):
    async def incompatible(self, deployment):
        raise NotFoundError("old generation entrypoint")

    monkeypatch.setattr(NanobodyAdapter, "preflight", incompatible)
    app = _app(tmp_path)
    _humanization_session(app)
    try:
        parents = [{"id": "one", "vhh": VHH}]
        preview = _request(
            app, "POST", ROOT + "/prepare", json={"parents": parents}
        ).json()
        response = _request(
            app,
            "POST",
            ROOT + "/jobs",
            json={
                "parents": parents,
                "preparation_digest": preview["preparation_digest"],
            },
            headers={"Origin": ORIGIN, "Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 409
        assert response.json()["code"] == "deployment_incompatible"
        assert _request(app, "GET", "/api/v1/jobs").json()["jobs"] == []
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())
