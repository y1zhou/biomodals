"""AF3 previews preserve native identity, axes, nulls and private delivery."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import replace
from uuid import uuid4

import numpy as np
import orjson
import pytest
from alphafold3_preview_fixture import CHAINS, CIF, PAE, RESIDUES, preview_archive
from test_api_contract import _app, _enabled_session, _request, _session

from biomodals.service.alphafold3 import results
from biomodals.service.artifacts import ArtifactLease
from biomodals.service.http_contract import require_session
from biomodals.service.store import JobState


def _read(tmp_path, **kwargs):
    content = preview_archive(**kwargs)
    path = tmp_path / f"{uuid4()}.tar.zst"
    path.write_bytes(content)
    lease = ArtifactLease(os.open(path, os.O_RDONLY), path=path)
    try:
        return results.read_prediction(
            lease,
            digest=hashlib.sha256(content).hexdigest(),
            display_name=kwargs.get("display_name", "Preview fixture"),
        )
    finally:
        lease.close()


@pytest.mark.parametrize("manifest_last", [False, True])
@pytest.mark.parametrize("display_name", ["zzz test", "af3-dddddddddddddddd-renamed"])
def test_best_identity_native_bytes_and_historical_order(
    tmp_path, manifest_last, display_name
):
    """Rounded-score ties cannot change the exact manifest-selected prediction."""
    data = _read(tmp_path, manifest_last=manifest_last, display_name=display_name)
    assert data.cif == CIF
    assert data.summary.seed == 2
    assert data.summary.sample_index == 0
    assert data.summary.ranking_score == 0.91234
    assert data.summary.prediction_count == 2
    assert data.summary.ptm == 0.81
    assert data.summary.iptm is None
    assert data.summary.has_clash is False
    assert data.summary.token_chain_ids == CHAINS
    assert data.summary.token_res_ids == RESIDUES
    assert data.summary.pae_error is None


def test_pae_exact_and_asymmetric_window(tmp_path):
    """Y rows stay aligned, X columns stay scored, including ligand tokens."""
    data = _read(tmp_path)
    whole = results.pae_window(data)
    assert whole.values == PAE
    assert whole.valid_counts[0][1] == 0
    assert whole.valid_counts[1][0] == 1
    assert whole.aggregation == "exact"
    window = results.pae_window(data, x_start=4, x_end=7, y_start=1, y_end=3)
    assert window.x_edges == [4, 5, 6, 7]
    assert window.y_edges == [1, 2, 3]
    assert window.values == [row[4:7] for row in PAE[1:3]]


def test_block_means_exclude_nulls_preserve_edges_and_all_missing(tmp_path):
    """A nondivisible block grid reports means and counts, not sampled values."""
    data = _read(tmp_path)
    matrix = np.array(
        [
            [None, None, 3, 4, 5],
            [None, None, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        dtype=np.float32,
    )
    page = results.pae_window(replace(data, pae=matrix), max_size=3)
    assert page.x_edges == page.y_edges == [0, 2, 4, 5]
    assert page.values == [[None, 6.0, 7.5], [14.0, 16.0, 17.5], [21.5, 23.5, 25.0]]
    assert page.valid_counts == [[0, 4, 2], [4, 4, 2], [2, 2, 1]]
    assert page.aggregation == "mean"


@pytest.mark.parametrize(
    "bounds",
    [
        {"x_end": 8},
        {"x_start": 2, "x_end": 2},
        {"y_start": -1},
        {"max_size": 0},
        {"max_size": 513},
    ],
)
def test_pae_rejects_invalid_window(tmp_path, bounds):
    """The response cell bound cannot be bypassed by query shape."""
    with pytest.raises(ValueError, match="outside"):
        results.pae_window(_read(tmp_path), **bounds)


@pytest.mark.parametrize(
    "document",
    [
        b"{",
        b"[]",
        b'{"pae": [], "token_chain_ids": [], "token_res_ids": []}',
        orjson.dumps({
            "pae": [["1.0"]],
            "token_chain_ids": ["A"],
            "token_res_ids": [1],
        }),
        orjson.dumps({"pae": [[True]], "token_chain_ids": ["A"], "token_res_ids": [1]}),
        orjson.dumps({"pae": [[32.1]], "token_chain_ids": ["A"], "token_res_ids": [1]}),
    ],
)
def test_invalid_pae_does_not_remove_structure_or_summary(tmp_path, document):
    """Broken confidence data is a local preview failure, not Job failure."""
    data = _read(tmp_path, confidence=document)
    assert data.cif == CIF
    assert data.summary.ptm == 0.81
    assert data.summary.pae_error == "pae_invalid"
    assert data.pae is None


def test_oversized_confidence_is_not_parsed(tmp_path, monkeypatch):
    """The decompressed source cap is enforced before calling orjson."""
    monkeypatch.setitem(
        results._SUFFIXES, "request_best_confidences", ("confidences.json", 8)
    )

    def forbidden(_content):
        raise AssertionError("Oversized confidence was parsed")

    monkeypatch.setattr(results, "_read_pae", forbidden)
    data = _read(tmp_path)
    assert data.cif == CIF
    assert data.summary.pae_error == "pae_too_large"


def test_token_limit_and_invalid_summary_are_independent(tmp_path, monkeypatch):
    """Resource and summary failures preserve other available output."""
    monkeypatch.setattr(results, "MAX_PAE_TOKENS", 6)
    data = _read(tmp_path, summary=b"{}")
    assert data.summary.pae_error == "pae_too_large"
    assert data.summary.summary_error == "summary_invalid"
    assert data.cif == CIF


def test_manifest_best_must_equal_exact_ranking(tmp_path):
    """A plausible but different seed cannot replace the selected prediction."""
    with pytest.raises(ValueError, match="best prediction"):
        _read(tmp_path, corrupt_best=True)


def test_member_integrity_failures_stay_scoped_to_the_affected_preview(tmp_path):
    """Changed confidence bytes cannot silently produce numbers or hide the CIF."""
    data = _read(tmp_path, corrupt_role="request_best_confidences")
    assert data.summary.pae_error == "pae_invalid"
    assert data.cif == CIF
    with pytest.raises(ValueError, match="does not match its manifest"):
        _read(tmp_path, corrupt_role="request_best_model_cif")


def test_prediction_reader_retains_only_two_immutable_previews(tmp_path, monkeypatch):
    """Opening many completed Jobs cannot retain every full token matrix."""
    reader = results.PredictionReader()
    observed = []
    data = _read(tmp_path)

    def read(_lease, *, digest, display_name):
        observed.append(digest)
        return data

    monkeypatch.setattr(results, "read_prediction", read)
    path = tmp_path / "cached.result"
    path.write_bytes(b"cached archive")
    lease = ArtifactLease(os.open(path, os.O_RDONLY), path=path)
    try:
        for digest in ("a", "b", "a", "c", "b"):
            reader.read(
                lease, job_id="test", digest=digest, display_name="Preview fixture"
            )
    finally:
        lease.close()
    assert observed == ["a", "b", "c", "b"]


def test_structure_and_archive_scan_limits(tmp_path, monkeypatch):
    """Oversized CIFs and scans fail before an unbounded preview is returned."""
    monkeypatch.setitem(results._SUFFIXES, "request_best_model_cif", ("model.cif", 1))
    with pytest.raises(results.PreviewTooLargeError, match="member exceeds"):
        _read(tmp_path)
    monkeypatch.setattr(results, "MAX_ARCHIVE_SCAN_BYTES", 1)
    with pytest.raises(results.PreviewTooLargeError, match="scan limit"):
        _read(tmp_path)


def _published(app, session, content, *, state=JobState.SUCCEEDED):
    job = app.state.store.admit_job(
        owner_user_id=session.principal.user_id,
        tool="alphafold3",
        display_name="Preview fixture",
        idempotency_key=str(uuid4()),
        request_digest="a" * 64,
        modal_environment="main",
        modal_app_name="AlphaFold3",
        modal_app_version=1,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=8,
        max_active_gpu_provider_calls=1,
        now=3,
    ).job
    digest = hashlib.sha256(content).hexdigest()
    cache = app.state.cache
    staging = cache.staging_path(str(job.job_id))
    staging.write_bytes(content)
    asyncio.run(
        cache.publish_staged(
            str(job.job_id), staging, size_bytes=len(content), sha256=digest
        )
    )
    app.state.store.complete_job(
        job.job_id,
        result_state=state,
        result_filename="result.tar.zst",
        result_media_type="application/zstd",
        result_size_bytes=len(content),
        result_sha256=digest,
        result_archive_schema="alphafold3-request/1",
        now=4,
    )
    return job.job_id


def test_private_http_delivery_reuses_preparation_and_requires_live_lease(
    tmp_path, monkeypatch
):
    """No anonymous/other-owner access or serving memory after cache removal."""
    app = _app(tmp_path)
    session = _enabled_session(app)
    job_id = _published(app, session, preview_archive())
    base = f"/api/v1/alphafold3/jobs/{job_id}/prediction"
    for suffix in ("", "/model.cif", "/pae"):
        response = _request(app, "GET", base + suffix)
        assert response.status_code == 401
        assert response.headers["Cache-Control"] == "private, no-store"
    app.dependency_overrides[require_session] = lambda: session
    summary = _request(app, "GET", base)
    assert summary.status_code == 200, summary.text
    assert summary.json()["seed"] == 2
    assert summary.headers["Cache-Control"] == "private, no-store"

    def no_reparse(*args, **kwargs):
        raise AssertionError("Cached prediction was parsed again")

    monkeypatch.setattr(results, "read_prediction", no_reparse)
    cif = _request(app, "GET", base + "/model.cif")
    assert cif.content == CIF
    assert cif.headers["Cache-Control"] == "private, no-store"
    page = _request(app, "GET", base + "/pae", params={"max_size": 2})
    assert page.status_code == 200, page.text
    assert page.json()["prediction_id"] == summary.json()["prediction_id"]
    assert len(page.json()["values"]) <= 2
    assert (
        _request(app, "GET", base + "/pae", params={"x_end": 8}).json()["code"]
        == "pae_window_invalid"
    )
    invalid = _request(app, "GET", base + "/pae", params={"max_size": 513})
    assert invalid.status_code == 422
    assert invalid.headers["Cache-Control"] == "private, no-store"
    app.dependency_overrides[require_session] = _session
    for suffix in ("", "/model.cif", "/pae"):
        response = _request(app, "GET", base + suffix)
        assert response.status_code == 404
        assert response.headers["Cache-Control"] == "private, no-store"
    app.dependency_overrides[require_session] = lambda: session
    app.state.cache.discard(str(job_id))
    for suffix in ("", "/model.cif", "/pae"):
        response = _request(app, "GET", base + suffix)
        assert response.json()["code"] == "result_not_cached"
        assert response.headers["Cache-Control"] == "private, no-store"


def test_pae_failure_and_partial_result_http_contract(tmp_path):
    """A missing PAE is recoverable, while unpublished partial AF3 is ineligible."""
    app = _app(tmp_path)
    session = _enabled_session(app)
    app.dependency_overrides[require_session] = lambda: session
    job_id = _published(app, session, preview_archive(confidence=b"{}"))
    base = f"/api/v1/alphafold3/jobs/{job_id}/prediction"
    assert _request(app, "GET", base).json()["pae_error"] == "pae_invalid"
    assert _request(app, "GET", base + "/model.cif").content == CIF
    assert _request(app, "GET", base + "/pae").json()["code"] == "pae_invalid"
    partial = _published(app, session, preview_archive(), state=JobState.PARTIAL)
    response = _request(app, "GET", f"/api/v1/alphafold3/jobs/{partial}/prediction")
    assert response.json()["code"] == "result_not_ready"
