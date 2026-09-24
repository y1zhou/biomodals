"""Analysis admission shares durable Jobs without changing a completed simulation."""

# ruff: noqa: D103

from uuid import UUID, uuid4

import pytest
from test_api_contract import (
    ORIGIN,
    GromacsAdapter,
    _app,
    _humanization_session,
    _request,
    _session,
)
from test_gromacs_continuation_api import _completed

from biomodals.app.bioinfo.gromacs.clustering import (
    ClusteringExecutionRequest,
    ClusteringSource,
)
from biomodals.schema import ArtifactFile
from biomodals.service.http_contract import require_session, require_unsafe_session
from biomodals.service.pending import PendingRequestStore


@pytest.fixture
def setup(tmp_path, monkeypatch):
    app = _app(tmp_path)
    session = _humanization_session(app)
    parent = _completed(app, session)
    source = ClusteringSource(
        execution_run_id=parent.job_id,
        run_name="source",
        publication_sha256="a" * 64,
        trajectory=ArtifactFile(
            path="production_source_nopbc.xtc", size_bytes=100, content_sha256="b" * 64
        ),
        template=ArtifactFile(
            path="production_source_nopbc_centered.pdb",
            size_bytes=100,
            content_sha256="c" * 64,
        ),
        frame_count=1_000_000,
        protein_atoms=100,
        ca_atoms=25,
    )
    reads = []

    async def inspect(self, job, deployment):
        reads.append(job.job_id)
        return source

    async def preflight(self, deployment):
        return None

    monkeypatch.setattr(GromacsAdapter, "clustering_source", inspect)
    monkeypatch.setattr(GromacsAdapter, "clustering_preflight", preflight)
    return app, session, parent, source, reads


def submit(app, source_id, *, key=None, **body):
    return _request(
        app,
        "POST",
        f"/api/v1/gromacs/jobs/{source_id}/clustering",
        json=body,
        headers={"Origin": ORIGIN, "Idempotency-Key": str(key or uuid4())},
    )


def test_large_source_warns_and_admits_exact_analysis_with_immutable_lineage(
    setup, tmp_path, monkeypatch
):
    app, _, parent, source, reads = setup
    metadata = _request(app, "GET", f"/api/v1/gromacs/jobs/{parent.job_id}/clustering")
    assert metadata.status_code == 200
    assert metadata.json()["eligible"] is True
    assert metadata.json()["warnings"]
    assert metadata.json()["deadline_seconds"] == 43200
    key = uuid4()
    response = submit(app, parent.job_id, key=key, cutoff_angstrom=3.5)
    assert response.status_code == 202, response.text
    view = response.json()
    assert view["operation"] == "trajectory_clustering"
    assert view["source_job_id"] == str(parent.job_id)
    pending = PendingRequestStore(tmp_path / "pending")
    saved = ClusteringExecutionRequest.from_bytes(pending.get(UUID(view["job_id"])))
    assert saved.source == source
    assert saved.cutoff_angstrom == 3.5
    assert saved.execution_plan.node_keys == ("cluster_trajectory", "prepare_result")
    assert app.state.store.get_job_by_id(parent.job_id) == parent

    async def unavailable(*args):
        pytest.fail("Exact replay must not inspect the source or deployment")

    monkeypatch.setattr(GromacsAdapter, "clustering_source", unavailable)
    monkeypatch.setattr(GromacsAdapter, "clustering_preflight", unavailable)
    replay = submit(app, parent.job_id, key=key, cutoff_angstrom=3.5)
    assert replay.status_code == 202
    assert replay.json()["job_id"] == view["job_id"]
    assert submit(app, parent.job_id, key=key, cutoff_angstrom=4).status_code == 409
    assert reads == [parent.job_id, parent.job_id]
    # Analysis jobs cannot themselves become MD/analysis sources or expose MD figures.
    info = _request(app, "GET", f"/api/v1/gromacs/jobs/{view['job_id']}/clustering")
    assert info.json()["code"] == "source_not_completed"
    assert submit(app, view["job_id"]).status_code == 409
    assert (
        _request(
            app, "GET", f"/api/v1/gromacs/jobs/{view['job_id']}/continuation"
        ).json()["eligible"]
        is False
    )


def test_owner_scope_missing_source_and_mismatched_identity(setup, monkeypatch):
    app, _, parent, source, reads = setup
    path = f"/api/v1/gromacs/jobs/{parent.job_id}/clustering"

    async def incorrect(*args):
        return source.model_copy(update={"execution_run_id": uuid4()})

    monkeypatch.setattr(GromacsAdapter, "clustering_source", incorrect)
    assert _request(app, "GET", path).json()["code"] == "source_unavailable"
    assert submit(app, parent.job_id).status_code == 409
    app.dependency_overrides[require_session] = _session
    app.dependency_overrides[require_unsafe_session] = _session
    assert _request(app, "GET", path).status_code == 404
    assert submit(app, parent.job_id).status_code == 404
    assert reads == []


@pytest.mark.parametrize("cutoff", [0, -1, "nan", "inf"])
def test_invalid_cutoff_rejected_before_inspection(setup, cutoff):
    app, _, parent, _, reads = setup
    assert submit(app, parent.job_id, cutoff_angstrom=cutoff).status_code == 422
    assert reads == []
