"""Production trajectory previews preserve native plots and private delivery."""

from __future__ import annotations

import asyncio
import hashlib
import io
import zipfile
from uuid import uuid4

import orjson
import pytest
from gromacs_preview_fixture import PLOTS, trajectory_archive
from test_api_contract import _app, _enabled_session, _request, _session
from test_gromacs_archive import PNG, _build_archive

from biomodals.service.gromacs import archive as gromacs_archive
from biomodals.service.gromacs.archive import read_trajectory_plot
from biomodals.service.http_contract import require_session
from biomodals.service.store import JobState


def test_reads_exact_native_production_pngs_without_trajectory_io(monkeypatch):
    """Each metric selects its published role, never an equilibration figure."""
    native, _metadata = _build_archive()
    synthetic = trajectory_archive()
    opened = []
    original = zipfile.ZipFile.open

    def record_open(self, name, *args, **kwargs):
        path = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert path == "metadata/manifest.json" or path.endswith(".png")
        opened.append(path)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", record_open)
    for metric in ("rmsd", "rg", "rmsf"):
        assert read_trajectory_plot(io.BytesIO(native), metric) == PNG
        assert read_trajectory_plot(io.BytesIO(synthetic), metric) == PLOTS[metric]
    assert all("_production_" in path for path in opened if path.endswith(".png"))


def _changed_archive(*, manifest_change=None, plot: bytes | None = None):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(trajectory_archive())) as original:
        with zipfile.ZipFile(output, "w") as changed:
            for info in original.infolist():
                content = original.read(info)
                if info.filename == "metadata/manifest.json" and manifest_change:
                    manifest = orjson.loads(content)
                    manifest_change(manifest)
                    content = orjson.dumps(manifest)
                if (
                    info.filename == "outputs/rmsd_production_fixture.png"
                    and plot is not None
                ):
                    content = plot
                changed.writestr(info, content)
    return output.getvalue()


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest.update(files=manifest["files"][1:]),
        lambda manifest: manifest["files"].append(manifest["files"][0]),
        lambda manifest: manifest["files"][0].update(path="input.pdb"),
    ],
    ids=["missing-plot", "ambiguous-role", "non-png-member"],
)
def test_rejects_unavailable_or_ambiguous_plot(change):
    """Do not substitute an unrelated file for an unavailable production plot."""
    with pytest.raises(ValueError):
        read_trajectory_plot(
            io.BytesIO(_changed_archive(manifest_change=change)), "rmsd"
        )


def _published(app, session, content, *, tool="gromacs", state=JobState.SUCCEEDED):
    job = app.state.store.admit_job(
        owner_user_id=session.principal.user_id,
        tool=tool,
        display_name="Trajectory fixture",
        idempotency_key=str(uuid4()),
        request_digest="a" * 64,
        modal_environment="main",
        modal_app_name="Gromacs",
        modal_app_version=1,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=8,
        max_active_gpu_provider_calls=1,
        now=3,
    ).job
    cache = app.state.cache
    staging = cache.staging_path(str(job.job_id))
    staging.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    asyncio.run(
        cache.publish_staged(
            str(job.job_id), staging, size_bytes=len(content), sha256=digest
        )
    )
    app.state.store.complete_job(
        job.job_id,
        result_state=state,
        result_filename="result.zip",
        result_media_type="application/zip",
        result_size_bytes=len(content),
        result_sha256=digest,
        result_archive_schema="gromacs/5",
        now=4,
    )
    return job.job_id


def test_private_png_delivery_and_cache_miss_do_not_run_science(tmp_path):
    """GETs return individual PNGs, enforce ownership, and release cache leases."""
    app = _app(tmp_path)
    session = _enabled_session(app)
    job_id = _published(app, session, trajectory_archive())
    base = f"/api/v1/gromacs/jobs/{job_id}/trajectory"
    anonymous = _request(app, "GET", base + "/rmsd.png")
    assert anonymous.status_code == 401
    assert anonymous.headers["Cache-Control"] == "private, no-store"
    app.dependency_overrides[require_session] = lambda: session
    for metric, content in PLOTS.items():
        response = _request(app, "GET", f"{base}/{metric}.png")
        assert response.status_code == 200, response.text
        assert response.content == content
        assert response.headers["Content-Type"] == "image/png"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Cache-Control"] == "private, no-store"
    assert _request(app, "GET", base + "/nvt.png").status_code == 422
    app.dependency_overrides[require_session] = _session
    foreign = _request(app, "GET", base + "/rmsd.png")
    assert foreign.status_code == 404
    assert foreign.headers["Cache-Control"] == "private, no-store"
    app.dependency_overrides[require_session] = lambda: session
    wrong_tool = _published(app, session, trajectory_archive(), tool="alphafold3")
    assert (
        _request(
            app, "GET", f"/api/v1/gromacs/jobs/{wrong_tool}/trajectory/rmsd.png"
        ).status_code
        == 404
    )
    app.state.cache.discard(str(job_id))
    missing = _request(app, "GET", base + "/rmsd.png")
    assert missing.status_code == 409
    assert missing.json()["code"] == "result_not_cached"
    assert app.state.remote_execution.preflights == 0


def test_invalid_and_oversized_plot_leave_other_results_usable(tmp_path, monkeypatch):
    """Preview failures are local to the plot and do not change the Job."""
    app = _app(tmp_path)
    session = _enabled_session(app)
    app.dependency_overrides[require_session] = lambda: session
    job_id = _published(app, session, _changed_archive(plot=b"not a PNG"))
    base = f"/api/v1/gromacs/jobs/{job_id}/trajectory"
    assert _request(app, "GET", base + "/rmsd.png").json()["code"] == "result_invalid"
    assert _request(app, "GET", base + "/rg.png").content == PLOTS["rg"]
    for limit, value in (
        ("MAX_TRAJECTORY_PNG_BYTES", 1),
        ("MAX_TRAJECTORY_PNG_PIXELS", 0),
    ):
        with monkeypatch.context() as bounded:
            bounded.setattr(gromacs_archive, limit, value)
            response = _request(app, "GET", base + "/rg.png")
            assert response.status_code == 413
            assert response.json()["code"] == "trajectory_plot_too_large"
    assert app.state.store.get_job_by_id(job_id).state == JobState.SUCCEEDED
    app.state.cache.discard(str(job_id))
    assert _request(app, "GET", base + "/rg.png").json()["code"] == "result_not_cached"
    partial = _published(app, session, trajectory_archive(), state=JobState.PARTIAL)
    response = _request(
        app, "GET", f"/api/v1/gromacs/jobs/{partial}/trajectory/rmsd.png"
    )
    assert response.json()["code"] == "result_not_ready"
