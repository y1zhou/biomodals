"""Offline exact native medoids and sparse per-frame export contracts."""

# ruff: noqa: D103

import asyncio
import os
import shutil
import zipfile
from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import orjson
import polars as pl
import pytest

from biomodals.app.bioinfo.gromacs.clustering import (
    ClusteringExecutionRequest,
    cluster_trajectory,
    inspect_clustering_source,
    read_memberships,
)
from biomodals.app.bioinfo.gromacs.execution import PREPARE_RESULT
from biomodals.app.bioinfo.gromacs.execution_runtime import (
    GromacsExecutionRequest,
    gromacs_node_paths,
    gromacs_publication_path,
    persist_execution_request,
)
from biomodals.execution import ContentBoundFileSet
from biomodals.helper.artifacts import file_size_sha256
from biomodals.schema import ArtifactFile
from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.gromacs import modal as gromacs_modal
from biomodals.service.pending import PendingRequestStore


def publish(root, request):
    records = []
    for name in gromacs_node_paths(request, PREPARE_RESULT):
        size, digest = file_size_sha256(request.run_root(root) / name)
        records.append(ArtifactFile(path=name, size_bytes=size, content_sha256=digest))
    ContentBoundFileSet(
        root=request.run_root(root),
        marker_path=root / gromacs_publication_path(request, PREPARE_RESULT),
        expected_paths=tuple(record.path for record in records),
        identity={
            "node_key": PREPARE_RESULT,
            "workload_plan_fingerprint": request.execution_plan.workload_plan_fingerprint,
        },
    ).write(tuple(records))


@pytest.fixture
def source(tmp_path):
    import biotite.structure as struc
    import biotite.structure.io as strucio

    request = GromacsExecutionRequest(
        run_name="source",
        pdb_content=b"ATOM\n",
        simulation_time_ns=5,
        run_pdbfixer=False,
        cpu_only=True,
        num_threads=1,
        use_openmp_threads=False,
        ld_seed=1,
        gen_seed=2,
        genion_seed=3,
        max_active_provider_calls=1,
        max_active_gpu_provider_calls=0,
    )
    source_id = uuid4()
    persist_execution_request(tmp_path, source_id, request)
    root = request.run_root(tmp_path)
    root.mkdir(exist_ok=True)
    for name in gromacs_node_paths(request, PREPARE_RESULT):
        (root / name).write_bytes(b"published")
    atoms = struc.AtomArray(3)
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    atoms.res_name[:] = "ALA"
    atoms.chain_id[:] = "A"
    atoms.res_id = np.arange(1, 4)
    atoms.coord = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0]], dtype=np.float32)
    strucio.save_structure(root / "production_source_nopbc_centered.pdb", atoms)
    (root / "rmsd_production_source.csv").write_bytes(b"time_ns,rmsd\n0,0\n1,0.2\n")
    publish(tmp_path, request)
    return inspect_clustering_source(tmp_path, source_id)


def test_inspection_checks_metadata_without_reading_trajectory(
    tmp_path, source, monkeypatch
):
    from biomodals.app.bioinfo.gromacs import clustering

    read = []

    def track(path):
        read.append(path.name)
        return file_size_sha256(path)

    monkeypatch.setattr(clustering, "file_size_sha256", track)
    inspected = inspect_clustering_source(tmp_path, source.execution_run_id)
    assert inspected.frame_count == 2
    assert inspected.ca_atoms == inspected.protein_atoms == 3
    assert read == [source.template.path, "rmsd_production_source.csv"]
    template = tmp_path / source.run_name / source.template.path
    template.write_bytes(template.read_bytes().replace(b"ALA", b"VAL"))
    with pytest.raises(ValueError, match="metadata changed"):
        inspect_clustering_source(tmp_path, source.execution_run_id)


def test_worker_reloads_checks_source_then_publishes(tmp_path, source, monkeypatch):
    from biomodals.app.bioinfo.gromacs import app, clustering

    events = []
    request = ClusteringExecutionRequest(run_name="analysis", source=source)
    monkeypatch.setattr(
        app,
        "CONF",
        SimpleNamespace(
            output_volume=SimpleNamespace(
                reload=lambda: events.append("reload"),
                commit=lambda: events.append("commit"),
            ),
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    original = clustering.inspect_clustering_source

    def inspect(*args):
        assert events == ["reload"]
        events.append("inspect")
        return original(*args)

    def compute(trajectory, template, destination, **kwargs):
        assert events == ["reload", "inspect"]
        assert trajectory == tmp_path / source.run_name / source.trajectory.path
        assert kwargs["expected_frames"] == source.frame_count
        events.append("compute")

    monkeypatch.setattr(clustering, "inspect_clustering_source", inspect)
    monkeypatch.setattr(clustering, "cluster_trajectory", compute)
    assert app.cluster_trajectory.get_raw_f()(request.to_bytes()) == str(
        tmp_path / "analysis"
    )
    assert events == ["reload", "inspect", "compute", "commit"]
    events.clear()
    path = tmp_path / source.run_name / source.trajectory.path
    path.write_bytes(
        b"corrupted"
    )  # Same size: inspection is cheap; worker verifies the digest.
    with pytest.raises(ValueError, match="trajectory changed"):
        app.cluster_trajectory.get_raw_f()(request.to_bytes())
    assert events == ["reload", "inspect"]


def test_cli_analysis_dispatch_preserves_coordinator_injection(monkeypatch):
    from biomodals.app.bioinfo.gromacs import app
    from biomodals.helper.cli_entrypoint import invoke_local_entrypoint

    calls = []
    monkeypatch.setattr(
        app, "_submit_clustering", lambda **kwargs: calls.append(kwargs)
    )
    overrides = dict(
        use_deployed_coordinator=True,
        deployment_environment="test",
        deployment_name="Gromacs",
        deployment_version=9,
        restart_from=None,
        max_containers=2,
        max_gpu_containers=1,
    )
    source_id = str(uuid4())
    invoke_local_entrypoint(
        module_name=app.__name__,
        entrypoint_name="submit_gromacs_task",
        flags=["--cluster-from", source_id, "--clustering-cutoff-angstrom", "3"],
        overrides=overrides,
        program_name="test",
    )
    assert calls[0] == {
        **{k: v for k, v in overrides.items() if k != "restart_from"},
        "source_run_id": source_id,
        "cutoff_angstrom": 3.0,
        "run_name": None,
    }
    with pytest.raises(ValueError, match="cannot be combined"):
        invoke_local_entrypoint(
            module_name=app.__name__,
            entrypoint_name="submit_gromacs_task",
            flags=["--cluster-from", source_id, "--cpu-only"],
            overrides=overrides,
            program_name="test",
        )
    assert len(calls) == 1


@pytest.mark.parametrize("tampered", [False, True])
def test_adapter_downloads_only_content_bound_archive(
    tmp_path, source, monkeypatch, tampered
):
    request = ClusteringExecutionRequest(run_name="analysis", source=source)
    root = request.run_root(tmp_path)
    root.mkdir()
    archive = root / "clusters.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "clusters.csv", "frame,time_ns,cluster_id,is_medoid\n0,0,1,yes\n"
        )
    original = archive.read_bytes()
    publish(tmp_path, request)
    if tampered:
        archive.write_bytes(original + b"tampered")
    reads = []

    async def read(path):
        yield (tmp_path / path).read_bytes()

    async def download(path, handle, **kwargs):
        reads.append(path)
        handle.write((tmp_path / path).read_bytes())

    volume = SimpleNamespace(
        read_file=SimpleNamespace(aio=read),
        _read_file_into_fileobj=SimpleNamespace(aio=download),
    )
    adapter = gromacs_modal.GromacsToolAdapter(
        PendingRequestStore(tmp_path / "pending")
    )
    monkeypatch.setattr(adapter, "_volume", lambda job: volume)
    monkeypatch.setattr(
        gromacs_modal, "load_execution_request_from_volume", lambda *args: request
    )
    job = SimpleNamespace(job_id=uuid4())

    async def prepare():
        cache = ArtifactCache(tmp_path / "cache")
        try:
            return await adapter.prepare_result(job, cache, completed_at=1)
        finally:
            await cache.shutdown()

    if tampered:
        with pytest.raises(ArtifactIntegrityError, match="changed after publication"):
            asyncio.run(prepare())
    else:
        result = asyncio.run(prepare())
        assert result.archive_schema == "gromacs-clustering/1"
        assert result.sha256 == sha256(original).hexdigest()
        assert result.size_bytes == len(original)
    assert reads == ["analysis/clusters.zip"]


def test_memberships_are_total_unique_and_one_based(tmp_path):
    path = tmp_path / "members.ndx"
    path.write_text("[Cluster_0001]\n1 3\n[Cluster_0002]\n2\n")
    assert read_memberships(path, 3).tolist() == [1, 2, 1]
    for content in (
        "[Cluster_0001]\n1 1 2 3\n",
        "[Cluster_0001]\n1 2\n",
        "[Cluster_0002]\n1 2 3\n",
    ):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_memberships(path, 3)


def test_native_cluster_archive_matches_pairwise_medoid_oracle(tmp_path):
    import biotite.structure as struc
    import biotite.structure.io as strucio
    from biotite.structure.io.xtc import XTCFile

    gmx = os.environ.get("BIOMODALS_GMX") or shutil.which("gmx")
    if not gmx:
        pytest.skip("Set BIOMODALS_GMX to GROMACS 2026.1")
    rng = np.random.default_rng(13)
    template = struc.AtomArray(32)
    template.atom_name = np.tile(["N", "CA", "C", "O"], 8)
    template.element = np.tile(["N", "C", "C", "O"], 8)
    template.res_id = np.repeat(np.arange(1, 9), 4)
    template.res_name[:] = "ALA"
    template.chain_id[:] = "A"
    reference = rng.normal(0, 10, (32, 3))
    deformation = rng.normal(0, 4, (32, 3))
    coordinates = np.array(
        [
            reference + (index % 3) * deformation + rng.normal(0, 0.05, (32, 3))
            for index in range(30)
        ],
        dtype=np.float32,
    )
    template.coord = coordinates[0]
    structure = tmp_path / "protein.pdb"
    strucio.save_structure(structure, template)
    xtc = XTCFile()
    xtc.set_coord(coordinates)
    xtc.set_time(1_000_000 + np.arange(30, dtype=np.float32) * 0.125)
    xtc.set_box(np.repeat((np.eye(3) * 100)[None], 30, axis=0))
    trajectory = tmp_path / "source.xtc"
    xtc.write(trajectory)
    before = sha256(trajectory.read_bytes()).hexdigest()
    output = tmp_path / "clusters.zip"
    cluster_trajectory(
        trajectory,
        structure,
        output,
        run_name="child",
        cutoff_angstrom=2,
        provenance={"job_id": "source"},
        expected_frames=30,
        gmx=gmx,
        timeout=30,
    )
    actual = XTCFile.read(trajectory).get_coord()
    with zipfile.ZipFile(output) as archive:
        table = pl.read_csv(archive.read("clusters.csv"))
        assert table.columns == ["frame", "time_ns", "cluster_id", "is_medoid"]
        assert table["frame"].to_list() == list(range(30))
        assert table["is_medoid"].drop_nulls().to_list() == ["yes"] * 3
        assert table["is_medoid"].null_count() == 27
        assert table["time_ns"].to_list() == pytest.approx(
            (1_000_000 + np.arange(30) * 0.125) / 1000
        )
        ca = template.atom_name == "CA"
        for members in table.partition_by("cluster_id"):
            indices = members["frame"].to_numpy()
            means = []
            for index in indices:
                aligned, _ = struc.superimpose(
                    actual[index, ca], actual[indices][:, ca]
                )
                means.append(struc.rmsd(actual[index, ca], aligned).mean())
            medoid = members.filter(pl.col("is_medoid") == "yes")["frame"].item()
            assert means[list(indices).index(medoid)] == pytest.approx(
                min(means), abs=0.001
            )
            import io

            pdb = strucio.pdb.PDBFile.read(
                io.StringIO(archive.read(f"medoids/child-frame_{medoid}.pdb").decode())
            ).get_structure(model=1)
            assert len(pdb) == len(template)
            assert pdb.coord == pytest.approx(actual[medoid], abs=0.0011)
        provenance = orjson.loads(archive.read("provenance.json"))
        assert provenance["cluster_count"] == 3
        assert set(archive.namelist()) == {
            "provenance.json",
            *(record["path"] for record in provenance["files"]),
        }
        for record in provenance["files"]:
            data = archive.read(record["path"])
            assert len(data) == record["size_bytes"]
            assert sha256(data).hexdigest() == record["sha256"]
    assert sha256(trajectory.read_bytes()).hexdigest() == before
