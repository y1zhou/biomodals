"""Tests for the API-specific GROMACS result archive."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from biomodals.app.bioinfo import gromacs_app
from biomodals.schema import ArtifactKind

VALID_PDB = (
    b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
    b"END\n"
)


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


class FakePublications:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def put(self, key: str, value: object, *, skip_if_exists: bool) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key: str) -> object:
        return self.values[key]


def _install_fake_gromacs_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    omit: str | None = None,
) -> tuple[FakeVolume, dict[str, int]]:
    volume = FakeVolume()
    calls = {"prepare": 0, "production": 0, "analysis": 0, "warmup": 0}
    mountpoint = tmp_path / "Gromacs-outputs"
    run_root = mountpoint / "api-123"

    monkeypatch.setattr(
        gromacs_app,
        "CONF",
        SimpleNamespace(
            name="Gromacs",
            version="2026.1",
            repo_url="https://github.com/gromacs/gromacs",
            output_volume=volume,
            output_volume_name="Gromacs-outputs",
            output_volume_mountpoint=str(mountpoint),
        ),
    )

    class FakePrepare:
        def remote(self, **_kwargs):
            calls["prepare"] += 1
            run_root.mkdir(parents=True, exist_ok=True)
            files = {
                "production.mdp": b"integrator = md\nnsteps = 1500000\n",
                "production_api-123.xtc": b"compressed trajectory\x00\x01",
                "production_api-123.tpr": b"binary topology\x00\x02",
                "production_api-123_nopbc_centered.pdb": VALID_PDB,
                "rmsd_production_api-123.csv": b"time_ns,rmsd\n0,0\n",
                "rg_production_api-123.csv": b"time_ns,rg\n0,1\n",
                "rmsf_production_api-123.csv": b"residue_index,rmsf\n1,0\n",
                "production_api-123.log": (
                    f"Working directory: {run_root}\nStep 1000 complete\n".encode()
                ),
                "production_api-123.cpt": b"must not be downloaded",
                "rmsd_production_api-123.png": b"must not be downloaded",
            }
            for name, content in files.items():
                if name != omit:
                    run_root.joinpath(name).write_bytes(content)
            return str(run_root)

    class FakeProduction:
        def remote(self, **_kwargs):
            calls["production"] += 1
            return str(run_root)

    class FakeStats:
        def remote(self, *_args, **_kwargs):
            calls["analysis"] += 1
            return str(run_root)

    monkeypatch.setattr(gromacs_app, "prepare_tpr_cpu", FakePrepare())
    monkeypatch.setattr(gromacs_app, "production_run_cpu", FakeProduction())
    monkeypatch.setattr(gromacs_app, "collect_traj_stats", FakeStats())
    monkeypatch.setattr(gromacs_app, "api_result_publications", FakePublications())
    monkeypatch.setattr(
        gromacs_app,
        "warmup_directory",
        lambda _path, *, file_pattern: calls.__setitem__(
            "warmup", calls["warmup"] + int(bool(file_pattern))
        ),
    )
    return volume, calls


def _run_api_job():
    return gromacs_app.run_gromacs_job.get_raw_f()(
        pdb_content=VALID_PDB,
        run_name="api-123",
        simulation_time_ns=3,
        run_pdbfixer=True,
        cpu_only=True,
        num_threads=2,
        use_openmp_threads=True,
        ld_seed=11,
        gen_seed=12,
        genion_seed=13,
    )


def test_run_gromacs_job_publishes_one_verified_allowlisted_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume, calls = _install_fake_gromacs_run(monkeypatch, tmp_path)

    result = _run_api_job()

    assert result.status.value == "succeeded"
    assert result.logs == []
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.name == "gromacs_run"
    assert output.kind == ArtifactKind.ARCHIVE
    assert output.storage.kind.value == "volume_path"
    assert output.storage.volume_name == "Gromacs-outputs"
    assert output.storage.path == "api-results/api-123/result.zip"
    assert output.storage.media_type == "application/zip"
    assert output.metadata["filename"] == "api-123.zip"
    assert output.metadata["run_name"] == "api-123"

    archive_path = Path(gromacs_app.CONF.output_volume_mountpoint) / output.storage.path
    expected_members = [
        "input.pdb",
        "parameters.json",
        "provenance.json",
        "run.log",
        "outputs/production.mdp",
        "outputs/production_api-123.xtc",
        "outputs/production_api-123.tpr",
        "outputs/production_api-123_nopbc_centered.pdb",
        "outputs/rmsd_production_api-123.csv",
        "outputs/rg_production_api-123.csv",
        "outputs/rmsf_production_api-123.csv",
        "manifest.json",
        "checksums.sha256",
    ]
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == expected_members
        assert archive.read("input.pdb") == VALID_PDB
        assert archive.getinfo("outputs/production_api-123.xtc").compress_type == (
            zipfile.ZIP_STORED
        )
        assert archive.getinfo("outputs/production_api-123.tpr").compress_type == (
            zipfile.ZIP_STORED
        )
        assert archive.getinfo("parameters.json").compress_type == zipfile.ZIP_DEFLATED

        parameters = orjson.loads(archive.read("parameters.json"))
        assert parameters == {
            "cpu_only": True,
            "gen_seed": 12,
            "genion_seed": 13,
            "ld_seed": 11,
            "num_threads": 2,
            "run_name": "api-123",
            "run_pdbfixer": True,
            "simulation_time_ns": 3,
            "use_openmp_threads": True,
        }
        provenance = orjson.loads(archive.read("provenance.json"))
        assert provenance["app"] == "Gromacs"
        assert provenance["gromacs_version"] == "2026.1"
        assert provenance["archive_schema_version"] == 1
        assert provenance["started_at"].endswith("Z")
        assert provenance["completed_at"].endswith("Z")

        manifest = orjson.loads(archive.read("manifest.json"))
        assert manifest["archive_schema_version"] == 1
        assert manifest["run_name"] == "api-123"
        assert {entry["path"] for entry in manifest["files"]} == set(
            expected_members[:-2]
        )

        checksum_lines = archive.read("checksums.sha256").decode().splitlines()
        checksums = {line[66:]: line[:64] for line in checksum_lines}
        assert set(checksums) == set(expected_members[:-1])
        for name, digest in checksums.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest

        log = archive.read("run.log").decode()
        assert "status: succeeded" in log
        assert "Step 1000 complete" in log
        assert "<run-directory>" in log
        assert str(tmp_path) not in log
        assert "Modal" not in log

    assert output.metadata["size_bytes"] == archive_path.stat().st_size
    assert (
        output.metadata["sha256"]
        == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    assert not any(
        ".cpt" in member or member.endswith(".png") for member in expected_members
    )
    assert calls == {"prepare": 1, "production": 1, "analysis": 3, "warmup": 1}
    assert volume.commit_count == 3
    assert volume.reload_count >= 2
    assert archive_path.with_name("result.json").is_file()


def test_run_gromacs_job_reuses_immutable_completed_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _volume, calls = _install_fake_gromacs_run(monkeypatch, tmp_path)
    first = _run_api_job()
    archive_path = (
        Path(gromacs_app.CONF.output_volume_mountpoint) / first.outputs[0].storage.path
    )
    first_bytes = archive_path.read_bytes()

    second = _run_api_job()

    assert second.outputs[0].metadata == first.outputs[0].metadata
    assert archive_path.read_bytes() == first_bytes
    assert calls == {"prepare": 1, "production": 1, "analysis": 3, "warmup": 1}


def test_publication_registry_restores_the_first_elected_archive_after_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gromacs_run(monkeypatch, tmp_path)
    first = _run_api_job()
    archive_path = (
        Path(gromacs_app.CONF.output_volume_mountpoint) / first.outputs[0].storage.path
    )
    marker_path = archive_path.with_name("result.json")
    first_bytes = archive_path.read_bytes()
    archive_path.unlink()
    marker_path.unlink()

    retried = _run_api_job()

    assert archive_path.read_bytes() == first_bytes
    assert (
        retried.outputs[0].metadata["sha256"] == hashlib.sha256(first_bytes).hexdigest()
    )
    run_root = archive_path.parents[2] / "api-123"
    assert len(list(run_root.glob(".api-result-candidate-*.zip"))) == 2


def test_run_gromacs_job_does_not_publish_incomplete_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gromacs_run(
        monkeypatch,
        tmp_path,
        omit="production_api-123.xtc",
    )

    with pytest.raises(FileNotFoundError, match="trajectory"):
        _run_api_job()

    result_dir = (
        Path(gromacs_app.CONF.output_volume_mountpoint) / "api-results" / "api-123"
    )
    assert not result_dir.joinpath("result.zip").exists()
    assert not result_dir.joinpath("result.json").exists()


def _rewrite_archive_member(archive_path: Path, name: str, content: bytes) -> None:
    temporary = archive_path.with_name(f".{archive_path.name}.rewrite")
    with zipfile.ZipFile(archive_path) as source:
        members = [
            (info, content if info.filename == name else source.read(info.filename))
            for info in source.infolist()
        ]
    with zipfile.ZipFile(temporary, mode="w", allowZip64=True) as destination:
        for info, member_content in members:
            destination.writestr(info, member_content)
    temporary.replace(archive_path)


@pytest.mark.parametrize("field", ["size_bytes", "sha256"])
def test_run_gromacs_job_rejects_manifest_that_disagrees_with_member_bytes(
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gromacs_run(monkeypatch, tmp_path)
    first = _run_api_job()
    archive_path = (
        Path(gromacs_app.CONF.output_volume_mountpoint) / first.outputs[0].storage.path
    )
    with zipfile.ZipFile(archive_path) as archive:
        manifest = orjson.loads(archive.read("manifest.json"))
    manifest["files"][0][field] = (
        manifest["files"][0][field] + 1 if field == "size_bytes" else "0" * 64
    )
    _rewrite_archive_member(
        archive_path,
        "manifest.json",
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n",
    )

    with pytest.raises(RuntimeError, match="manifest does not match input.pdb"):
        _run_api_job()


def test_run_gromacs_job_rejects_inconsistent_checksum_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gromacs_run(monkeypatch, tmp_path)
    first = _run_api_job()
    archive_path = (
        Path(gromacs_app.CONF.output_volume_mountpoint) / first.outputs[0].storage.path
    )
    with zipfile.ZipFile(archive_path) as archive:
        checksums = archive.read("checksums.sha256")
    _rewrite_archive_member(
        archive_path,
        "checksums.sha256",
        b"0" * 64 + checksums[64:],
    )

    with pytest.raises(RuntimeError, match="checksums are inconsistent"):
        _run_api_job()
