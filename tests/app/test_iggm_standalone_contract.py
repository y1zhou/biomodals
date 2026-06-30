"""Tests for IgGM app run layout behavior."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.design import iggm_app


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


def test_merge_pdb_chains_uses_app_run_layout(tmp_path: Path, monkeypatch) -> None:
    calls = {}
    output_volume = FakeOutputVolume()
    monkeypatch.setattr(
        iggm_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name="IgGM-outputs",
            git_clone_dir=tmp_path / "IgGM",
        ),
    )

    def fake_run_command(cmd, *, output_mode, log_file, cwd):
        calls["cmd"] = cmd
        calls["output_mode"] = output_mode
        calls["log_file"] = log_file
        calls["cwd"] = cwd
        return []

    monkeypatch.setattr(iggm_app, "run_command", fake_run_command)

    assert (
        iggm_app.merge_pdb_chains.get_raw_f()(
            antigen_pdb_bytes=b"ATOM\n",
            run_name="merge",
        )
        == "merge"
    )

    run_root = tmp_path / "merge"
    assert (run_root / "inputs" / "antigen.pdb").read_bytes() == b"ATOM\n"
    assert (run_root / "outputs").is_dir()
    assert calls["log_file"] == run_root / "logs" / "iggm_merge_chains.log"
    assert calls["cwd"] == tmp_path / "IgGM"
    assert calls["output_mode"] == "tee"
    assert output_volume.commit_count == 2


def test_iggm_inference_uses_app_run_layout(tmp_path: Path, monkeypatch) -> None:
    calls = {}
    output_volume = FakeOutputVolume()
    monkeypatch.setattr(
        iggm_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name="IgGM-outputs",
            git_clone_dir=tmp_path / "IgGM",
        ),
    )

    def fake_run_command(cmd, *, output_mode, log_file, cwd):
        calls["cmd"] = cmd
        calls["output_mode"] = output_mode
        calls["log_file"] = log_file
        calls["cwd"] = cwd
        return []

    monkeypatch.setattr(iggm_app, "run_command", fake_run_command)

    assert (
        iggm_app.iggm_inference.get_raw_f()(
            input_fasta_bytes=b">H\nAAA\n",
            task="design",
            run_name="design-run",
            antigen_pdb_bytes=b"ATOM\n",
            fasta_origin_bytes=b">origin\nAAA\n",
            num_samples=3,
        )
        == "design-run"
    )

    run_root = tmp_path / "design-run"
    assert (run_root / "inputs" / "input.fasta").read_bytes() == b">H\nAAA\n"
    assert (run_root / "inputs" / "antigen.pdb").read_bytes() == b"ATOM\n"
    assert (run_root / "inputs" / "original.fasta").read_bytes() == b">origin\nAAA\n"
    assert (run_root / "outputs").is_dir()
    assert calls["log_file"] == run_root / "logs" / "iggm.log"
    assert calls["cmd"][-2:] == ["--output", str(run_root / "outputs")]
    assert output_volume.commit_count == 2
