"""Tests for standalone GROMACS app behavior used by workflows."""

# ruff: noqa: D101,D102,D103,D107

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.app.bioinfo.gromacs import analysis
from biomodals.app.bioinfo.gromacs import app as gromacs_app


def test_gromacs_preparation_rebuilds_an_incomplete_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_name = "demo"
    run_root = tmp_path / run_name
    run_root.mkdir()
    (run_root / f"production_{run_name}.tpr").write_bytes(b"tpr")
    (run_root / "production.mdp").write_bytes(b"mdp")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "prepare-tpr.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    calls = []

    def run_command(command, **kwargs):
        assert not (run_root / f"production_{run_name}.tpr").exists()
        calls.append((command, kwargs))
        for relative_path in gromacs_app.preparation_execution_paths(run_name):
            (run_root / relative_path).write_bytes(b"result")

    monkeypatch.setattr(
        gromacs_app.CONF,
        "output_volume_mountpoint",
        str(tmp_path),
    )
    monkeypatch.setattr(
        gromacs_app.CONF,
        "output_volume",
        SimpleNamespace(commit=lambda: None, reload=lambda: None),
    )
    monkeypatch.setattr(gromacs_app.APP_INFO, "gmx_scripts", str(scripts))
    monkeypatch.setattr(gromacs_app, "run_command", run_command)

    gromacs_app.prepare_tpr_gpu.get_raw_f()(
        pdb_content=b"ATOM\n",
        run_name=run_name,
    )

    assert len(calls) == 1


def test_fresh_production_run_uses_mdp_nsteps(tmp_path: Path, monkeypatch) -> None:
    work_path = tmp_path / "fresh"
    captured = {}

    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            # Model a warm mount that sees prepared inputs only after reload.
            work_path.mkdir(exist_ok=True)
            work_path.joinpath("production_fresh.tpr").write_bytes(b"tpr")

    volume = FakeVolume()
    monkeypatch.setattr(
        gromacs_app,
        "CONF",
        SimpleNamespace(output_volume_mountpoint=str(tmp_path), output_volume=volume),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/gmx")

    def fake_run_command(cmd, *, cwd, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return []

    monkeypatch.setattr(gromacs_app, "run_command", fake_run_command)

    result = gromacs_app.production_run_cpu.get_raw_f()(
        run_name="fresh",
        simulation_time_ns=2,
    )

    nsteps_index = captured["cmd"].index("-nsteps")
    assert captured["cmd"][nsteps_index + 1] == "-2"
    assert captured["cwd"] == str(work_path)
    assert result == str(work_path)
    assert volume.commit_count == 1


def test_analysis_volume_handoff_refreshes_inputs_and_stale_deletion(
    tmp_path, monkeypatch
):
    """Model distinct warm analysis/postprocessing mounts across two jobs."""
    events = []
    active_root = tmp_path

    class StopAfterHandoff(Exception):
        pass

    def analyze_trajectory(_trajectory, path, **kwargs):
        assert Path(path).read_bytes() == b"new centered structure"
        assert kwargs["run_name"] == active_root.name
        assert kwargs["prefix"] == "production_parent"
        events.append("read")
        raise StopAfterHandoff

    monkeypatch.setattr(analysis, "analyze_trajectory", analyze_trajectory)

    def analysis_reload():
        events.append("analysis reload")
        active_root.mkdir(exist_ok=True)
        raw = active_root / "production_parent.xtc"
        if not raw.exists():
            raw.write_bytes(b"new raw trajectory")
            os.utime(raw, (20, 20))
            stale = active_root / "production_parent_nopbc.xtc"
            stale.write_bytes(b"old processed trajectory")
            os.utime(stale, (10, 10))

    volume = SimpleNamespace(
        reload=analysis_reload, commit=lambda: events.append("analysis commit")
    )
    monkeypatch.setattr(gromacs_app.CONF, "output_volume_mountpoint", str(tmp_path))
    monkeypatch.setattr(gromacs_app.CONF, "output_volume", volume)
    monkeypatch.setattr(gromacs_app.APP_INFO, "gmx_scripts", str(tmp_path))
    (tmp_path / "postprocess-traj.sh").write_bytes(b"script")

    def native_command(command, **kwargs):
        assert events[-1] == "postprocess reload"
        output = Path(command[command.index("--output-file") + 1])
        assert not output.exists()
        assert Path(command[command.index("--xtc-file") + 1]).is_file()
        output.write_bytes(b"new processed trajectory")
        output.with_name(output.stem + "_centered.pdb").write_bytes(
            b"new centered structure"
        )
        events.append("postprocess")

    monkeypatch.setattr(gromacs_app, "run_command", native_command)
    postprocess = gromacs_app.postprocess_traj.get_raw_f()

    def remote(*args, **kwargs):
        assert events[-1] == "analysis commit"
        with monkeypatch.context() as worker:
            worker.setattr(
                gromacs_app.CONF,
                "output_volume",
                SimpleNamespace(
                    reload=lambda: events.append("postprocess reload"),
                    commit=lambda: events.append("postprocess commit"),
                ),
            )
            postprocess(*args, **kwargs)

    monkeypatch.setattr(gromacs_app, "postprocess_traj", SimpleNamespace(remote=remote))
    for name in ("child-one", "child-two"):
        active_root = tmp_path / name
        events.clear()
        with pytest.raises(StopAfterHandoff):
            gromacs_app.collect_traj_stats.get_raw_f()(
                "production_", name, file_stem="parent"
            )
        assert events == [
            "analysis reload",
            "analysis commit",
            "postprocess reload",
            "postprocess",
            "postprocess commit",
            "analysis reload",
            "read",
        ]
