"""Tests for standalone GROMACS app behavior used by workflows."""

# ruff: noqa: D101,D102,D103,D107

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from biomodals.app.bioinfo.gromacs import analysis
from biomodals.app.bioinfo.gromacs import app as gromacs_app
from biomodals.execution import RunStatus


def test_gromacs_declares_workflow_expected_files() -> None:
    prepared = gromacs_app.prepared_workflow_files("demo")
    production = gromacs_app.production_workflow_files("demo-rep1")

    assert [(item.path, item.role) for item in prepared] == [
        ("demo.pdb", "input_structure"),
        ("production_demo.tpr", "production_topology"),
        ("production.mdp", "production_parameters"),
    ]
    assert [(item.path, item.role) for item in production] == [
        ("production_demo-rep1.xtc", "trajectory"),
        ("production_demo-rep1.edr", "production_energy"),
        ("production_demo-rep1.tpr", "production_topology"),
        ("production_demo-rep1_nopbc_centered.pdb", "centered_structure"),
        ("rmsd_production_demo-rep1.csv", "rmsd"),
        ("rg_production_demo-rep1.csv", "radius_of_gyration"),
        ("rmsf_production_demo-rep1.csv", "rmsf"),
    ]


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


def test_submit_gromacs_task_launches_one_remote_execution_coordinator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("ATOM\n", encoding="utf-8")
    execution_run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    staged = {}
    launched = {}

    class FakeMethod:
        def spawn(self, **kwargs):
            launched["run_kwargs"] = kwargs
            return SimpleNamespace(
                object_id="fc-1",
                get=lambda: SimpleNamespace(
                    run=SimpleNamespace(
                        status=RunStatus.SUCCEEDED,
                        status_message=None,
                        status_reason=None,
                    )
                ),
            )

    def stage(volume, run_id, request):
        staged.update(volume=volume, run_id=run_id, request=request)

    monkeypatch.setattr(gromacs_app, "uuid4", lambda: execution_run_id)
    monkeypatch.setattr(gromacs_app, "stage_execution_request", stage)
    monkeypatch.setattr(
        gromacs_app,
        "submit_staged_execution_run",
        lambda volume, **kwargs: (
            launched.update(submit=(volume, kwargs)) or FakeMethod().spawn().get()
        ),
    )

    submit_task_info = gromacs_app.submit_gromacs_task.info
    assert submit_task_info is not None
    submit_task_raw_f = submit_task_info.raw_f
    assert submit_task_raw_f is not None
    submit_task_raw_f(
        input_pdb=str(pdb_path),
        run_name="single",
        simulation_time_ns=3,
        cpu_only=True,
        num_threads=2,
    )

    request = staged["request"]
    assert staged["run_id"] == execution_run_id
    assert request.run_name == "single"
    assert request.pdb_content == b"ATOM\n"
    assert request.simulation_time_ns == 3
    assert request.ld_seed != -1
    assert request.gen_seed != -1
    assert request.num_threads == 2
    assert request.max_active_provider_calls == 3
    assert request.max_active_gpu_provider_calls == 0
    _, submit_kwargs = launched["submit"]
    assert submit_kwargs["execution_run_id"] == execution_run_id
    assert submit_kwargs["predecessor_execution_run_id"] is None
    assert submit_kwargs["use_deployed_coordinator"] is False


def test_prepare_tpr_cpu_stages_input_with_app_run_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("prepare-tpr.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    captured = {}

    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            pass

    volume = FakeVolume()
    monkeypatch.setattr(
        gromacs_app,
        "APP_INFO",
        SimpleNamespace(gmx_scripts=str(scripts_dir)),
    )
    monkeypatch.setattr(
        gromacs_app,
        "CONF",
        SimpleNamespace(output_volume_mountpoint=str(tmp_path), output_volume=volume),
    )

    def fake_run_command(cmd, *, cwd, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return []

    monkeypatch.setattr(gromacs_app, "run_command", fake_run_command)

    result = gromacs_app.prepare_tpr_cpu.get_raw_f()(
        pdb_content=b"ATOM\n",
        run_name="prep",
        simulation_time_ns=1,
        num_threads=2,
    )

    run_root = tmp_path / "prep"
    input_path = run_root / "prep.pdb"
    staged_input_path = run_root / "inputs" / "prep.pdb"
    assert result == str(run_root)
    assert input_path.read_bytes() == b"ATOM\n"
    assert staged_input_path.read_bytes() == b"ATOM\n"
    assert captured["cmd"][captured["cmd"].index("-i") + 1] == str(input_path)
    assert captured["cwd"] == str(run_root)
    assert captured["env"] == {"OMP_NUM_THREADS": None}
    assert volume.commit_count == 1


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
