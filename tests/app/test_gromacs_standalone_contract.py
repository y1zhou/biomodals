"""Tests for standalone GROMACS app behavior used by workflows."""

# ruff: noqa: D101,D102,D103,D107

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.app.bioinfo import gromacs_app
from biomodals.schema import VolumePath

VALID_PDB = (
    b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
    b"END\n"
)


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
        ("production_demo-rep1.tpr", "production_topology"),
        ("production_demo-rep1_nopbc_centered.pdb", "centered_structure"),
        ("rmsd_production_demo-rep1.csv", "rmsd"),
        ("rg_production_demo-rep1.csv", "radius_of_gyration"),
        ("rmsf_production_demo-rep1.csv", "rmsf"),
    ]


def test_submit_gromacs_task_keeps_single_run_standalone_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_bytes(VALID_PDB)
    prepare_kwargs = {}
    production_kwargs = {}
    analysis_stats = []

    class FakePrepare:
        def remote(self, **kwargs):
            prepare_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/single"

    class FakeProduction:
        def remote(self, **kwargs):
            production_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/single"

    class FakeStats:
        def remote(self, traj_prefix, **kwargs):
            analysis_stats.append((traj_prefix, kwargs))
            return f"stats-{traj_prefix}"

    monkeypatch.setattr(gromacs_app, "prepare_tpr_cpu", FakePrepare())
    monkeypatch.setattr(gromacs_app, "production_run_cpu", FakeProduction())
    monkeypatch.setattr(gromacs_app, "collect_traj_stats", FakeStats())

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

    assert prepare_kwargs["run_name"] == "single"
    assert prepare_kwargs["pdb_content"] == VALID_PDB
    assert production_kwargs == {
        "run_name": "single",
        "simulation_time_ns": 3,
        "num_threads": 2,
        "use_openmp_threads": False,
    }
    assert analysis_stats == [
        ("nvt_", {"run_name": "single"}),
        ("npt_", {"run_name": "single"}),
        (
            "production_",
            {"run_name": "single", "save_processed_traj": True},
        ),
    ]


def test_internal_gromacs_job_keeps_durable_directory_output(monkeypatch) -> None:
    analysis_stats = []

    class FakePrepare:
        def remote(self, **_kwargs):
            return f"{gromacs_app.CONF.output_volume_mountpoint}/api-123"

    class FakeProduction:
        def remote(self, **_kwargs):
            return f"{gromacs_app.CONF.output_volume_mountpoint}/api-123"

    class FakeStats:
        def remote(self, traj_prefix, **kwargs):
            analysis_stats.append((traj_prefix, kwargs))
            return f"{gromacs_app.CONF.output_volume_mountpoint}/api-123"

    monkeypatch.setattr(gromacs_app, "prepare_tpr_cpu", FakePrepare())
    monkeypatch.setattr(gromacs_app, "production_run_cpu", FakeProduction())
    monkeypatch.setattr(gromacs_app, "collect_traj_stats", FakeStats())

    result = gromacs_app._run_gromacs_job(
        pdb_content=VALID_PDB,
        run_name="api-123",
        simulation_time_ns=3,
        cpu_only=True,
    )

    assert result.status.value == "succeeded"
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.name == "gromacs_run"
    assert output.storage.kind.value == "volume_path"
    assert isinstance(output.storage, VolumePath)
    assert output.storage.path == "api-123"
    assert output.metadata["run_name"] == "api-123"
    assert output.metadata["files"][0] == {
        "path": "production_api-123.xtc",
        "role": "trajectory",
    }
    assert analysis_stats == [
        ("nvt_", {"run_name": "api-123"}),
        ("npt_", {"run_name": "api-123"}),
        (
            "production_",
            {"run_name": "api-123", "save_processed_traj": True},
        ),
    ]


def test_run_gromacs_job_rejects_unsafe_run_name() -> None:
    with pytest.raises(ValueError, match="run_name must be a safe filename"):
        gromacs_app.run_gromacs_job.get_raw_f()(
            pdb_content=VALID_PDB,
            run_name="../escape",
        )


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
            self.reload_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            self.reload_count += 1

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
        pdb_content=VALID_PDB,
        run_name="prep",
        simulation_time_ns=1,
        num_threads=2,
    )

    run_root = tmp_path / "prep"
    input_path = run_root / "prep.pdb"
    staged_input_path = run_root / "inputs" / "prep.pdb"
    assert result == str(run_root)
    assert input_path.read_bytes() == VALID_PDB
    assert staged_input_path.read_bytes() == VALID_PDB
    assert captured["cmd"][captured["cmd"].index("-i") + 1] == str(input_path)
    assert captured["cwd"] == str(run_root)
    assert captured["env"] == {"OMP_NUM_THREADS": None}
    assert volume.commit_count == 2
    assert volume.reload_count == 1


def test_fresh_production_run_uses_mdp_nsteps(tmp_path: Path, monkeypatch) -> None:
    work_path = tmp_path / "fresh"
    work_path.mkdir()
    work_path.joinpath("production_fresh.tpr").write_text("tpr\n", encoding="utf-8")
    captured = {}

    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0
            self.reload_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            self.reload_count += 1

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
    assert volume.reload_count == 1
