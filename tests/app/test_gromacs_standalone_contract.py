"""Tests for standalone GROMACS app behavior used by workflows."""

# ruff: noqa: D101,D102,D103,D107

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from biomodals.app.bioinfo import gromacs_app
from biomodals.execution import RunStatus


def test_analysis_csv_preserves_the_established_checkpoint_format(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rmsf.csv"

    gromacs_app.write_analysis_csv(
        output,
        {
            "residue_index": [1.0, 2.0],
            "rmsf": [0.123456, 2.0],
        },
    )

    assert output.read_text(encoding="utf-8") == (
        "residue_index,rmsf\n1.00000,0.12346\n2.00000,2.00000\n"
    )


def test_analysis_pair_invalidates_each_stale_member_independently(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.xtc"
    csv = tmp_path / "analysis.csv"
    figure = tmp_path / "analysis.png"
    for path in (trajectory, csv, figure):
        path.write_bytes(b"data")
    os.utime(trajectory, (20, 20))
    os.utime(csv, (10, 10))
    os.utime(figure, (30, 30))

    gromacs_app.remove_stale_analysis_outputs(
        csv,
        figure,
        trajectory,
        make_figures=True,
    )

    assert not csv.exists()
    assert figure.exists()

    csv.write_bytes(b"new")
    os.utime(csv, (30, 30))
    os.utime(figure, (10, 10))
    gromacs_app.remove_stale_analysis_outputs(
        csv,
        figure,
        trajectory,
        make_figures=True,
    )

    assert csv.exists()
    assert not figure.exists()


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

    def coordinator_handle(**kwargs):
        launched["handle_kwargs"] = kwargs
        return SimpleNamespace(run=FakeMethod())

    monkeypatch.setattr(gromacs_app, "uuid4", lambda: execution_run_id)
    monkeypatch.setattr(gromacs_app, "stage_execution_request", stage)
    monkeypatch.setattr(
        gromacs_app,
        "_execution_coordinator_handle",
        coordinator_handle,
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
    assert request.num_threads == 2
    assert request.max_active_provider_calls == 3
    assert request.max_active_gpu_provider_calls == 0
    assert launched["handle_kwargs"]["execution_run_id"] == execution_run_id
    assert launched["run_kwargs"] == {"development": True}


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
    assert volume.commit_count == 2


def test_fresh_production_run_uses_mdp_nsteps(tmp_path: Path, monkeypatch) -> None:
    work_path = tmp_path / "fresh"
    work_path.mkdir()
    work_path.joinpath("production_fresh.tpr").write_text("tpr\n", encoding="utf-8")
    captured = {}

    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

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
