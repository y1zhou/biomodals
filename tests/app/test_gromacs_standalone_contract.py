"""Tests for standalone GROMACS app behavior used by workflows."""

# ruff: noqa: D101,D102,D103,D107

import shutil
from pathlib import Path
from types import SimpleNamespace

from biomodals.app.bioinfo import gromacs_app


def test_submit_gromacs_task_keeps_single_run_standalone_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("ATOM\n", encoding="utf-8")
    prepare_kwargs = {}
    production_kwargs = {}
    spawned_stats = []

    class FakePrepare:
        def remote(self, **kwargs):
            prepare_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/single"

    class FakeProduction:
        def remote(self, **kwargs):
            production_kwargs.update(kwargs)
            return f"{gromacs_app.CONF.output_volume_mountpoint}/single"

    class FakeStats:
        def spawn(self, traj_prefix, **kwargs):
            spawned_stats.append((traj_prefix, kwargs))
            return f"stats-{traj_prefix}"

    class FakeFunctionCall:
        @staticmethod
        def gather(*tasks):
            return list(tasks)

    monkeypatch.setattr(gromacs_app, "prepare_tpr_cpu", FakePrepare())
    monkeypatch.setattr(gromacs_app, "production_run_cpu", FakeProduction())
    monkeypatch.setattr(gromacs_app, "collect_traj_stats", FakeStats())
    monkeypatch.setattr(gromacs_app.modal, "FunctionCall", FakeFunctionCall)

    gromacs_app.submit_gromacs_task.info.raw_f(
        input_pdb=str(pdb_path),
        run_name="single",
        simulation_time_ns=3,
        cpu_only=True,
        num_threads=2,
    )

    assert prepare_kwargs["run_name"] == "single"
    assert prepare_kwargs["pdb_content"] == b"ATOM\n"
    assert production_kwargs == {
        "run_name": "single",
        "simulation_time_ns": 3,
        "num_threads": 2,
        "use_openmp_threads": False,
    }
    assert spawned_stats == [
        ("nvt_", {"run_name": "single"}),
        ("npt_", {"run_name": "single"}),
        (
            "production_",
            {"run_name": "single", "save_processed_traj": True},
        ),
    ]


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
