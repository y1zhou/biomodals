"""Tests for PPIFlow workflow-compatible run layout behavior."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.design import ppiflow_app


def test_ppiflow_run_uses_app_run_layout_for_outputs_and_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeOutputVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    def fake_run_command(cmd, *, output_mode, log_file):
        calls["cmd"] = cmd
        calls["output_mode"] = output_mode
        calls["log_file"] = log_file

    output_volume = FakeOutputVolume()
    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            name=ppiflow_app.CONF.name,
            model_volume_mountpoint=str(tmp_path / "models"),
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    monkeypatch.setattr(ppiflow_app, "run_command", fake_run_command)

    args = ppiflow_app.PPIFlowArgs(
        args=ppiflow_app.SampleBinderConfig(
            name="target",
            specified_hotspots="A1",
            input_pdb="/inputs/target.pdb",
            binder_chain="B",
        )
    )

    remote_workdir = ppiflow_app.ppiflow_run.get_raw_f()(
        args=args,
        run_name="ppiflow-run",
    )

    layout_root = tmp_path / "ppiflow-run"
    assert remote_workdir == str(layout_root)
    assert (layout_root / "outputs").is_dir()
    assert (layout_root / "logs").is_dir()
    assert calls["output_mode"] == "capture"
    assert calls["log_file"] == layout_root / "logs" / "PPIFlow-run.log"
    assert f"--output_dir={layout_root / 'outputs'}" in calls["cmd"]
    assert output_volume.commit_count == 1
