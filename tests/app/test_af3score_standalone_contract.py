"""Tests for AF3Score standalone app contracts."""

# ruff: noqa: D101,D102,D103,D107

import inspect
from pathlib import Path
from types import SimpleNamespace

from biomodals.app.score import af3score_app


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def test_af3score_remote_functions_do_not_accept_path_payloads() -> None:
    for function_name in (
        "af3score_prepare",
        "af3score_run",
        "af3score_postprocess",
    ):
        signature = inspect.signature(getattr(af3score_app, function_name).get_raw_f())
        assert "paths" not in signature.parameters
        assert "Path" not in str(signature)


def test_af3score_prepare_reports_app_run_layout_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeOutputVolume()
    monkeypatch.setattr(
        af3score_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume, output_volume_mountpoint=str(tmp_path)
        ),
    )

    run_root = tmp_path / "demo"
    run_root.joinpath("inputs").mkdir(parents=True)
    run_root.joinpath("inputs", "target.pdb").write_text("ATOM\n", encoding="utf-8")
    sample_dir = (
        run_root / "outputs" / "target" / af3score_app.APP_INFO.completion_sample_subdir
    )
    sample_dir.mkdir(parents=True)
    for file_name in af3score_app.APP_INFO.completion_required_files:
        sample_dir.joinpath(file_name).write_text("{}", encoding="utf-8")

    result = af3score_app.af3score_prepare.get_raw_f()(
        run_name="demo", input_files=["target.pdb"], num_jobs=1, prepare_workers=1
    )

    assert result.pending == 0
    assert result.skipped == 1
    assert result.output_dir == str(run_root / "outputs")
    assert result.failed_dir == str(run_root / "outputs" / "failed_records")
    assert output_volume.reload_count == 1


def test_af3score_postprocess_uses_layout_and_run_root_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeOutputVolume()
    monkeypatch.setattr(
        af3score_app,
        "CONF",
        SimpleNamespace(
            git_clone_dir=tmp_path / "AF3Score",
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )

    run_root = tmp_path / "demo"
    run_root.joinpath("inputs").mkdir(parents=True)
    run_root.joinpath("prepare").mkdir()
    sample_dir = (
        run_root / "outputs" / "target" / af3score_app.APP_INFO.completion_sample_subdir
    )
    sample_dir.mkdir(parents=True)
    for file_name in af3score_app.APP_INFO.completion_required_files:
        sample_dir.joinpath(file_name).write_text("{}", encoding="utf-8")

    def fake_run_command(cmd):
        save_arg = next(arg for arg in cmd if arg.startswith("--save_metric_csv="))
        Path(save_arg.split("=", maxsplit=1)[1]).write_text(
            "name,score\ntarget,1.0\n", encoding="utf-8"
        )
        return []

    monkeypatch.setattr(af3score_app, "run_command", fake_run_command)

    result = af3score_app.af3score_postprocess.get_raw_f()(
        run_name="demo", input_files=["target.pdb"]
    )

    assert result["output_dir"] == str(run_root / "outputs")
    assert result["failed_dir"] == str(run_root / "outputs" / "failed_records")
    assert result["metrics_csv"] == str(
        run_root / af3score_app.APP_INFO.metrics_filename
    )
    assert result["metrics_rows"] == 1
    assert not run_root.joinpath("prepare").exists()
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1
