"""Tests for AF3Score standalone app contracts."""

# ruff: noqa: D101,D102,D103,D107

import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from biomodals.app.score import af3score_app
from biomodals.execution import RunStatus


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def test_af3score_removed_the_volume_directory_scheduler_lock() -> None:
    assert not hasattr(af3score_app, "af3score_manage_lock")


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
    af3score_app._write_input_publication(
        run_root / "outputs",
        "target",
        publication_key="request-key",
        input_sha256="a" * 64,
    )

    result = af3score_app.af3score_prepare.get_raw_f()(
        run_name="demo",
        input_files=["target.pdb"],
        input_digests={"target": "a" * 64},
        publication_key="request-key",
        num_jobs=1,
        prepare_workers=1,
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
    af3score_app._write_input_publication(
        run_root / "outputs",
        "target",
        publication_key="request-key",
        input_sha256="a" * 64,
    )

    def fake_run_command(cmd):
        save_arg = next(arg for arg in cmd if arg.startswith("--save_metric_csv="))
        Path(save_arg.split("=", maxsplit=1)[1]).write_text(
            "name,score\ntarget,1.0\n", encoding="utf-8"
        )
        return []

    monkeypatch.setattr(af3score_app, "run_command", fake_run_command)

    result = af3score_app.af3score_postprocess.get_raw_f()(
        run_name="demo",
        input_files=["target.pdb"],
        input_digests={"target": "a" * 64},
        publication_key="request-key",
    )

    assert result["output_dir"] == str(run_root / "outputs")
    assert result["failed_dir"] == str(run_root / "outputs" / "failed_records")
    assert result["metrics_csv"] == str(
        run_root / af3score_app.APP_INFO.metrics_filename
    )
    assert result["metrics_rows"] == 1
    assert af3score_app._metrics_publication_ready(run_root, "request-key")
    assert not run_root.joinpath("prepare").exists()
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1


def test_af3score_run_binds_outputs_to_the_current_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_volume = FakeOutputVolume()
    model_root = tmp_path / "models"
    model_path = model_root / af3score_app.APP_INFO.af3_weights
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"weights")
    batch_json_dir = tmp_path / "batch" / "json"
    batch_pdb_dir = tmp_path / "batch" / "pdb"
    batch_json_dir.mkdir(parents=True)
    batch_pdb_dir.mkdir(parents=True)
    batch_json_dir.joinpath("target.json").write_text("{}", encoding="utf-8")
    run_root = tmp_path / "demo"

    def fake_run_command(_cmd, **_kwargs):
        sample = (
            run_root
            / "outputs"
            / "target"
            / af3score_app.APP_INFO.completion_sample_subdir
        )
        sample.mkdir(parents=True, exist_ok=True)
        for file_name in af3score_app.APP_INFO.completion_required_files:
            sample.joinpath(file_name).write_text("{}", encoding="utf-8")
        return []

    monkeypatch.setattr(
        af3score_app,
        "CONF",
        SimpleNamespace(
            git_clone_dir=tmp_path / "AF3Score",
            model_volume_mountpoint=str(model_root),
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    monkeypatch.setattr(af3score_app, "run_command", fake_run_command)

    af3score_app.af3score_run.get_raw_f()(
        run_name="demo",
        batch_name="batch-0",
        batch_json_dir=str(batch_json_dir),
        batch_pdb_dir=str(batch_pdb_dir),
        input_digests={"target": "a" * 64},
        publication_key="request-key",
    )

    assert af3score_app._input_publication_ready(
        run_root / "outputs",
        "target",
        publication_key="request-key",
        input_sha256="a" * 64,
    )
    assert output_volume.commit_count == 1


def test_af3score_local_entrypoint_launches_one_execution_coordinator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    output_dir = tmp_path / "results"
    execution_run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    captured = {}

    class FakeBatch:
        def put_file(self, source, destination):
            captured["upload"] = (source.name, source.read_bytes(), destination)

    class FakeVolume:
        @contextmanager
        def batch_upload(self, *, force):
            captured["force"] = force
            yield FakeBatch()

        def read_file(self, path):
            captured["download"] = path
            yield b"name,score\ninput,1\n"

    class FakeMethod:
        def spawn(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return SimpleNamespace(
                object_id="fc-1",
                get=lambda: SimpleNamespace(
                    run=SimpleNamespace(
                        status=RunStatus.SUCCEEDED,
                        status_message=None,
                        status_reason=None,
                    ),
                    provider_calls=(),
                ),
            )

    def stage(volume, run_id, request):
        captured.update(volume=volume, run_id=run_id, request=request)

    def coordinator_handle(**kwargs):
        captured["handle_kwargs"] = kwargs
        return SimpleNamespace(run=FakeMethod())

    volume = FakeVolume()
    monkeypatch.setattr(
        af3score_app,
        "CONF",
        SimpleNamespace(
            name="AF3Score",
            version=None,
            repo_commit_hash="b0764aa",
            output_volume=volume,
            output_volume_mountpoint="/af3score-output",
            output_volume_name="AF3Score-outputs",
        ),
    )
    monkeypatch.setattr(af3score_app, "uuid4", lambda: execution_run_id)
    monkeypatch.setattr(af3score_app, "stage_execution_request", stage)
    monkeypatch.setattr(
        af3score_app,
        "stage_execution_launch",
        lambda _volume, run_id, predecessor: captured.update(
            launch=(run_id, predecessor)
        ),
    )
    monkeypatch.setattr(
        af3score_app,
        "_execution_coordinator_handle",
        coordinator_handle,
    )

    raw = af3score_app.submit_af3score_task.info.raw_f
    assert raw is not None
    raw(
        input_dir=str(input_pdb),
        run_name="scores",
        output_dir=str(output_dir),
        max_batches=2,
    )

    assert captured["run_id"] == execution_run_id
    assert captured["force"] is False
    uploaded_name, uploaded_content, uploaded_destination = captured["upload"]
    assert uploaded_name == input_pdb.name
    assert uploaded_content == input_pdb.read_bytes()
    assert uploaded_destination == (
        f"/.biomodals/execution/runs/{execution_run_id}/inputs/input.pdb"
    )
    assert captured["request"].inputs == (
        ("input.pdb", captured["request"].inputs[0][1]),
    )
    assert captured["request"].staged_input_execution_run_id == str(execution_run_id)
    assert captured["request"].max_batches == 2
    assert captured["launch"] == (execution_run_id, None)
    assert captured["run_kwargs"] == {"development": True}
    assert output_dir.joinpath("scores_af3score_metrics.csv").is_file()


def test_af3score_entrypoint_validates_request_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")

    class Volume:
        @contextmanager
        def batch_upload(self, *, force):
            del force
            pytest.fail("oversized request must fail before input upload")
            yield

    monkeypatch.setattr(
        af3score_app,
        "CONF",
        SimpleNamespace(
            name="AF3Score",
            version=None,
            repo_commit_hash="b0764aa",
            output_volume=Volume(),
            output_volume_mountpoint="/af3score-output",
            output_volume_name="AF3Score-outputs",
        ),
    )

    def reject_oversized_request(_self) -> bytes:
        raise ValueError("byte limit")

    monkeypatch.setattr(
        af3score_app.AF3ScoreExecutionRequest,
        "to_bytes",
        reject_oversized_request,
    )
    raw = af3score_app.submit_af3score_task.info.raw_f
    assert raw is not None

    with pytest.raises(ValueError, match="byte limit"):
        raw(
            input_dir=str(input_pdb),
            run_name="scores",
            output_dir=str(tmp_path / "results"),
        )
