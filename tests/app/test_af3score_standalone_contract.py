"""Tests for AF3Score standalone app contracts."""

# ruff: noqa: D101,D102,D103,D107

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.app.score import af3score_app, af3score_publications
from biomodals.app.score.af3score_execution import (
    af3score_staged_input_directory,
    af3score_staged_input_key,
)


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def _stage_input(root: Path, name: str, content: bytes) -> tuple[str, Path]:
    inputs = ((name, sha256(content).hexdigest()),)
    staged_input_key = af3score_staged_input_key(inputs)
    directory = root.joinpath(*af3score_staged_input_directory(staged_input_key).parts)
    directory.mkdir(parents=True)
    path = directory / name
    path.write_bytes(content)
    return staged_input_key, path


def test_af3score_prepare_uses_staged_inputs_without_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_volume = FakeOutputVolume()
    input_content = b"ATOM\n"
    staged_input_key, staged_input = _stage_input(tmp_path, "target.pdb", input_content)
    monkeypatch.setattr(
        af3score_app,
        "CONF",
        SimpleNamespace(
            git_clone_dir=tmp_path / "AF3Score",
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )

    def fake_run_command(command):
        input_dir = Path(
            next(arg for arg in command if arg.startswith("--input_dir=")).split(
                "=", maxsplit=1
            )[1]
        )
        pending_input = input_dir / "target.pdb"
        assert pending_input.is_symlink()
        assert pending_input.resolve() == staged_input
        batch_root = Path(
            next(arg for arg in command if arg.startswith("--batch_dir=")).split(
                "=", maxsplit=1
            )[1]
        )
        (batch_root / "json" / "batch_0").mkdir(parents=True)
        (batch_root / "json" / "batch_0" / "target.json").write_text("{}")
        batch_pdb_dir = batch_root / "pdb" / "batch_0"
        batch_pdb_dir.mkdir(parents=True)
        batch_pdb_dir.joinpath("target.pdb").symlink_to(pending_input)
        return []

    monkeypatch.setattr(af3score_app, "run_command", fake_run_command)

    result = af3score_app.af3score_prepare.get_raw_f()(
        run_name="demo",
        staged_input_key=staged_input_key,
        input_files=["target.pdb"],
        input_digests={"target": sha256(input_content).hexdigest()},
        publication_key="request-key",
        num_jobs=1,
        prepare_workers=1,
    )

    assert result.pending == 1
    assert len(result.chunk_specs) == 1
    assert not tmp_path.joinpath("demo", "inputs").exists()
    batch_pdb = Path(result.chunk_specs[0].batch_pdb_dir) / "target.pdb"
    assert batch_pdb.resolve(strict=True) == staged_input


def test_af3score_prepare_rejects_changed_staged_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_input_key, _staged_input = _stage_input(tmp_path, "target.pdb", b"CHANGED\n")
    monkeypatch.setattr(
        af3score_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeOutputVolume(),
            output_volume_mountpoint=str(tmp_path),
        ),
    )

    with pytest.raises(ValueError, match="digest changed: target.pdb"):
        af3score_app.af3score_prepare.get_raw_f()(
            run_name="demo",
            staged_input_key=staged_input_key,
            input_files=["target.pdb"],
            input_digests={"target": sha256(b"ORIGINAL\n").hexdigest()},
            publication_key="request-key",
            num_jobs=1,
            prepare_workers=1,
        )


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

    assert af3score_publications._input_publication_ready(
        run_root / "outputs",
        "target",
        publication_key="request-key",
        input_sha256="a" * 64,
    )
    assert output_volume.commit_count == 1


def test_af3score_coordinator_can_outlive_worker_timeout() -> None:
    assert af3score_app._COORDINATOR_TIMEOUT_SECONDS > af3score_app.CONF.timeout


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
