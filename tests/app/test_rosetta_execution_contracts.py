"""Tests for Rosetta Task commands and completion publications."""

# ruff: noqa: D103

from hashlib import sha256
from pathlib import Path

import pytest

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    execute_rosetta_task,
    task_publication_path,
    validate_task_publication,
    validate_task_publication_from_volume,
)


def _task(*, expected_files: tuple[str, ...] = ()) -> RosettaTaskSpec:
    return RosettaTaskSpec(
        task_key="candidate-a",
        index=1,
        binary="/usr/bin/relax",
        pdb="inputs/1/input.pdb",
        rosetta_script="inputs/_script/workflow.xml",
        flags_file="inputs/_flags/options.flags",
        output_dir="outputs/1",
        worker_log="logs/1.log",
        expected_files=expected_files,
        input_sha256=sha256(b"ATOM\n").hexdigest(),
        script_sha256=sha256(b"<ROSETTASCRIPTS />").hexdigest(),
        flags_sha256=sha256(b"-nstruct 1").hexdigest(),
        candidate_id="candidate-a",
    )


def test_execute_preserves_command_and_reuses_valid_publication(
    tmp_path: Path,
) -> None:
    task = _task(expected_files=("outputs/1/score.sc",))
    calls = []
    checkpoints: list[str] = []

    def run_command(command, *, output_mode, log_file):
        calls.append((command, output_mode, log_file))
        Path(log_file).write_text("log\n", encoding="utf-8")
        score = tmp_path / "outputs" / "1" / "score.sc"
        score.parent.mkdir(parents=True, exist_ok=True)
        score.write_text("score\n", encoding="utf-8")

    def checkpoint_outputs() -> None:
        assert (tmp_path / "outputs" / "1" / "score.sc").is_file()
        assert not task_publication_path(tmp_path, task.task_key).exists()
        checkpoints.append("outputs")

    expected_command = [
        "/usr/bin/relax",
        "-parser:protocol",
        str(tmp_path / "inputs" / "_script" / "workflow.xml"),
        f"@{tmp_path / 'inputs' / '_flags' / 'options.flags'}",
        "-s",
        str(tmp_path / "inputs" / "1" / "input.pdb"),
        "-out:path:all",
        str(tmp_path / "outputs" / "1"),
    ]

    first = execute_rosetta_task(
        run_root=tmp_path,
        task=task,
        task_fingerprint="fingerprint",
        run_command=run_command,
        checkpoint_outputs=checkpoint_outputs,
    )
    second = execute_rosetta_task(
        run_root=tmp_path,
        task=task,
        task_fingerprint="fingerprint",
        run_command=run_command,
    )

    assert first == second
    assert calls == [(expected_command, "capture", tmp_path / "logs" / "1.log")]
    assert checkpoints == ["outputs"]
    assert validate_task_publication(tmp_path, task, "fingerprint")
    assert not validate_task_publication(tmp_path, task, "other-fingerprint")


def test_missing_declared_output_never_publishes_success(tmp_path: Path) -> None:
    task = _task(expected_files=("outputs/1/score.sc",))

    def run_command(command, *, output_mode, log_file):
        del command, output_mode
        Path(log_file).write_text("log\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="required output"):
        execute_rosetta_task(
            run_root=tmp_path,
            task=task,
            task_fingerprint="fingerprint",
            run_command=run_command,
        )

    assert not validate_task_publication(tmp_path, task, "fingerprint")


def test_empty_undeclared_output_never_publishes_success(tmp_path: Path) -> None:
    task = _task()

    def run_command(command, *, output_mode, log_file):
        del command, output_mode
        Path(log_file).write_text("log\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without output files"):
        execute_rosetta_task(
            run_root=tmp_path,
            task=task,
            task_fingerprint="fingerprint",
            run_command=run_command,
        )

    assert not validate_task_publication(tmp_path, task, "fingerprint")


def test_volume_probe_revalidates_marker_and_required_files(
    tmp_path: Path,
) -> None:
    task = _task(expected_files=("outputs/1/score.sc",))

    def run_command(command, *, output_mode, log_file):
        del command, output_mode
        Path(log_file).write_text("log\n", encoding="utf-8")
        score = tmp_path / "outputs" / "1" / "score.sc"
        score.parent.mkdir(parents=True, exist_ok=True)
        score.write_text("score\n", encoding="utf-8")

    execute_rosetta_task(
        run_root=tmp_path,
        task=task,
        task_fingerprint="fingerprint",
        run_command=run_command,
    )

    class VolumeReader:
        def read_file(self, path: str):
            selected = tmp_path.parent / path
            if not selected.is_file():
                raise FileNotFoundError(path)
            yield selected.read_bytes()

    volume = VolumeReader()
    assert validate_task_publication_from_volume(
        volume,
        tmp_path.name,
        task,
        "fingerprint",
    )
    assert not validate_task_publication_from_volume(
        volume,
        tmp_path.name,
        task,
        "different",
    )
    (tmp_path / "outputs" / "1" / "score.sc").write_text(
        "corrupt\n",
        encoding="utf-8",
    )
    assert not validate_task_publication_from_volume(
        volume,
        tmp_path.name,
        task,
        "fingerprint",
    )
    (tmp_path / "outputs" / "1" / "score.sc").unlink()
    assert not validate_task_publication_from_volume(
        volume,
        tmp_path.name,
        task,
        "fingerprint",
    )
