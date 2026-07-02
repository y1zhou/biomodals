"""Tests for shell command helpers."""

# ruff: noqa: D103

from __future__ import annotations

import subprocess as sp
import sys
import warnings

import pytest

from biomodals.helper.shell import run_command


def test_run_command_tee_streams_raw_child_output(capfd) -> None:
    lines = run_command(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write("
                "b'\\x1b[32mok\\x1b[0m\\n[ranked_designs]\\n'"
                "); "
                "sys.stdout.flush(); "
                "sys.stderr.buffer.write(b'err\\n'); "
                "sys.stderr.flush()"
            ),
        ],
        output_mode="tee",
    )

    captured = capfd.readouterr()
    assert "\x1b[32mok\x1b[0m\n[ranked_designs]\n" in captured.out
    assert "err\n" in captured.out
    assert captured.err == ""
    assert lines == ["\x1b[32mok\x1b[0m", "[ranked_designs]", "err"]


def test_run_command_capture_returns_output_without_streaming(capfd) -> None:
    lines = run_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write(chr(91) + 'literal' + chr(93) + '\\n')",
        ],
        output_mode="capture",
    )

    captured = capfd.readouterr()
    assert captured.out.startswith("Running command: ")
    assert "[literal]" not in captured.out
    assert captured.err == ""
    assert lines == ["[literal]"]


def test_run_command_can_hide_command_banner(capfd) -> None:
    lines = run_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('hidden\\n')",
        ],
        output_mode="capture",
        show_command=False,
    )

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert lines == ["hidden"]


def test_run_command_log_mode_writes_without_streaming_or_collecting(
    tmp_path, capfd
) -> None:
    log_path = tmp_path / "command.log"
    lines = run_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('logged\\n'); sys.stderr.write('err\\n')",
        ],
        output_mode="log",
        log_file=log_path,
        show_command=False,
    )

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert lines == []
    log_text = log_path.read_text()
    assert "logged\n" in log_text
    assert "err\n" in log_text


def test_run_command_hidden_failure_warning_omits_command(tmp_path) -> None:
    log_path = tmp_path / "command.log"
    with (
        pytest.raises(sp.CalledProcessError),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        run_command(
            [sys.executable, "-c", "raise SystemExit(3)"],
            output_mode="log",
            log_file=log_path,
            show_command=False,
        )

    assert len(caught) == 1
    assert "Command failed with return code 3." in str(caught[0].message)
    assert sys.executable not in str(caught[0].message)


def test_run_command_can_suppress_failure_warning(tmp_path) -> None:
    log_path = tmp_path / "command.log"
    with (
        pytest.raises(sp.CalledProcessError),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        run_command(
            [sys.executable, "-c", "raise SystemExit(3)"],
            output_mode="log",
            log_file=log_path,
            show_command=False,
            warn_on_error=False,
        )

    assert caught == []


def test_run_command_log_mode_requires_log_file() -> None:
    with pytest.raises(ValueError, match="requires log_file"):
        run_command(
            [sys.executable, "-c", "print('ignored')"],
            output_mode="log",
        )


def test_run_command_inherit_uses_parent_streams(capfd) -> None:
    lines = run_command(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.write('out\\n'); "
                "sys.stdout.flush(); "
                "sys.stderr.write('err\\n'); "
                "sys.stderr.flush()"
            ),
        ],
        output_mode="inherit",
    )

    captured = capfd.readouterr()
    assert captured.out.startswith("Running command: ")
    assert captured.out.endswith("out\n")
    assert captured.err == "err\n"
    assert lines == []


def test_run_command_capture_logs_raw_child_output_without_returning_log_metadata(
    tmp_path, capfd, monkeypatch
) -> None:
    def fail_print(*_args, **_kwargs):
        raise AssertionError("run_command should not call print")

    log_path = tmp_path / "command.log"
    monkeypatch.setattr("builtins.print", fail_print)

    lines = run_command(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.write(chr(91) + 'ranked_designs' + chr(93) + '\\n')"
            ),
        ],
        output_mode="capture",
        log_file=log_path,
    )

    captured = capfd.readouterr()
    assert captured.out.startswith("Running command: ")
    assert f"Saving logs to: {log_path}\n" in captured.out
    assert "[ranked_designs]" not in captured.out
    assert captured.err == ""
    assert lines == ["[ranked_designs]"]

    log_text = log_path.read_text()
    assert "Running command: " in log_text
    assert "Time: " in log_text
    assert "Finished at: " in log_text
    assert "Elapsed time: " in log_text
    assert "[ranked_designs]\n" in log_text
    assert all("Elapsed time:" not in line for line in lines)


def test_run_command_tee_logs_and_streams_raw_child_output_without_print(
    tmp_path, capfd, monkeypatch
) -> None:
    def fail_print(*_args, **_kwargs):
        raise AssertionError("run_command should not call print")

    log_path = tmp_path / "command.log"
    monkeypatch.setattr("builtins.print", fail_print)

    lines = run_command(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write("
                "b'\\x1b[32mok\\x1b[0m\\n[ranked_designs]\\n'"
                "); "
                "sys.stdout.flush(); "
                "sys.stderr.buffer.write(b'err\\n'); "
                "sys.stderr.flush()"
            ),
        ],
        log_file=log_path,
        output_mode="tee",
    )

    captured = capfd.readouterr()
    assert captured.out.startswith("Running command: ")
    assert "\x1b[32mok\x1b[0m\n[ranked_designs]\n" in captured.out
    assert "err\n" in captured.out
    assert captured.err == ""
    assert lines == ["\x1b[32mok\x1b[0m", "[ranked_designs]", "err"]

    log_bytes = log_path.read_bytes()
    assert b"\x1b[32mok\x1b[0m\n[ranked_designs]\n" in log_bytes
    assert b"err\n" in log_bytes


def test_run_command_discard_waits_and_drops_child_output(capfd) -> None:
    lines = run_command(
        [
            sys.executable,
            "-c",
            ("import sys; sys.stdout.write('out\\n'); sys.stderr.write('err\\n')"),
        ],
        output_mode="discard",
    )

    captured = capfd.readouterr()
    assert captured.out.startswith("Running command: ")
    assert "out\n" not in captured.out
    assert captured.err == ""
    assert lines == []


@pytest.mark.parametrize("output_mode", ["inherit", "discard"])
def test_run_command_rejects_log_file_for_uncaptured_output_modes(
    tmp_path, output_mode
) -> None:
    with pytest.raises(ValueError, match="log_file requires"):
        run_command(
            [sys.executable, "-c", "print('ignored')"],
            output_mode=output_mode,
            log_file=tmp_path / "command.log",
        )
