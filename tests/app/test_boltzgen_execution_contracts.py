"""Tests for BoltzGen publication claims and completion validation."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.app.design import boltzgen_app
from biomodals.app.design.boltzgen.execution_contracts import (
    acquire_output_claim,
    is_boltzgen_run_complete,
)


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


def test_claim_requires_same_owner_or_explicit_terminal_replacement(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    claim = acquire_output_claim(run_dir, owner="call-1")
    assert acquire_output_claim(run_dir, owner="call-1") == claim
    with pytest.raises(RuntimeError, match="another Provider Call"):
        acquire_output_claim(run_dir, owner="call-2")

    assert (
        acquire_output_claim(
            run_dir,
            owner="call-2",
            replace_owner="call-1",
        )
        == claim
    )
    assert (claim / "owner").read_text() == "call-2"


def test_worker_preserves_claim_on_failure_and_redelivery_finishes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = FakeVolume()
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        boltzgen_app,
        "CONF",
        SimpleNamespace(
            output_volume=volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name="outputs",
        ),
    )

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("preempted")

    monkeypatch.setattr(boltzgen_app, "run_command", fail)
    worker = boltzgen_app.run_boltzgen_task.get_raw_f()
    with pytest.raises(RuntimeError, match="preempted"):
        worker(
            out_dir=str(out_dir),
            input_yaml_path=str(tmp_path / "input.yaml"),
            claim_owner="call-1",
        )

    assert (out_dir / ".lock" / "owner").read_text() == "call-1"

    def finish(*args, **kwargs):
        del args, kwargs
        final = out_dir / "final_ranked_designs"
        final.mkdir(parents=True)
        (final / "results_overview.pdf").write_bytes(b"pdf")

    monkeypatch.setattr(boltzgen_app, "run_command", finish)
    assert worker(
        out_dir=str(out_dir),
        input_yaml_path=str(tmp_path / "input.yaml"),
        claim_owner="call-1",
    ) == str(out_dir)
    assert is_boltzgen_run_complete(out_dir)
    assert not (out_dir / ".lock").exists()
