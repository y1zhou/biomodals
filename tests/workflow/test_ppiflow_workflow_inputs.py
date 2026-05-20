"""Tests for PPIFlow workflow input staging helpers."""

# ruff: noqa: D103

from pathlib import Path

import pytest

from biomodals.workflow.ppiflow_workflow import _collect_local_inputs


def test_collect_local_inputs_rejects_sanitized_name_collisions(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("task: demo\n", encoding="utf-8")
    nested = tmp_path / "a"
    nested.mkdir()
    nested.joinpath("b.pdb").write_text("nested", encoding="utf-8")
    tmp_path.joinpath("a_b.pdb").write_text("flat", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate staged input path: a_b.pdb"):
        _collect_local_inputs(
            {
                "task": {
                    "nested_path": "a/b.pdb",
                    "flat_path": "a_b.pdb",
                }
            },
            task_path,
        )


def test_collect_local_inputs_returns_sanitized_keys(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("task: demo\n", encoding="utf-8")
    nested = tmp_path / "a"
    nested.mkdir()
    nested.joinpath("b.pdb").write_text("nested", encoding="utf-8")

    staged = _collect_local_inputs({"task": {"input_path": "a/b.pdb"}}, task_path)

    assert staged == [("a_b.pdb", b"nested")]
