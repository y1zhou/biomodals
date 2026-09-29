"""Tests for defensive public Job projection decoding."""

# ruff: noqa: D103

from biomodals.service.jobs import _stage_from_projection


def test_malformed_disposable_stage_projection_is_ignored() -> None:
    assert _stage_from_projection({"code": "prepare", "task_counts": []}) is None
    assert _stage_from_projection({"code": 7}) is None
