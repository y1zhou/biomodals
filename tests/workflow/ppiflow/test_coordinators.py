"""Tests for PPIFlow coordinator helpers."""

# ruff: noqa: D103


def test_coordinator_module_is_importable() -> None:
    from biomodals.workflow.ppiflow import coordinators

    assert coordinators.__doc__
