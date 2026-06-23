"""Tests for PPIFlow manifest helpers."""

# ruff: noqa: D103


def test_manifest_module_is_importable() -> None:
    from biomodals.workflow.ppiflow import manifests

    assert manifests.__doc__
