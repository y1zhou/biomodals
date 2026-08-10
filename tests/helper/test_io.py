"""Shared path-validation helper tests."""

# ruff: noqa: D103

import pytest

from biomodals.helper.io import require_safe_filename_component


def test_require_safe_filename_component() -> None:
    require_safe_filename_component("run-1", field_name="run name")

    with pytest.raises(ValueError, match="run name must be a safe filename component"):
        require_safe_filename_component("../run-1", field_name="run name")
