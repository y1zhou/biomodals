"""Tests for standalone DockQ app helper behavior."""

# ruff: noqa: D103

from pathlib import Path

import pytest

from biomodals.app.score import dockq_app


def test_build_local_output_path_reports_blank_run_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        dockq_app.build_local_output_path(tmp_path, run_name=" ")
