"""Tests for standalone DockQ app helper behavior."""

# ruff: noqa: D103

from pathlib import Path

import pytest

from biomodals.app.score import dockq_app
from biomodals.helper import io as helper_io


def test_dockq_reuses_shared_local_output_helpers() -> None:
    assert dockq_app.build_local_output_path is helper_io.build_local_output_path
    assert dockq_app.resolve_local_output_dir is helper_io.resolve_local_output_dir
    assert dockq_app.write_local_tarball is helper_io.write_local_tarball


def test_build_local_output_path_reports_blank_run_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        dockq_app.build_local_output_path(tmp_path, run_name=" ")
