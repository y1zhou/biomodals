"""Tests for reusable app run helpers."""

from __future__ import annotations

import pytest

from biomodals.helper.app_run import (
    volume_path_from_mount_path,
)


def test_volume_path_from_mount_path_rejects_paths_outside_mount_root() -> None:
    """Paths outside the mounted volume root are rejected."""
    with pytest.raises(ValueError, match="outside mounted volume root"):
        volume_path_from_mount_path(
            remote_path="/other/run-1",
            mount_root="/outputs",
            volume_name="Gromacs-outputs",
        )


def test_volume_path_from_mount_path_rejects_mount_root_itself() -> None:
    """The mount root itself is not a valid artifact storage path."""
    with pytest.raises(ValueError, match="below mounted volume root"):
        volume_path_from_mount_path(
            remote_path="/outputs",
            mount_root="/outputs",
            volume_name="Gromacs-outputs",
        )
