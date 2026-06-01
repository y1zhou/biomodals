"""Tests for reusable volume run helpers."""

from __future__ import annotations

import pytest

from biomodals.helper.volume_run import volume_path_from_mount_path
from biomodals.schema import VolumePath


def test_volume_path_from_mount_path_returns_relative_volume_path() -> None:
    """Mount paths are converted to volume-relative storage paths."""
    assert volume_path_from_mount_path(
        remote_path="/outputs/run-1/production",
        mount_root="/outputs",
        volume_name="Gromacs-outputs",
    ) == VolumePath(volume_name="Gromacs-outputs", path="run-1/production")


def test_volume_path_from_mount_path_preserves_media_type() -> None:
    """Optional media type is preserved on the returned storage object."""
    assert volume_path_from_mount_path(
        remote_path="/outputs/run-1/archive.tar.zst",
        mount_root="/outputs",
        volume_name="FlowPacker-outputs",
        media_type="application/zstd",
    ) == VolumePath(
        volume_name="FlowPacker-outputs",
        path="run-1/archive.tar.zst",
        media_type="application/zstd",
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
