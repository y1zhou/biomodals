"""Tests for reusable app run helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from biomodals.helper.app_run import (
    AppRunLayout,
    has_completed_output_files,
    inline_zstd_output,
    volume_app_output,
    volume_path_from_mount_path,
)
from biomodals.schema import (
    AppOutput,
    ArtifactFile,
    ArtifactKind,
    InlineBytes,
    VolumePath,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


def test_app_run_layout_builds_standard_paths_without_creating_dirs(
    tmp_path: Path,
) -> None:
    """The app run layout describes paths only."""
    run_root = tmp_path / "run-1"
    layout = AppRunLayout.from_run_root(run_root)

    assert layout.run_root == run_root
    assert layout.inputs_dir == run_root / "inputs"
    assert layout.prep_dir == run_root / "prepare"
    assert layout.outputs_dir == run_root / "outputs"
    assert layout.logs_dir == run_root / "logs"
    assert layout.failures_dir == run_root / "outputs" / "failed_records"
    assert layout.metrics_dir == run_root / "metrics"
    assert layout.markers_dir == run_root / ".markers"
    assert not run_root.exists()


def test_volume_app_output_builds_storage_and_expected_file_metadata() -> None:
    """Volume outputs carry storage and expected child-file metadata together."""
    output = volume_app_output(
        name="structures",
        kind=ArtifactKind.STRUCTURES,
        remote_path="/outputs/run-1/outputs",
        mount_root="/outputs",
        volume_name="Design-outputs",
        metadata={"run_name": "run-1"},
        files=[
            ArtifactFile(path="model.pdb", role="structure"),
            "scores.csv",
        ],
    )

    assert output == AppOutput(
        name="structures",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(volume_name="Design-outputs", path="run-1/outputs"),
        metadata={
            "run_name": "run-1",
            "files": [
                {"path": "model.pdb", "role": "structure"},
                {"path": "scores.csv"},
            ],
        },
    )


def test_inline_zstd_output_builds_archive_metadata() -> None:
    """Inline zstd archive outputs use the standard archive metadata."""
    output = inline_zstd_output(
        name="scores",
        kind=ArtifactKind.SCORES,
        data=b"archive",
        filename="scores.tar.zst",
        metadata={"run_name": "run-1"},
    )

    assert output == AppOutput(
        name="scores",
        kind=ArtifactKind.SCORES,
        storage=InlineBytes(
            data=b"archive",
            filename="scores.tar.zst",
            media_type=ZSTD_MEDIA_TYPE,
        ),
        metadata={"archive_format": "tar.zst", "run_name": "run-1"},
    )


def test_has_completed_output_files_checks_required_files(tmp_path: Path) -> None:
    """Completion is based on required output files, not just a directory."""
    sample_dir = tmp_path / "input-1" / "ranked_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "model.cif").write_text("data")

    assert has_completed_output_files(
        tmp_path,
        "input-1",
        sample_subdir="ranked_0",
        required_files=("model.cif",),
    )
    assert not has_completed_output_files(
        tmp_path,
        "input-1",
        sample_subdir="ranked_0",
        required_files=("model.cif", "scores.json"),
    )


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
