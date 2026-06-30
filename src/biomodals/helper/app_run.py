"""Helpers for Biomodals app run paths and volume-backed artifacts.

These helpers intentionally cover only path and completion policies. Locking and
queue semantics stay owned by app code unless a caller explicitly moves to a
Modal-supported atomic primitive.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from biomodals.schema import (
    AppOutput,
    ArtifactFile,
    ArtifactKind,
    InlineBytes,
    VolumePath,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


@dataclass(frozen=True)
class AppRunLayout:
    """Standard directory contract for one Biomodals app run.

    The layout only describes paths and never creates directories. Callers can
    use the same contract for container-local scratch directories and mounted
    Modal output volumes.
    """

    run_root: Path
    inputs_dir: Path
    prep_dir: Path
    outputs_dir: Path
    logs_dir: Path
    failures_dir: Path
    metrics_dir: Path
    markers_dir: Path

    @classmethod
    def from_run_root(cls, run_root: str | Path) -> "AppRunLayout":
        """Build the standard layout below a resolved per-run root directory."""
        root = Path(run_root)
        outputs_dir = root / "outputs"
        return cls(
            run_root=root,
            inputs_dir=root / "inputs",
            prep_dir=root / "prepare",
            outputs_dir=outputs_dir,
            logs_dir=root / "logs",
            failures_dir=outputs_dir / "failed_records",
            metrics_dir=root / "metrics",
            markers_dir=root / ".markers",
        )


def volume_path_from_mount_path(
    remote_path: str,
    mount_root: str,
    volume_name: str,
    media_type: str | None = None,
) -> VolumePath:
    """Convert an app mount path into a volume-relative workflow storage path."""
    resolved_remote_path = PurePosixPath(remote_path)
    resolved_mount_root = PurePosixPath(mount_root)
    try:
        relative_path = resolved_remote_path.relative_to(resolved_mount_root)
    except ValueError as exc:
        raise ValueError(
            f"Remote path is outside mounted volume root {mount_root}: {remote_path}"
        ) from exc
    if str(relative_path) == ".":
        raise ValueError(
            f"Remote path must be below mounted volume root {mount_root}: {remote_path}"
        )
    return VolumePath(
        volume_name=volume_name, path=str(relative_path), media_type=media_type
    )


def _file_metadata(files: list[ArtifactFile | str]) -> list[dict[str, Any]]:
    return [
        (
            ArtifactFile(path=file).model_dump(
                exclude_defaults=True,
                exclude_none=True,
            )
            if isinstance(file, str)
            else file.model_dump(exclude_defaults=True, exclude_none=True)
        )
        for file in files
    ]


def volume_app_output(
    *,
    name: str,
    kind: ArtifactKind,
    remote_path: str,
    mount_root: str,
    volume_name: str,
    media_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    files: list[ArtifactFile | str] | None = None,
) -> AppOutput:
    """Build a workflow-compatible app output backed by a mounted volume path."""
    output_metadata = dict(metadata or {})
    if files is not None:
        if "files" in output_metadata:
            raise ValueError(
                "Provide expected files either in metadata or files, not both"
            )
        output_metadata["files"] = _file_metadata(files)
    return AppOutput(
        name=name,
        kind=kind,
        storage=volume_path_from_mount_path(
            remote_path=remote_path,
            mount_root=mount_root,
            volume_name=volume_name,
            media_type=media_type,
        ),
        metadata=output_metadata,
    )


def inline_zstd_output(
    *,
    name: str,
    kind: ArtifactKind,
    data: bytes,
    filename: str,
    metadata: dict[str, Any] | None = None,
) -> AppOutput:
    """Build a workflow-compatible inline zstd archive output."""
    output_metadata = {"archive_format": "tar.zst"} | dict(metadata or {})
    return AppOutput(
        name=name,
        kind=kind,
        storage=InlineBytes(
            data=data,
            filename=filename,
            media_type=ZSTD_MEDIA_TYPE,
        ),
        metadata=output_metadata,
    )


def has_completed_output_files(
    output_dir: str | Path,
    input_id: str,
    *,
    sample_subdir: str,
    required_files: tuple[str, ...],
) -> bool:
    """Return whether all required completion files exist for one input.

    This encodes artifact-based completion only. It does not create marker files
    or infer success from the presence of a run directory.
    """
    sample_dir = Path(output_dir) / input_id / sample_subdir
    return all((sample_dir / file_name).exists() for file_name in required_files)
