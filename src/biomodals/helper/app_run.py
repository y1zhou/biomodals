"""Helpers for Biomodals app run paths and volume-backed artifacts.

These helpers intentionally cover only path and completion policies. Locking and
queue semantics stay owned by app code unless a caller explicitly moves to a
Modal-supported atomic primitive.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from biomodals.schema import VolumePath


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


def build_volume_run_paths(
    mount_root: str | Path,
    run_name: str,
    *,
    metrics_filename: str | None = None,
) -> dict[str, Path]:
    """Return legacy volume run path keys without creating directories.

    The returned keys match the current AF3Score and IgGM run-state policy:
    ``mount_root``, ``run_root``, ``inputs_dir``, ``prep_dir``, ``output_dir``,
    ``failed_dir``, and optionally ``metrics_csv`` when ``metrics_filename`` is
    provided.
    """
    mount_root_path = Path(mount_root)
    layout = AppRunLayout.from_run_root(mount_root_path / run_name)
    paths = {
        "mount_root": mount_root_path,
        "run_root": layout.run_root,
        "inputs_dir": layout.inputs_dir,
        "prep_dir": layout.prep_dir,
        "output_dir": layout.outputs_dir,
        "failed_dir": layout.failures_dir,
    }
    if metrics_filename is not None:
        paths["metrics_csv"] = layout.run_root / metrics_filename
    return paths


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
