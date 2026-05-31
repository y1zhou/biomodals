"""Pure helpers for Modal volume-backed run state.

These helpers intentionally cover only path and completion policies. Locking
semantics stay owned by the app code unless a caller explicitly moves to a
Modal-supported atomic primitive.
"""

from pathlib import Path, PurePosixPath

from biomodals.schema import VolumePath


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
        volume_name=volume_name,
        path=str(relative_path),
        media_type=media_type,
    )


def build_volume_run_paths(
    mount_root: str | Path,
    run_name: str,
    *,
    metrics_filename: str | None = None,
) -> dict[str, Path]:
    """Return standard volume run paths without creating directories.

    The returned keys match the current AF3Score run-state policy:
    ``mount_root``, ``run_root``, ``inputs_dir``, ``prep_dir``, ``output_dir``,
    ``failed_dir``, and optionally ``metrics_csv`` when ``metrics_filename`` is
    provided.
    """
    mount_root_path = Path(mount_root)
    run_root = mount_root_path / run_name
    output_dir = run_root / "outputs"
    paths = {
        "mount_root": mount_root_path,
        "run_root": run_root,
        "inputs_dir": run_root / "inputs",
        "prep_dir": run_root / "prepare",
        "output_dir": output_dir,
        "failed_dir": output_dir / "failed_records",
    }
    if metrics_filename is not None:
        paths["metrics_csv"] = run_root / metrics_filename
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
