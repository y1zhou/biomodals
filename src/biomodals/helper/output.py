"""Local output materialization helpers."""

from pathlib import Path


def resolve_local_output_dir(out_dir: str | Path | None) -> Path:
    """Resolve a local output directory without creating it."""
    if out_dir is None:
        return Path.cwd()
    return Path(out_dir).expanduser().resolve()


def build_local_output_path(
    out_dir: str | Path,
    *,
    run_name: str,
    suffix: str = "",
    extension: str = ".tar.zst",
) -> Path:
    """Build a local output path from a run name, suffix, and extension."""
    return Path(out_dir) / f"{run_name}{suffix}{extension}"


def ensure_output_file_available(
    out_file: str | Path, *, overwrite: bool = False
) -> Path:
    """Return the output path or raise if it would overwrite an existing file."""
    out_path = Path(out_file)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {out_path}")
    return out_path


def write_local_tarball(
    out_file: str | Path, content: bytes, *, overwrite: bool = False
) -> Path:
    """Write tarball bytes to a local path and return the final path."""
    out_path = ensure_output_file_available(out_file, overwrite=overwrite)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)
    return out_path
