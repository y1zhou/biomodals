"""Local I/O materialization helpers."""

import re
from pathlib import Path

from biomodals.helper.shell import sanitize_filename


def resolve_local_output_dir(out_dir: str | Path | None) -> Path:
    """Resolve a local output directory without creating it."""
    if out_dir is None:
        return Path.cwd()
    return Path(out_dir).expanduser().resolve()


def _clean_filename_part(value: str | Path | None) -> str:
    """Return one clean filename component."""
    if value is None:
        return ""
    raw_value = str(value)
    if not raw_value.strip():
        return ""
    cleaned = sanitize_filename(raw_value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned


def _clean_extension(extension: str) -> str:
    """Return a clean file extension that cannot escape the output directory."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", extension.strip())
    cleaned = cleaned.replace("/", "_").replace("\\", "_").strip("_")
    if not cleaned:
        return ""
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned


def build_local_output_path(
    out_dir: str | Path,
    *,
    run_name: str,
    prefix: str | Path | None = None,
    suffix: str | Path | None = None,
    extension: str = ".tar.zst",
    overwrite: bool = False,
) -> Path:
    """Build a clean local output path and raise if it would overwrite a file."""
    parts = [
        p for part in (prefix, run_name, suffix) if (p := _clean_filename_part(part))
    ]
    if not parts:
        raise ValueError(
            "At least one of prefix, run_name, or suffix must be non-empty"
        )

    out_path = resolve_local_output_dir(out_dir) / (
        "_".join(parts) + _clean_extension(extension)
    )
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {out_path}")
    return out_path


def write_local_tarball(
    out_file: str | Path, content: bytes, *, overwrite: bool = False
) -> Path:
    """Write tarball bytes to a local path and return the final path."""
    out_path = Path(out_file).expanduser().resolve()
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)
    return out_path
