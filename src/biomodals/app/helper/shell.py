"""Helper functions for operations using external shell commands."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

from biomodals.app.helper.internal import timed_function


def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:  # noqa: S603
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buffered_output)


@timed_function
def package_outputs(
    root: str | Path,
    *,
    paths_to_bundle: Iterable[str | Path] | None = None,
    tar_args: list[str] | None = None,
    num_threads: int = 16,
) -> bytes:
    """Package directory into a tar.zst archive and return as bytes.

    We make an assumption here that all paths to bundle are under the same root.

    Args:
        root: Root directory in the archive. All paths will be relative to this.
        paths_to_bundle: Specific paths (relative to root) to include in the archive.
        tar_args: Additional arguments to pass to `tar`. For example, you can
            use `--exclude` to skip certain files or directories, or `-h` to
            follow symlinks. See `man tar` for details.
        num_threads: Number of threads to use for zstd compression.
    """
    import shutil
    import subprocess as sp

    # Ensure zstd is available
    if shutil.which("zstd") is None:
        raise RuntimeError("zstd is not installed or not found in PATH.")

    # TODO: check effect of symlinks
    root_path = Path(root).resolve()
    cmd = ["tar", "-I", f"zstd -T{num_threads}", "-O"]  # ZSTD_NBTHREADS
    if tar_args is not None:
        cmd.extend(tar_args)

    # We want to preserve the relative paths
    cmd_paths: list[str] = []
    if paths_to_bundle is None:
        paths_to_bundle = []
    for p in paths_to_bundle:
        out_path = root_path.joinpath(p)
        if not out_path.exists():
            warnings.warn(
                f"Path '{out_path}' does not exist and will be skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not out_path.resolve().is_relative_to(root_path):
            warnings.warn(
                f"'{p}' is not under the root '{root_path}' and will be skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        cmd_paths.append(str(out_path.relative_to(root_path.parent, walk_up=False)))

    # If no valid subpaths, use all of the root directory
    if not cmd_paths:
        cmd_paths = [str(root_path.name)]

    cmd.extend(["-c", *cmd_paths])
    return sp.check_output(cmd, cwd=root_path.parent)  # noqa: S603


def softlink_dir(src: str | Path, dst: str | Path) -> None:
    """Create a soft link from src to dst if dst does not exist."""
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists():
        print(f"Destination path {dst} already exists. Skipping link creation.")
        return

    src_path.mkdir(parents=True, exist_ok=True)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.symlink_to(src_path, target_is_directory=True)
