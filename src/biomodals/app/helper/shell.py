"""Helper functions for operations using external shell commands.

Runtime environment dependency:
- rich (optional)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

from biomodals.app.helper.internal import timed_function


def run_command(
    cmd: list[str] | str, *, verbose: bool = True, try_rich_print: bool = True, **kwargs
) -> list[str]:
    """Run a shell command and stream output to stdout.

    Args:
        cmd: Command to run, either as a string or a list of arguments.
        verbose: If True, print the command output to stdout in real time.
        try_rich_print: If True, attempt to use `rich.print` for output formatting.
        **kwargs: Additional keyword arguments to pass to `subprocess.Popen`.
            For example, you can use `cwd` to specify the working directory, or
            `env` to specify environment variables.

    Returns:
        A list of output lines from the command. Note that both STDOUT and STDERR
        are captured.
    """
    import os
    import shlex
    import subprocess as sp

    if try_rich_print:
        try:
            import builtins

            import rich

            builtins.print = rich.print  # ty:ignore[invalid-assignment]
        except ImportError:
            pass

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    print(f"Running command: {shlex.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    default_env = os.environ | {"SYSTEMD_COLORS": "1"}
    if "env" not in kwargs:
        kwargs["env"] = default_env
    else:
        kwargs["env"] = default_env | kwargs["env"]

    all_outputs: list[str] = []
    with sp.Popen(cmd, **kwargs) as p:  # noqa: S603
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            if verbose:
                print(buffered_output, end="", flush=True)
            all_outputs.append(buffered_output.rstrip("\n"))

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd)

        return all_outputs


def run_command_with_log(cmd: list[str] | str, log_file: str | Path, **kwargs) -> None:
    """Run a shell command and log output to a file."""
    import shlex
    import subprocess as sp
    from datetime import UTC, datetime
    from time import time

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    cmd_str = shlex.join(cmd)
    print(f"Running command: {cmd_str}")

    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    log_path = Path(log_file)
    banner = "=" * 100
    now = time()
    with (
        log_path.open("a", buffering=1) as f,
        sp.Popen(cmd, **kwargs) as p,  # noqa: S603
    ):
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        f.write(f"\n{banner}\nTime: {str(datetime.now(UTC))}\n")
        f.write(f"Running command: {cmd_str}\n{banner}\n")

        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            f.write(buffered_output)

        f.write(f"\n{banner}\nFinished at: {str(datetime.now(UTC))}\n")
        f.write(f"Elapsed time: {time() - now:.2f} seconds\n")

        if p.returncode != 0:
            warnings.warn(
                f"Command '{cmd_str}' failed with return code {p.returncode}. "
                f"Check log file {log_path} for details.",
                RuntimeWarning,
                stacklevel=2,
            )
            raise sp.CalledProcessError(p.returncode, cmd)


def find_with_fd(dir_path: str | Path, file_pattern: str = ".", *args) -> list[str]:
    """Find files in a directory matching a pattern using fd.

    Args:
        dir_path: Directory to search in.
        file_pattern: Pattern to match files against.
        *args: Additional arguments to pass to fd.

    Returns:
        List of matching file paths as strings. Note that the paths are relative
        to ``dir_path``.
    """
    import shutil

    fd_binary = shutil.which("fd") or shutil.which("fdfind")
    if fd_binary is None:
        raise FileNotFoundError(
            "Neither 'fd' nor 'fdfind' is installed. Please install one of them to use this function."
        )
    if not Path(dir_path).exists():
        raise FileNotFoundError(dir_path)

    cmd = [fd_binary, file_pattern, str(dir_path), *args]
    return run_command(cmd, verbose=False)


def warmup_directory(dir_path: str | Path, file_pattern: str = ".") -> None:
    """Warm up the disk cache for all files in a directory matching a pattern."""
    if not Path(dir_path).exists():
        raise FileNotFoundError(dir_path)
    fd_args = [
        "-tf",
        "-j256",
        "-x",
        "dd",
        "if={}",
        "of=/dev/null",
        "bs=1M",
        "status=none",
    ]
    try:
        find_with_fd(dir_path, file_pattern, *fd_args)
    except FileNotFoundError as e:
        warnings.warn(str(e), RuntimeWarning, stacklevel=2)
        return


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
            It is also okay if these paths are absolute, as long as they are under
            the root directory. If None, the entire root directory will be included.
        tar_args: Additional arguments to pass to `tar`. For example, you can
            use `--exclude` to skip certain files or directories, or `-h` to
            follow symlinks. See `man tar` for details.
        num_threads: Number of threads to use for zstd compression.
    """
    import shutil
    import subprocess as sp
    import tempfile

    # Ensure zstd is available
    if shutil.which("zstd") is None:
        raise RuntimeError("zstd is not installed or not found in PATH.")

    root_path = Path(root).resolve()
    if root_path.is_file():
        warnings.warn(
            f"root_path '{root_path}' should be a directory; skipping 'paths_to_bundle'.",
            RuntimeWarning,
            stacklevel=2,
        )
        paths_to_bundle = [root_path.name]
        root_path = root_path.parent

    workdir = root_path.parent  # We want the tarball to contain a top-level dir
    cmd = ["tar", "-I", f"zstd -T{num_threads}", "-O"]  # $ZSTD_NBTHREADS
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

        # TODO: deal with symlinks
        if not out_path.resolve().is_relative_to(root_path):
            warnings.warn(
                f"'{p}' is not under the root '{root_path}' and will be skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        cmd_paths.append(str(out_path.relative_to(workdir, walk_up=False)))

    # If no valid subpaths, use all of the root directory
    if not cmd_paths:
        return sp.check_output([*cmd, "-c", root_path.name], cwd=workdir)  # noqa: S603

    # Write the list of paths to a temporary file and use --files-from to pass to tar
    # We use this instead of passing paths directly to avoid issues
    # with very long command lines when there are many files in cmd_paths
    with tempfile.NamedTemporaryFile(mode="w", suffix=".list") as tmp_file:
        tmp_file.write("\n".join(cmd_paths))
        tmp_file.flush()
        return sp.check_output([*cmd, "-c", "-T", tmp_file.name], cwd=workdir)  # noqa: S603


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
