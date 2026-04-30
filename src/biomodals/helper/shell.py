"""Helper functions for operations using external shell commands."""

from __future__ import annotations

import shutil
import subprocess as sp
import warnings
from collections.abc import Iterable
from pathlib import Path

from biomodals.helper.internal import timed_function


def _build_env(env: dict[str, str] | None) -> dict[str, str]:
    """Build environment variables for subprocesses."""
    import os

    default_env = os.environ | {"SYSTEMD_COLORS": "1"}
    if env is None:
        return default_env
    new_env = default_env | env
    # Remove keys with None values to avoid issues with subprocesses
    return {k: v for k, v in new_env.items() if v is not None}


def run_background_command(cmd: list[str] | str, **kwargs) -> sp.Popen:
    """Run a shell command in the background without waiting for it to finish."""
    import shlex

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    print(f"Running background command: {shlex.join(cmd)}")
    kwargs.setdefault("stdout", sp.DEVNULL)
    kwargs.setdefault("stderr", sp.DEVNULL)
    kwargs["env"] = _build_env(kwargs.get("env", None))
    return sp.Popen(cmd, **kwargs)  # noqa: S603


def run_command(
    cmd: list[str] | str,
    *,
    verbose: bool = True,
    try_rich_print: bool = False,
    **kwargs,
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

    if verbose:
        print(f"Running command: {shlex.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")
    kwargs["env"] = _build_env(kwargs.get("env", None))

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


def run_command_with_log(
    cmd: list[str] | str, log_file: str | Path, verbose: bool = False, **kwargs
) -> None:
    """Run a shell command and log output to a file."""
    import shlex
    import subprocess as sp
    from datetime import UTC, datetime, timedelta
    from time import time

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    cmd_str = shlex.join(cmd)
    print(f"Running command: {cmd_str}")

    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")
    kwargs["env"] = _build_env(kwargs.get("env", None))

    log_path = Path(log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
            if verbose:
                print(buffered_output, end="", flush=True)

        f.write(f"\n{banner}\nFinished at: {str(datetime.now(UTC))}\n")

        elapsed_seconds = float(time() - now)
        elapsed_time = timedelta(seconds=elapsed_seconds)
        f.write(f"Elapsed time: {elapsed_time}\n")

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
    cmd = ["tar", "-I", f"zstd -T{num_threads}", "-f", "-"]  # $ZSTD_NBTHREADS
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

        cmd_paths.append(str(out_path.relative_to(workdir)))

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


@timed_function
def copy_files(src_dst_mapping: dict[str | Path, str | Path]) -> None:
    """Copy files from source to destination paths.

    Args:
        src_dst_mapping: A dictionary mapping source file paths to destination file paths.
            Both keys and values can be either strings or Path objects. The function
            will create any necessary parent directories for the destination paths.
    """
    import shlex
    import shutil
    import subprocess as sp

    subprocesses: list[sp.Popen] = []
    cp_binary = shutil.which("cp")
    if cp_binary is None:
        raise FileNotFoundError("The 'cp' command is not available on this system.")
    for src, dst in src_dst_mapping.items():
        src_path = Path(src)
        dst_path = Path(dst)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file '{src_path}' does not exist.")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        subprocesses.append(
            sp.Popen(  # noqa: S603
                [cp_binary, "-an", str(src_path), str(dst_path)],
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )
        )

    err_msgs: list[str] = []
    for p in subprocesses:
        _, p_stderr = p.communicate()
        if p.returncode != 0:
            p_cmd = shlex.join(p.args)
            p_err_msg = p_stderr.decode().strip()
            err_msgs.append(
                f"'{p_cmd}' failed with return code {p.returncode}: {p_err_msg}"
            )
    if err_msgs:
        raise RuntimeError("\n".join(err_msgs))


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


def sanitize_filename(filename: str, separator: str = "_") -> str:
    """Sanitize a filename by replacing unsafe characters with a specified separator."""
    import os

    root_dir = Path(os.sep)
    f = (root_dir / filename.strip()).resolve().relative_to(root_dir)
    return separator.join(f.parts)
