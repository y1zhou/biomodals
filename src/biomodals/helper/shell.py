"""Helper functions for operations using external shell commands."""

from __future__ import annotations

import shutil
import subprocess as sp
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from biomodals.helper.internal import timed_function
from biomodals.helper.styling import print_rich, styled_text


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

    print_rich(
        styled_text(("Running background command: ", None), (shlex.join(cmd), "yellow"))
    )
    kwargs.setdefault("stdout", sp.DEVNULL)
    kwargs.setdefault("stderr", sp.DEVNULL)
    kwargs["env"] = _build_env(kwargs.get("env", None))
    return sp.Popen(cmd, **kwargs)  # noqa: S603


def run_command(
    cmd: list[str] | str,
    *,
    output_mode: Literal["tee", "capture", "inherit", "discard"] = "tee",
    log_file: str | Path | None = None,
    **kwargs,
) -> list[str]:
    """Run a shell command with explicit subprocess output handling.

    Args:
        cmd: Command to run, either as a string or a list of arguments.
        output_mode: ``tee`` captures and streams raw output, ``capture`` captures
            without streaming, ``inherit`` attaches the child to parent streams,
            and ``discard`` drops child output while waiting for completion.
        log_file: Optional file that receives command timing metadata and raw
            child output. Only valid with ``tee`` or ``capture`` modes.
        **kwargs: Additional keyword arguments to pass to `subprocess.Popen`.
            For example, you can use `cwd` to specify the working directory, or
            `env` to specify environment variables.

    Returns:
        Captured output lines. ``inherit`` mode returns an empty list.
    """
    import shlex
    import subprocess as sp
    from datetime import datetime, timedelta
    from time import time

    if sys.version_info >= (3, 11):  # noqa: UP036
        from datetime import UTC
    else:
        from datetime import timezone

        UTC = timezone.utc  # noqa: UP017

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    cmd_str = shlex.join(cmd)
    print_rich(styled_text(("Running command: ", None), (cmd_str, "yellow")))
    if output_mode not in {"tee", "capture", "inherit", "discard"}:
        raise ValueError(f"Unsupported command output mode: {output_mode}")
    if log_file is not None and output_mode in {"inherit", "discard"}:
        raise ValueError("log_file requires output_mode='tee' or 'capture'")
    kwargs["env"] = _build_env(kwargs.get("env", None))
    if output_mode == "inherit":
        kwargs.setdefault("stdout", None)
        kwargs.setdefault("stderr", None)
        with sp.Popen(cmd, **kwargs) as p:  # noqa: S603
            p.wait()
            if p.returncode != 0:
                raise sp.CalledProcessError(p.returncode, cmd)
        return []
    if output_mode == "discard":
        kwargs.setdefault("stdout", sp.DEVNULL)
        kwargs.setdefault("stderr", sp.DEVNULL)
        with sp.Popen(cmd, **kwargs) as p:  # noqa: S603
            p.wait()
            if p.returncode != 0:
                raise sp.CalledProcessError(p.returncode, cmd)
        return []

    all_outputs: list[str] = []
    output_buffer = ""
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 0)
    log_path = Path(log_file).expanduser() if log_file is not None else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    now = time()
    banner = "=" * 100
    log_handle = log_path.open("ab", buffering=0) if log_path is not None else None
    return_code: int | None = None
    try:
        if log_handle is not None:
            log_handle.write(f"\n{banner}\nTime: {str(datetime.now(UTC))}\n".encode())
            log_handle.write(f"Running command: {cmd_str}\n{banner}\n".encode())

        with sp.Popen(cmd, **kwargs) as p:  # noqa: S603
            if p.stdout is None:
                raise RuntimeError("Failed to capture stdout from the command.")

            read_chunk = getattr(p.stdout, "read1", p.stdout.read)
            while chunk := read_chunk(8192):
                if isinstance(chunk, str):
                    text = chunk
                    raw = chunk.encode()
                else:
                    raw = chunk
                    text = chunk.decode("utf-8", errors="replace")
                if log_handle is not None:
                    log_handle.write(raw)
                if output_mode == "tee":
                    try:
                        sys.stdout.buffer.write(raw)
                    except AttributeError:
                        sys.stdout.write(text)
                    sys.stdout.flush()
                output_buffer += text
                while "\n" in output_buffer:
                    line, output_buffer = output_buffer.split("\n", 1)
                    all_outputs.append(line.removesuffix("\r"))

            if output_buffer:
                all_outputs.append(output_buffer.removesuffix("\r"))

            p.wait()
            return_code = p.returncode
    finally:
        if log_handle is not None:
            log_handle.write(
                f"\n{banner}\nFinished at: {str(datetime.now(UTC))}\n".encode()
            )
            elapsed_seconds = float(time() - now)
            elapsed_time = timedelta(seconds=elapsed_seconds)
            log_handle.write(f"Elapsed time: {elapsed_time}\n".encode())
            log_handle.close()

    if return_code != 0:
        if log_path is not None:
            warnings.warn(
                f"Command '{cmd_str}' failed with return code {return_code}. "
                f"Check log file {log_path} for details.",
                RuntimeWarning,
                stacklevel=2,
            )
        raise sp.CalledProcessError(return_code, cmd)

    return all_outputs


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

    cmd = [fd_binary, file_pattern, *args]
    return run_command(cmd, output_mode="capture", cwd=str(dir_path))


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
            p_cmd = shlex.join(p.args)  # type: ignore[ty:invalid-argument-type]
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
    sanitized = separator.join(f.parts)
    if not sanitized:
        raise ValueError("Value must contain at least one safe filename component")
    return sanitized
