"""Pure command builders for the Biomodals CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def _modal_base_command(
    *,
    modal_mode: str,
    detach: bool,
    python_executable: str | None,
) -> list[str]:
    cmd = [python_executable or sys.executable, "-m", "modal", modal_mode]
    if detach:
        cmd.append("-d")
    return cmd


def build_app_run_command(
    *,
    app_path: str | Path,
    entrypoint: str | None,
    modal_mode: str,
    detach: bool,
    flags: list[str] | tuple[str, ...] | None = None,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    """Build the command for `biomodals app run` without side effects."""
    target = str(app_path) if entrypoint is None else f"{app_path}::{entrypoint}"
    return tuple([
        *_modal_base_command(
            modal_mode=modal_mode,
            detach=detach,
            python_executable=python_executable,
        ),
        target,
        *(flags or ()),
    ])


def build_workflow_run_command(
    *,
    workflow_module: str,
    entrypoint: str,
    modal_mode: str,
    detach: bool,
    dry_run: bool,
    flags: list[str] | tuple[str, ...] | None = None,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    """Build the command for `biomodals workflow run` without side effects."""
    entrypoint_flags = list(flags or ())
    if dry_run and "--dry-run" not in entrypoint_flags:
        entrypoint_flags.insert(0, "--dry-run")
    return tuple([
        *_modal_base_command(
            modal_mode=modal_mode,
            detach=detach,
            python_executable=python_executable,
        ),
        "-m",
        f"{workflow_module}::{entrypoint}",
        *entrypoint_flags,
    ])


def modal_env_overrides(*, gpu: str | None, timeout: int | None) -> dict[str, str]:
    """Return environment overrides for Modal subprocess commands."""
    env: dict[str, str] = {}
    if gpu is not None:
        env["GPU"] = gpu
    if timeout is not None:
        env["TIMEOUT"] = str(timeout)
    return env


def build_modal_deploy_command(
    *,
    app_path: str | Path,
    name: str | None,
    tag: str | None,
) -> tuple[str, ...]:
    """Build the command for `biomodals app deploy` without side effects."""
    cmd = ["modal", "deploy"]
    if name:
        cmd.extend(["--name", name])
    if tag:
        cmd.extend(["--tag", tag])
    cmd.append(str(app_path))
    return tuple(cmd)
