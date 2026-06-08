"""Shared Rich-based terminal styling helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import TextIO

from rich.console import Console
from rich.text import Text

_FALSE_COLOR_VALUES = {"0", "false", "no", "off", "never"}
_TRUE_COLOR_VALUES = {"1", "true", "yes", "on", "always", "force"}
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def console_color_enabled(
    environ: Mapping[str, str] | None = None,
    *,
    color_env_var: str = "BIOMODALS_COLOR",
) -> bool:
    """Return whether Biomodals console output may use color."""
    environ = os.environ if environ is None else environ
    if "NO_COLOR" in environ:
        return False
    color_setting = environ.get(color_env_var, "").strip().lower()
    return color_setting not in _FALSE_COLOR_VALUES


def _console_force_terminal(
    environ: Mapping[str, str] | None = None,
    *,
    color_env_var: str = "BIOMODALS_COLOR",
) -> bool | None:
    """Return Rich force-terminal mode from Biomodals color environment settings."""
    environ = os.environ if environ is None else environ
    if "NO_COLOR" in environ:
        return False
    color_setting = environ.get(color_env_var, "").strip().lower()
    if color_setting in _FALSE_COLOR_VALUES:
        return False
    if color_setting in _TRUE_COLOR_VALUES:
        return True
    return None


def rich_console(
    *,
    file: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    color_env_var: str = "BIOMODALS_COLOR",
) -> Console:
    """Create a Rich console with project color environment handling."""
    force_terminal = _console_force_terminal(environ, color_env_var=color_env_var)
    return Console(
        file=file,
        force_terminal=force_terminal,
        no_color=force_terminal is False,
        color_system="standard" if force_terminal is True else "auto",
        highlight=False,
        markup=False,
        soft_wrap=True,
    )


def print_rich(
    renderable: object,
    *,
    style: str | None = None,
    file: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    color_env_var: str = "BIOMODALS_COLOR",
) -> None:
    """Print one Rich renderable with Biomodals color handling."""
    color_enabled = console_color_enabled(environ, color_env_var=color_env_var)
    if not color_enabled:
        style = None
        if isinstance(renderable, Text):
            renderable = renderable.plain
    console = rich_console(
        file=file,
        environ=environ,
        color_env_var=color_env_var,
    )
    console.print(renderable, style=style)
    console.file.flush()


def styled_text(*segments: tuple[str, str | None]) -> Text:
    """Build a Rich Text renderable from ``(content, style)`` segments."""
    text = Text()
    for content, style in segments:
        text.append(content, style=style)
    return text


def strip_ansi(text: str) -> str:
    """Remove ANSI control sequences from terminal output."""
    return _ANSI_ESCAPE_RE.sub("", text)
