"""Terminal text helpers."""

from __future__ import annotations

import re

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Remove ANSI control sequences from terminal output."""
    return _ANSI_ESCAPE_RE.sub("", text)
