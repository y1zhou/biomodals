"""Tests for shared console styling helpers."""

from __future__ import annotations

from biomodals.helper.styling import strip_ansi


def test_strip_ansi_removes_escape_sequences() -> None:
    """ANSI escape sequences should be removed from terminal text."""
    assert strip_ansi("\x1b[31mred\x1b[0m\n") == "red\n"
