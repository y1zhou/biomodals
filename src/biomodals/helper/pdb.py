"""Small, dependency-free validation helpers for uploaded PDB structures."""

from __future__ import annotations

from math import isfinite


def validate_pdb_content(content: bytes, *, max_bytes: int) -> None:
    """Reject content without at least one parseable fixed-width atom record."""
    if not content:
        raise ValueError("PDB content is empty")
    if len(content) > max_bytes:
        raise ValueError(f"PDB content exceeds the {max_bytes}-byte limit")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("PDB content must be UTF-8 text") from exc
    if "\x00" in text:
        raise ValueError("PDB content must not contain NUL bytes")

    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        try:
            serial = int(line[6:11])
            coordinates = tuple(
                float(line[start : start + 8]) for start in (30, 38, 46)
            )
        except ValueError:
            continue
        if serial > 0 and line[12:16].strip() and all(map(isfinite, coordinates)):
            return

    raise ValueError("PDB content does not contain a parseable atom record")
