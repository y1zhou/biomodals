"""Small, dependency-free validation helpers for uploaded PDB structures."""

from __future__ import annotations

from math import isfinite


def validate_pdb_content(content: bytes, *, max_bytes: int) -> None:
    """Require at least one atom and reject every malformed coordinate record."""
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

    atom_records = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            if line.startswith(("ATOM  ", "HETATM")):
                raise ValueError(f"PDB atom record on line {line_number} is malformed")
            continue
        try:
            serial = int(line[6:11])
            coordinates = tuple(
                float(line[start : start + 8]) for start in (30, 38, 46)
            )
        except ValueError as exc:
            raise ValueError(
                f"PDB atom record on line {line_number} is malformed"
            ) from exc
        if (
            serial <= 0
            or not line[12:16].strip()
            or not all(map(isfinite, coordinates))
        ):
            raise ValueError(f"PDB atom record on line {line_number} is malformed")
        atom_records += 1

    if not atom_records:
        raise ValueError("PDB content does not contain a parseable atom record")
