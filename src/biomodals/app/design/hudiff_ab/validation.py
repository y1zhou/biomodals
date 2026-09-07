"""Trust-boundary validation shared by the CLI and execution coordinator."""

from __future__ import annotations

from io import BytesIO

import polars as pl

from biomodals.helper.io import fasta_identifier

CSV_COLUMNS = ("id", "vh", "vl")
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_PAIRS = 1000
MAX_INPUT_BYTES = 3 * 1024 * 1024
MAX_ID_LENGTH = 200
MAX_SEQUENCE_LENGTH = 200


def parse_hudiff_ab_csv(content: bytes) -> pl.DataFrame:
    """Parse the strict bounded wide-CSV interface."""
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_INPUT_BYTES:
        raise ValueError(f"Input CSV must be between 1 and {MAX_INPUT_BYTES} bytes")
    try:
        content.decode("utf-8")
        frame = pl.read_csv(
            BytesIO(content),
            schema_overrides={column: pl.String for column in CSV_COLUMNS},
            infer_schema=False,
        )
    except Exception as exc:
        raise ValueError(f"Input CSV could not be parsed: {exc}") from exc
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ValueError("Input CSV must have exactly these columns: id,vh,vl")
    if not 1 <= frame.height <= MAX_PAIRS:
        raise ValueError(f"Input CSV must contain between 1 and {MAX_PAIRS} pairs")
    seen: set[str] = set()
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Row {row_number}: id must be non-empty")
        if identifier != identifier.strip() or len(identifier) > MAX_ID_LENGTH:
            raise ValueError(f"Row {row_number}: id is invalid")
        if any(ord(character) < 32 or character == ">" for character in identifier):
            raise ValueError(f"Row {row_number}: id contains unsupported characters")
        if fasta_identifier(identifier) in seen:
            raise ValueError(
                f"Row {row_number}: duplicate id after FASTA whitespace normalization: {identifier!r}"
            )
        seen.add(fasta_identifier(identifier))
        for column in ("vh", "vl"):
            sequence = row[column]
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"Row {row_number}: {column} must be non-empty")
            if len(sequence) > MAX_SEQUENCE_LENGTH or set(sequence) - AMINO_ACIDS:
                raise ValueError(
                    f"Row {row_number}: {column} must be <=200 uppercase canonical residues"
                )
    return frame


def validate_antibody_chains(frame: pl.DataFrame) -> None:
    """Require complete IMGT-numbered H plus K/L chains before dispatch."""
    import anarci  # type: ignore[ty:unresolved-import]

    for row in frame.iter_rows(named=True):
        for column, allowed in (("vh", {"H"}), ("vl", {"K", "L"})):
            result = anarci.number(row[column], scheme="imgt")
            if not result or result[0] is None or result[1] not in allowed:
                raise ValueError(f"{row['id']}: {column} has the wrong chain role")
            numbered = "".join(residue for _, residue in result[0] if residue != "-")
            if numbered != row[column]:
                raise ValueError(
                    f"{row['id']}: {column} has residues outside the IMGT variable region"
                )
