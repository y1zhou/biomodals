"""Tests for uploaded PDB structure validation."""

# ruff: noqa: D103

import pytest

from biomodals.helper.pdb import validate_pdb_content

VALID_ATOM = (
    b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
)
VALID_HETATM = (
    b"HETATM    2  O   HOH A   2       1.000   2.000   3.000  1.00 20.00           O\n"
)
MALFORMED_ATOM = b"ATOM      3  CB  ALA A   1       invalid coordinates\n"


def test_validate_pdb_content_accepts_multiple_valid_coordinate_records() -> None:
    validate_pdb_content(VALID_ATOM + VALID_HETATM + b"END\n", max_bytes=4096)


@pytest.mark.parametrize(
    "content",
    [VALID_ATOM + MALFORMED_ATOM, MALFORMED_ATOM + VALID_ATOM],
)
def test_validate_pdb_content_rejects_any_malformed_coordinate_record(
    content: bytes,
) -> None:
    with pytest.raises(ValueError, match="atom record on line"):
        validate_pdb_content(content, max_bytes=4096)
