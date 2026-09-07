"""Cross-method FASTA identifier validation."""

# ruff: noqa: D103

import polars as pl
import pytest

from biomodals.app.design.hudiff_ab.validation import parse_hudiff_ab_csv
from biomodals.app.design.humatch.app import parse_humatch_csv
from biomodals.app.design.pabnativ2.app import parse_pabnativ2_csv
from biomodals.app.design.sapiens.app import parse_sapiens_csv


@pytest.mark.parametrize(
    "parse",
    (parse_sapiens_csv, parse_humatch_csv, parse_pabnativ2_csv, parse_hudiff_ab_csv),
)
@pytest.mark.parametrize("space", (" ", "\u2003"))
def test_fasta_normalization_preserves_ids_and_rejects_collisions(parse, space) -> None:
    identifier = f"clone{space}1"
    records = [{"id": identifier, "vh": "AAAA", "vl": "CCCC"}]
    assert parse(pl.DataFrame(records).write_csv().encode()).item(0, "id") == identifier
    records.append({"id": "clone_1", "vh": "AAAA", "vl": "CCCC"})
    with pytest.raises(ValueError, match="duplicate id after FASTA"):
        parse(pl.DataFrame(records).write_csv().encode())
