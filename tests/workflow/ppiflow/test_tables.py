"""Tests for PPIFlow table helpers."""

# ruff: noqa: D103

import pytest

from biomodals.workflow.ppiflow import tables


def test_candidate_key_recovers_original_structure_stem() -> None:
    assert tables.candidate_key("artifact__nested__design-1.pdb") == "design-1"
    assert tables.candidate_key("design-2.cif") == "design-2"


def test_row_passes_filters_supports_numeric_clauses() -> None:
    row = {"iptm": "0.81", "dockq": 0.55}

    assert tables.row_passes_filters(row, {"iptm": "> 0.8", "dockq": ">= 0.5"})
    assert not tables.row_passes_filters(row, {"iptm": "> 0.9"})


def test_row_passes_filters_rejects_bad_clause() -> None:
    with pytest.raises(ValueError, match="Invalid filter clause"):
        tables.row_passes_filters({"iptm": 0.8}, {"iptm": "roughly 0.8"})
