"""Frozen single-domain annotations use the existing full-sequence science."""

# ruff: noqa: D103

import polars as pl
import pytest

from biomodals.helper.antibody import sequence_pi
from biomodals.helper.antibody_tables import GERMLINE_TABLE_SCHEMA
from biomodals.workflow.nanobody_humanization import annotation

OZORALIZUMAB = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSV"
    "KGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"
)


def test_native_annotation_deduplicates_work_but_preserves_parent_identity(monkeypatch):
    calls = []
    native = annotation.assign_germlines

    def assign(sequence):
        calls.append(sequence)
        return native(sequence)

    monkeypatch.setattr(annotation, "assign_germlines", assign)
    candidates = pl.DataFrame({
        "parent_id": ["p", "q"],
        "candidate_id": ["one", "two"],
        "vh": [OZORALIZUMAB, OZORALIZUMAB],
    })
    summary, germlines = annotation.annotation_tables(candidates)
    assert calls == [OZORALIZUMAB]
    assert summary["candidate_id"].to_list() == ["one", "two"]
    assert summary["vh_pI"].to_list() == pytest.approx([sequence_pi(OZORALIZUMAB)] * 2)
    assert summary["vh_v_gene"].null_count() == 0
    assert summary["vh_j_gene"].null_count() == 0
    assert germlines.schema == pl.Schema(GERMLINE_TABLE_SCHEMA)
    assert germlines["chain"].to_list() == ["vh", "vh"]
    assert germlines["sequence_sha256"].n_unique() == 1
    assert germlines["v"].list.len().min() > 0
    assert germlines["j"].list.len().min() > 0
