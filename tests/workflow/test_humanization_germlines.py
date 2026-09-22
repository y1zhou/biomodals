"""Frozen gene evidence must not affect the humanization ranking contract."""

import hashlib

import orjson
import polars as pl
import pytest

from biomodals.schema import AppRunResult
from biomodals.workflow.humanization.annotation import annotate_humanization_candidate
from biomodals.workflow.humanization.contracts import AntibodyPair, CandidateAnnotation
from biomodals.workflow.humanization.germlines import (
    GENE_COLUMNS,
    GERMLINE_SCHEMA,
    SEQUENCE_COLUMNS,
    add_gene_columns,
    add_sequence_columns,
    candidate_germlines,
    complete_germline_table,
)
from biomodals.workflow.humanization.tables import candidate_union, selection_table

VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGGYFDYWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"


def test_gene_evidence_bound_to_sequences_and_joined_after_rank():
    """Native species/allele ties survive one typed table and scalar CSV labels."""
    candidates = candidate_union([AntibodyPair(id="parent", vh=VH, vl=VL)], [])
    rows = candidate_germlines(candidates[0])
    assert rows[0]["sequence_sha256"] == hashlib.sha256(VH.encode()).hexdigest()
    evidence = pl.DataFrame(rows, schema=GERMLINE_SCHEMA)
    complete = complete_germline_table(candidates, [evidence])
    original = selection_table(candidates, [], []).with_columns(
        pl.lit(5).alias("panel_order")
    )
    annotated = add_gene_columns(original, complete)
    assert annotated.select(original.columns).equals(original)
    assert set(GENE_COLUMNS) <= set(annotated.columns)
    assert annotated["vh_v_gene"][0] == "IGHV1-2"
    with pytest.raises(pl.exceptions.ComputeError):
        complete_germline_table(candidates, [evidence, evidence])
    with pytest.raises(ValueError, match="identities"):
        complete_germline_table(
            candidates,
            [evidence.with_columns(pl.lit("wrong").alias("sequence_sha256"))],
        )
    absent = complete_germline_table(candidates, [])
    assert absent["error"].to_list() == ["Germline annotation unavailable"] * 2
    assert add_gene_columns(original, absent)["evaluation_complete"].equals(
        original["evaluation_complete"]
    )


def test_imgt_failure_does_not_discard_independent_germlines(monkeypatch):
    """One CPU call publishes independent Task outcomes, including on recovery."""
    parent = AntibodyPair(id="parent", vh=VH, vl=VL)
    candidate = candidate_union([parent], [])[0]
    monkeypatch.setattr(
        "biomodals.workflow.humanization.annotation.annotate_candidate",
        lambda *args: (
            CandidateAnnotation(
                parent_id=parent.id,
                candidate_id=candidate.candidate_id,
                cdr_preservation="unknown",
                error="ANARCI unavailable",
            ),
            [],
        ),
    )
    result = annotate_humanization_candidate(
        parent.model_dump(), candidate.model_dump()
    )
    assert result["annotation"]["status"] == "failed"
    assert result["germline"]["status"] == "succeeded"
    outputs = {
        output.name: orjson.loads(output.storage.data)
        for output in AppRunResult.model_validate(result["germline"]).outputs
    }
    assert outputs["germlines"][0]["v_gene"] == "IGHV1-2"
    monkeypatch.setattr(
        "biomodals.workflow.humanization.annotation.candidate_germlines",
        lambda *_: pytest.fail("Recovery must reuse successful germline evidence"),
    )
    assert set(
        annotate_humanization_candidate(
            parent.model_dump(), candidate.model_dump(), operations=("annotation",)
        )
    ) == {"annotation"}


def test_sequence_columns_preserve_rank_and_compute_each_unique_input_once(monkeypatch):
    """Native pIs join by exact sequence without affecting candidate ranking."""
    from biomodals.helper.antibody import combined_pi, sequence_pi

    candidates = candidate_union(
        [AntibodyPair(id=parent, vh=VH, vl=VL) for parent in ("one", "two")], []
    )
    evidence = pl.DataFrame(
        [row for candidate in candidates for row in candidate_germlines(candidate)],
        schema=GERMLINE_SCHEMA,
    )
    chains, pairs = [], []

    def chain_pi(sequence):
        chains.append(sequence)
        return sequence_pi(sequence)

    def pair_pi(vh, vl):
        pairs.append((vh, vl))
        return combined_pi(vh, vl)

    monkeypatch.setattr(
        "biomodals.workflow.humanization.germlines.sequence_pi", chain_pi
    )
    monkeypatch.setattr(
        "biomodals.workflow.humanization.germlines.combined_pi", pair_pi
    )
    original = selection_table(candidates, [], []).with_columns(
        pl.lit(3).alias("quality_tier")
    )
    table = add_sequence_columns(original, evidence)
    assert table.select(original.columns).equals(original)
    at = table.columns.index("vh") + 1
    assert table.columns[at : at + 7] == list(SEQUENCE_COLUMNS)
    assert set(chains) == {VH, VL} and len(chains) == 2
    assert pairs == [(VH, VL)]
    assert table["vh_pi"].to_list() == [sequence_pi(VH)] * 2
    assert table["vl_pi"].to_list() == [sequence_pi(VL)] * 2
    assert table["vh_vl_pi"].to_list() == [combined_pi(VH, VL)] * 2
    assert table["vh_v_gene"].to_list() == ["IGHV1-2"] * 2
    empty = add_sequence_columns(original.clear(), evidence.clear())
    assert empty.schema == table.schema and empty.height == 0
