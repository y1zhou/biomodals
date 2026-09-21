"""Frozen gene evidence must not affect the humanization ranking contract."""

import hashlib

import orjson
import polars as pl
import pytest

from biomodals.workflow.humanization.annotation import annotate_humanization_candidate
from biomodals.workflow.humanization.contracts import AntibodyPair, CandidateAnnotation
from biomodals.workflow.humanization.germlines import (
    GENE_COLUMNS,
    GERMLINE_SCHEMA,
    add_gene_columns,
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
    """CPU Task publishes both outcomes; finalization owns partial-result policy."""
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
    assert result.status == "succeeded"
    outputs = {
        output.name: orjson.loads(output.storage.data) for output in result.outputs
    }
    assert outputs["annotation"]["cdr_preservation"] == "unknown"
    assert outputs["germlines"][0]["v_gene"] == "IGHV1-2"
