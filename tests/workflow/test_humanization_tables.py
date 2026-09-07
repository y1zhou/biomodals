"""Local candidate/selection contracts; no model or Modal invocation."""

import tarfile
from io import BytesIO

import polars as pl
import pytest
import zstandard
from pydantic import ValidationError

from biomodals.workflow.humanization.annotation import annotate_candidate
from biomodals.workflow.humanization.artifacts import archive_members
from biomodals.workflow.humanization.contracts import (
    AntibodyPair,
    CandidateAnnotation,
    CandidateEvaluation,
    CandidateOrigin,
)
from biomodals.workflow.humanization.tables import (
    candidate_union,
    parse_parents,
    selection_table,
)


def test_union_preserves_pairs_parent_membership_and_all_origins():
    """Union is order-independent and never merges parental contexts."""
    parents = [AntibodyPair(id=name, vh="ACD", vl="EFG") for name in ("a", "b")]
    first = CandidateOrigin(method="sapiens", source_id="a")
    second = CandidateOrigin(
        method="hudiff_ab", source_id="a_1", attempt_index=1, seed=42
    )
    generated = [
        ("a", "ACD", "EFG", first),
        ("a", "ACH", "EFG", first),
        ("a", "ACH", "EFG", second),
    ]
    union = candidate_union(parents, generated)
    assert union == candidate_union(parents[::-1], generated[::-1])
    assert len(union) == 3
    assert len({candidate.candidate_id for candidate in union}) == 3
    variant = next(candidate for candidate in union if not candidate.is_parent)
    assert set(variant.origins) == {first, second}
    assert next(
        candidate
        for candidate in union
        if candidate.parent_id == "a" and candidate.is_parent
    ).origins == (first,)
    with pytest.raises(ValueError, match="Unknown parent"):
        candidate_union(parents, [("other", "ACD", "EFG", first)])


@pytest.mark.parametrize(
    "content",
    [
        b"id,vh,vl\na,ACD,\n",
        b"id,vh,vl\na,ACX,EFG\n",
        b"id,vh,vl\na b,ACD,EFG\na_b,ACD,EFG\n",
        b"vh,vl,id\nACD,EFG,a\n",
    ],
)
def test_invalid_request_rejected(content):
    """Reject malformed pairs before remote work."""
    with pytest.raises(ValueError):
        parse_parents(content)


def test_common_parser_does_not_apply_sapiens_specific_length_limit():
    """A method-specific limit is not a common request error."""
    pair = parse_parents(f"id,vh,vl\na,{'A' * 150},{'C' * 130}\n".encode())[0]
    assert len(pair.vh) == 150


def test_sortable_summary_nulls_deltas_and_no_silent_loss():
    """Retain all rows and typed nulls alongside comparable deltas."""
    parent = AntibodyPair(id="a", vh="ACD", vl="EFG")
    origin = CandidateOrigin(method="sapiens", source_id="a")
    union = candidate_union([parent], [("a", "ACH", "EFG", origin)])
    evaluations = [
        CandidateEvaluation(
            parent_id="a",
            candidate_id=candidate.candidate_id,
            evaluator="sapiens",
            status="succeeded",
            scores={
                "vh_mean_probability": 0.2 if candidate.is_parent else 0.8,
                "vl_mean_probability": 0.4,
            },
        )
        for candidate in union
    ]
    annotations = [
        CandidateAnnotation(
            parent_id="a",
            candidate_id=candidate.candidate_id,
            cdr_preservation="preserved",
            cdr_mutations=0,
            vh_mutations=0 if candidate.is_parent else 1,
            vl_mutations=0,
        )
        for candidate in union
    ]
    table = selection_table(union, evaluations, annotations)
    assert table.height == 2
    assert table.schema["sapiens_vh_mean_probability"] == pl.Float64
    assert table.schema["humatch_pairing_score"] == pl.Float64
    assert table["humatch_pairing_score"].null_count() == 2
    assert table["humatch_error"].to_list() == ["Evaluation unavailable"] * 2
    assert table["sapiens_error"].to_list() == [None, None]
    assert table["generating_methods"].to_list() == [None, "sapiens"]
    assert table["evaluation_complete"].to_list() == [False, False]
    assert table["sapiens_vh_mean_probability_delta"].to_list() == pytest.approx([
        0,
        0.6,
    ])
    assert (
        table.sort("sapiens_vh_mean_probability", descending=True)["is_parent"][0]
        is False
    )
    # CSV stays scalar and spreadsheet-readable, including when all scores are absent.
    assert "humatch_pairing_score" in table.write_csv()
    missing = selection_table(union, [], [])
    assert missing["cdr_preservation"].to_list() == ["unknown", "unknown"]
    assert missing["sapiens_vh_mean_probability_delta"].null_count() == 2
    with pytest.raises(ValueError, match="duplicate evaluation"):
        selection_table(union, evaluations + evaluations, annotations)


def test_humatch_reference_switch_cannot_be_reported_as_a_delta():
    """A candidate must not silently change its comparison family."""
    union = candidate_union(
        [AntibodyPair(id="a", vh="ACD", vl="EFG")],
        [
            ("a", "ACH", "EFG", CandidateOrigin(method="hudiff_ab", source_id="1")),
        ],
    )
    evaluations = [
        CandidateEvaluation(
            parent_id="a",
            candidate_id=candidate.candidate_id,
            evaluator="humatch",
            status="succeeded",
            scores={"vh_target_probability": 0.8},
            labels={"vh_target_family": "hv1" if candidate.is_parent else "hv3"},
        )
        for candidate in union
    ]
    with pytest.raises(ValueError, match="identical target families"):
        selection_table(union, evaluations, [])


def test_invalid_status_and_nonfinite_scores_are_rejected():
    """Nonfinite scores and evidence-free statuses cannot enter artifacts."""
    with pytest.raises(ValidationError):
        CandidateEvaluation(
            parent_id="a",
            candidate_id="b",
            evaluator="sapiens",
            status="succeeded",
            scores={"vh_mean_probability": float("nan")},
        )
    with pytest.raises(ValidationError):
        CandidateEvaluation(
            parent_id="a", candidate_id="b", evaluator="sapiens", status="failed"
        )
    with pytest.raises(ValidationError):
        CandidateAnnotation(parent_id="a", candidate_id="b", cdr_preservation="unknown")


def test_imgt_check_reports_framework_cdr_and_insertion_changes():
    """Compare numbered positions rather than shifting raw sequence indexes."""
    parent = AntibodyPair(id="a", vh="ACD", vl="EFG")
    union = candidate_union(
        [parent],
        [
            ("a", "ACHK", "EFG", CandidateOrigin(method="hudiff_ab", source_id="1")),
        ],
    )
    candidate = next(row for row in union if not row.is_parent)

    def number(sequence, *, scheme):
        assert scheme == "imgt"
        positions = [(26, " "), (27, " "), (39, " ")]
        if sequence == "ACHK":
            positions = [(26, " "), (27, " "), (27, "A"), (39, " ")]
        return list(
            zip(positions, sequence, strict=True)
        ), "L" if sequence == "EFG" else "H"

    annotation, mutations = annotate_candidate(parent, candidate, number=number)
    assert annotation.cdr_preservation == "changed"
    assert annotation.vh_mutations == 2
    assert annotation.cdr_mutations == 1
    assert {row["change_type"] for row in mutations} == {"insertion", "substitution"}
    assert candidate.vh == "ACHK"

    unknown, partial = annotate_candidate(
        parent, candidate, number=lambda *args, **kwargs: (None, None)
    )
    assert unknown.cdr_preservation == "unknown"
    assert unknown.error
    assert unknown.vh_mutations is None
    assert partial == []


@pytest.mark.parametrize(
    "names", [("../escape.csv",), ("/absolute.csv",), ("a/scores.csv", "b/scores.csv")]
)
def test_native_archive_rejects_unsafe_or_ambiguous_members(names):
    """Never extract paths or silently accept two files with the same logical name."""
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in names:
            entry = tarfile.TarInfo(name)
            entry.size = 1
            archive.addfile(entry, BytesIO(b"x"))
    with pytest.raises(ValueError):
        archive_members(zstandard.ZstdCompressor().compress(buffer.getvalue()))
