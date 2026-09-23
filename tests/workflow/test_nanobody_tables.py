"""Parent-scoped exact union, mutation integrity and nullable baseline deltas."""

# ruff: noqa: D103

import polars as pl
import pytest

from biomodals.workflow.nanobody_humanization.preparation import PreparedVH
from biomodals.workflow.nanobody_humanization.tables import (
    GENERATION_SCHEMA,
    MUTATION_SCHEMA,
    add_scores,
    candidate_union,
    mutation_table,
)


def _parent(identifier="p"):
    # A short synthetic domain is sufficient for these pure table operations;
    # biological preparation/recognition is tested separately with native inputs.
    return PreparedVH(
        id=identifier,
        original_sequence="ACDE",
        sequence="ACDE",
        original_indices=(0, 1, 2, 3),
        imputation_references=((), (), (), ()),
        v_reference="reference_v",
        j_reference="reference_j",
        imgt_positions=("1", "2", "3", "4"),
        aho_positions=(1, 2, 3, 4),
        protected_indices=(1,),
    )


def test_union_keeps_noop_and_duplicate_provenance_without_cross_parent_merge():
    generation = pl.DataFrame(
        [
            ("p", "hudiff_nb", 1, 0, "ACDF", None),
            ("p", "abnativ2_vhh", 1, None, "ACDF", None),
            ("p", "hudiff_nb", 2, 0, "ACDE", None),
            ("p", "hudiff_nb", 3, 0, None, "Invalid sample"),
            ("q", "abnativ2_vhh", 1, None, "ACDF", None),
        ],
        schema=GENERATION_SCHEMA,
        orient="row",
    )
    parents = [_parent(), _parent("q")]
    table, origins = candidate_union(parents, generation)
    assert table["parent_id"].to_list() == ["p", "p", "q", "q"]
    assert table["vh"].to_list() == ["ACDE", "ACDF", "ACDE", "ACDF"]
    assert table["generating_methods"].to_list() == [
        None,
        "abnativ2_vhh;hudiff_nb",
        None,
        "abnativ2_vhh",
    ]
    assert table["candidate_id"].n_unique() == 4
    assert origins["candidate_id"][0] == origins["candidate_id"][1]
    assert origins["candidate_id"][2] == table["candidate_id"][0]
    assert origins["error"][3] == "Invalid sample"
    assert candidate_union(parents, generation.reverse())[0].equals(table)
    mutations = mutation_table(table, parents)
    assert mutations.schema == pl.Schema(MUTATION_SCHEMA)
    assert mutations["sequence_index"].to_list() == [3, 3]
    assert mutations["imgt_position"].to_list() == ["4", "4"]
    assert mutations["parent_residue"].to_list() == ["E", "E"]
    assert mutations["candidate_residue"].to_list() == ["F", "F"]


@pytest.mark.parametrize("sequence", ["AADE", "ACDEE", "ACDX"])
def test_mutation_evidence_rejects_invalid_or_protected_changes(sequence):
    generation = pl.DataFrame(
        [("p", "hudiff_nb", 1, 0, sequence, None)],
        schema=GENERATION_SCHEMA,
        orient="row",
    )
    table, _ = candidate_union([_parent()], generation)
    with pytest.raises(ValueError, match="protected|length or alphabet"):
        mutation_table(table, [_parent()])


def test_complete_candidate_scores_do_not_require_available_parent_scores():
    generation = pl.DataFrame(
        [("p", "hudiff_nb", 1, 0, "ACDF", None)], schema=GENERATION_SCHEMA, orient="row"
    )
    parents = [_parent()]
    candidates, _ = candidate_union(parents, generation)
    candidate_id = candidates.filter(~pl.col("is_parent"))["candidate_id"][0]
    score = pl.DataFrame(
        {"candidate_id": [candidate_id], "score": [-0.5], "error": [None]},
        schema={"candidate_id": pl.String, "score": pl.Float64, "error": pl.String},
    )
    table = add_scores(
        candidates, mutation_table(candidates, parents), {"VH2": score, "VHH2": score}
    )
    assert table["vh_mutations"].to_list() == [0, 1]
    assert table["evaluation_complete"].to_list() == [False, True]
    assert table["abnativ2_vh_nativeness"].to_list() == [None, -0.5]
    assert table["abnativ2_vh_nativeness_delta"].to_list() == [None, None]
    assert table["abnativ2_vhh_nativeness_error"].to_list() == [
        "Evaluation unavailable",
        None,
    ]


def test_parent_only_no_generation_retains_typed_empty_mutations():
    candidates, origins = candidate_union(
        [_parent()], pl.DataFrame(schema=GENERATION_SCHEMA)
    )
    mutations = mutation_table(candidates, [_parent()])
    assert mutations.schema == pl.Schema(MUTATION_SCHEMA) and mutations.height == 0
    table = add_scores(candidates, mutations, {})
    assert table["is_parent"].to_list() == [True]
    assert origins.height == 0
    assert table["evaluation_complete"].to_list() == [False]
