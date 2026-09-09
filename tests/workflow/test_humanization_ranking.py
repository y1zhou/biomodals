"""Deterministic, quality-constrained diversity without model calls."""

from itertools import permutations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from biomodals.workflow.humanization.ranking import OBJECTIVES, rank_panel


def _rank(table, mutations):
    schema = {
        "parent_id": pl.String,
        "candidate_id": pl.String,
        "chain": pl.String,
        "position": pl.Int64,
        "insertion_code": pl.String,
        "candidate_residue": pl.String,
        "region": pl.String,
    }
    return rank_panel(table, pl.DataFrame(mutations, schema=schema))


def _row(name, *, score=0.8, changes=1, parent="a", **overrides):
    return {
        "parent_id": parent,
        "candidate_id": name,
        "is_parent": name == "parent",
        "evaluation_complete": True,
        "cdr_preservation": "preserved",
        "vh_mutations": 0 if name == "parent" else changes,
        "vl_mutations": 0,
        **dict.fromkeys(OBJECTIVES[:-1], score),
        "humatch_vh_best_family_probability": score,
        "humatch_vl_best_family_probability": score,
        **overrides,
    }


def _mutation(name, position, residue="C", *, parent="a", **overrides):
    return {
        "parent_id": parent,
        "candidate_id": name,
        "chain": "vh",
        "position": position,
        "insertion_code": "",
        "candidate_residue": residue,
        "region": "framework",
        **overrides,
    }


def test_diversity_can_promote_tier_two_over_a_near_duplicate():
    """A distinct acceptable alternative can precede a near-copy of the seed."""
    rows = [
        _row("parent"),
        _row("best", score=0.9),
        _row("clone", score=0.9),
        _row("diverse", score=0.85),
    ]
    mutations = [
        _mutation("best", 10),
        _mutation("clone", 10, "D"),
        _mutation("diverse", 20),
    ]
    expected = _rank(pl.DataFrame(rows), mutations)
    assert expected["candidate_id"].to_list() == ["parent", "best", "diverse", "clone"]
    assert expected["quality_tier"].to_list() == [None, 1, 2, 1]
    assert expected["panel_order"].to_list() == [None, 1, 2, 3]
    assert expected.schema["panel_order"] == pl.Int64
    assert expected.schema["quality_tier"] == pl.Int64
    for order in permutations(rows):
        assert_frame_equal(_rank(pl.DataFrame(order), mutations[::-1]), expected)
    assert_frame_equal(_rank(expected, mutations), expected)


def test_missing_changed_and_pairing_regressions_remain_unranked():
    """No dropped rows or zero-valued ranks for review-required candidates."""
    rows = [
        _row("parent"),
        _row("valid"),
        _row("failed", evaluation_complete=False),
        _row("changed", cdr_preservation="changed"),
        _row("unknown", cdr_preservation="unknown"),
        _row("regressed", pabnativ2_pairing_score=0.1),
        _row("humatch_regressed", humatch_pairing_score=0.1),
        _row("missing", pabnativ2_pair_nativeness=None),
        _row("nonfinite", pabnativ2_pair_nativeness=float("nan")),
        _row("missing_family", humatch_vh_best_family_probability=None),
        _row("nonfinite_family", humatch_vl_best_family_probability=float("inf")),
        _row("no_evidence"),
    ]
    mutations = [_mutation(row["candidate_id"], 10) for row in rows[1:-1]]
    table = _rank(pl.DataFrame(rows), mutations)
    assert table.height == len(rows)
    assert table.filter(pl.col("panel_order").is_not_null())[
        "candidate_id"
    ].to_list() == ["valid"]
    assert table["quality_tier"].null_count() == len(rows) - 1


def test_rank_restarts_per_parent_and_raw_scores_are_unchanged():
    """The same candidate label and IMGT coordinate are isolated by parent."""
    rows = [
        _row(name, parent=parent) for parent in ("a", "b") for name in ("parent", "one")
    ]
    mutations = [_mutation("one", 10, parent=parent) for parent in ("a", "b")]
    original = pl.DataFrame(rows)
    table = _rank(original, mutations)
    assert table["panel_order"].to_list() == [None, 1, None, 1]
    assert_frame_equal(table.select(original.columns), original)
    empty = _rank(original.head(0), [])
    assert empty.height == 0
    assert empty.schema["quality_tier"] == pl.Int64


def test_rounding_ties_and_mutation_count_objective():
    """Numerical jitter does not dominate; fewer edits remain a separate objective."""
    rows = [
        _row("parent"),
        _row("a", score=0.90000001),
        _row("b", score=0.90000002),
        _row("c", score=0.9, changes=2),
    ]
    mutations = [_mutation(name, 10) for name in ("a", "b", "c")] + [_mutation("c", 20)]
    table = _rank(pl.DataFrame(rows), mutations)
    assert dict(table.select("candidate_id", "quality_tier").iter_rows()) == {
        "parent": None,
        "a": 1,
        "b": 1,
        "c": 2,
    }


def test_insertion_deletion_and_chain_coordinates_drive_diversity():
    """An insertion code and a light-chain coordinate are distinct positions."""
    rows = [_row("parent"), _row("a"), _row("b"), _row("c", changes=0, vl_mutations=1)]
    mutations = [
        _mutation("a", 10, "-"),
        _mutation("b", 10, insertion_code="A"),
        _mutation("c", 10, chain="vl"),
    ]
    table = _rank(pl.DataFrame(rows), mutations)
    assert table["panel_order"].to_list() == [None, 1, 2, 3]
    with pytest.raises(ValueError, match="Duplicate IMGT"):
        _rank(pl.DataFrame(rows), mutations + mutations[:1])
    with pytest.raises(ValueError, match="unknown candidate"):
        _rank(pl.DataFrame(rows), [_mutation("absent", 10)])


@pytest.mark.parametrize("value", [0.0, None, float("nan")])
def test_unusable_parental_pairing_never_implies_a_pass(value):
    """Zero, absent, and nonfinite baselines cannot pass the relative guardrail."""
    rows = [_row("parent", humatch_pairing_score=value), _row("one")]
    table = _rank(pl.DataFrame(rows), [_mutation("one", 10)])
    assert table["panel_order"].null_count() == 2


def test_ranking_does_not_materialize_python_rows(monkeypatch):
    """Native ranking preserves wide payloads without converting them to objects."""

    def forbid(*args, **kwargs):
        pytest.fail("Ranking materialized Python rows")

    monkeypatch.setattr(pl.DataFrame, "to_dicts", forbid)
    monkeypatch.setattr(pl.DataFrame, "iter_rows", forbid)
    table = pl.DataFrame([_row("parent"), _row("one")]).with_columns(
        pl.lit("A" * 200).alias("vh"),
        pl.lit("C" * 200).alias("vl"),
    )
    ranked = _rank(table, [_mutation("one", 10)])
    assert ranked["panel_order"].to_list() == [None, 1]
    assert ranked["vh"].equals(table["vh"])


def test_sequential_fronts_and_exact_pairing_boundary():
    """All Pareto layers terminate; the ten-percent boundary itself is admitted."""
    rows = [_row("parent")] + [
        _row("a", score=0.95),
        _row("b", score=0.90),
        _row("c", score=0.85),
        _row("d", score=0.72),
    ]
    mutations = [_mutation(name, 10) for name in ("a", "b", "c", "d")]
    table = _rank(pl.DataFrame(rows), mutations)
    assert table["quality_tier"].to_list() == [None, 1, 2, 3, 4]
    assert table["panel_order"].to_list() == [None, 1, 2, 3, 4]


def test_mean_family_probability_is_one_independent_objective():
    """Average both chains, retain tradeoffs, and do not add a family guardrail."""
    rows = [
        _row(
            "parent",
            humatch_vh_best_family_probability=1.0,
            humatch_vl_best_family_probability=1.0,
        ),
        _row(
            "balanced",
            humatch_vh_best_family_probability=0.6,
            humatch_vl_best_family_probability=0.6,
        ),
        _row(
            "asymmetric",
            humatch_vh_best_family_probability=0.9,
            humatch_vl_best_family_probability=0.3,
        ),
        _row(
            "lower",
            humatch_vh_best_family_probability=0.9,
            humatch_vl_best_family_probability=0.1,
        ),
        _row(
            "tradeoff",
            pabnativ2_pair_nativeness=0.9,
            humatch_vh_best_family_probability=0.1,
            humatch_vl_best_family_probability=0.1,
        ),
    ]
    table = _rank(
        pl.DataFrame(rows), [_mutation(r["candidate_id"], 10) for r in rows[1:]]
    )
    assert dict(table.select("candidate_id", "quality_tier").iter_rows()) == {
        "parent": None,
        "balanced": 1,
        "asymmetric": 1,
        "lower": 2,
        "tradeoff": 1,
    }


def test_family_mean_breaks_seed_ties_and_rounds_after_averaging():
    """Use raw-chain arithmetic before six-decimal dominance and tie breaking."""
    rows = [
        _row("parent"),
        _row(
            "a",
            humatch_vh_best_family_probability=0.50000049,
            humatch_vl_best_family_probability=0.50000049,
        ),
        _row(
            "b",
            humatch_vh_best_family_probability=0.50000051,
            humatch_vl_best_family_probability=0.50000001,
        ),
        _row(
            "z",
            humatch_vh_best_family_probability=0.8,
            humatch_vl_best_family_probability=0.9,
        ),
    ]
    table = _rank(
        pl.DataFrame(rows), [_mutation(r["candidate_id"], 10) for r in rows[1:]]
    )
    assert table["candidate_id"].to_list() == ["parent", "z", "a", "b"]
    assert table["quality_tier"].to_list() == [None, 1, 2, 2]
