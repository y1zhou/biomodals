"""Independent single-domain Pareto and diversity examples."""

# ruff: noqa: D103

import numpy as np
import polars as pl
import pytest

from biomodals.workflow.nanobody_humanization.ranking import SCORES, rank_panel


def _panel():
    table = pl.DataFrame({
        "parent_id": ["p"] * 6,
        "candidate_id": ["parent", "a", "b", "c", "d", "missing"],
        "is_parent": [True, False, False, False, False, False],
        SCORES[0]: [None, 0.9, 0.8, 0.8, 0.7, None],
        SCORES[1]: [None, 0.8, 0.9, 0.9, 0.7, 1.0],
        "vh_mutations": [0, 1, 1, 1, 2, 1],
    })
    mutations = pl.DataFrame(
        [
            ("p", "a", 0, "D"),
            ("p", "b", 0, "E"),
            ("p", "c", 1, "D"),
            ("p", "d", 1, "D"),
            ("p", "d", 2, "D"),
            ("p", "missing", 3, "E"),
        ],
        schema=[
            ("parent_id", pl.String),
            ("candidate_id", pl.String),
            ("sequence_index", pl.Int64),
            ("candidate_residue", pl.String),
        ],
        orient="row",
    )
    return table, mutations


def test_pareto_tiers_diversity_and_missing_parent_scores():
    table, mutations = _panel()
    result = rank_panel(table, mutations)
    # c differs at two positions from a; b changes the same site as a.
    assert result["candidate_id"].to_list() == ["parent", "a", "c", "b", "d", "missing"]
    assert result["quality_tier"].to_list() == [None, 1, 1, 1, 2, None]
    assert result["panel_order"].to_list() == [None, 1, 2, 3, 4, None]
    assert rank_panel(table.reverse(), mutations.reverse()).equals(result)
    assert result[SCORES[0]].to_list() == [None, 0.9, 0.8, 0.8, 0.7, None]


def test_finite_negative_scores_are_ranked_without_biological_cutoffs():
    table, mutations = _panel()
    shifted = table.with_columns(pl.col(name) - 2 for name in SCORES)
    assert rank_panel(shifted, mutations)["panel_order"].to_list() == [
        None,
        1,
        2,
        3,
        4,
        None,
    ]


def test_parent_only_and_incomplete_panels_remain_unranked():
    table, mutations = _panel()
    result = rank_panel(
        table.filter(pl.col("candidate_id").is_in(["parent", "missing"])),
        mutations.filter(pl.col("candidate_id") == "missing"),
    )
    assert result["candidate_id"].to_list() == ["parent", "missing"]
    assert result["panel_order"].to_list() == [None, None]


def test_mutation_counts_require_complete_evidence():
    table, mutations = _panel()
    with pytest.raises(ValueError, match="evidence"):
        rank_panel(table, mutations.filter(pl.col("candidate_id") != "d"))


@pytest.mark.parametrize("seed", range(10))
def test_linear_memory_ranking_matches_brute_force_layers_and_distance(seed):
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 4, size=(40, 8))
    states[:, 0] = 1  # Every candidate differs from the parent.
    ids = [f"c{index:02d}" for index in range(len(states))]
    scores = rng.integers(0, 4, size=(len(states), 2)) / 10
    counts = np.count_nonzero(states, axis=1)
    objectives = np.column_stack([scores, -counts])
    remaining = set(range(len(states)))
    selected, layers = [], {}
    layer = 0
    while remaining:
        layer += 1
        front = {
            i
            for i in remaining
            if not any(
                np.all(objectives[j] >= objectives[i])
                and np.any(objectives[j] > objectives[i])
                for j in remaining
            )
        }
        remaining -= front
        while front:
            chosen = min(
                front,
                key=lambda i: (
                    -min(
                        (np.count_nonzero(states[i] != states[j]) for j in selected),
                        default=100,
                    ),
                    -scores[i, 0],
                    -scores[i, 1],
                    counts[i],
                    ids[i],
                ),
            )
            front.remove(chosen)
            selected.append(chosen)
            layers[ids[chosen]] = layer
    table = pl.DataFrame({
        "parent_id": ["p"] * 41,
        "candidate_id": ["parent", *ids],
        "is_parent": [True, *([False] * 40)],
        SCORES[0]: [0.0, *scores[:, 0]],
        SCORES[1]: [0.0, *scores[:, 1]],
        "vh_mutations": [0, *counts],
    })
    mutations = pl.DataFrame(
        [
            ("p", ids[i], j, "ACD"[states[i, j] - 1])
            for i, j in zip(*np.nonzero(states), strict=True)
        ],
        schema=[
            ("parent_id", pl.String),
            ("candidate_id", pl.String),
            ("sequence_index", pl.Int64),
            ("candidate_residue", pl.String),
        ],
        orient="row",
    )
    ranked = rank_panel(table.reverse(), mutations.reverse()).filter(
        ~pl.col("is_parent")
    )
    assert ranked["candidate_id"].to_list() == [ids[i] for i in selected]
    assert ranked["quality_tier"].to_list() == [layers[ids[i]] for i in selected]
