"""Exact small-space oracles and bounded huge-space sampling."""

# ruff: noqa: D103

from collections import Counter
from itertools import product

from biomodals.app.design.abnativ2_vhh.sampling import parent_seed, sample_combinations


def test_complete_space_matches_cartesian_oracle():
    parent, choices = "AAA-A", ["ACD", "AG", "A", "-", "AVW"]
    expected = {"".join(row) for row in product(*choices)} - {parent}
    result = sample_combinations(parent, choices, budget=100, seed=3)
    assert set(result.sequences) == expected
    assert result.possible_count == len(expected) == 17
    assert len(result.sequences) == len(expected)
    assert all(row[2:4] == "A-" for row in result.sequences)


def test_balanced_quotas_redistribute_small_groups_and_are_reproducible():
    result = sample_combinations("AAAA", ["AC"] * 4, budget=13, seed=7)
    counts = Counter(sum(a != "A" for a in row) for row in result.sequences)
    assert counts == {1: 4, 2: 4, 3: 4, 4: 1}
    assert result == sample_combinations("AAAA", ["CA"] * 4, budget=13, seed=7)
    assert len(set(result.sequences)) == 13
    assert "AAAA" not in result.sequences


def test_huge_space_and_small_budget_do_not_enumerate():
    choices = ["ACDEFGHIKL"] * 100
    result = sample_combinations("A" * 100, choices, budget=20, seed=12)
    assert result.possible_count == 10**100 - 1
    assert len(set(result.sequences)) == 20
    assert sorted(result.counts_by_edits) == [0] * 80 + [1] * 20
    assert result != sample_combinations("A" * 100, choices, budget=20, seed=13)


def test_empty_mutational_space_is_successful_noop():
    result = sample_combinations("AAA", ["A"] * 3, budget=1000, seed=0)
    assert result.sequences == ()
    assert result.possible_count == 0


def test_seed_binds_parent_and_policy_not_batch_position():
    seed = parent_seed(0, "parent", "ACDE", [1])
    assert seed == parent_seed(0, "parent", "ACDE", [1])
    assert (
        len({
            seed,
            parent_seed(1, "parent", "ACDE", [1]),
            parent_seed(0, "other", "ACDE", [1]),
            parent_seed(0, "parent", "ACDE", [2]),
        })
        == 4
    )
