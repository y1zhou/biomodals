"""Exact small-space oracle and enormous-space sampling without enumeration."""

from collections import Counter
from itertools import product

import pytest

from biomodals.app.design.mutation_ridge.inputs import build_dataset, parse_mutations
from biomodals.workflow.protein_optimization.exploration import ExplorationSpace
from biomodals.workflow.protein_optimization.settings import (
    OptimizationSettings,
    PositionChoices,
)


def test_sampling_matches_exhaustive_oracle_and_keeps_position_constraints():
    """Every and only allowed novel variant appears when budget covers the space."""
    data = build_dataset(
        b"mutations,label\nA:A1C,1\nA:A2V,2\nB:C1M,3\n", ">A\nAA\n>B\nC\n"
    )
    settings = OptimizationSettings(
        mode="exploration",
        candidate_budget=100,
        max_mutations=2,
        max_new_mutations=1,
        positions=(
            PositionChoices(chain_id="A", position=1, amino_acids="CG"),
            PositionChoices(chain_id="A", position=2, amino_acids="MV"),
            PositionChoices(chain_id="B", position=1, amino_acids="AF"),
        ),
    )
    space = ExplorationSpace.build(data, settings)
    known = set(data.vocabulary)
    groups = [[None, *[m for m, _ in group]] for group in space.groups]
    oracle = {
        tuple(m for m in row if m is not None)
        for row in product(*groups)
        if 1 <= sum(m is not None for m in row) <= 2
        and sum(m is not None and m not in known for m in row) == 1
    }
    assert set(space.sample(100, seed=2)) == oracle
    assert len(oracle) == sum(space.counts)
    assert len(space.sample(100, seed=2)) == len(oracle)
    sampled = space.sample(7, seed=15)
    assert len(sampled) == len(set(sampled)) == 7
    assert space.sample(7, seed=15) == sampled
    assert set(sampled) <= oracle
    counts = Counter(map(len, sampled))
    assert abs(counts[1] - counts[2]) <= 1


def test_default_cm_restriction_only_affects_proposals_not_training_or_parent():
    """Known C/M replacements are blocked in exploration backgrounds by default."""
    data = build_dataset(b"mutations,label\nA:C1M,1\nA:M2C,2\n", ">A\nCMA\n")
    space = ExplorationSpace.build(
        data, OptimizationSettings(mode="exploration", candidate_budget=100)
    )
    assert data.measurements["label"].to_list() == [1, 2]
    assert data.parents["A"] == "CMA"
    assert all(m.replacement not in "CM" for v in space.sample(100, seed=0) for m in v)
    override = ExplorationSpace.build(
        data,
        OptimizationSettings(
            mode="exploration",
            candidate_budget=100,
            positions=(
                PositionChoices(chain_id="A", position=1, amino_acids="MF"),
                PositionChoices(chain_id="A", position=2, amino_acids="CG"),
            ),
        ),
    )
    assert parse_mutations("A:C1M,A:M2G") in override.sample(100, seed=0)


def test_strata_redistribute_exhausted_capacity_and_support_huge_integer_spaces():
    """Floyd sampling accepts sizes above platform len(range) without allocation."""
    data = build_dataset(b"mutations,label\n,0\nA:A1V,1\n", ">A\n" + "A" * 1000)
    settings = OptimizationSettings(
        mode="exploration",
        candidate_budget=23,
        max_mutations=10,
        max_new_mutations=10,
        positions=tuple(
            PositionChoices(chain_id="A", position=p, amino_acids="G")
            for p in range(1, 1001)
        ),
    )
    space = ExplorationSpace.build(data, settings)
    assert sum(space.counts) > 2**63
    rows = space.sample(23, seed=0)
    assert len(rows) == len(set(rows)) == 23
    assert set(map(len, rows)) == set(range(1, 11))
    tiny = ExplorationSpace.build(
        data,
        settings.model_copy(
            update={
                "positions": (
                    PositionChoices(chain_id="A", position=1, amino_acids="VG"),
                    PositionChoices(chain_id="A", position=2, amino_acids="VF"),
                ),
                "max_new_mutations": 1,
            }
        ),
    )
    assert len(tiny.sample(23, seed=0)) == sum(tiny.counts)


def test_new_position_and_extra_chain_are_valid_but_masks_must_match_parent():
    """New chains are explicit FASTA entries and invalid design sites fail early."""
    data = build_dataset(b"mutations,label\nA:A1V,1\n", ">A\nA\n>Partner\nMC\n")
    settings = OptimizationSettings(
        mode="exploration",
        candidate_budget=2,
        positions=(PositionChoices(chain_id="Partner", position=2, amino_acids="CA"),),
    )
    assert ExplorationSpace.build(data, settings).sample(2, seed=0) == (
        parse_mutations("Partner:C2A"),
    )
    for positions in (
        (PositionChoices(chain_id="missing", position=1),),
        (PositionChoices(chain_id="A", position=2),),
        (PositionChoices(chain_id="A", position=1),) * 2,
    ):
        with pytest.raises(ValueError):
            ExplorationSpace.build(
                data, settings.model_copy(update={"positions": positions})
            )


def test_freezing_all_sites_is_an_explicit_empty_space():
    """Empty allowed choices never silently restore a default residue set."""
    data = build_dataset(b"mutations,label\nA:A1V,1\n", ">A\nA\n")
    settings = OptimizationSettings(
        mode="exploration",
        candidate_budget=1,
        positions=(PositionChoices(chain_id="A", position=1, amino_acids=""),),
    )
    space = ExplorationSpace.build(data, settings)
    assert space.counts == (0,)
    assert space.sample(1, seed=0) == ()
