"""Budgeted, edit-count-balanced exploration of native substitution choices."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass

import orjson

SAMPLING_VERSION = "balanced-edit-count-v1"
MAX_EXPLORATION_CANDIDATES = 5000


def parent_seed(
    root_seed: int, parent_id: str, sequence: str, protected: Sequence[int]
) -> int:
    """Use a private stream unaffected by batch order, run IDs or HuDiff's RNG."""
    content = orjson.dumps([
        SAMPLING_VERSION,
        root_seed,
        parent_id,
        sequence,
        list(protected),
    ])
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "big")


@dataclass(frozen=True)
class ExplorationSample:
    """Distinct nonparent aligned sequences; possible_count excludes the parent."""

    sequences: tuple[str, ...]
    possible_count: int
    counts_by_edits: tuple[int, ...]


def sample_combinations(
    parent: str, choices: Sequence[str], *, budget: int, seed: int
) -> ExplorationSample:
    """Balance edit counts, then uniformly sample without replacement within each.

    Suffix coefficients count exact-k-edit combinations. Integer rank decoding
    avoids constructing the Cartesian product, even when its size exceeds int64.
    Quotas differ by at most one among nonexhausted groups; a seeded group order
    resolves remainders, including budgets smaller than the number of groups.
    """
    if not 1 <= budget <= MAX_EXPLORATION_CANDIDATES:
        raise ValueError("Exploration budget is outside the supported range")
    if len(parent) != len(choices) or any(
        aa not in values for aa, values in zip(parent, choices, strict=True)
    ):
        raise ValueError("Every position must retain its parental residue")
    positions, alternatives = [], []
    for index, (aa, values) in enumerate(zip(parent, choices, strict=True)):
        other = "".join(sorted(set(values) - {aa}))
        if other:
            positions.append(index)
            alternatives.append(other)
    width = len(positions)
    suffix = [[0] * (width + 1) for _ in range(width + 1)]
    suffix[width][0] = 1
    for index in range(width - 1, -1, -1):
        suffix[index][0] = 1
        for edits in range(1, width - index + 1):
            suffix[index][edits] = (
                suffix[index + 1][edits]
                + len(alternatives[index]) * suffix[index + 1][edits - 1]
            )
    possible = sum(suffix[0][1:])
    rng = random.Random(seed)  # noqa: S311 - reproducible scientific sampling.
    groups = list(range(1, width + 1))
    rng.shuffle(groups)
    quotas = [0] * (width + 1)
    remaining = min(budget, possible)
    while remaining:
        for edits in groups:
            if quotas[edits] < suffix[0][edits]:
                quotas[edits] += 1
                remaining -= 1
                if not remaining:
                    break
    sequences = []
    for edits in range(1, width + 1):
        size, count = suffix[0][edits], quotas[edits]
        # Floyd sampling works with arbitrary-size integers and never retries
        # duplicate draws. random.sample(range(...)) overflows on huge populations.
        ranks: set[int] = set()
        for upper in range(size - count, size):
            rank = rng.randrange(upper + 1)
            ranks.add(upper if rank in ranks else rank)
        for rank in sorted(ranks):
            residues, left = list(parent), edits
            for index, position in enumerate(positions):
                unchanged = suffix[index + 1][left]
                if rank < unchanged:
                    continue
                rank -= unchanged
                block = suffix[index + 1][left - 1]
                replacement, rank = divmod(rank, block)
                residues[position] = alternatives[index][replacement]
                left -= 1
            sequences.append("".join(residues))
    return ExplorationSample(tuple(sequences), possible, tuple(quotas[1:]))
