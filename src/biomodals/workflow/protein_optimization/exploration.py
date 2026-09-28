"""Count and sample novel substitution spaces without enumerating a huge pool."""

from __future__ import annotations

import random
from dataclasses import dataclass

from biomodals.app.design.mutation_ridge.inputs import (
    MutationDataset,
    Substitution,
    Variant,
)
from biomodals.workflow.protein_optimization.settings import (
    OptimizationSettings,
    PositionChoices,
)


def resolved_positions(
    dataset: MutationDataset, settings: OptimizationSettings
) -> tuple[PositionChoices, ...]:
    """Use measured sites by default and validate explicitly supplied replacements."""
    settings.validate_mode_budget()
    positions = settings.positions
    if positions is None:
        positions = tuple(
            PositionChoices(chain_id=chain, position=position)
            for chain, position in sorted({m.site for m in dataset.vocabulary})
        )
    seen = set()
    for site in positions:
        key = (site.chain_id, site.position)
        if key in seen:
            raise ValueError("Specify each editable chain position only once")
        seen.add(key)
        sequence = dataset.parents.get(site.chain_id)
        if sequence is None or site.position > len(sequence):
            raise ValueError(
                "An editable position is outside the supplied parental chains"
            )
    return tuple(sorted(positions, key=lambda p: (p.chain_id, p.position)))


@dataclass
class ExplorationSpace:
    """Suffix counts support exact integer unranking, including enormous spaces."""

    groups: tuple[tuple[tuple[Substitution, bool], ...], ...]
    suffix: list[dict[tuple[int, int], int]]
    max_mutations: int
    max_new_mutations: int

    @classmethod
    def build(
        cls, dataset: MutationDataset, settings: OptimizationSettings
    ) -> ExplorationSpace:
        """Construct feasible counts by total/new substitutions, independent of labels."""
        known = set(dataset.vocabulary)
        groups = []
        for site in resolved_positions(dataset, settings):
            original = dataset.parents[site.chain_id][site.position - 1]
            alternatives = tuple(
                (mutation, mutation not in known)
                for replacement in site.amino_acids
                if replacement != original
                for mutation in (
                    Substitution(site.chain_id, site.position, original, replacement),
                )
            )
            if alternatives:
                groups.append(alternatives)
        n = min(settings.max_mutations, len(groups))
        max_new = min(settings.max_new_mutations, n)
        suffix: list[dict[tuple[int, int], int]] = [{} for _ in range(len(groups) + 1)]
        suffix[-1] = {(0, 0): 1}
        for i in range(len(groups) - 1, -1, -1):
            new_count = sum(is_new for _, is_new in groups[i])
            known_count = len(groups[i]) - new_count
            current = dict(suffix[i + 1])
            for (total, novel), count in suffix[i + 1].items():
                if total >= n:
                    continue
                if known_count:
                    key = (total + 1, novel)
                    current[key] = current.get(key, 0) + count * known_count
                if new_count and novel < max_new:
                    key = (total + 1, novel + 1)
                    current[key] = current.get(key, 0) + count * new_count
            suffix[i] = current
        return cls(tuple(groups), suffix, n, max_new)

    @property
    def counts(self) -> tuple[int, ...]:
        """Count only designs with at least one unmeasured exact substitution."""
        return tuple(
            sum(
                self.suffix[0].get((size, novel), 0)
                for novel in range(1, self.max_new_mutations + 1)
            )
            for size in range(self.max_mutations + 1)
        )

    def unrank(self, size: int, rank: int) -> Variant:
        """Map a bounded integer to one compatible candidate without rejection loops."""
        if not 0 <= size <= self.max_mutations or not 0 <= rank < self.counts[size]:
            raise ValueError("Candidate rank is outside its mutation-count stratum")
        novel = 1
        for novel in range(1, self.max_new_mutations + 1):
            count = self.suffix[0].get((size, novel), 0)
            if rank < count:
                break
            rank -= count
        result = []
        for i, alternatives in enumerate(self.groups):
            skipped = self.suffix[i + 1].get((size, novel), 0)
            if rank < skipped:
                continue
            rank -= skipped
            for mutation, is_new in alternatives:
                count = self.suffix[i + 1].get((size - 1, novel - is_new), 0)
                if rank < count:
                    result.append(mutation)
                    size -= 1
                    novel -= is_new
                    break
                rank -= count
        return tuple(result)

    def sample(self, budget: int, *, seed: int) -> tuple[Variant, ...]:
        """Evenly allocate across feasible sizes, redistributing exhausted capacity."""
        if budget < 1:
            raise ValueError("Candidate budget must be positive")
        counts = self.counts
        allocation = [0] * len(counts)
        remaining = min(budget, sum(counts))
        while remaining:
            active = [
                size for size, count in enumerate(counts) if count > allocation[size]
            ]
            quota = max(1, remaining // len(active))
            for size in active:
                take = min(quota, counts[size] - allocation[size], remaining)
                allocation[size] += take
                remaining -= take
        rng = random.Random(seed)  # noqa: S311 - reproducible scientific sampling
        result = []
        for size, take in enumerate(allocation):
            count = counts[size]
            # Floyd sampling supports arbitrary-size integers unlike len(range(N)).
            selected: set[int] = set()
            for j in range(count - take, count):
                proposal = rng.randrange(j + 1)
                selected.add(j if proposal in selected else proposal)
            result.extend(self.unrank(size, rank) for rank in sorted(selected))
        return tuple(result)
