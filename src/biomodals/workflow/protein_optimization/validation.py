"""Label-blind holdouts of exact substitutions at experimentally tested sites."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from biomodals.app.design.mutation_ridge.inputs import (
    Substitution,
    Variant,
    variant_key,
)
from biomodals.app.design.mutation_ridge.regression import (
    ValidationPoint,
    ValidationSummary,
    validation_metrics,
)


@dataclass(frozen=True)
class SubstitutionFold:
    """One withheld alternative, every affected variant, and extra novelty evidence."""

    substitution: Substitution
    training_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    extra_unsupported: tuple[Substitution, ...]


def exploration_folds(
    variants: Sequence[Variant], *, seed: int = 0, max_folds: int = 5
) -> tuple[SubstitutionFold, ...]:
    """Remove all occurrences; retain another alternative and at least two rows.

    Repeated held-out variants across different substitution tests are allowed,
    but final metrics average their predictions rather than count them twice.
    Labels, candidate settings, and uploaded replicate multiplicity do not enter
    split construction.
    """
    if not 1 <= max_folds <= 5:
        raise ValueError("Validation supports one to five folds")
    occurrences: dict[Substitution, set[int]] = defaultdict(set)
    sites: dict[tuple[str, int], set[Substitution]] = defaultdict(set)
    for index, variant in enumerate(variants):
        for mutation in variant:
            occurrences[mutation].add(index)
            sites[mutation.site].add(mutation)
    eligible = sorted(m for m in occurrences if len(sites[m.site]) > 1)
    random.Random(seed).shuffle(eligible)  # noqa: S311 - label-blind reproducible split
    folds = []
    all_indices = set(range(len(variants)))
    for held in eligible:
        test = occurrences[held]
        train = all_indices - test
        if len(train) < 2:
            continue
        supported = {m for index in train for m in variants[index]}
        if not (sites[held.site] - {held}) & supported:
            continue
        extra = {m for index in test for m in variants[index]} - supported - {held}
        folds.append(
            SubstitutionFold(
                held, tuple(sorted(train)), tuple(sorted(test)), tuple(sorted(extra))
            )
        )
        if len(folds) == max_folds:
            break
    return tuple(folds)


def exploration_summary(
    variants: Sequence[Variant],
    labels,
    folds: Sequence[SubstitutionFold],
    predictions: Sequence,
) -> ValidationSummary:
    """Report each unique measured variant once, with explicit transfer limitations."""
    import numpy as np

    totals = np.zeros(len(variants), dtype=np.float64)
    counts = np.zeros(len(variants), dtype=np.int32)
    for fold, values in zip(folds, predictions, strict=True):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(fold.test_indices),) or not np.isfinite(values).all():
            raise ValueError("Invalid held-out predictions")
        indices = list(fold.test_indices)
        totals[indices] += values
        counts[indices] += 1
    tested = np.flatnonzero(counts)
    warnings = [
        "Validation tests new amino-acid alternatives at measured positions, not entirely new positions or chains.",
        "Separate-chain sequence features do not model interchain attention or multimer structure.",
    ]
    if not folds:
        warnings.append(
            "No same-position alternative holdout leaves sufficient training data; exploration validation is unavailable."
        )
    extra = sorted({m.token for fold in folds for m in fold.extra_unsupported})
    if extra:
        warnings.append(
            f"Some held-out variants also contain {len(extra)} other unsupported substitutions; these are not clean one-new-substitution tests."
        )
    if np.any(counts > 1):
        warnings.append(
            "Predictions for variants evaluated in multiple substitution holdouts are averaged before computing metrics."
        )
    return ValidationSummary(
        regime="same_position_alternatives",
        training_variants=len(variants),
        evaluated_variants=len(tested),
        folds=len(folds),
        evaluated_mutation_counts=tuple(sorted({len(variants[i]) for i in tested})),
        warnings=tuple(warnings),
        points=tuple(
            ValidationPoint(
                mutations=variant_key(variants[i]),
                measured_label=labels[i],
                predicted_label=totals[i] / counts[i],
                prediction_count=int(counts[i]),
            )
            for i in tested
        ),
        **validation_metrics(
            np.asarray(labels)[tested], totals[tested] / counts[tested]
        ),
    )
