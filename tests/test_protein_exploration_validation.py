"""Novel-substitution evaluation keeps training support and final-fit data intact."""

import numpy as np
import pytest

from biomodals.app.design.mutation_ridge.inputs import build_dataset
from biomodals.workflow.protein_optimization.validation import (
    exploration_folds,
    exploration_summary,
)


def _dataset():
    return build_dataset(
        b'mutations,label\n,0\nA:A1V,1\nA:A1V,3\nA:A1L,4\nA:A2V,5\nA:A2L,6\n"A:A1V,A:A2V,A:A3W",7\n',
        ">A\nAAA\n",
    )


def test_all_occurrences_removed_and_same_site_support_retained():
    """An exact mutation never leaks through another multi-mutant training row."""
    dataset = _dataset()
    folds = exploration_folds(dataset.variants, seed=17)
    assert folds == exploration_folds(dataset.variants, seed=17)
    assert len(folds) == 4
    for fold in folds:
        assert len(fold.training_indices) >= 2
        assert set(fold.training_indices).isdisjoint(fold.test_indices)
        assert set(fold.training_indices) | set(fold.test_indices) == set(
            range(len(dataset.variants))
        )
        train = {m for i in fold.training_indices for m in dataset.variants[i]}
        assert fold.substitution not in train
        assert any(m.site == fold.substitution.site for m in train)
        assert all(fold.substitution in dataset.variants[i] for i in fold.test_indices)
    held_v = next(f for f in folds if f.substitution.token == "A:A1V")  # noqa: S105 - mutation notation
    assert [m.token for m in held_v.extra_unsupported] == ["A:A3W"]
    # Replicate aggregation precedes splitting; changing labels cannot change folds.
    changed = build_dataset(
        b'mutations,label\n,100\nA:A1V,-100\nA:A1L,-40\nA:A2V,50\nA:A2L,60\n"A:A1V,A:A2V,A:A3W",70\n',
        ">A\nAAA\n",
    )
    assert exploration_folds(changed.variants, seed=17) == folds


def test_metrics_count_unique_variants_not_repeated_holdout_occurrences():
    """Overlapping substitution tests do not inflate unique-variant coverage."""
    dataset = _dataset()
    folds = exploration_folds(dataset.variants)
    labels = dataset.measurements["label"].to_numpy()
    predictions = [labels[list(f.test_indices)] + 1 for f in folds]
    summary = exploration_summary(dataset.variants, labels, folds, predictions)
    tested = {i for f in folds for i in f.test_indices}
    assert summary.evaluated_variants == len(tested)
    assert len(tested) < sum(len(f.test_indices) for f in folds)
    assert summary.training_variants == len(dataset.variants)
    assert summary.mae == pytest.approx(1)
    assert summary.rmse == pytest.approx(1)
    assert any("unsupported" in warning for warning in summary.warnings)
    assert any("averaged" in warning for warning in summary.warnings)
    predictions[0] = np.full(len(folds[0].test_indices), np.nan)
    with pytest.raises(ValueError, match="predictions"):
        exploration_summary(dataset.variants, labels, folds, predictions)


def test_unavailable_validation_is_explicit_and_not_a_random_split():
    """Single-alternative sites cannot support the accepted transfer validation."""
    dataset = build_dataset(b"mutations,label\n,0\nA:A1V,1\nA:A2L,2\n", ">A\nAA\n")
    folds = exploration_folds(dataset.variants)
    summary = exploration_summary(
        dataset.variants, dataset.measurements["label"].to_numpy(), folds, []
    )
    assert summary.evaluated_variants == 0
    assert summary.mae is None
    assert any("unavailable" in warning for warning in summary.warnings)
