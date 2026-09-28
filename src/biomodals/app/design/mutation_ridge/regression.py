"""Sparse additive ridge fitting, support-aware evaluation and streamed designs."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from biomodals.app.design.mutation_ridge.inputs import (
    MAX_INPUT_BYTES,
    MAX_MEASUREMENT_ROWS,
    MutationDataset,
    Substitution,
    Variant,
    combination_space,
    iter_combinations,
    variant_key,
    variant_sequences,
)

SCIKIT_LEARN_VERSION = "1.9.1"
RIDGE_VERSION = "3"
# Worst-case JSON escaping of input text plus 10,000 numeric point records.
MAX_VALIDATION_BYTES = 6 * MAX_INPUT_BYTES + 2 * 1024 * 1024


class ValidationPoint(BaseModel):
    """One unique variant, using only predictions made while it was held out."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    mutations: str
    measured_label: float
    predicted_label: float
    prediction_count: int = Field(ge=1)


class ValidationSummary(BaseModel):
    """Compact held-out evidence, never training-fit performance."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    regime: str
    training_variants: int
    evaluated_variants: int
    folds: int
    mae: float | None = None
    rmse: float | None = None
    spearman: float | None = None
    evaluated_mutation_counts: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    points: tuple[ValidationPoint, ...] = Field(
        default=(), max_length=MAX_MEASUREMENT_ROWS
    )


def combination_folds(
    variants: Sequence[Variant], *, seed: int = 0, max_folds: int = 5
) -> tuple[tuple[int, ...], ...]:
    """Hold out multi-mutants only if the entire fold retains exact support.

    Parent/singles anchor every training fold. Each canonical variant is tested
    at most once; label values do not influence assignment.
    """
    if not 1 <= max_folds <= 5:
        raise ValueError("Validation supports between one and five folds")
    support = Counter(mutation for variant in variants for mutation in variant)
    eligible = [i for i, variant in enumerate(variants) if len(variant) >= 2]
    random.Random(seed).shuffle(eligible)  # noqa: S311 - reproducible scientific split
    folds: list[list[int]] = [[] for _ in range(max_folds)]
    held_support: list[Counter] = [Counter() for _ in folds]
    for index in eligible:
        variant = variants[index]
        for fold_index in sorted(range(max_folds), key=lambda i: (len(folds[i]), i)):
            held = held_support[fold_index]
            if all(support[m] - held[m] > 1 for m in variant):
                folds[fold_index].append(index)
                held.update(variant)
                break
    return tuple(tuple(sorted(fold)) for fold in folds if fold)


def mutation_matrix(variants: Sequence[Variant], vocabulary: Sequence[Substitution]):
    """Build sparse exact-substitution features; unseen alternatives are errors."""
    import numpy as np
    from scipy.sparse import csr_matrix

    columns = {mutation: i for i, mutation in enumerate(vocabulary)}
    indices = []
    indptr = [0]
    for variant in variants:
        try:
            indices.extend(columns[mutation] for mutation in variant)
        except KeyError as exc:
            raise ValueError(
                "Cannot encode an unmeasured substitution as parental"
            ) from exc
        indptr.append(len(indices))
    return csr_matrix(
        (np.ones(len(indices), dtype=np.float64), indices, indptr),
        shape=(len(variants), len(vocabulary)),
    )


def fit_ridge(matrix, labels, *, alpha: float = 1.0):
    """Fit unscaled binary features with an unpenalized intercept."""
    from sklearn.linear_model import Ridge

    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    return Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-10).fit(
        matrix, labels
    )


def validation_metrics(actual, predicted) -> dict[str, float | None]:
    """Calculate defined metrics only, with no constant-vector rank claims."""
    import numpy as np
    from scipy.stats import spearmanr

    if not len(actual):
        return {"mae": None, "rmse": None, "spearman": None}
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if not np.isfinite(predicted).all():
        raise ValueError("Estimator returned nonfinite validation predictions")
    residual = actual - predicted
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": (
            float(spearmanr(actual, predicted).statistic)
            if len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1
            else None
        ),
    }


def validate_ridge(
    dataset: MutationDataset, *, alpha: float = 1.0, seed: int = 0
) -> ValidationSummary:
    """Evaluate held-out combinations without dropping their final-fit labels."""
    import numpy as np

    variants = dataset.variants
    vocabulary = dataset.vocabulary
    matrix = mutation_matrix(variants, vocabulary)
    labels = dataset.measurements["label"].to_numpy()
    folds = combination_folds(variants, seed=seed)
    actual, predictions, tested = [], [], []
    for fold in folds:
        train = np.ones(len(variants), dtype=bool)
        train[list(fold)] = False
        model = fit_ridge(matrix[train], labels[train], alpha=alpha)
        actual.extend(labels[list(fold)])
        predictions.extend(model.predict(matrix[list(fold)]))
        tested.extend(fold)
    warnings = [
        "Additive ridge does not model interactions; validation at lower mutation counts does not establish higher-order accuracy."
    ]
    if not tested:
        warnings.append(
            "No supported multi-mutant holdout is available; predictions have no empirical combination validation."
        )
    occurrences: dict[Substitution, list[int]] = {m: [] for m in vocabulary}
    for index, variant in enumerate(variants):
        for mutation in variant:
            occurrences[mutation].append(index)
    supports = Counter(tuple(rows) for rows in occurrences.values())
    if any(count > 1 for count in supports.values()):
        warnings.append(
            "Some substitutions always co-occur; their individual additive effects are confounded."
        )
    return ValidationSummary(
        regime="supported_combinations",
        training_variants=len(variants),
        evaluated_variants=len(tested),
        folds=len(folds),
        evaluated_mutation_counts=tuple(sorted({len(variants[i]) for i in tested})),
        warnings=tuple(warnings),
        points=tuple(
            ValidationPoint(
                mutations=variant_key(variants[index]),
                measured_label=measured,
                predicted_label=predicted,
                prediction_count=1,
            )
            for index, measured, predicted in zip(
                tested, actual, predictions, strict=True
            )
        ),
        **validation_metrics(actual, predictions),
    )


def write_combinations(
    dataset: MutationDataset,
    output_path: Path,
    *,
    max_mutations: int = 2,
    budget: int = 1_000_000,
    alpha: float = 1.0,
    higher_is_better: bool = True,
    seed: int = 0,
    batch_size: int = 1024,
) -> ValidationSummary:
    """Refit all measurements and write novel scored designs, bounded by batches.

    Candidate scoring sums active coefficients instead of building a dense
    candidate matrix. Polars performs the final streaming sort/write; no full
    sequence table is collected into Python memory.
    """
    import numpy as np

    _, counts = combination_space(dataset, max_mutations)
    count = sum(counts)
    if budget < 1 or count > budget:
        raise ValueError(
            f"Novel combination count {count} exceeds candidate budget {budget}"
        )
    if not dataset.vocabulary:
        raise ValueError("At least one measured substitution is required")
    if not 1 <= batch_size <= 10_000:
        raise ValueError("Candidate batch size must be between 1 and 10000")
    summary = validate_ridge(dataset, alpha=alpha, seed=seed)
    matrix = mutation_matrix(dataset.variants, dataset.vocabulary)
    model = fit_ridge(matrix, dataset.measurements["label"].to_numpy(), alpha=alpha)
    coefficients = dict(zip(dataset.vocabulary, model.coef_, strict=True))
    if not np.isfinite(model.coef_).all() or not math.isfinite(model.intercept_):
        raise ValueError("Ridge returned nonfinite parameters")
    schema = {
        "id": pl.String,
        "mutations": pl.String,
        "predicted_label": pl.Float64,
        "n_mutations": pl.UInt32,
        "n_new_mutations": pl.UInt32,
        "warnings": pl.String,
        **{f"sequence_{chain}": pl.String for chain in dataset.parents},
    }
    variants = iter_combinations(dataset, max_mutations=max_mutations, budget=budget)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ridge-candidates-") as staging:
        spool = Path(staging) / "candidates.csv"
        offset = 0
        with spool.open("wb") as handle:
            pl.DataFrame(schema=schema).write_csv(handle)
            for batch in batched(variants, batch_size):
                rows = []
                for index, variant in enumerate(batch, offset + 1):
                    score = float(
                        model.intercept_ + sum(coefficients[m] for m in variant)
                    )
                    if not math.isfinite(score):
                        raise ValueError(
                            "Ridge returned a nonfinite candidate prediction"
                        )
                    rows.append({
                        "id": f"candidate_{index:09d}",
                        "mutations": variant_key(variant),
                        "predicted_label": score,
                        "n_mutations": len(variant),
                        "n_new_mutations": 0,
                        "warnings": None,
                        **{
                            f"sequence_{chain}": seq
                            for chain, seq in (
                                variant_sequences(dataset.parents, variant)
                                if dataset.parents
                                else {}
                            ).items()
                        },
                    })
                pl.DataFrame(rows, schema=schema).write_csv(
                    handle, include_header=False
                )
                offset += len(batch)
        pl.scan_csv(spool, schema=schema).sort(
            ["predicted_label", "id"], descending=[higher_is_better, False]
        ).sink_csv(output_path, engine="streaming")
    return summary
