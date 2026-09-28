"""Native sparse ridge, leakage boundaries and complete combination CSVs."""

from itertools import combinations

import numpy as np
import polars as pl
import pytest

from biomodals.app.design.mutation_ridge.inputs import (
    build_dataset,
    parse_mutations,
    variant_sequences,
)
from biomodals.app.design.mutation_ridge.regression import (
    combination_folds,
    fit_ridge,
    mutation_matrix,
    validate_ridge,
    write_combinations,
)


def _landscape():
    rows = [{"mutations": "", "label": 7.0}]
    tokens = ["A:A1C", "A:A2M", "B:C1V"]
    effects = dict(zip(tokens, [1.0, -2.0, 4.0], strict=True))
    for size in (1, 2):
        rows.extend(
            {"mutations": ",".join(group), "label": 7 + sum(effects[t] for t in group)}
            for group in combinations(tokens, size)
        )
    return build_dataset(pl.DataFrame(rows).write_csv().encode(), ">A\nAA\n>B\nC\n")


def test_native_sparse_ridge_recovers_additive_landscape_and_validates(tmp_path):
    """Heldout doubles predict well and a novel triple uses the final all-data fit."""
    dataset = _landscape()
    path = tmp_path / "candidates.csv"
    summary = write_combinations(
        dataset, path, max_mutations=3, alpha=1e-8, batch_size=1
    )
    assert summary.evaluated_variants == 3
    assert summary.evaluated_mutation_counts == (2,)
    assert summary.mae < 1e-6
    frame = pl.read_csv(path)
    assert frame["mutations"].to_list() == ["A:A1C,A:A2M,B:C1V"]
    assert frame["predicted_label"].item() == pytest.approx(10, abs=1e-6)
    assert frame["n_mutations"].item() == 3
    assert frame["n_new_mutations"].item() == 0
    assert frame["sequence_A"].item() == "CM"
    assert frame["sequence_B"].item() == "V"


def test_interaction_failure_is_reported_but_all_labels_return_to_final_fit(tmp_path):
    """A bad heldout prediction is not selected as the final scoring model."""
    dataset = build_dataset(
        b'mutations,label\n,0\nA:A1C,1\nA:A2C,1\nA:A3C,1\n"A:A1C,A:A2C",100\n',
        ">A\nAAA\n",
    )
    summary = write_combinations(dataset, tmp_path / "result.csv", max_mutations=3)
    assert summary.evaluated_variants == 1
    assert summary.mae > 90
    assert len(summary.points) == 1
    point = summary.points[0]
    assert point.mutations == "A:A1C,A:A2C"
    assert point.measured_label == 100
    assert point.prediction_count == 1
    assert abs(point.measured_label - point.predicted_label) == pytest.approx(
        summary.mae
    )
    result = pl.read_csv(tmp_path / "result.csv")
    matrix = mutation_matrix(dataset.variants, dataset.vocabulary)
    final = fit_ridge(matrix, dataset.measurements["label"].to_numpy())
    measured_variant = mutation_matrix(
        [parse_mutations(point.mutations)], dataset.vocabulary
    )
    assert final.predict(measured_variant)[0] != pytest.approx(point.predicted_label)
    candidates = [parse_mutations(value) for value in result["mutations"]]
    expected = final.predict(mutation_matrix(candidates, dataset.vocabulary))
    np.testing.assert_allclose(result["predicted_label"].to_numpy(), expected)
    assert result["predicted_label"].to_list() == sorted(expected, reverse=True)


def test_complete_holdout_fold_retains_support_not_just_individual_rows():
    """Two individually eligible rows must not remove each other's only support."""
    variants = tuple(
        parse_mutations(text)
        for text in ("A:A1C,A:A2C", "A:A1C,A:A3C", "A:A2C", "A:A3C")
    )
    folds = combination_folds(variants, max_folds=1)
    assert len(folds[0]) == 1
    for fold in folds:
        supported = {m for i, v in enumerate(variants) if i not in fold for m in v}
        assert all(set(variants[i]) <= supported for i in fold)
    assert combination_folds(variants, seed=15) == combination_folds(variants, seed=15)


def test_singles_only_and_confounded_inputs_show_truthful_warnings():
    """No random-row validation substitutes for missing combination evidence."""
    singles = build_dataset(b"mutations,label\nA:A1C,1\nA:A2C,2\n", ">A\nAA\n")
    summary = validate_ridge(singles)
    assert summary.evaluated_variants == 0
    assert summary.mae is None
    assert any("No supported multi-mutant" in warning for warning in summary.warnings)
    confounded = build_dataset(b'mutations,label\n,1\n"A:A1C,A:A2C",2\n', ">A\nAA\n")
    assert any(
        "confounded" in warning for warning in validate_ridge(confounded).warnings
    )


def test_batch_size_and_direction_do_not_change_scores_or_sequence_identity(tmp_path):
    """Bounded writing changes neither exhaustive coverage nor predictions."""
    dataset = build_dataset(
        b"mutations,label\n,0\nA:A1C,3\nA:A2M,2\nB:C1V,-1\n", ">A\nAA\n>B\nC\n"
    )
    write_combinations(dataset, tmp_path / "one.csv", max_mutations=3, batch_size=1)
    write_combinations(
        dataset,
        tmp_path / "many.csv",
        max_mutations=3,
        batch_size=10,
        higher_is_better=False,
    )
    one, many = pl.read_csv(tmp_path / "one.csv"), pl.read_csv(tmp_path / "many.csv")
    assert one.height == 4
    assert one.sort("id").equals(many.sort("id"))
    assert many["predicted_label"].to_list() == sorted(many["predicted_label"])
    for row in one.iter_rows(named=True):
        sequences = variant_sequences(
            dataset.parents, parse_mutations(row["mutations"])
        )
        assert {chain: row[f"sequence_{chain}"] for chain in sequences} == sequences


def test_unknown_substitution_and_oversized_proposal_fail_before_writing(tmp_path):
    """Never map an unseen replacement to zero or silently sample exhaustive work."""
    dataset = _landscape()
    with pytest.raises(ValueError, match="unmeasured substitution"):
        mutation_matrix([parse_mutations("A:A1V")], dataset.vocabulary)
    with pytest.raises(ValueError, match="budget"):
        write_combinations(dataset, tmp_path / "result.csv", max_mutations=3, budget=0)


def test_zero_novel_candidates_still_write_typed_csv_header(tmp_path):
    """An entirely measured space has a valid empty scientific output."""
    dataset = build_dataset(b"mutations,label\n,1\nA:A1C,2\n", ">A\nA\n")
    summary = write_combinations(dataset, tmp_path / "empty.csv")
    frame = pl.read_csv(tmp_path / "empty.csv")
    assert frame.height == 0
    assert frame.columns == [
        "id",
        "mutations",
        "predicted_label",
        "n_mutations",
        "n_new_mutations",
        "warnings",
        "sequence_A",
    ]
    assert summary.training_variants == 2


def test_table_only_fit_preserves_normalized_labels_and_predictions(tmp_path):
    """Sequences are irrelevant to ridge; mutation identities fully specify features."""
    content = b"mutations,label\nA:A1C,-0.2\nA:A2V,-0.4\nB:C1A,0.1\n"
    sparse = build_dataset(content)
    complete = build_dataset(content, ">A\nAA\n>B\nC\n")
    write_combinations(sparse, tmp_path / "sparse.csv")
    write_combinations(complete, tmp_path / "complete.csv")
    output = pl.read_csv(tmp_path / "sparse.csv")
    assert output.height == 3
    assert output.columns == [
        "id",
        "mutations",
        "predicted_label",
        "n_mutations",
        "n_new_mutations",
        "warnings",
    ]
    assert output.equals(pl.read_csv(tmp_path / "complete.csv").select(output.columns))
    assert sparse.measurements["label"].to_list() == [-0.2, -0.4, 0.1]
