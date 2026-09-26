"""Mutation identity, replicate aggregation and exhaustive candidate boundaries."""

import math

import pytest

from biomodals.app.design.mutation_ridge.inputs import (
    build_dataset,
    combination_space,
    iter_combinations,
    parent_issues,
    parse_mutations,
    parse_parents,
    review_measurements,
    variant_key,
    variant_sequences,
)


def test_replicates_preserve_observations_and_numeric_chain_positions():
    """Order-equivalent observations share a mean without losing raw evidence."""
    content = (
        b"id,mutations,label\nparent,,1\n"
        b'first,"B:C2M,A:A10V,A:A2C",2\n'
        b'again,"A:A2C,A:A10V,B:C2M",4\n'
    )
    dataset = build_dataset(content, ">B extra description\nacde\n>A\n aaaa aaaaaa \n")
    assert dataset.parents == {"A": "AAAAAAAAAA", "B": "ACDE"}
    assert dataset.observations["id"].to_list() == ["parent", "first", "again"]
    assert dataset.measurements["mutations"].to_list() == ["", "A:A2C,A:A10V,B:C2M"]
    assert dataset.measurements["label"].to_list() == [1, 3]
    assert dataset.measurements["replicate_count"].to_list() == [1, 2]
    assert dataset.measurements["replicate_std"].to_list() == [None, math.sqrt(2)]


def test_review_keeps_invalid_rows_and_reports_conflicting_references():
    """Invalid inputs retain row-associated explanations rather than disappear."""
    review = review_measurements(
        b'mutations,label\nA:A1C,>1000\nA:G1V,nan\nA:A2M,\n"A:A1C,A:A1V",4\n,inf\n'
    )
    assert review.observations.height == 5
    assert review.required_chains == ("A",)
    assert {(issue.row_index, issue.code) for issue in review.issues} == {
        (0, "invalid_label"),
        (1, "invalid_label"),
        (2, "invalid_label"),
        (4, "invalid_label"),
        (3, "invalid_mutations"),
        (0, "conflicting_original"),
        (1, "conflicting_original"),
    }
    with pytest.raises(ValueError, match="Measurement row"):
        build_dataset(b"mutations,label\nA:A1C,>1000\n", ">A\nA\n")


@pytest.mark.parametrize(
    "token",
    [
        "A:A0C",
        "A:A01C",
        "A:A1A",
        "A:A1X",
        "A:a1C",
        "A:A1C,",
        "A:A1C,A:A1C",
        "A:A1C,A:A1M",
        "A:A1C\x00",
    ],
)
def test_malformed_or_incompatible_substitutions_are_rejected(token):
    """Only real, unambiguous standard-AA replacements are supported."""
    with pytest.raises(ValueError):
        parse_mutations(token)


@pytest.mark.parametrize(
    "fasta", [">A\nAAA\n>A\nCCC", ">\nAAA", "AAA", ">A\nAXA", ">A\n", ">A:B\nAAA"]
)
def test_invalid_parent_records_fail(fasta):
    """Malformed FASTA never silently repairs or discards biological residues."""
    with pytest.raises(ValueError):
        parse_parents(fasta)


def test_parent_issues_are_row_specific_and_extra_chains_are_retained():
    """All referenced sites must match, while unmutated partners are accepted."""
    review = review_measurements(b"mutations,label\nA:A2C,1\nA:A10C,2\nB:A1C,3\n")
    parents = parse_parents(">A\nAC\n>Partner\nMCAC\n")
    assert [(i.row_index, i.code) for i in parent_issues(review, parents)] == [
        (0, "original_mismatch"),
        (1, "position_out_of_range"),
        (2, "missing_chain"),
    ]
    assert parents["Partner"] == "MCAC"


def test_all_compatible_novel_combinations_have_exact_count_and_sequences():
    """Known C/M replacements are allowed, measured identities are excluded."""
    dataset = build_dataset(
        b'mutations,label\nA:A1C,1\nA:A1M,2\nB:C2A,3\n"A:A1C,B:C2A",4\n',
        ">A\nAA\n>B\nAC\n",
    )
    _, counts = combination_space(dataset, 10)
    assert counts == (0, 0, 1)
    variants = list(iter_combinations(dataset, max_mutations=10, budget=1))
    assert [variant_key(v) for v in variants] == ["A:A1M,B:C2A"]
    assert variant_sequences(dataset.parents, variants[0]) == {"A": "MA", "B": "AA"}


def test_count_rejects_oversize_and_does_not_drop_higher_order_training_rows():
    """Proposal bounds do not alter the measurement snapshot used for fitting."""
    dataset = build_dataset(
        b'mutations,label\nA:A1C,1\nA:A2C,2\nA:A3C,3\n"A:A1C,A:A2C,A:A3C",4\n',
        ">A\nAAA\n",
    )
    assert combination_space(dataset, 2)[1] == (0, 0, 3)
    assert parse_mutations("A:A1C,A:A2C,A:A3C") in dataset.variants
    assert len(dataset.measurements) == 4
    stream = iter_combinations(dataset, max_mutations=2, budget=2)
    with pytest.raises(ValueError, match="count 3 exceeds"):
        next(stream)
    assert len(list(iter_combinations(dataset, max_mutations=2, budget=3))) == 3


@pytest.mark.parametrize(
    "content",
    [
        b"mutations,label\n",
        b"mutations,label,extra\n,1,x\n",
        b"mutations,label,label\n,1,2\n",
        b"mutations,label\n\xff,1\n",
    ],
)
def test_invalid_csv_schema_or_encoding_is_not_silently_repaired(content):
    """Unknown/duplicate fields and invalid UTF-8 fail the CSV boundary."""
    with pytest.raises(ValueError):
        review_measurements(content)


def test_row_limit_rejects_rather_than_truncating():
    """A bounded read detects overflow and reports it instead of fitting a prefix."""
    with pytest.raises(ValueError, match="between 1 and 1"):
        review_measurements(b"mutations,label\n,1\n,2\n", max_rows=1)


def test_chain_ids_are_case_sensitive_and_never_path_sanitized():
    """Display identities remain exact even when unsuitable for filenames."""
    dataset = build_dataset(
        b"id,mutations,label\n../../sample,../A:A1C,1\n", ">../A\nA\n>a\nC\n"
    )
    assert dataset.parents == {"../A": "A", "a": "C"}
    assert dataset.observations["id"].item() == "../../sample"
