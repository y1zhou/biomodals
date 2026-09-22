"""Frozen, candidate-bound germline evidence and post-ranking scalar joins."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import polars as pl

from biomodals.helper.antibody import (
    GermlineAssignment,
    assign_germlines,
    combined_pi,
    sequence_pi,
)
from biomodals.workflow.humanization.contracts import HumanizationCandidate

GENE_COLUMNS = {
    f"{chain}_{segment}_gene": pl.String
    for chain in ("vh", "vl")
    for segment in ("v", "j")
}
SEQUENCE_COLUMNS = {
    "vh_pi": pl.Float64,
    "vl_pi": pl.Float64,
    "vh_vl_pi": pl.Float64,
    **GENE_COLUMNS,
}
_EVIDENCE = pl.Struct({
    "reference_id": pl.String,
    "species": pl.String,
    "gene": pl.String,
    "allele": pl.String,
    "accession": pl.String,
    "score": pl.Float64,
    "known_pairs": pl.Int64,
    "known_matches": pl.Int64,
    "known_fr4_pairs": pl.Int64,
    "reference_coverage": pl.Float64,
    "query_coverage": pl.Float64,
    "query_input_start": pl.Int64,
    "imgt_span": pl.List(pl.Int64),
})
GERMLINE_SCHEMA = {
    "parent_id": pl.String,
    "candidate_id": pl.String,
    "chain": pl.String,
    "sequence_sha256": pl.String,
    "chain_type": pl.String,
    "v_gene": pl.String,
    "j_gene": pl.String,
    "v": pl.List(_EVIDENCE),
    "j": pl.List(_EVIDENCE),
    "diagnostics": pl.List(pl.String),
    "error": pl.String,
}
ASSIGNMENT_COLUMNS = tuple(
    key
    for key in GERMLINE_SCHEMA
    if key not in {"parent_id", "candidate_id", "chain", "sequence_sha256"}
)
IDENTITY_COLUMNS = ("parent_id", "candidate_id", "chain", "sequence_sha256")


def candidate_germlines(candidate: HumanizationCandidate) -> list[dict[str, Any]]:
    """Annotate two exact supplied chains; no scoring, networking or filtering."""
    rows = []
    for chain in ("vh", "vl"):
        sequence = getattr(candidate, chain)
        assignment = assign_germlines(sequence)
        if assignment["chain_type"] not in (
            (None, "H") if chain == "vh" else (None, "K", "L")
        ):
            assignment = unavailable_assignment(
                f"{chain}: germline chain role mismatch"
            )
        rows.append({
            "parent_id": candidate.parent_id,
            "candidate_id": candidate.candidate_id,
            "chain": chain,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            **assignment,
        })
    return rows


def unavailable_assignment(error: str) -> GermlineAssignment:
    """Represent absent native evidence explicitly rather than a zero score."""
    return {
        "chain_type": None,
        "v_gene": None,
        "j_gene": None,
        "v": [],
        "j": [],
        "diagnostics": [],
        "error": error,
    }


def complete_germline_table(
    candidates: Sequence[HumanizationCandidate], frames: Sequence[pl.DataFrame]
) -> pl.DataFrame:
    """Fill missing Tasks, rejecting evidence bound to a different sequence."""
    expected = pl.DataFrame(
        [
            (
                candidate.parent_id,
                candidate.candidate_id,
                chain,
                hashlib.sha256(getattr(candidate, chain).encode()).hexdigest(),
            )
            for candidate in candidates
            for chain in ("vh", "vl")
        ],
        schema={name: GERMLINE_SCHEMA[name] for name in IDENTITY_COLUMNS},
        orient="row",
    )
    evidence = pl.concat(frames) if frames else pl.DataFrame(schema=GERMLINE_SCHEMA)
    if evidence.join(expected, on=list(IDENTITY_COLUMNS), how="anti").height:
        raise ValueError("Germline evidence does not match candidate identities")
    result = expected.join(
        evidence,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    return result.with_columns(
        pl
        .when(pl.col("v").is_null())
        .then(pl.lit("Germline annotation unavailable"))
        .otherwise(pl.col("error"))
        .alias("error"),
        pl.col("v", "j", "diagnostics").fill_null([]),
    ).select(*GERMLINE_SCHEMA)


def add_gene_columns(selection: pl.DataFrame, germlines: pl.DataFrame) -> pl.DataFrame:
    """Append labels only after ranking, retaining exact row order and scores."""
    result = selection
    for chain in ("vh", "vl"):
        labels = germlines.filter(pl.col("chain") == chain).select(
            "parent_id",
            "candidate_id",
            pl.col("v_gene").alias(f"{chain}_v_gene"),
            pl.col("j_gene").alias(f"{chain}_j_gene"),
        )
        result = result.join(
            labels,
            on=["parent_id", "candidate_id"],
            how="left",
            validate="1:1",
            maintain_order="left",
        )
    return result


def add_sequence_columns(
    selection: pl.DataFrame, germlines: pl.DataFrame
) -> pl.DataFrame:
    """Join unique-sequence pIs after ranking and place annotations after VH."""
    # Native scientific work runs once per unique chain/pair. Polars owns the
    # deduplication and joins; scores, rank and original row order are untouched.
    chains = (
        pl
        .concat([
            selection.select(pl.col(chain).alias("sequence")) for chain in ("vh", "vl")
        ])
        .unique()
        .with_columns(
            pl
            .col("sequence")
            .map_elements(sequence_pi, return_dtype=pl.Float64)
            .alias("pi")
        )
    )
    result = add_gene_columns(selection, germlines)
    for chain in ("vh", "vl"):
        result = result.join(
            chains.rename({"sequence": chain, "pi": f"{chain}_pi"}),
            on=chain,
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    pairs = (
        selection
        .select("vh", "vl")
        .unique()
        .with_columns(
            pl
            .struct("vh", "vl")
            .map_elements(
                lambda pair: combined_pi(pair["vh"], pair["vl"]),
                return_dtype=pl.Float64,
            )
            .alias("vh_vl_pi")
        )
    )
    result = result.join(
        pairs, on=["vh", "vl"], how="left", validate="m:1", maintain_order="left"
    )
    columns = selection.columns
    at = columns.index("vh") + 1
    return result.select(*columns[:at], *SEQUENCE_COLUMNS, *columns[at:])
