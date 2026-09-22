"""Thin-table Pareto and mutation-distance operations shared by design workflows."""

from collections.abc import Sequence

import polars as pl


def pareto_layers(group: pl.DataFrame, objectives: Sequence[str]) -> pl.DataFrame:
    """Number nondominated layers for finite maximization objectives in one parent."""
    pairs = group.select("candidate_id", *objectives).join(
        group.select("candidate_id", *objectives), how="cross", suffix="_other"
    )
    edges = pairs.filter(
        pl.all_horizontal(
            pl.col(name) >= pl.col(f"{name}_other") for name in objectives
        ),
        pl.any_horizontal(
            pl.col(name) > pl.col(f"{name}_other") for name in objectives
        ),
    ).select("candidate_id", "candidate_id_other")
    remaining, fronts = group, []
    while remaining.height:
        front = remaining.join(
            edges.select(pl.col("candidate_id_other").alias("candidate_id")).unique(),
            on="candidate_id",
            how="anti",
        )
        fronts.append(
            front.with_columns(
                pl.lit(len(fronts) + 1, dtype=pl.Int64).alias("quality_tier")
            )
        )
        remaining = remaining.join(
            front.select("candidate_id"), on="candidate_id", how="anti"
        )
        edges = edges.join(front.select("candidate_id"), on="candidate_id", how="anti")
    return (
        pl.concat(fronts)
        if fronts
        else group.with_columns(pl.lit(None, dtype=pl.Int64).alias("quality_tier"))
    )


def mutation_distances(
    candidates: pl.DataFrame, patterns: pl.DataFrame, positions: Sequence[str]
) -> pl.DataFrame:
    """Count differing mutated positions, including distinct substitutions once.

    Inputs contain one parent, unique candidate IDs and mutation_count; patterns
    contain candidate_id, position identity and candidate_residue. Callers own
    validation and define which scientific positions participate.
    """
    pairs = candidates.select("candidate_id", "mutation_count").join(
        candidates.select("candidate_id", "mutation_count"),
        how="cross",
        suffix="_other",
    )
    overlap = (
        patterns
        .join(patterns, on=list(positions), how="inner", suffix="_other")
        .group_by("candidate_id", "candidate_id_other")
        .agg(
            pl.len().cast(pl.Int64).alias("_shared"),
            (pl.col("candidate_residue") == pl.col("candidate_residue_other"))
            .sum()
            .cast(pl.Int64)
            .alias("_same"),
        )
    )
    return pairs.join(
        overlap, on=["candidate_id", "candidate_id_other"], how="left"
    ).select(
        "candidate_id",
        "candidate_id_other",
        (
            pl.col("mutation_count")
            + pl.col("mutation_count_other")
            - pl.col("_shared").fill_null(0)
            - pl.col("_same").fill_null(0)
        ).alias("_distance"),
    )
