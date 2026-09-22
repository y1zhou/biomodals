"""Exact per-parent candidate identity and thin Polars mutation evidence."""

import hashlib
from collections.abc import Sequence

import polars as pl

from biomodals.workflow.nanobody_humanization.preparation import PreparedVH

KEY = ["parent_id", "candidate_id"]
GENERATION_SCHEMA = {
    "parent_id": pl.String,
    "method": pl.String,
    "attempt_index": pl.Int64,
    "seed": pl.Int64,
    "vh": pl.String,
    "error": pl.String,
}
MUTATION_SCHEMA = {
    "parent_id": pl.String,
    "candidate_id": pl.String,
    "sequence_index": pl.Int64,
    "imgt_position": pl.String,
    "parent_residue": pl.String,
    "candidate_residue": pl.String,
}


def candidate_union(
    parents: Sequence[PreparedVH], generation: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deduplicate exact sequences within parents, retaining every attempt's origin."""
    baselines = pl.DataFrame(
        {
            "parent_id": [parent.id for parent in parents],
            "vh": [parent.sequence for parent in parents],
        },
        schema={"parent_id": pl.String, "vh": pl.String},
    )
    if not parents or baselines["parent_id"].is_duplicated().any():
        raise ValueError("Expected nonempty, unique prepared parents")
    if generation.join(
        baselines.select("parent_id"), on="parent_id", how="anti"
    ).height:
        raise ValueError("Generation references an unknown parent")
    valid = generation.filter(pl.col("error").is_null() & pl.col("vh").is_not_null())
    candidates = (
        pl
        .concat([baselines, valid.select("parent_id", "vh")])
        .unique(maintain_order=True)
        .with_columns(
            # A portable scientific identity, unlike Polars' version-dependent hash.
            pl
            .concat_str("parent_id", "vh", separator="\x00")
            .map_elements(
                lambda value: hashlib.sha256(value.encode()).hexdigest(),
                return_dtype=pl.String,
            )
            .alias("candidate_id")
        )
        .join(
            baselines.with_columns(pl.lit(True).alias("is_parent")),
            on=["parent_id", "vh"],
            how="left",
            validate="m:1",
        )
        .with_columns(pl.col("is_parent").fill_null(False))
    )
    origins = generation.join(
        candidates.select(*KEY, "vh"),
        on=["parent_id", "vh"],
        how="left",
        validate="m:1",
    )
    methods = valid.group_by("parent_id", "vh").agg(
        pl.col("method").unique().sort().str.join(";").alias("generating_methods")
    )
    candidates = (
        candidates
        .join(methods, on=["parent_id", "vh"], how="left", validate="1:1")
        .with_columns(
            pl
            .when(~pl.col("is_parent"))
            .then(pl.col("generating_methods"))
            .alias("generating_methods")
        )
        .sort("parent_id", "is_parent", "candidate_id", descending=[False, True, False])
        .select(*KEY, "vh", "is_parent", "generating_methods")
    )
    return candidates, origins


def mutation_table(
    candidates: pl.DataFrame, parents: Sequence[PreparedVH]
) -> pl.DataFrame:
    """Validate every accepted substitution against the frozen parental mask."""
    references = pl.DataFrame({
        "parent_id": [parent.id for parent in parents],
        "_parent": [parent.sequence for parent in parents],
        "imgt_position": [list(parent.imgt_positions) for parent in parents],
        "_protected": [
            [index in parent.protected_indices for index in range(len(parent.sequence))]
            for parent in parents
        ],
    })
    joined = candidates.join(references, on="parent_id", how="left", validate="m:1")
    if joined.filter(
        pl.col("_parent").is_null()
        | (pl.col("vh").str.len_chars() != pl.col("_parent").str.len_chars())
        | ~pl.col("vh").str.contains("^[ACDEFGHIKLMNPQRSTVWY]+$")
    ).height:
        raise ValueError("Candidate changed prepared domain length or alphabet")
    mutations = (
        joined
        .select(
            *KEY,
            pl.int_ranges(0, pl.col("vh").str.len_chars(), dtype=pl.Int64).alias(
                "sequence_index"
            ),
            "imgt_position",
            "_protected",
            pl.col("_parent").str.split("").alias("parent_residue"),
            pl.col("vh").str.split("").alias("candidate_residue"),
        )
        .explode(
            "sequence_index",
            "imgt_position",
            "_protected",
            "parent_residue",
            "candidate_residue",
            empty_as_null=True,
        )
        .filter(pl.col("parent_residue") != pl.col("candidate_residue"))
    )
    if mutations["_protected"].any():
        raise ValueError("Candidate changed a protected parental residue")
    return mutations.select(*MUTATION_SCHEMA)


def add_scores(
    candidates: pl.DataFrame, mutations: pl.DataFrame, scores: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """Left-join all candidates and compute deltas only from available baselines."""
    counts = mutations.group_by(KEY).len(name="vh_mutations")
    result = candidates.join(counts, on=KEY, how="left", validate="1:1").with_columns(
        pl.col("vh_mutations").fill_null(0).cast(pl.Int64)
    )
    for model, column in (
        ("VH2", "abnativ2_vh_nativeness"),
        ("VHH2", "abnativ2_vhh_nativeness"),
    ):
        frame = scores.get(
            model,
            pl.DataFrame(
                schema={
                    "candidate_id": pl.String,
                    "score": pl.Float64,
                    "error": pl.String,
                }
            ),
        )
        if frame.join(
            candidates.select("candidate_id"), on="candidate_id", how="anti"
        ).height:
            raise ValueError("Evaluation references an unknown candidate")
        result = result.join(
            frame.select(
                "candidate_id",
                pl.col("score").alias(column),
                pl.col("error").alias(f"{column}_error"),
            ),
            on="candidate_id",
            how="left",
            validate="1:1",
        ).with_columns(
            pl.when(pl.col(column).is_finite()).then(pl.col(column)).alias(column),
            pl
            .when(pl.col(column).is_finite())
            .then(pl.col(f"{column}_error"))
            .otherwise(pl.col(f"{column}_error").fill_null("Evaluation unavailable"))
            .alias(f"{column}_error"),
        )
        baseline = result.filter(pl.col("is_parent")).select(
            "parent_id", pl.col(column).alias("_baseline")
        )
        result = (
            result
            .join(baseline, on="parent_id", how="left", validate="m:1")
            .with_columns(
                (pl.col(column) - pl.col("_baseline")).alias(f"{column}_delta")
            )
            .drop("_baseline")
        )
    return result.with_columns(
        pl
        .all_horizontal(
            pl.col("abnativ2_vh_nativeness").is_finite(),
            pl.col("abnativ2_vhh_nativeness").is_finite(),
        )
        .fill_null(False)
        .alias("evaluation_complete")
    )
