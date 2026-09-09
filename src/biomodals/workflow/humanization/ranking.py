"""Versioned, per-parent experimental-panel ordering; not a fitness predictor."""

from __future__ import annotations

import polars as pl

RANKING_VERSION = "2"
SCORE_DECIMALS = 6
MAX_PAIRING_DROP = 0.10
PAIRING_SCORES = (
    "pabnativ2_pairing_score",
    "humatch_pairing_score",
)
OBJECTIVES = (
    "pabnativ2_pair_nativeness",
    *PAIRING_SCORES,
    "humatch_mean_best_family_probability",
)
KEY = ["parent_id", "candidate_id"]
POSITION = ["chain", "position", "insertion_code"]
RANKING_POLICY = {
    "version": RANKING_VERSION,
    "maximize": list(OBJECTIVES),
    "derived_objectives": {
        "humatch_mean_best_family_probability": "(humatch_vh_best_family_probability + humatch_vl_best_family_probability) / 2",
    },
    "minimize": ["vh_mutations + vl_mutations"],
    "score_decimal_places": SCORE_DECIMALS,
    "max_relative_parental_pairing_decrease": MAX_PAIRING_DROP,
    "eligibility": "nonparent, complete candidate and parent evaluations, preserved CDRs, complete IMGT mutation evidence, positive parental pairing scores, pairing within parental guardrail",
    "quality_tier": "Pareto layers among eligible candidates per parent",
    "panel_order": "seed by tier then objective order then fewer mutations then candidate ID; subsequently maximize minimum IMGT framework distance to selected candidates within the best two remaining tiers; ties use seed order",
    "distance": "number of differing framework positions across both chains, including insertion codes and deletions; independent of generating method",
    "interpretation": "provisional triage heuristic, not calibrated experimental success or immunogenicity",
}


def rank_panel(table: pl.DataFrame, mutations: pl.DataFrame) -> pl.DataFrame:
    """Rank thin numeric frames, joining only the two ranks back to the full table."""
    if table.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate candidate identities in panel table")
    if mutations.join(table.select(KEY), on=KEY, how="anti").height:
        raise ValueError("Mutation references an unknown candidate")
    patterns = mutations.filter(pl.col("region") == "framework").select(
        *KEY, *POSITION, "candidate_residue"
    )
    if patterns.select(*KEY, *POSITION).is_duplicated().any():
        raise ValueError("Duplicate IMGT mutation position")
    if (
        table
        .group_by("parent_id")
        .agg(pl.col("is_parent").sum())
        .filter(pl.col("is_parent") != 1)
        .height
    ):
        raise ValueError("Every panel requires exactly one parental baseline")
    numeric = table.select(
        *KEY,
        "is_parent",
        "evaluation_complete",
        "cdr_preservation",
        *OBJECTIVES[:-1],
        (
            (
                pl.col("humatch_vh_best_family_probability")
                + pl.col("humatch_vl_best_family_probability")
            )
            / 2
        ).alias("humatch_mean_best_family_probability"),
        "vh_mutations",
        "vl_mutations",
    )
    parents = numeric.filter(
        pl.col("is_parent")
        & pl.col("evaluation_complete")
        & pl.all_horizontal(
            pl.col(name).is_finite() & (pl.col(name) > 0) for name in PAIRING_SCORES
        )
    ).select(
        "parent_id", *[pl.col(name).alias(f"{name}_parent") for name in PAIRING_SCORES]
    )
    counts = patterns.group_by(KEY).agg(
        (pl.col("chain") == "vh").sum().alias("_vh_count"),
        (pl.col("chain") == "vl").sum().alias("_vl_count"),
    )
    eligible = (
        numeric
        .lazy()
        .filter(
            ~pl.col("is_parent")
            & pl.col("evaluation_complete")
            & (pl.col("cdr_preservation") == "preserved")
        )
        .select(*KEY, *OBJECTIVES, "vh_mutations", "vl_mutations")
        .join(parents.lazy(), on="parent_id", how="inner")
        .join(counts.lazy(), on=KEY, how="inner")
        .filter(
            pl.all_horizontal(pl.col(name).is_finite() for name in OBJECTIVES),
            pl.col("vh_mutations") == pl.col("_vh_count"),
            pl.col("vl_mutations") == pl.col("_vl_count"),
            pl.col("vh_mutations") + pl.col("vl_mutations") > 0,
            *[
                pl.col(name).round(SCORE_DECIMALS)
                >= (pl.col(f"{name}_parent") * (1 - MAX_PAIRING_DROP)).round(
                    SCORE_DECIMALS
                )
                for name in PAIRING_SCORES
            ],
        )
        .select(
            *KEY,
            *[pl.col(name).round(SCORE_DECIMALS) for name in OBJECTIVES],
            (-(pl.col("vh_mutations") + pl.col("vl_mutations"))).alias(
                "_negative_edits"
            ),
        )
        .collect()
    )
    # Pairwise tables are bounded per parent, never
    # cross-joined across parents or with full sequences/native residue profiles.
    pattern_groups = patterns.join(
        eligible.select(KEY), on=KEY, how="semi"
    ).partition_by("parent_id", as_dict=True)
    ranks = [
        _rank_parent(group, pattern_groups[(group["parent_id"][0],)])
        for group in eligible.partition_by("parent_id")
    ]
    ranked = (
        pl.concat(ranks)
        if ranks
        else pl.DataFrame(
            schema={
                "parent_id": pl.String,
                "candidate_id": pl.String,
                "quality_tier": pl.Int64,
                "panel_order": pl.Int64,
            }
        )
    )
    columns = [*KEY, "quality_tier", "panel_order"]
    return (
        table
        .drop("quality_tier", "panel_order", strict=False)
        .join(ranked, on=KEY, how="left", validate="1:1")
        .select(*columns, pl.exclude(columns))
        .sort(
            "parent_id",
            "is_parent",
            "panel_order",
            "candidate_id",
            descending=[False, True, False, False],
            nulls_last=True,
        )
    )


def _rank_parent(group: pl.DataFrame, patterns: pl.DataFrame) -> pl.DataFrame:
    objectives = [*OBJECTIVES, "_negative_edits"]
    pairs = group.drop("parent_id").join(
        group.drop("parent_id"), how="cross", suffix="_other"
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
    patterns = patterns.drop("parent_id")
    overlap = (
        patterns
        .join(patterns, on=POSITION, how="inner", suffix="_other")
        .group_by("candidate_id", "candidate_id_other")
        .agg(
            pl.len().cast(pl.Int64).alias("_shared"),
            (pl.col("candidate_residue") == pl.col("candidate_residue_other"))
            .sum()
            .cast(pl.Int64)
            .alias("_same"),
        )
    )
    # Distinct replacements at a shared position count once; identical ones zero.
    distances = pairs.join(
        overlap, on=["candidate_id", "candidate_id_other"], how="left"
    ).select(
        "candidate_id",
        "candidate_id_other",
        (
            -pl.col("_negative_edits")
            - pl.col("_negative_edits_other")
            - pl.col("_shared").fill_null(0)
            - pl.col("_same").fill_null(0)
        ).alias("_distance"),
    )
    remaining = pl.concat(fronts).with_columns(pl.lit(2**31 - 1).alias("_nearest"))
    picks = []
    while remaining.height:
        chosen = (
            remaining
            .filter(pl.col("quality_tier") <= pl.col("quality_tier").min() + 1)
            .sort(
                "_nearest",
                "quality_tier",
                *objectives,
                "candidate_id",
                descending=[True, False, *[True for _ in objectives], False],
            )
            .head(1)
        )
        picks.append(
            chosen.select(*KEY, "quality_tier").with_columns(
                pl.lit(len(picks) + 1, dtype=pl.Int64).alias("panel_order")
            )
        )
        remaining = (
            remaining
            .join(chosen.select("candidate_id"), on="candidate_id", how="anti")
            .join(
                distances.filter(
                    pl.col("candidate_id_other") == chosen["candidate_id"][0]
                ).select("candidate_id", "_distance"),
                on="candidate_id",
            )
            .with_columns(pl.min_horizontal("_nearest", "_distance").alias("_nearest"))
            .drop("_distance")
        )
    return pl.concat(picks)
