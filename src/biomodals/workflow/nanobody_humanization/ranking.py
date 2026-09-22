"""Three-objective, per-parent experimental panels for single-domain candidates."""

import polars as pl

from biomodals.helper.panel import mutation_distances, pareto_layers

RANKING_VERSION = "1"
SCORES = ("abnativ2_vh_nativeness", "abnativ2_vhh_nativeness")
KEY = ["parent_id", "candidate_id"]
RANKING_POLICY = {
    "version": RANKING_VERSION,
    "maximize": list(SCORES),
    "minimize": ["vh_mutations"],
    "score_decimal_places": 6,
    "eligibility": "nonparent with both finite candidate scores and verified protected-residue preservation; no parent-score requirement or nativeness guardrail",
    "quality_tier": "one-based Pareto layer within each parent",
    "panel_order": "finish each tier before the next; maximize minimum mutation distance to already selected designs; ties use descending VH2 then VHH2, fewer mutations, ascending candidate ID",
    "distance": "number of differing prepared-parent sequence positions, counting distinct replacements at one site once",
    "interpretation": "experimental-panel triage, not calibrated functional preservation or immunogenicity",
}


def rank_panel(table: pl.DataFrame, mutations: pl.DataFrame) -> pl.DataFrame:
    """Keep wide sequence/evidence columns out of all pairwise calculations."""
    if table.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate candidate identities")
    if mutations.join(table.select(KEY), on=KEY, how="anti").height:
        raise ValueError("Mutation references an unknown candidate")
    if (
        table
        .group_by("parent_id")
        .agg(pl.col("is_parent").sum())
        .filter(pl.col("is_parent") != 1)
        .height
    ):
        raise ValueError("Each panel requires one prepared parent")
    eligible = table.filter(
        ~pl.col("is_parent"),
        pl.all_horizontal(pl.col(name).is_finite() for name in SCORES),
        pl.col("vh_mutations") > 0,
    ).select(*KEY, *[pl.col(name).round(6) for name in SCORES], "vh_mutations")
    patterns = mutations.join(eligible.select(KEY), on=KEY, how="semi").select(
        *KEY, "sequence_index", "candidate_residue"
    )
    if patterns.select(*KEY, "sequence_index").is_duplicated().any():
        raise ValueError("Duplicate mutation position")
    counts = patterns.group_by(KEY).len()
    if (
        eligible
        .join(counts, on=KEY, how="left")
        .filter(pl.col("vh_mutations") != pl.col("len").fill_null(0))
        .height
    ):
        raise ValueError("Mutation evidence does not match the candidate count")
    pattern_groups = patterns.partition_by("parent_id", as_dict=True)
    results = []
    for group in eligible.partition_by("parent_id"):
        objectives = [*SCORES, "_negative_edits"]
        distances = mutation_distances(
            group.rename({"vh_mutations": "mutation_count"}),
            pattern_groups[(group["parent_id"][0],)].drop("parent_id"),
            ["sequence_index"],
        )
        remaining = pareto_layers(
            group.with_columns((-pl.col("vh_mutations")).alias("_negative_edits")),
            objectives,
        ).with_columns(pl.lit(2**31 - 1).alias("_nearest"))
        order = 0
        while remaining.height:
            chosen = (
                remaining
                .filter(pl.col("quality_tier") == pl.col("quality_tier").min())
                .sort(
                    "_nearest",
                    *objectives,
                    "candidate_id",
                    descending=[True, True, True, True, False],
                )
                .head(1)
            )
            order += 1
            results.append(
                chosen.select(*KEY, "quality_tier").with_columns(
                    pl.lit(order, dtype=pl.Int64).alias("panel_order")
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
                .with_columns(
                    pl.min_horizontal("_nearest", "_distance").alias("_nearest")
                )
                .drop("_distance")
            )
    ranks = (
        pl.concat(results)
        if results
        else pl.DataFrame(
            schema={
                "parent_id": pl.String,
                "candidate_id": pl.String,
                "quality_tier": pl.Int64,
                "panel_order": pl.Int64,
            }
        )
    )
    return table.join(ranks, on=KEY, how="left", validate="1:1").sort(
        "parent_id",
        "is_parent",
        "panel_order",
        "candidate_id",
        descending=[False, True, False, False],
        nulls_last=True,
    )
