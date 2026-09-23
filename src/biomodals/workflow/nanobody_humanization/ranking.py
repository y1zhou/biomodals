"""Three-objective, per-parent experimental panels for single-domain candidates."""

import numpy as np
import polars as pl

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
    """Exact ranking with linear working memory, never an N-by-N distance table."""
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
        group = group.sort(
            *SCORES,
            "vh_mutations",
            "candidate_id",
            descending=[True, True, False, False],
        )
        objectives = group.select(*SCORES, -pl.col("vh_mutations")).to_numpy()
        layers = np.ones(group.height, dtype=np.int64)
        # Sorted objectives put every strict dominator before its descendants.
        # Identical objective tuples belong to the same layer, not a chain.
        first = 0
        for index in range(group.height):
            if index and np.array_equal(objectives[index], objectives[index - 1]):
                layers[index] = layers[first]
                continue
            first = index
            dominates = (objectives[:index, 1] >= objectives[index, 1]) & (
                objectives[:index, 2] >= objectives[index, 2]
            )
            layers[index] = 1 + layers[:index][dominates].max(initial=0)
        patterns_for_parent = pattern_groups[(group["parent_id"][0],)]
        encoded = patterns_for_parent.with_columns(
            pl.col("candidate_residue").replace_strict(
                {
                    residue: index
                    for index, residue in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)
                },
                return_dtype=pl.UInt8,
            )
        ).pivot(on="sequence_index", index="candidate_id", values="candidate_residue")
        states = (
            group
            .select("candidate_id")
            .join(encoded, on="candidate_id", maintain_order="left")
            .drop("candidate_id")
            .fill_null(0)
            .to_numpy()
        )
        # Zero represents the prepared-parent residue at a site. Different
        # substitutions at that same site contribute one difference, not two.
        nearest = np.full(group.height, states.shape[1] + 1, dtype=np.int64)
        orders = np.zeros(group.height, dtype=np.int64)
        for order in range(1, group.height + 1):
            available = np.flatnonzero(
                (orders == 0) & (layers == layers[orders == 0].min())
            )
            chosen = available[np.argmax(nearest[available])]
            orders[chosen] = order
            nearest = np.minimum(
                nearest, np.count_nonzero(states != states[chosen], axis=1)
            )
        results.append(
            group.select(KEY).with_columns(
                pl.Series("quality_tier", layers), pl.Series("panel_order", orders)
            )
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
