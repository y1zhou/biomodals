"""Compose pinned native science without unconsumed structural reporting."""

# ruff: noqa: PLC0415 - native packages exist only in the scientific image.

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from biomodals.app.design.abnativ2_vhh.contracts import (
    HumanizationSettings,
    SearchSummary,
)
from biomodals.app.design.abnativ2_vhh.sampling import sample_combinations


def generate(
    sequence: str,
    aligned: str,
    mutable: list[int],
    settings: HumanizationSettings,
    directory: Path,
) -> tuple[list[str], SearchSummary | None]:
    """Return the enhanced endpoint or every native-accepted explored candidate."""
    from abnativ.humanisation import (  # type: ignore[ty:unresolved-import]
        humanisation_utils as native,
    )
    from abnativ.model.scoring_functions import (  # type: ignore[ty:unresolved-import]
        abnativ_scoring,
    )
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    structures = directory / "structures"
    structures.mkdir(parents=True)
    if not settings.explore:
        candidate = native.humanise_enhanced_sampling(
            wt_seq=sequence,
            name_seq="vhh",
            nat_to_hum="VH2",
            is_VHH=True,
            pdb_file=None,
            ch_id=None,
            seq_dir=str(directory),
            allowed_user_aho_positions=mutable,
            threshold_abnativ_score=settings.residue_score_threshold,
            threshold_rasa_score=settings.rasa_threshold,
            nat_vhh="VHH2",
            perc_allowed_decrease_vhh=settings.max_relative_vhh_score_decrease,
            forbidden_mut=["C", "M"],
            a=2,
            b=1,
            pdb_dir=str(structures),
            verbose=True,
        )
        return [candidate], None

    records = [SeqRecord(Seq(sequence), id="vhh")]
    parent_scores, profiles = {}, {}
    for model in ("VH2", "VHH2"):
        means, profile = abnativ_scoring(
            model,
            records,
            batch_size=1,
            mean_score_only=False,
            do_align=True,
            is_VHH=True,
            verbose=False,
        )
        if means["aligned_seq"].tolist() != [aligned]:
            raise ValueError("Native parental scoring changed the prepared alignment")
        parent_scores[model] = means[f"AbNatiV {model} Score"].iloc[0]
        profiles[model] = profile
    if not all(np.isfinite(value) for value in parent_scores.values()):
        raise ValueError("Native parent scores are not finite")
    exposed = (
        range(1, 150)
        if settings.rasa_threshold == 0
        else native.rasa_selection_posi_to_humanise(
            sequence,
            "VH2",
            True,
            None,
            "vhh",
            None,
            settings.rasa_threshold,
            pdb_dir=str(structures),
        )
    )
    options = native.exhaustive_selection_mutation_pposi_to_humanise(
        native.get_dict_pposi_allowed_muts(0.01, "VH2", ["C", "M", "-"]),
        native.get_dict_pposi_allowed_muts(0.01, "VHH2", ["C", "M", "-"]),
        aligned,
        "VH2",
        True,
        profiles["VH2"],
        profiles["VHH2"],
        settings.residue_score_threshold,
        nat_vhh="VHH2",
        allowed_aho_positions=sorted(set(mutable).intersection(exposed)),
        verbose=True,
    )
    sample = sample_combinations(
        aligned, options, budget=settings.candidate_budget, seed=settings.sampling_seed
    )
    candidates = []
    if sample.sequences:
        records = [
            SeqRecord(Seq(value), id=f"variant-{index}")
            for index, value in enumerate(sample.sequences)
        ]
        scores = {}
        for model in ("VH2", "VHH2"):
            means, _ = abnativ_scoring(
                model,
                records,
                batch_size=128,
                mean_score_only=True,
                do_align=False,
                is_VHH=True,
                verbose=False,
            )
            if means["seq_id"].tolist() != [row.id for row in records] or means[
                "aligned_seq"
            ].tolist() != list(sample.sequences):
                raise ValueError("Native exploration scores changed candidate identity")
            scores[model] = means[f"AbNatiV {model} Score"].to_numpy()
        # The pinned native routine calls round on NumPy/pandas scalars. Use
        # NumPy's same rounding, rather than changing boundary behavior in Polars.
        accepted = (
            np.isfinite(scores["VH2"])
            & np.isfinite(scores["VHH2"])
            & (np.round(scores["VH2"] - parent_scores["VH2"], 5) >= 0)
            & (
                np.round(scores["VHH2"] - parent_scores["VHH2"], 5)
                >= -settings.max_relative_vhh_score_decrease * parent_scores["VHH2"]
            )
        )
        candidates = (
            pl
            .Series("sequence", sample.sequences)
            .filter(pl.Series(accepted))
            .str.replace_all("-", "", literal=True)
            .to_list()
        )
    return candidates, SearchSummary(
        possible_candidates=str(sample.possible_count),
        evaluated_candidates=len(sample.sequences),
        accepted_candidates=len(candidates),
        coverage="complete"
        if sample.possible_count == len(sample.sequences)
        else "sampled",
    )
