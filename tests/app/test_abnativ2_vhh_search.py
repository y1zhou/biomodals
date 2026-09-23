"""Native composition and scalar acceptance oracle, without model inference."""

# ruff: noqa: D103

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biomodals.app.design.abnativ2_vhh.contracts import HumanizationSettings
from biomodals.app.design.abnativ2_vhh.search import generate


def test_enhanced_delegates_unchanged_native_search_controls(monkeypatch, tmp_path):
    calls = []

    def enhanced(**kwargs):
        calls.append(kwargs)
        return "ACDF"

    monkeypatch.setitem(
        sys.modules,
        "abnativ.humanisation",
        SimpleNamespace(
            humanisation_utils=SimpleNamespace(humanise_enhanced_sampling=enhanced)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "abnativ.model.scoring_functions",
        SimpleNamespace(abnativ_scoring=None),
    )
    candidates, summary = generate(
        "ACDE",
        "ACDE" + "-" * 145,
        [4],
        HumanizationSettings(rasa_threshold=0),
        tmp_path,
    )
    assert candidates == ["ACDF"]
    assert summary is None
    call = calls[0]
    assert (call["nat_to_hum"], call["nat_vhh"], call["is_VHH"]) == (
        "VH2",
        "VHH2",
        True,
    )
    assert (call["a"], call["b"], call["forbidden_mut"]) == (2, 1, ["C", "M"])
    assert (
        call["threshold_abnativ_score"],
        call["threshold_rasa_score"],
        call["perc_allowed_decrease_vhh"],
    ) == (0.98, 0, 0.05)
    assert call["allowed_user_aho_positions"] == [4]


@pytest.mark.parametrize("rasa", [0, 0.15])
@pytest.mark.parametrize("budget", [3, 1000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exploration_gates_all_samples_against_original_parent(
    monkeypatch, tmp_path, rasa, budget, dtype
):
    aligned = "AAAA" + "-" * 145
    calls, evaluated, exposure_calls = [], {}, []

    def score(model, records, **kwargs):
        calls.append((model, kwargs))
        if kwargs["do_align"]:
            assert [str(row.seq) for row in records] == ["AAAA"]
            return pd.DataFrame({
                "aligned_seq": [aligned],
                f"AbNatiV {model} Score": np.array([0.8], dtype=dtype),
            }), pd.DataFrame({"model": [model]})
        # Rejections at either native threshold; finite negative values remain
        # meaningful, but NaN/inf can never be an accepted candidate.
        values = np.array(
            (
                [-0.000004, -0.000006, 0.1, 0.2, 0.1, np.inf, 0.1, 0.1]
                if model == "VH2"
                else [-0.04, 0, -0.040006, -0.03, np.nan, 0, -0.039994, 0]
            ),
            dtype=dtype,
        )[: len(records)] + dtype(0.8)
        evaluated[model] = ([str(row.seq) for row in records], values)
        return pd.DataFrame({
            "seq_id": [row.id for row in records],
            "aligned_seq": [str(row.seq) for row in records],
            f"AbNatiV {model} Score": values,
        }), None

    def pssm(threshold, model, forbidden):
        assert (threshold, forbidden) == (0.01, ["C", "M", "-"])
        return model

    def options(*args, **kwargs):
        assert args[:5] == ("VH2", "VHH2", aligned, "VH2", True)
        assert args[5]["model"][0] == "VH2"
        assert args[6]["model"][0] == "VHH2"
        assert args[7] == 0.98
        assert kwargs["nat_vhh"] == "VHH2"
        assert kwargs["allowed_aho_positions"] == ([1, 2] if rasa else [1, 2, 3])
        return ["ADE", "AGH", *aligned[2:]]

    def exposure(*args, **kwargs):
        exposure_calls.append(args)
        return [1, 2]

    native = SimpleNamespace(
        get_dict_pposi_allowed_muts=pssm,
        exhaustive_selection_mutation_pposi_to_humanise=options,
        rasa_selection_posi_to_humanise=exposure,
    )
    monkeypatch.setitem(
        sys.modules, "abnativ.humanisation", SimpleNamespace(humanisation_utils=native)
    )
    monkeypatch.setitem(
        sys.modules,
        "abnativ.model.scoring_functions",
        SimpleNamespace(abnativ_scoring=score),
    )
    candidates, summary = generate(
        "AAAA",
        aligned,
        [1, 2, 3],
        HumanizationSettings(
            explore=True, candidate_budget=budget, rasa_threshold=rasa
        ),
        tmp_path,
    )
    sequences, vh = evaluated["VH2"]
    _, vhh = evaluated["VHH2"]
    # Exact pinned native scalar operations, including its five-place rounding.
    expected = [
        sequence.replace("-", "")
        for sequence, h, n in zip(sequences, vh, vhh, strict=True)
        if np.isfinite(h)
        and np.isfinite(n)
        and round(h - dtype(0.8), 5) >= 0
        and round(n - dtype(0.8), 5) >= -0.05 * dtype(0.8)
    ]
    assert candidates == expected
    assert summary.evaluated_candidates == min(8, budget)
    assert summary.accepted_candidates == len(expected)
    assert summary.possible_candidates == "8"
    assert summary.coverage == ("complete" if budget >= 8 else "sampled")
    assert len(exposure_calls) == int(rasa > 0)
    assert [call[1]["mean_score_only"] for call in calls] == [False, False, True, True]
    assert len(calls) == 4  # No acceptance-driven refill or scoring per candidate.
