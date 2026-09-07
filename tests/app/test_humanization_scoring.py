"""Independent scoring contracts against controlled upstream responses."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from biomodals.app.design.humatch import app as humatch
from biomodals.app.design.pabnativ2 import app as pab
from biomodals.workflow.humanization.scoring import scoring_result


def test_humatch_scoring_keeps_parent_targets_when_candidate_best_changes(monkeypatch):
    """Full classifier outputs and best-family summaries do not change reference."""

    def best(values, kind):
        classes = humatch.HEAVY_CLASSES if kind == "heavy" else humatch.LIGHT_CLASSES
        index = 1 + int(np.argmax(values[0, 1:]))
        return [(classes[index], values[0, index])]

    gl_calls = []
    upstream = {
        "get_max_class": best,
        "gl_dir": "reference",
        "gl_score": lambda seq, target, directory: (
            gl_calls.append((seq, target)) or 0.4
        ),
    }
    monkeypatch.setattr(humatch, "_load_upstream", lambda: upstream)
    monkeypatch.setattr(
        humatch, "_load_models", lambda upstream: ((None, None, None), 0)
    )
    monkeypatch.setattr(
        humatch,
        "_align_pair_record",
        lambda record, **kwargs: humatch._AlignedPair(
            record["id"], record["vh"], record["vl"]
        ),
    )

    def predict(vh, vl, *args, **kwargs):
        heavy, light = (
            np.zeros(len(humatch.HEAVY_CLASSES)),
            np.zeros(len(humatch.LIGHT_CLASSES)),
        )
        heavy[1], heavy[3] = (0.6, 0.3) if vh == "ACD" else (0.1, 0.9)
        light[1] = 0.8
        return heavy, light, np.array([0.2, 0.8])

    monkeypatch.setattr(humatch, "_predict_distributions", predict)
    inputs, summary, details = humatch._score_humatch_pairs(
        b"id,vh,vl\na,ACH,EFG\n",
        reference_vh="ACD",
        reference_vl="EFG",
    )
    assert inputs["vh"].to_list() == ["ACH"]
    row = summary.row(0, named=True)
    assert row["vh_target_family"] == "hv1"
    assert row["vh_best_family"] == "hv3"
    assert row["vh_target_probability"] == 0.1
    assert row["vh_best_family_probability"] == 0.9
    assert gl_calls[0] == ("ACH", "hv1")
    assert details["h_hv3"].to_list() == [0.9]


def test_pab_scoring_preserves_ids_scores_and_has_no_structure_stage(monkeypatch):
    """Native sequence-only scoring retains negative scores and raw pairing units."""

    def score(frame, **kwargs):
        assert kwargs["is_plotting_profiles"] is False
        assert kwargs["mean_score_only"] is False
        assert frame["ID"].to_list() == ["pair_0000"]
        values = {key: -0.2 for key in pab.SEQUENCE_SCORE_COLUMNS}
        values["AbNatiV Pairing Score (%)"] = 0.85
        mean = pd.DataFrame([
            {
                "seq_id": "pair_0000",
                "input_seq_vh": "ACD",
                "input_seq_vl": "EFG",
                **values,
            }
        ])
        profile = pd.DataFrame({
            "seq_id": ["pair_0000"] * 298,
            "AHo position": [
                f"{chain}-{position}"
                for chain in ("H", "L")
                for position in range(1, 150)
            ],
        })
        return mean, profile

    monkeypatch.setitem(
        sys.modules,
        "abnativ.model.scoring_functions",
        SimpleNamespace(abnativ_scoring_paired=score),
    )
    monkeypatch.setattr(pab, "_validate_antibody_chains", lambda frame: None)
    monkeypatch.setattr(pab, "_assert_scoring_checkpoint", lambda: None)
    monkeypatch.setattr(
        pab,
        "_humanize_pair",
        lambda **kwargs: pytest.fail("No humanization or structure prediction"),
    )
    inputs, summary, details, profile = pab._score_pabnativ2_pairs(
        b"id,vh,vl\nraw.id,ACD,EFG\n"
    )
    assert summary["id"].to_list() == ["raw.id"]
    assert summary["pair_nativeness"].to_list() == [-0.2]
    assert summary["pairing_score"].to_list() == [0.85]
    assert profile["id"].unique().to_list() == ["raw.id"]
    assert "structure_scaffold_rmsd_angstrom" not in details.columns
    assert inputs["vh"].to_list() == ["ACD"]

    monkeypatch.setitem(
        sys.modules,
        "abnativ.model.scoring_functions",
        SimpleNamespace(
            abnativ_scoring_paired=lambda *args, **kwargs: (
                pd.DataFrame({"seq_id": []}),
                pd.DataFrame(),
            )
        ),
    )
    with pytest.raises(ValueError, match="omitted or duplicated"):
        pab._score_pabnativ2_pairs(b"id,vh,vl\na,ACD,EFG\n")


def test_shared_scoring_archive_contains_separate_tables_and_file_digests(monkeypatch):
    """Summary and detailed artifacts are both covered by the manifest."""
    import orjson

    from biomodals.workflow.humanization import scoring as humanization

    inputs = pl.DataFrame({"id": ["a"], "vh": ["ACD"], "vl": ["EFG"]})
    summary = pl.DataFrame({"id": ["a"], "pairing_score": [0.5]})

    def package(root, **kwargs):
        manifest = orjson.loads((root / "manifest.json").read_bytes())
        assert set(manifest["files"]) == {
            "input.csv",
            "summary.csv",
            "classifier_scores.parquet",
        }
        assert all(len(value["sha256"]) == 64 for value in manifest["files"].values())
        assert pl.read_csv(root / "summary.csv").equals(summary)
        return b"archive"

    monkeypatch.setattr(humanization, "package_outputs", package)
    result = scoring_result(
        "humatch",
        inputs,
        summary,
        {"classifier_scores": summary},
        {"runtime": "pinned"},
    )
    assert result.outputs[0].storage.data == b"archive"
