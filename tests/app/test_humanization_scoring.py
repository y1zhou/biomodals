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


def test_humatch_scoring_keeps_parent_targets_when_candidate_best_changes(
    monkeypatch, request
):
    """Full classifier outputs and best-family summaries do not change reference."""
    humatch._reference_targets.cache_clear()
    request.addfinalizer(humatch._reference_targets.cache_clear)

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

    predictions = []
    failures = set()

    def predict(vh, vl, *args, **kwargs):
        predictions.append((vh, vl))
        if (vh, vl) in failures:
            failures.remove((vh, vl))
            raise ValueError("Native scoring failed")
        heavy, light = (
            np.zeros(len(humatch.HEAVY_CLASSES)),
            np.zeros(len(humatch.LIGHT_CLASSES)),
        )
        heavy[1], heavy[3] = (0.6, 0.3) if vh == "ACD" else (0.1, 0.9)
        light[1], light[3] = (0.8, 0.1) if vl == "EFG" else (0.2, 0.9)
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

    def score(vh="ACD", vl="EFG", **targets):
        return humatch._score_humatch_pairs(
            b"id,vh,vl\na,ACH,EFG\n",
            reference_vh=vh,
            reference_vl=vl,
            **targets,
        )

    for _ in range(3):
        repeated = score()
        assert all(
            actual.equals(expected)
            for actual, expected in zip(
                repeated, (inputs, summary, details), strict=True
            )
        )
    assert predictions.count(("ACD", "EFG")) == 1
    assert predictions.count(("ACH", "EFG")) == 4
    assert score(vh="ACC")[1]["vh_target_family"].to_list() == ["hv3"]
    assert score(vl="EFA")[1]["vl_target_family"].to_list() == ["lv3"]
    assert score(vh_target_family="hv2")[1]["vh_target_family"].to_list() == ["hv2"]
    assert score(vl_target_family="kv1")[1]["vl_target_family"].to_list() == ["kv1"]

    failures.add(("ACH", "EFG"))
    before = predictions.count(("ACD", "EFG"))
    with pytest.raises(ValueError, match="Native scoring failed"):
        score()
    assert score()[1].equals(summary)
    assert predictions.count(("ACD", "EFG")) == before
    failures.add(("AAA", "EFG"))
    with pytest.raises(ValueError, match="Native scoring failed"):
        score(vh="AAA")
    score(vh="AAA")
    assert predictions.count(("AAA", "EFG")) == 2

    # More distinct parents than the bounded cache can retain must evict old work.
    for suffix in range(129):
        sequence = "C" + "".join(
            "A" if suffix & (1 << bit) else "D" for bit in range(8)
        )
        score(vh=sequence)
    before = predictions.count(("ACD", "EFG"))
    assert score()[1].equals(summary)
    assert predictions.count(("ACD", "EFG")) == before + 1


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
    inputs, summary, _details, profile = pab._score_pabnativ2_pairs(
        b"id,vh,vl\nraw.id,ACD,EFG\n"
    )
    assert summary["id"].to_list() == ["raw.id"]
    assert summary["pair_nativeness"].to_list() == [-0.2]
    assert summary["pairing_score"].to_list() == [0.85]
    assert profile["id"].unique().to_list() == ["raw.id"]
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
