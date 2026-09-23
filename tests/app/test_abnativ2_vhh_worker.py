"""Native call contracts without model downloads, structure prediction or GPUs."""

# ruff: noqa: D103

import sys
import tarfile
from io import BytesIO
from types import SimpleNamespace

import orjson
import pandas as pd
import polars as pl
import pytest
import zstandard

from biomodals.app.design.abnativ2_vhh import worker
from biomodals.app.design.abnativ2_vhh.contracts import SearchSummary
from biomodals.app.design.vhh import VHHInput


@pytest.mark.parametrize("aligned", ["ACDE" + "-" * 145, "ACD" + "-" * 146, "ACDE"])
def test_native_alignment_preserves_exact_complete_domain(monkeypatch, aligned):
    calls = []

    def align(records, **kwargs):
        calls.append((records, kwargs))
        heavy = SimpleNamespace(to_recs=lambda: [SimpleNamespace(seq=aligned)])
        empty = SimpleNamespace(to_recs=lambda: [])
        return heavy, empty, empty, [], []

    monkeypatch.setitem(
        sys.modules,
        "abnativ.model.alignment.mybio",
        SimpleNamespace(anarci_alignments_of_Fv_sequences_iter=align),
    )
    if len(aligned) == 149 and aligned.replace("-", "") == "ACDE":
        assert worker.native_grid("ACDE") == aligned
    else:
        with pytest.raises(ValueError, match="changed the prepared"):
            worker.native_grid("ACDE")
    assert calls[0][1] == {
        "isVHH": True,
        "verbose": False,
        "run_parallel": False,
        "check_AHo_CDR_gaps": True,
        "del_cyst_misalign": False,
        "check_terms": False,
    }


def test_native_allowed_positions_intersect_framework_and_frozen_mask(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "abnativ.model.alignment.aho_consensus",
        SimpleNamespace(fr_aho_indices=[1, 3, 4, 6]),
    )
    parent = VHHInput(sequence="ACDE", protected_indices=(1, 3))
    assert worker.allowed_positions(parent, "A-CD-E" + "-" * 143) == [1, 4]


@pytest.mark.parametrize("candidate", ["ACDE", "ACDF", "AADE"])
def test_generation_retains_native_endpoint_and_rechecks_protection(
    monkeypatch, candidate
):
    calls = []
    monkeypatch.setattr(worker, "prepare_runtime", lambda: None)
    monkeypatch.setattr(worker, "native_grid", lambda value: value + "-" * 145)
    monkeypatch.setattr(worker, "allowed_positions", lambda *args: [1, 4])

    def generate(*args):
        calls.append(args)
        return [candidate], None

    monkeypatch.setattr(worker, "generate", generate)
    result = worker.humanize_vhh(
        parent=VHHInput(sequence="ACDE", protected_indices=(1,)).model_dump(),
        settings={},
    )
    report = orjson.loads(result.outputs[0].storage.data)
    assert report["attempts"][0]["sequence"] == candidate
    assert (report["attempts"][0]["error"] is None) == (candidate != "AADE")
    assert result.metrics["valid_attempts"] == int(candidate != "AADE")
    assert calls[0][:3] == ("ACDE", "ACDE" + "-" * 145, [1, 4])
    assert report["settings"] == calls[0][3].model_dump()


@pytest.mark.parametrize("model_type", ["VH2", "VHH2"])
def test_scoring_retains_raw_values_and_marks_missing_values(monkeypatch, model_type):
    monkeypatch.setattr(worker, "prepare_runtime", lambda: None)

    def align(sequence, identifier):
        if identifier == "invalid":
            raise ValueError("unsupported native grid")
        return sequence + "-" * (149 - len(sequence))

    calls = []
    monkeypatch.setattr(worker, "native_grid", align)

    def score(model, records, **kwargs):
        calls.append((model, records, kwargs))
        return pd.DataFrame({
            "seq_id": [row.id for row in records],
            "aligned_seq": [str(row.seq) for row in records],
            f"AbNatiV {model} Score": [-1.2, float("nan")],
        }), pd.DataFrame({"seq_id": ["negative"], "AHo position": [1]})

    monkeypatch.setitem(
        sys.modules,
        "abnativ.model.scoring_functions",
        SimpleNamespace(abnativ_scoring=score),
    )
    result = worker.score_vhh(
        sequences=[
            {"id": key, "sequence": "ACDE"}
            for key in ("invalid", "negative", "missing")
        ],
        model_type=model_type,
    )
    with (
        zstandard.ZstdDecompressor().stream_reader(
            BytesIO(result.outputs[0].storage.data)
        ) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        table = next(
            pl.read_parquet(BytesIO(archive.extractfile(member).read()))
            for member in archive
            if member.name.endswith("/summary.parquet")
        )
    assert table["candidate_id"].to_list() == ["invalid", "negative", "missing"]
    assert table["score"].to_list() == [None, -1.2, None]
    assert table["error"].to_list() == [
        "unsupported native grid",
        None,
        "Native model did not return a finite score",
    ]
    assert result.metrics == {"candidates": 3, "scored": 1}
    assert calls[0][0] == model_type
    assert calls[0][2] == {
        "batch_size": 32,
        "mean_score_only": False,
        "do_align": False,
        "is_VHH": True,
        "verbose": False,
    }


def test_invalid_parameters_fail_before_model_access(monkeypatch):
    def no_models():
        pytest.fail("Invalid controls must fail before reading model assets")

    monkeypatch.setattr(worker, "prepare_runtime", no_models)
    with pytest.raises(ValueError):
        worker.humanize_vhh(
            parent={"sequence": "ACDE", "protected_indices": [1]},
            settings={"rasa_threshold": float("nan")},
        )
    with pytest.raises(ValueError, match="Candidate identities"):
        worker.score_vhh(
            sequences=[{"id": "same", "sequence": "ACDE"}] * 2, model_type="VH2"
        )


def test_zero_passing_exploration_retains_successful_parent(monkeypatch):
    monkeypatch.setattr(worker, "prepare_runtime", lambda: None)
    monkeypatch.setattr(worker, "native_grid", lambda value: value + "-" * 145)
    monkeypatch.setattr(worker, "allowed_positions", lambda *_: [1, 4])
    summary = SearchSummary(
        possible_candidates="1000000000000000000000",
        evaluated_candidates=1000,
        accepted_candidates=0,
        coverage="sampled",
    )
    monkeypatch.setattr(worker, "generate", lambda *_: ([], summary))
    result = worker.humanize_vhh(
        parent={"sequence": "ACDE", "protected_indices": [1]},
        settings={"explore": True},
    )
    report = orjson.loads(result.outputs[0].storage.data)
    assert report["attempts"] == [
        {"attempt_index": 1, "sequence": "ACDE", "error": None}
    ]
    assert report["search"] == summary.model_dump()
    assert result.metrics == {"attempts": 1, "valid_attempts": 1}
