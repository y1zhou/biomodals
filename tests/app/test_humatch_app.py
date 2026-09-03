"""Contracts for the Humatch humanization app."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from biomodals.app.design.humatch import app as humatch_app
from biomodals.app.design.humatch.execution import (
    HUMANIZE_NODE,
    HumatchExecutionRequest,
    _HumatchHumanizeNode,
)
from biomodals.app.design.humatch.models import ASSETS
from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import AppRunStatus, InlineBytes
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

VALID_CSV = (
    b"id,vh,vl\n"
    b"pair-1,QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMG,"
    b"DIQMTQSPSSLSASVGDRVTITCRASQSI\n"
)


def _request() -> HumatchExecutionRequest:
    return HumatchExecutionRequest(
        run_name="example",
        csv_bytes=VALID_CSV,
        vh_target_family="auto",
        vl_target_family="auto",
        germline_likeness_target=0.4,
        vh_classifier_target=0.95,
        vl_classifier_target=0.95,
        pair_classifier_target=0.95,
        max_edits=60,
        mutate_cdrs=False,
        fixed_vh_positions="",
        fixed_vl_positions="",
        app_version="humatch-source",
        asset_record="asset-record",
        runtime_identity="runtime-identity",
    )


def test_modal_group_tag_does_not_depend_on_source_path() -> None:
    assert humatch_app.CONF.tags == {"group": "design"}


def test_parse_humatch_csv_accepts_complete_unique_pairs() -> None:
    frame = humatch_app.parse_humatch_csv(VALID_CSV)

    assert frame.schema == {"id": pl.String, "vh": pl.String, "vl": pl.String}
    assert frame.to_dicts()[0]["id"] == "pair-1"


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"id,vh\na,AAAA\n", "exactly these columns"),
        (b"id,vh,vl\na,AAAA,CCCC\na,DDDD,EEEE\n", "duplicate id"),
        (b"id,vh,vl\na,AAAA,\n", "vl must be non-empty"),
        (b"id,vh,vl\na,AAAa,CCCC\n", "non-canonical or lowercase"),
    ),
)
def test_parse_humatch_csv_rejects_invalid_batches(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        humatch_app.parse_humatch_csv(content)


def test_fixed_positions_are_normalized_without_exposing_upstream_spacing() -> None:
    assert humatch_app._normalize_fixed_positions("27, 33a,120", "fixed") == (
        "27 ",
        "33A",
        "120 ",
    )
    with pytest.raises(ValueError, match="duplicate"):
        humatch_app._normalize_fixed_positions("27,27", "fixed")


def test_execution_request_roundtrips_and_plans_one_cpu_node() -> None:
    request = _request()

    assert HumatchExecutionRequest.from_bytes(request.to_bytes()) == request
    assert request.execution_plan.nodes[0].node_key == HUMANIZE_NODE
    call = _HumatchHumanizeNode(request).prepare_remote(cast(NodeRunContext, None))
    assert call.function_name == "humatch_humanize"
    assert call.uses_gpu is False
    assert call.kwargs["max_edits"] == 60


def test_execution_request_rejects_boolean_edit_limit() -> None:
    with pytest.raises(ValueError, match="max_edits"):
        replace(_request(), max_edits=True)


def test_align_chain_requires_type_coverage_and_conserved_cysteines() -> None:
    sequence = "ACCEF"
    numbering = [
        ((1, " "), "A"),
        ((23, " "), "C"),
        ((104, " "), "C"),
        ((127, " "), "E"),
        ((128, " "), "F"),
    ]
    upstream = {
        "anarci": SimpleNamespace(number=lambda *_args, **_kwargs: (numbering, "H")),
        "canonical_numbering": ("1 ", "23 ", "104 ", "127 ", "128 "),
    }

    aligned = humatch_app._align_chain(
        identifier="pair",
        chain_label="vh",
        sequence=sequence,
        upstream=upstream,
    )

    assert aligned == "ACCEF"
    upstream["anarci"] = SimpleNamespace(
        number=lambda *_args, **_kwargs: (numbering, "L")
    )
    with pytest.raises(ValueError, match="heavy chain"):
        humatch_app._align_chain(
            identifier="pair",
            chain_label="vh",
            sequence=sequence,
            upstream=upstream,
        )


def test_humanize_pair_preserves_upstream_result_and_reports_endpoints() -> None:
    canonical = (
        "1 ",
        "23 ",
        "27 ",
        "38 ",
        "56 ",
        "65 ",
        "104 ",
        "105 ",
        "117 ",
        "127 ",
        "128 ",
    )
    parental_vh = "ACCCCCCCCCC"
    parental_vl = "ACCCCCCCCCC"
    calls: dict[str, Any] = {}

    def predict(sequences, model, *, num_cpus):
        assert num_cpus == 16
        if model == "h":
            return np.asarray([[0.01, 0.90, *([0.015] * 6)]])
        if model == "l":
            return np.asarray([[0.01, 0.80, *([0.01] * 16)]])
        return np.asarray([[0.10, 0.90]])

    def max_class(probabilities, classifier_type):
        classes = (
            humatch_app.HEAVY_CLASSES
            if classifier_type == "heavy"
            else humatch_app.LIGHT_CLASSES
        )
        index = int(np.argmax(probabilities[0, 1:])) + 1
        return [(classes[index], probabilities[0, index])]

    def humanise(vh, vl, _h, _l, _p, config):
        calls["config"] = config
        return {
            "Humatch_H": "V" + vh[1:],
            "Humatch_L": vl,
            "Edit": 1,
            "HV": "hv1",
            "LV": "lv1",
            "CNN_H": np.float32(0.96),
            "CNN_L": np.float32(0.97),
            "CNN_P": np.float32(0.98),
        }

    upstream = {
        "canonical_numbering": canonical,
        "predict": predict,
        "get_max_class": max_class,
        "gl_dir": "/models/gl",
        "gl_mutate": lambda sequence, *_args, **_kwargs: sequence,
        "gl_score": lambda *_args, **_kwargs: 0.4,
        "humanise": humanise,
    }
    result = humatch_app._humanize_pair(
        pair=humatch_app._AlignedPair("pair", parental_vh, parental_vl),
        models=("h", "l", "p"),
        upstream=upstream,
        vh_target_family="auto",
        vl_target_family="auto",
        germline_likeness_target=0.4,
        vh_classifier_target=0.95,
        vl_classifier_target=0.95,
        pair_classifier_target=0.95,
        max_edits=60,
        mutate_cdrs=False,
        fixed_vh_positions=(),
        fixed_vl_positions=(),
    )

    assert result["humanized"]["vh"].startswith("V")
    assert result["summary"]["humanization_success"] is True
    assert result["summary"]["edit_count"] == 1
    assert len(result["classifier_scores"]) == 2
    assert result["mutations"] == [
        {
            "id": "pair",
            "chain": "vh",
            "imgt_position": "1",
            "region": "FR1",
            "from_aa": "A",
            "to_aa": "V",
        }
    ]
    assert calls["config"]["target_gene_H"] == "hv1"
    assert calls["config"]["target_gene_L"] == "lv1"


def test_result_bundle_has_the_stable_workflow_files(monkeypatch) -> None:
    captured: dict[str, bytes] = {}

    def fake_package_outputs(result_dir: Any, *, num_threads: int) -> bytes:
        assert num_threads == 2
        captured.update(
            (path.name, path.read_bytes()) for path in Path(result_dir).iterdir()
        )
        return b"archive"

    monkeypatch.setattr(humatch_app, "package_outputs", fake_package_outputs)
    pair_result = {
        "humanized": {"id": "pair", "vh": "AAAA", "vl": "CCCC"},
        "summary": {
            "id": "pair",
            "edit_count": 0,
            "humanization_success": True,
        },
        "classifier_scores": [{"id": "pair", "endpoint": "input", "h_neg": 0.0}],
        "alignment": [
            {
                "id": "pair",
                "chain": "vh",
                "imgt_position": "1",
                "region": "FR1",
                "input_aa": "A",
                "final_aa": "A",
                "mutated": False,
                "protected": False,
            }
        ],
        "mutations": [],
    }
    frame = pl.DataFrame([{"id": "pair", "vh": "AAAA", "vl": "CCCC"}])

    archive = humatch_app._write_result_bundle(
        run_name="demo",
        input_frame=frame,
        pair_results=[pair_result],
        parameters={},
    )

    assert archive == b"archive"
    assert set(captured) == {
        "input.csv",
        "humanized.csv",
        "humanized.fasta",
        "summary.csv",
        "classifier_scores.csv",
        "alignment.csv",
        "mutations.csv",
        "manifest.json",
    }
    assert b"accepted_mutations" not in b"".join(captured.values())


def test_workflow_function_returns_inline_archive(monkeypatch) -> None:
    metrics = {
        "pair_count": 3,
        "mutation_count": 7,
        "success_count": 2,
        "model_load_seconds": 1.0,
        "humanization_seconds": 2.0,
        "elapsed_seconds": 3.0,
    }
    monkeypatch.setattr(
        humatch_app,
        "_run_humatch_humanization",
        lambda **kwargs: (b"archive", metrics),
    )

    result = humatch_app.humatch_humanize.get_raw_f()(
        run_name="../example",
        csv_bytes=VALID_CSV,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert result.metrics == metrics
    assert result.outputs[0].storage == InlineBytes(
        data=b"archive",
        filename="example_humatch.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )


def test_image_asset_manifest_contains_only_inference_assets() -> None:
    assert len(ASSETS) == 27
    assert {asset.subdirectory for asset in ASSETS} == {
        "trained_models",
        "germline_likeness_lookup_arrays",
    }
    assert all(not asset.filename.endswith(".zip") for asset in ASSETS)
