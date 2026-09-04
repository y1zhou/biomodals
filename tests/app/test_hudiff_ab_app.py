"""Contracts for the HuDiff-Ab humanization app."""

# ruff: noqa: D103

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType
from typing import cast

import polars as pl
import pytest

from biomodals.app.design.hudiff_ab import app as hudiff_app
from biomodals.app.design.hudiff_ab import (
    execution,
    models,
    patches,
    upstream_runtime,
    worker,
)
from biomodals.app.design.hudiff_ab.execution import (
    COLLECT_NODE,
    HUMANIZE_NODE,
    HuDiffAbExecutionRequest,
    _HuDiffAbHumanizeNode,
)
from biomodals.execution.nodes import NodeRunContext

VH = "QVQLKQSGPGLVAPSQSLSITCTVSGFSLINYAISWVRQPPGKGLEWLGVIWTGGGTNYNSALKSRLSISKDNSKSQVFLKMNSLQTDDTARYYCARKDYYGRYYGMDYWGQGTSVTVS"
VL = "QAVVTQESALTTSPGETVTLTCRSSTGAVTTSNYANWVQEKPDHLFTGLIGGTNNRAPGVPARFSGSLIGDKAALTITGAQTEDEAIYFCALWYNNHWVFGGGTKLTVL"
VALID_CSV = f"id,vh,vl\n7k9i,{VH},{VL}\n".encode()


def _request() -> HuDiffAbExecutionRequest:
    return HuDiffAbExecutionRequest(
        run_name="example",
        csv_bytes=VALID_CSV,
        candidate_count=10,
        seed=7,
        sampling_order="shuffle",
        upstream_inference_dropout=True,
        source_commit=models.SOURCE_COMMIT,
        checkpoint_sha256=models.ANTIBODY_CHECKPOINT_SHA256,
        patch_identity=patches.patch_identity(),
        runtime_identity=models.RUNTIME_IDENTITY,
    )


def test_package_app_is_discoverable_by_contract() -> None:
    assert hudiff_app.CONF.tags == {"group": "design"}
    assert hudiff_app.CONF.name == "HuDiff-Ab"
    assert hudiff_app.EXECUTION_COORDINATOR_ENTRYPOINTS == {"submit_hudiff_ab_task"}


def test_parse_accepts_complete_unique_wide_csv() -> None:
    frame = hudiff_app.parse_hudiff_ab_csv(VALID_CSV)

    assert frame.schema == {"id": pl.String, "vh": pl.String, "vl": pl.String}
    assert frame.item(0, "id") == "7k9i"


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"id,vh\na,AAAA\n", "exactly these columns"),
        (b"id,vh,vl\na,AAAA,CCCC\na,DDDD,EEEE\n", "duplicate id"),
        (b"id,vh,vl\na,AAAA,\n", "vl must be non-empty"),
        (b"id,vh,vl\na,AAAa,CCCC\n", "uppercase canonical"),
    ),
)
def test_parse_rejects_invalid_batches(content: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        hudiff_app.parse_hudiff_ab_csv(content)


def test_controls_reject_total_attempt_overflow_and_boolean_seed() -> None:
    with pytest.raises(ValueError, match="10,000"):
        hudiff_app._validate_controls(
            pair_count=1001,
            candidate_count=10,
            seed=0,
            sampling_order="shuffle",
            upstream_inference_dropout=True,
        )
    with pytest.raises(ValueError, match="seed"):
        hudiff_app._validate_controls(
            pair_count=1,
            candidate_count=1,
            seed=True,
            sampling_order="shuffle",
            upstream_inference_dropout=True,
        )


def test_request_roundtrips_and_plans_one_gpu_task_per_pair(monkeypatch) -> None:
    request = _request()
    records = tuple(hudiff_app.parse_hudiff_ab_csv(VALID_CSV).to_dicts())
    monkeypatch.setattr(execution, "_pair_records", lambda _content: records)

    assert HuDiffAbExecutionRequest.from_bytes(request.to_bytes()) == request
    assert [node.node_key for node in request.execution_plan.nodes] == [
        HUMANIZE_NODE,
        COLLECT_NODE,
    ]
    node = _HuDiffAbHumanizeNode(request)
    task = node.discover_remote_tasks(cast(NodeRunContext, None))[0]
    call = node.prepare_remote_task(cast(NodeRunContext, None), task)
    assert call.function_name == "hudiff_ab_humanize_pair"
    assert call.uses_gpu is True
    assert call.runtime_image_key == "hudiff_ab-a10g"
    assert call.kwargs["candidate_count"] == 10
    assert call.kwargs["upstream_inference_dropout"] is True


def test_pair_seed_is_repeatable_and_input_specific() -> None:
    pair = {"id": "x", "vh": VH, "vl": VL}
    first = worker.pair_seed(4, pair)

    assert first == worker.pair_seed(4, pair)
    assert first != worker.pair_seed(5, pair)
    assert first != worker.pair_seed(4, {**pair, "id": "y"})


def test_attempt_validation_rejects_protected_position_changes() -> None:
    output = {
        "input_vh_aligned": "A" + "-" * 151,
        "input_vl_aligned": "C" + "-" * 138,
        "mutable_indices": [],
    }
    attempt = {
        "vh": "D",
        "vl": "C",
        "vh_aligned": "D" + "-" * 151,
        "vl_aligned": "C" + "-" * 138,
    }

    assert (
        worker._attempt_error(attempt, output, {"id": "x", "vh": "A", "vl": "C"})
        == "changed protected CDR or terminal position"
    )


def test_normalization_deduplicates_only_valid_attempts(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_asset_manifest", lambda: {"schema_version": 1})
    monkeypatch.setattr(worker, "_attempt_error", lambda *_args: None)
    aligned_vh = "A" + "-" * 151
    aligned_vl = "C" + "-" * 138
    output = {
        "schema_version": 1,
        "id": "x",
        "pair_seed": 9,
        "sampling_order": "shuffle",
        "upstream_inference_dropout": True,
        "device": "A10G",
        "runtime_versions": {"python": "3.10.18"},
        "input_vh_aligned": aligned_vh,
        "input_vl_aligned": aligned_vl,
        "mutable_indices": [0],
        "region_indices": [0] * 291,
        "vh_positions": ["1", *([None] * 151)],
        "vl_positions": ["1", *([None] * 138)],
        "attempts": [
            {
                "attempt_index": index,
                "vh": "A",
                "vl": "C",
                "vh_aligned": aligned_vh,
                "vl_aligned": aligned_vl,
            }
            for index in (1, 2)
        ],
    }

    result = worker._normalize_output(output, {"id": "x", "vh": "A", "vl": "C"})

    assert len(result["candidates"]) == 1
    assert [row["status"] for row in result["attempts"]] == ["valid", "duplicate"]
    assert result["attempts"][1]["duplicate_of"] == "x__candidate_1"


def test_fully_accounted_zero_yield_is_not_an_execution_failure(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_asset_manifest", lambda: {"schema_version": 1})
    monkeypatch.setattr(worker, "_attempt_error", lambda *_args: "sampled gap")
    output = {
        "schema_version": 1,
        "id": "x",
        "pair_seed": 9,
        "sampling_order": "shuffle",
        "upstream_inference_dropout": True,
        "device": "A10G",
        "runtime_versions": {"python": "3.10.18"},
        "input_vh_aligned": "A" + "-" * 151,
        "input_vl_aligned": "C" + "-" * 138,
        "mutable_indices": [0],
        "region_indices": [0] * 291,
        "vh_positions": ["1", *([None] * 151)],
        "vl_positions": ["1", *([None] * 138)],
        "attempts": [
            {
                "attempt_index": 1,
                "vh": "-",
                "vl": "C",
                "vh_aligned": "-" * 152,
                "vl_aligned": "C" + "-" * 138,
            }
        ],
    }

    result = worker._normalize_output(output, {"id": "x", "vh": "A", "vl": "C"})

    assert result["candidates"] == []
    assert result["candidate_generation_status"] == "no_valid_candidates"
    assert result["attempts"][0]["status"] == "invalid"


def test_guarded_patch_rejects_changed_preimage(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("different", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Expected one pinned HuDiff preimage"):
        patches._replace_once(source, "expected", "replacement")


def test_guarded_deletion_uses_an_idempotent_marker(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("from pymol import cmd\n", encoding="utf-8")
    marker = "# Biomodals: PyMOL omitted.\n"

    patches._replace_once(source, "from pymol import cmd\n", marker)
    patches._replace_once(source, "from pymol import cmd\n", marker)

    assert source.read_text(encoding="utf-8") == marker


def test_verified_checkpoint_audit_values_are_pinned() -> None:
    assert models.ARCHIVE_SIZE_BYTES == 2_070_382_005
    assert len(models.CHECKPOINTS) == 6
    antibody = next(
        item for item in models.CHECKPOINTS if item.path == models.ANTIBODY_CHECKPOINT
    )
    assert antibody.size_bytes == 479_136_082
    assert antibody.sha256 == models.ANTIBODY_CHECKPOINT_SHA256


@pytest.mark.parametrize(
    "module",
    (models, patches, upstream_runtime, worker),
)
def test_worker_import_closure_parses_as_python_310(module: ModuleType) -> None:
    path = Path(cast(str, module.__file__))
    ast.parse(
        path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10)
    )
