"""Contracts for the HuDiff-Ab humanization app."""

# ruff: noqa: D103

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import orjson
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
from biomodals.schema import AppRunResult, AppRunStatus

VH = "QVQLKQSGPGLVAPSQSLSITCTVSGFSLINYAISWVRQPPGKGLEWLGVIWTGGGTNYNSALKSRLSISKDNSKSQVFLKMNSLQTDDTARYYCARKDYYGRYYGMDYWGQGTSVTVS"
VL = "QAVVTQESALTTSPGETVTLTCRSSTGAVTTSNYANWVQEKPDHLFTGLIGGTNNRAPGVPARFSGSLIGDKAALTITGAQTEDEAIYFCALWYNNHWVFGGGTKLTVL"
VALID_CSV = f"id,vh,vl\n7k9i,{VH},{VL}\n".encode()


def _install_fake_anarci(
    monkeypatch: pytest.MonkeyPatch,
    chain_types: dict[str, str],
) -> list[str]:
    calls: list[str] = []
    fake_anarci = ModuleType("anarci")

    def anarci(
        sequences: list[tuple[str, str]],
        *,
        scheme: str,
        output: bool,
        allow: set[str],
    ) -> tuple[Any, Any, Any]:
        assert scheme == "imgt"
        assert output is False
        assert allow == {"H", "K", "L"}
        assert len(sequences) == 1
        sequence = sequences[0][1]
        calls.append(sequence)
        numbering = [
            ((index, ""), residue) for index, residue in enumerate(sequence, 1)
        ]
        return (
            [[(numbering, 0, len(sequence) - 1)]],
            [[{"chain_type": chain_types[sequence]}]],
            [[object()]],
        )

    fake_anarci.__dict__["anarci"] = anarci
    monkeypatch.setitem(sys.modules, "anarci", fake_anarci)
    return calls


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


def test_multi_pair_tasks_prepare_fixed_batches_of_two(monkeypatch) -> None:
    records = tuple(
        {"id": f"pair-{index}", "vh": VH, "vl": VL} for index in range(1, 4)
    )
    monkeypatch.setattr(execution, "_pair_records", lambda _content: records)
    node = _HuDiffAbHumanizeNode(_request())
    tasks = node.discover_remote_tasks(cast(NodeRunContext, None))

    pair_call = node.prepare_remote_task_batch(cast(NodeRunContext, None), tasks[:2])
    singleton_call = node.prepare_remote_task(cast(NodeRunContext, None), tasks[2])

    assert pair_call.function_name == "hudiff_ab_humanize_batch"
    assert pair_call.max_tasks_per_call == 2
    assert [pair["id"] for pair in pair_call.kwargs["pairs"]] == [
        "pair-1",
        "pair-2",
    ]
    assert singleton_call.function_name == "hudiff_ab_humanize_batch"
    assert singleton_call.kwargs["pairs"][0]["id"] == "pair-3"


def test_batch_worker_preserves_pair_success_and_failure(monkeypatch) -> None:
    pairs = [
        {"id": "successful", "vh": VH, "vl": VL},
        {"id": "failed", "vh": VH, "vl": VL},
    ]
    monkeypatch.setattr(worker, "validate_pair", lambda _pair: None)
    monkeypatch.setattr(worker, "_prepare_upstream_runtime", lambda: None)
    monkeypatch.setattr(worker, "_asset_manifest", lambda: {})

    def fake_humanize(*, pair: dict[str, str], **_kwargs: object) -> AppRunResult:
        if pair["id"] == "failed":
            raise RuntimeError("expected failure")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    monkeypatch.setattr(worker, "hudiff_ab_humanize_pair", fake_humanize)

    raw = worker.hudiff_ab_humanize_batch(pairs=pairs)
    results = {
        pair_id: AppRunResult.model_validate(result) for pair_id, result in raw.items()
    }

    assert results["successful"].status == AppRunStatus.SUCCEEDED
    assert results["failed"].status == AppRunStatus.FAILED
    assert results["failed"].warnings == ["failed: RuntimeError: expected failure"]


def test_batch_initializes_runtime_once_before_concurrent_helpers(monkeypatch) -> None:
    pairs = [
        {"id": "one", "vh": VH, "vl": VL},
        {"id": "two", "vh": VH, "vl": VL},
    ]
    calls = {"patch": 0, "assets": 0}
    monkeypatch.setattr(worker, "validate_pair", lambda _pair: None)
    monkeypatch.setattr(
        worker,
        "apply_hudiff_inference_patches",
        lambda: calls.__setitem__("patch", calls["patch"] + 1),
    )
    monkeypatch.setattr(
        worker,
        "assert_hudiff_assets",
        lambda _root: calls.__setitem__("assets", calls["assets"] + 1) or {},
    )
    worker._prepare_upstream_runtime.cache_clear()
    worker._asset_manifest.cache_clear()

    def fake_humanize(**_kwargs: object) -> AppRunResult:
        worker._prepare_upstream_runtime()
        worker._asset_manifest()
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    monkeypatch.setattr(worker, "hudiff_ab_humanize_pair", fake_humanize)
    try:
        worker.hudiff_ab_humanize_batch(pairs=pairs)
    finally:
        worker._prepare_upstream_runtime.cache_clear()
        worker._asset_manifest.cache_clear()

    assert calls == {"patch": 1, "assets": 1}


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
        worker._attempt_error(
            attempt,
            output,
            {"id": "x", "vh": "A", "vl": "C"},
            "K",
        )
        == "changed protected CDR or terminal position"
    )


def test_attempt_validation_checks_independent_imgt_grid(monkeypatch) -> None:
    _install_fake_anarci(monkeypatch, {"AC": "H", "D": "K"})
    output = {
        "input_vh_aligned": "A-C" + "-" * 149,
        "input_vl_aligned": "D" + "-" * 138,
        "mutable_indices": [],
        "vh_positions": ["1", "2", "3", *([None] * 149)],
        "vl_positions": ["1", *([None] * 138)],
    }
    attempt = {
        "vh": "AC",
        "vl": "D",
        "vh_aligned": output["input_vh_aligned"],
        "vl_aligned": output["input_vl_aligned"],
    }

    assert (
        worker._attempt_error(
            attempt,
            output,
            {"id": "x", "vh": "AC", "vl": "D"},
            "K",
        )
        == "decoded sequence does not match independent IMGT numbering"
    )


def test_attempt_validation_preserves_parental_light_type(monkeypatch) -> None:
    _install_fake_anarci(monkeypatch, {"A": "H", "D": "L"})
    output = {
        "input_vh_aligned": "A" + "-" * 151,
        "input_vl_aligned": "C" + "-" * 138,
        "mutable_indices": [152],
        "vh_positions": ["1", *([None] * 151)],
        "vl_positions": ["1", *([None] * 138)],
    }
    attempt = {
        "vh": "A",
        "vl": "D",
        "vh_aligned": output["input_vh_aligned"],
        "vl_aligned": "D" + "-" * 138,
    }

    assert worker._attempt_error(
        attempt,
        output,
        {"id": "x", "vh": "A", "vl": "C"},
        "K",
    ) == ("generated VL changed the input chain type")


def test_normalization_deduplicates_only_valid_attempts(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_attempt_error", lambda *_args: None)
    number_calls: list[str] = []

    def fake_grid(sequence: str, positions: list[str | None]) -> tuple[str, str]:
        number_calls.append(sequence)
        return (
            sequence + "-" * (len(positions) - len(sequence)),
            "H" if sequence == "A" else "K",
        )

    monkeypatch.setattr(worker, "_imgt_grid", fake_grid)
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
    assert number_calls == ["A", "C"]


def test_fully_accounted_zero_yield_is_not_an_execution_failure(monkeypatch) -> None:
    monkeypatch.setattr(worker, "_attempt_error", lambda *_args: "sampled gap")
    monkeypatch.setattr(
        worker,
        "_imgt_grid",
        lambda sequence, positions: (
            sequence + "-" * (len(positions) - len(sequence)),
            "H" if sequence == "A" else "K",
        ),
    )
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


def test_normalization_rejects_stale_parental_grid(monkeypatch) -> None:
    _install_fake_anarci(monkeypatch, {"AC": "H", "D": "K"})
    output = {
        "schema_version": 1,
        "id": "x",
        "runtime_versions": {"python": "3.10.18"},
        "input_vh_aligned": "A-C" + "-" * 149,
        "input_vl_aligned": "D" + "-" * 138,
        "mutable_indices": [],
        "region_indices": [0] * 291,
        "vh_positions": ["1", "2", "3", *([None] * 149)],
        "vl_positions": ["1", *([None] * 138)],
        "attempts": [{"attempt_index": 1}],
    }

    with pytest.raises(ValueError, match="stale input IMGT grid"):
        worker._normalize_output(output, {"id": "x", "vh": "AC", "vl": "D"})


def test_upstream_runtime_enforces_and_reports_determinism(monkeypatch) -> None:
    calls: list[bool] = []
    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    torch = SimpleNamespace(
        use_deterministic_algorithms=lambda enabled: calls.append(enabled),
        backends=SimpleNamespace(cudnn=cudnn),
    )
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG", upstream_runtime.CUBLAS_WORKSPACE_CONFIG
    )

    upstream_runtime._configure_torch_determinism(torch)

    assert calls == [True]
    assert cudnn.deterministic is True
    assert cudnn.benchmark is False
    assert upstream_runtime.CUBLAS_WORKSPACE_CONFIG == models.CUBLAS_WORKSPACE_CONFIG
    assert upstream_runtime.CUDA_DETERMINISM_POLICY == models.CUDA_DETERMINISM_POLICY


def test_upstream_runtime_rejects_unpinned_cublas_policy(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="cuBLAS workspace policy"):
        upstream_runtime._configure_torch_determinism(SimpleNamespace())


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
    assert "hmmer=3.3.2" in models.RUNTIME_IDENTITY
    assert (
        f"resolved-environment={models.RUNTIME_ENVIRONMENT_SHA256}"
        in models.RUNTIME_IDENTITY
    )
    assert f"cuda-determinism={models.CUDA_DETERMINISM_POLICY}" in (
        models.RUNTIME_IDENTITY
    )
    assert "wrapper-protocol=2" in models.RUNTIME_IDENTITY


def test_checkpoint_publication_replaces_files_without_deleting_tree(
    monkeypatch, tmp_path: Path
) -> None:
    content = b"new checkpoint"
    checkpoint = models.CheckpointSpec(
        "checkpoints/antibody/test.pt",
        len(content),
        hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setattr(models, "CHECKPOINTS", (checkpoint,))
    root = tmp_path / "models"
    staging = tmp_path / "staging"
    staged_file = staging / checkpoint.path
    staged_file.parent.mkdir(parents=True)
    staged_file.write_bytes(content)
    retained = root / "checkpoints" / "other" / "retained.pt"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"retained")

    models._publish_checkpoints(root, staging)

    assert (root / checkpoint.path).read_bytes() == content
    assert retained.read_bytes() == b"retained"
    assert orjson.loads((root / models.MODEL_MANIFEST).read_bytes()) == (
        models._expected_manifest()
    )


@pytest.mark.parametrize(
    "module",
    (models, patches, upstream_runtime, worker),
)
def test_worker_import_closure_parses_as_python_310(module: ModuleType) -> None:
    path = Path(cast(str, module.__file__))
    ast.parse(
        path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10)
    )
