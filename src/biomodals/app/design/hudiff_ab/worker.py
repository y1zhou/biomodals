"""One-pair HuDiff-Ab worker and generated-output validation."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import orjson

from biomodals.app.design.hudiff_ab.models import (
    ANTIBODY_CHECKPOINT,
    SOURCE_COMMIT,
    assert_hudiff_assets,
)
from biomodals.app.design.hudiff_ab.patches import (
    apply_hudiff_inference_patches,
    patch_identity,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
CSV_COLUMNS = frozenset({"id", "vh", "vl"})
MODEL_ROOT = Path("/biomodals-store/hudiff")
SOURCE_ROOT = Path("/opt/HuDiff")
MAX_PAIR_RESULT_BYTES = 4 * 1024 * 1024
PAIRS_PER_GPU_CALL = 2
REGION_NAMES = ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4")


def validate_pair(pair: dict[str, str]) -> None:
    """Reject malformed pair payloads before loading the checkpoint."""
    if not isinstance(pair, dict) or set(pair) != CSV_COLUMNS:
        raise ValueError("HuDiff-Ab pair must contain exactly id, vh, and vl")
    if not all(isinstance(value, str) and value for value in pair.values()):
        raise ValueError("HuDiff-Ab pair values must be non-empty text")
    if len(pair["id"]) > 200 or pair["id"] != pair["id"].strip():
        raise ValueError("HuDiff-Ab pair ID is invalid")
    if any(ord(character) < 32 or character == ">" for character in pair["id"]):
        raise ValueError("HuDiff-Ab pair ID contains unsupported characters")
    from abnumber import Chain  # type: ignore[ty:unresolved-import]

    for name, expected in (("vh", {"H"}), ("vl", {"K", "L"})):
        sequence = pair[name]
        if len(sequence) > 200 or set(sequence) - AMINO_ACIDS:
            raise ValueError(f"HuDiff-Ab {name} must be <=200 canonical residues")
        try:
            chain = Chain(sequence, scheme="imgt")
        except Exception as exc:
            raise ValueError(f"HuDiff-Ab {name} cannot be IMGT-numbered") from exc
        if chain.chain_type not in expected:
            raise ValueError(f"HuDiff-Ab {name} has the wrong chain role")


def pair_seed(base_seed: int, pair: dict[str, str]) -> int:
    """Derive scheduling-independent unsigned pair randomness."""
    digest = hashlib.sha256(
        b"\0".join(
            value.encode("utf-8")
            for value in (str(base_seed), pair["id"], pair["vh"], pair["vl"])
        )
    ).digest()
    return int.from_bytes(digest[:4], "big")


@cache
def _asset_manifest() -> dict[str, object]:
    return assert_hudiff_assets(MODEL_ROOT)


@cache
def _prepare_upstream_runtime() -> None:
    """Apply and verify the guarded patch once in each worker container."""
    apply_hudiff_inference_patches()


def _validate_controls(
    candidate_count: int,
    seed: int,
    sampling_order: str,
    upstream_inference_dropout: bool,
) -> None:
    if type(candidate_count) is not int or not 1 <= candidate_count <= 10:
        raise ValueError("candidate_count must be between 1 and 10")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if sampling_order not in {"shuffle", "left_to_right"}:
        raise ValueError("sampling_order must be shuffle or left_to_right")
    if type(upstream_inference_dropout) is not bool:
        raise ValueError("upstream_inference_dropout must be boolean")


def _attempt_error(
    attempt: dict[str, Any],
    output: dict[str, Any],
    pair: dict[str, str],
) -> str | None:
    try:
        vh = attempt["vh"]
        vl = attempt["vl"]
        vh_aligned = attempt["vh_aligned"]
        vl_aligned = attempt["vl_aligned"]
        if not all(
            isinstance(value, str) for value in (vh, vl, vh_aligned, vl_aligned)
        ):
            return "non-text sequence"
        if len(vh_aligned) != 152 or len(vl_aligned) != 139:
            return "incorrect aligned-grid length"
        if set(vh + vl) - AMINO_ACIDS:
            return "non-canonical decoded residue"
        if set(vh_aligned + vl_aligned) - (AMINO_ACIDS | {"-"}):
            return "non-canonical aligned token"
        if vh_aligned.replace("-", "") != vh or vl_aligned.replace("-", "") != vl:
            return "decoded/aligned sequence mismatch"
        input_aligned = output["input_vh_aligned"] + output["input_vl_aligned"]
        final_aligned = vh_aligned + vl_aligned
        if any(
            (before == "-") != (after == "-")
            for before, after in zip(input_aligned, final_aligned, strict=True)
        ):
            return "changed IMGT grid occupancy"
        mutable = set(output["mutable_indices"])
        if any(
            before != after and index not in mutable
            for index, (before, after) in enumerate(
                zip(input_aligned, final_aligned, strict=True)
            )
        ):
            return "changed protected CDR or terminal position"
        from abnumber import Chain  # type: ignore[ty:unresolved-import]

        if Chain(vh, scheme="imgt").chain_type != "H":
            return "generated VH has the wrong chain role"
        if Chain(vl, scheme="imgt").chain_type not in {"K", "L"}:
            return "generated VL has the wrong chain role"
        if not vh or not vl or len(vh) != len(pair["vh"]) or len(vl) != len(pair["vl"]):
            return "generated chain length changed"
    except Exception as exc:
        return f"validation failed: {type(exc).__name__}: {exc}"
    return None


def _normalize_output(output: dict[str, Any], pair: dict[str, str]) -> dict[str, Any]:
    if output.get("schema_version") != 1 or output.get("id") != pair["id"]:
        raise ValueError("HuDiff-Ab subprocess returned an invalid identity")
    for name, length in (("input_vh_aligned", 152), ("input_vl_aligned", 139)):
        value = output.get(name)
        if not isinstance(value, str) or len(value) != length:
            raise ValueError("HuDiff-Ab subprocess returned an invalid input grid")
    attempts = output.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("HuDiff-Ab subprocess returned no attempts")
    if not isinstance(output.get("runtime_versions"), dict) or not all(
        isinstance(name, str)
        and isinstance(value, (str, int))
        and not isinstance(value, bool)
        for name, value in output["runtime_versions"].items()
    ):
        raise ValueError("HuDiff-Ab subprocess returned invalid runtime versions")
    if (
        not isinstance(output.get("mutable_indices"), list)
        or not all(
            type(index) is int and 0 <= index < 291
            for index in output["mutable_indices"]
        )
        or len(set(output["mutable_indices"])) != len(output["mutable_indices"])
        or not isinstance(output.get("region_indices"), list)
        or len(output["region_indices"]) != 291
        or not all(
            type(region) is int and 0 <= region < 7
            for region in output["region_indices"]
        )
        or not isinstance(output.get("vh_positions"), list)
        or len(output["vh_positions"]) != 152
        or not isinstance(output.get("vl_positions"), list)
        or len(output["vl_positions"]) != 139
    ):
        raise ValueError("HuDiff-Ab subprocess returned invalid grid metadata")

    accepted: dict[tuple[str, str], str] = {}
    candidates: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    mutations: list[dict[str, str]] = []
    positions = output["vh_positions"] + output["vl_positions"]
    regions = output["region_indices"]
    input_aligned = output["input_vh_aligned"] + output["input_vl_aligned"]
    for source in attempts:
        if not isinstance(source, dict) or type(source.get("attempt_index")) is not int:
            raise ValueError("HuDiff-Ab subprocess returned a malformed attempt")
        error = _attempt_error(source, output, pair)
        key = (source.get("vh", ""), source.get("vl", ""))
        duplicate_of = accepted.get(key) if error is None else None
        candidate_id = None
        if error is None and duplicate_of is None:
            candidate_id = f"{pair['id']}__candidate_{len(candidates) + 1}"
            accepted[key] = candidate_id
            candidates.append({
                "id": pair["id"],
                "candidate_id": candidate_id,
                "attempt_index": source["attempt_index"],
                "vh": source["vh"],
                "vl": source["vl"],
            })
            final_aligned = source["vh_aligned"] + source["vl_aligned"]
            for index, (before, after) in enumerate(
                zip(input_aligned, final_aligned, strict=True)
            ):
                if before == after:
                    continue
                chain = "vh" if index < 152 else "vl"
                label = positions[index]
                if not isinstance(label, str):
                    raise ValueError("Mutation lacks an IMGT position")
                region_index = regions[index]
                mutations.append({
                    "id": pair["id"],
                    "candidate_id": candidate_id,
                    "chain": chain,
                    "imgt_position": label,
                    "region": REGION_NAMES[region_index],
                    "input_aa": before,
                    "final_aa": after,
                })
        status = "invalid" if error else ("duplicate" if duplicate_of else "valid")
        attempt_rows.append({
            "id": pair["id"],
            "pair_seed": output["pair_seed"],
            "attempt_index": source["attempt_index"],
            "status": status,
            "rejection_reason": error,
            "duplicate_of": duplicate_of,
            "candidate_id": candidate_id,
            "vh": source.get("vh"),
            "vl": source.get("vl"),
            "vh_aligned": source.get("vh_aligned"),
            "vl_aligned": source.get("vl_aligned"),
        })
    return {
        "schema_version": 1,
        "id": pair["id"],
        "input": pair,
        "pair_seed": output["pair_seed"],
        "sampling_order": output["sampling_order"],
        "upstream_inference_dropout": output["upstream_inference_dropout"],
        "device": output["device"],
        "runtime_versions": output["runtime_versions"],
        "attempts": attempt_rows,
        "candidates": candidates,
        "mutations": mutations,
        "candidate_generation_status": (
            "candidates_available" if candidates else "no_valid_candidates"
        ),
        "asset_manifest": _asset_manifest(),
        "patch_identity": patch_identity(),
    }


def hudiff_ab_humanize_pair(
    *,
    pair: dict[str, str],
    candidate_count: int = 10,
    seed: int = 42,
    sampling_order: str = "shuffle",
    upstream_inference_dropout: bool = True,
) -> AppRunResult:
    """Execute the pinned upstream sampler and publish one bounded pair report."""
    validate_pair(pair)
    _validate_controls(
        candidate_count, seed, sampling_order, upstream_inference_dropout
    )
    _prepare_upstream_runtime()
    manifest = _asset_manifest()
    checkpoint = MODEL_ROOT / ANTIBODY_CHECKPOINT
    derived_seed = pair_seed(seed, pair)
    request = {
        "pair": pair,
        "candidate_count": candidate_count,
        "seed": derived_seed,
        "sampling_order": sampling_order,
        "upstream_inference_dropout": upstream_inference_dropout,
    }
    with TemporaryDirectory(prefix="hudiff_ab_") as temporary:
        root = Path(temporary)
        input_path = root / "request.json"
        output_path = root / "response.json"
        input_path.write_bytes(orjson.dumps(request, option=orjson.OPT_SORT_KEYS))
        subprocess.run(  # noqa: S603 - argv and executable paths are app-owned.
            [
                sys.executable,
                str(SOURCE_ROOT / "antibody_scripts/sample_for_anti_cdr.py"),
                "--biomodals-input-json",
                str(input_path),
                "--biomodals-output-json",
                str(output_path),
                "--ckpt",
                str(checkpoint),
            ],
            cwd=SOURCE_ROOT,
            check=True,
            timeout=23 * 60 * 60,
        )
        if (
            not output_path.is_file()
            or output_path.stat().st_size > MAX_PAIR_RESULT_BYTES
        ):
            raise ValueError("HuDiff-Ab subprocess output is missing or oversized")
        raw = orjson.loads(output_path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError("HuDiff-Ab subprocess output must be an object")
    if (
        not isinstance(raw.get("attempts"), list)
        or len(raw["attempts"]) != candidate_count
    ):
        raise ValueError("HuDiff-Ab subprocess returned the wrong attempt count")
    normalized = _normalize_output(raw, pair)
    normalized["source_commit"] = SOURCE_COMMIT
    normalized["asset_manifest"] = manifest
    content = orjson.dumps({"schema_version": 1, "pair_result": normalized})
    if len(content) > MAX_PAIR_RESULT_BYTES:
        raise ValueError("HuDiff-Ab pair report exceeds its byte limit")
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="hudiff_ab_pair_result",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=content,
                    filename="hudiff-ab-pair.json",
                    media_type="application/json",
                ),
                metadata={"pair_id": pair["id"]},
            )
        ],
        metrics={
            "pair_count": 1,
            "attempt_count": len(normalized["attempts"]),
            "candidate_count": len(normalized["candidates"]),
            "mutation_count": len(normalized["mutations"]),
        },
    )


def hudiff_ab_humanize_batch(
    *,
    pairs: list[dict[str, str]],
    candidate_count: int = 10,
    seed: int = 42,
    sampling_order: str = "shuffle",
    upstream_inference_dropout: bool = True,
) -> dict[str, dict[str, Any]]:
    """Humanize at most two validated pairs in concurrent subprocesses."""
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= PAIRS_PER_GPU_CALL:
        raise ValueError("HuDiff-Ab GPU batch must contain one or two pairs")
    for pair in pairs:
        validate_pair(pair)
    if len({pair["id"] for pair in pairs}) != len(pairs):
        raise ValueError("HuDiff-Ab GPU batch IDs must be unique")
    _validate_controls(
        candidate_count, seed, sampling_order, upstream_inference_dropout
    )
    results: dict[str, AppRunResult] = {}
    with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
        futures = {
            pair["id"]: executor.submit(
                hudiff_ab_humanize_pair,
                pair=pair,
                candidate_count=candidate_count,
                seed=seed,
                sampling_order=sampling_order,
                upstream_inference_dropout=upstream_inference_dropout,
            )
            for pair in pairs
        }
        for pair_id, future in futures.items():
            try:
                results[pair_id] = future.result()
            except Exception as exc:
                results[pair_id] = AppRunResult(
                    status=AppRunStatus.FAILED,
                    warnings=[f"{pair_id}: {type(exc).__name__}: {exc}"],
                )
    return {pair["id"]: results[pair["id"]].model_dump(mode="json") for pair in pairs}
