"""Durable AlphaFold 3 seed predictions and accumulated run summaries.

Seed markers are the sole reusable prediction boundary. Workers stage upstream
output privately, promote complete seed directories, commit them, and publish
one marker per seed last. Global summaries are derived mutable state guarded by
their own append-only generation claim.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import orjson
import polars as pl

from biomodals.app.fold.alphafold3.artifacts import (
    VolumeHandle,
    artifact_record,
    require_regular_file,
    sha256_bytes,
    utc_now,
    validate_artifact_record,
    write_json_atomic,
)
from biomodals.app.fold.alphafold3.generation_claims import (
    ActiveGenerationError,
    ClaimStore,
    GenerationClaim,
    acquire_generation_claim,
    assert_generation_current,
    finish_generation_claim,
)
from biomodals.app.fold.alphafold3.inference_inputs import hash_sequences

SEED_PREDICTION_CLAIM_DICT_NAME = "AlphaFold3-inference-claims"

SEED_MARKER_SCHEMA_VERSION = 1
SUMMARY_MARKER_SCHEMA_VERSION = 1
SEED_CLAIM_SCHEMA_VERSION = 1
SUMMARY_CLAIM_SCHEMA_VERSION = 1

CORE_OUTPUT_SUFFIXES = (
    "model.cif",
    "confidences.json",
    "summary_confidences.json",
)
_SUMMARY_ARTIFACT_FILENAMES = {
    "data": "data.json",
    "ranking": "ranking_scores.csv",
    "model": "model.cif",
    "confidences": "confidences.json",
    "summary_confidences": "summary_confidences.json",
    "terms": "TERMS_OF_USE.md",
}


type PredictionExecutor = Callable[[Path, str, tuple[int, ...]], None]


@dataclass(frozen=True, slots=True)
class InferenceRuntime:
    """Mounted output store and coordination state for inference publications."""

    output_root: Path
    volume: VolumeHandle
    claims: ClaimStore
    container_id: str
    maximum_age_seconds: int | float
    summary_maximum_age_seconds: int | float
    wait_timeout_seconds: int | float
    claim_poll_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class RankingRow:
    """One seed/sample ranking result."""

    seed: int
    sample_index: int
    ranking_score: float

    def to_dict(self) -> dict[str, int | float]:
        """Return the marker/table representation."""
        return {
            "seed": self.seed,
            "sample_index": self.sample_index,
            "ranking_score": self.ranking_score,
        }


@dataclass(frozen=True, slots=True)
class SeedMarker:
    """One validated marker-complete seed prediction."""

    run_id: str
    seed: int
    generation_id: str
    rankings: tuple[RankingRow, ...]
    marker_sha256: str

    def summary(self, status: str) -> dict[str, object]:
        """Return compact reconciliation metadata."""
        return {
            "status": status,
            "run_id": self.run_id,
            "seed": self.seed,
            "generation_id": self.generation_id,
            "marker_sha256": self.marker_sha256,
            "rankings": [row.to_dict() for row in self.rankings],
        }


@dataclass(frozen=True, slots=True)
class ClaimedSeed:
    """One seed generation owned by this request."""

    seed: int
    claim: GenerationClaim

    def to_dict(self) -> dict[str, object]:
        """Return the Modal-serializable claim record."""
        return {
            "seed": self.seed,
            "claim": {
                "scope_key": self.claim.scope_key,
                "generation_id": self.claim.generation_id,
                "owner": self.claim.owner,
            },
        }


@dataclass(frozen=True, slots=True)
class ActiveSeed:
    """One seed currently owned by another request."""

    seed: int
    generation_id: str


@dataclass(frozen=True, slots=True)
class SeedClaimPlan:
    """Current marker reuse, owned claims, and external active owners."""

    reused_seeds: tuple[int, ...]
    owned: tuple[ClaimedSeed, ...]
    active: tuple[ActiveSeed, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a Modal-serializable claim plan."""
        return {
            "reused_seeds": list(self.reused_seeds),
            "owned": [item.to_dict() for item in self.owned],
            "active": [
                {
                    "seed": item.seed,
                    "generation_id": item.generation_id,
                }
                for item in self.active
            ],
        }


@dataclass(frozen=True, slots=True)
class SeedWorkerTask:
    """One disjoint multi-seed GPU worker assignment."""

    run_id: str
    sample_count: int
    claimed_seeds: tuple[ClaimedSeed, ...]


@dataclass(frozen=True, slots=True)
class SummaryEntry:
    """One validated accumulated run summary."""

    run_id: str
    included_seeds: tuple[int, ...]
    best: RankingRow
    artifacts: dict[str, dict[str, object]]
    marker_sha256: str

    def summary(self, status: str) -> dict[str, object]:
        """Return compact summary metadata."""
        return {
            "status": status,
            "run_id": self.run_id,
            "included_seeds": list(self.included_seeds),
            "best": self.best.to_dict(),
            "artifacts": self.artifacts,
            "marker_sha256": self.marker_sha256,
        }


def validate_run_id(run_id: str) -> str:
    """Validate one full seed-independent inference run digest."""
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{64}", run_id) is None:
        raise ValueError("run_id must be a lowercase SHA-256 digest")
    return run_id


def _validate_sample_count(sample_count: int) -> int:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer")
    return sample_count


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


def canonical_output_name(run_id: str) -> str:
    """Return the caller-independent upstream output prefix."""
    return f"af3-{validate_run_id(run_id)[:16]}"


def inference_run_root(output_root: Path, run_id: str) -> Path:
    """Return the hash-fanned run root below the output Volume mount."""
    selected_run = validate_run_id(run_id)
    return Path(output_root) / selected_run[:2] / selected_run


def _seed_marker_path(run_root: Path, seed: int) -> Path:
    return run_root / ".markers" / "seeds" / f"{_validate_seed(seed)}.json"


def _parse_ranking_rows(
    value: object,
    *,
    expected_seed: int | None = None,
    expected_sample_count: int | None = None,
) -> tuple[RankingRow, ...] | None:
    if not isinstance(value, list):
        return None
    rows: list[RankingRow] = []
    for raw_row in value:
        if not isinstance(raw_row, dict):
            return None
        seed = raw_row.get("seed", expected_seed)
        sample_index = raw_row.get("sample_index")
        ranking_score = raw_row.get("ranking_score")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
            or isinstance(ranking_score, bool)
            or not isinstance(ranking_score, int | float)
            or not math.isfinite(float(ranking_score))
        ):
            return None
        if expected_seed is not None and seed != expected_seed:
            return None
        rows.append(
            RankingRow(
                seed=seed,
                sample_index=sample_index,
                ranking_score=float(ranking_score),
            )
        )
    rows.sort(key=lambda row: (row.seed, row.sample_index))
    if len({(row.seed, row.sample_index) for row in rows}) != len(rows):
        return None
    if expected_sample_count is not None:
        expected_indices = tuple(range(expected_sample_count))
        if tuple(row.sample_index for row in rows) != expected_indices:
            return None
    return tuple(rows)


def load_seed_marker(
    run_root: Path,
    run_id: str,
    seed: int,
    *,
    sample_count: int,
) -> SeedMarker | None:
    """Load one matching marker without inspecting prediction directories."""
    selected_run = validate_run_id(run_id)
    selected_seed = _validate_seed(seed)
    selected_samples = _validate_sample_count(sample_count)
    marker_path = _seed_marker_path(run_root, selected_seed)
    if not marker_path.is_file():
        return None
    try:
        marker_bytes = marker_path.read_bytes()
        marker = orjson.loads(marker_bytes)
    except (OSError, orjson.JSONDecodeError):
        return None
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != SEED_MARKER_SCHEMA_VERSION
        or marker.get("status") != "complete"
        or marker.get("run_id") != selected_run
        or marker.get("seed") != selected_seed
        or marker.get("sample_count") != selected_samples
        or not isinstance(marker.get("generation_id"), str)
        or not marker["generation_id"]
    ):
        return None
    rankings = _parse_ranking_rows(
        marker.get("rankings"),
        expected_seed=selected_seed,
        expected_sample_count=selected_samples,
    )
    if rankings is None:
        return None
    return SeedMarker(
        run_id=selected_run,
        seed=selected_seed,
        generation_id=cast(str, marker["generation_id"]),
        rankings=rankings,
        marker_sha256=sha256_bytes(marker_bytes),
    )


def inspect_seed_predictions(
    runtime: InferenceRuntime,
    run_id: str,
    seeds: tuple[int, ...],
    *,
    sample_count: int,
) -> list[dict[str, object]]:
    """Inspect requested seed markers without walking output artifacts."""
    selected_run = validate_run_id(run_id)
    selected_samples = _validate_sample_count(sample_count)
    selected_seeds = tuple(_validate_seed(seed) for seed in seeds)
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seed inspection inputs must be unique")
    runtime.volume.reload()
    run_root = inference_run_root(runtime.output_root, selected_run)
    statuses: list[dict[str, object]] = []
    for seed in selected_seeds:
        marker = load_seed_marker(
            run_root,
            selected_run,
            seed,
            sample_count=selected_samples,
        )
        statuses.append(
            marker.summary("reused")
            if marker is not None
            else {
                "status": "missing",
                "run_id": selected_run,
                "seed": seed,
            }
        )
    return statuses


def _seed_claim_scope(run_id: str, seed: int) -> str:
    return f"seed:{validate_run_id(run_id)}:{_validate_seed(seed)}"


def _seed_claim_identity(run_id: str, seed: int) -> dict[str, object]:
    return {
        "schema_version": SEED_CLAIM_SCHEMA_VERSION,
        "run_id": validate_run_id(run_id),
        "seed": _validate_seed(seed),
    }


def claim_seed_predictions(
    runtime: InferenceRuntime,
    run_id: str,
    seeds: tuple[int, ...],
    *,
    sample_count: int,
) -> SeedClaimPlan:
    """Reuse marked seeds and atomically claim every currently missing seed."""
    selected_run = validate_run_id(run_id)
    selected_samples = _validate_sample_count(sample_count)
    selected_seeds = tuple(sorted(_validate_seed(seed) for seed in seeds))
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("claim inputs must be a non-empty unique seed set")
    runtime.volume.reload()
    run_root = inference_run_root(runtime.output_root, selected_run)
    reused: list[int] = []
    claimed: list[ClaimedSeed] = []
    active: list[ActiveSeed] = []
    for seed in selected_seeds:
        if (
            load_seed_marker(
                run_root,
                selected_run,
                seed,
                sample_count=selected_samples,
            )
            is not None
        ):
            reused.append(seed)
            continue
        try:
            claim = acquire_generation_claim(
                runtime.claims,
                scope_key=_seed_claim_scope(selected_run, seed),
                generation_id=uuid.uuid4().hex,
                identity=_seed_claim_identity(selected_run, seed),
                container_id=runtime.container_id,
                maximum_age_seconds=runtime.maximum_age_seconds,
            )
        except ActiveGenerationError as exc:
            active.append(
                ActiveSeed(
                    seed=seed,
                    generation_id=cast(str, exc.owner["generation_id"]),
                )
            )
            continue

        claimed.append(ClaimedSeed(seed=seed, claim=claim))

    owned: list[ClaimedSeed] = []
    if claimed:
        runtime.volume.reload()
    for item in claimed:
        raced_marker = load_seed_marker(
            run_root,
            selected_run,
            item.seed,
            sample_count=selected_samples,
        )
        if raced_marker is not None:
            finish_generation_claim(
                runtime.claims,
                item.claim,
                status="complete",
                detail={
                    "publication": "raced",
                    "marker_sha256": raced_marker.marker_sha256,
                },
            )
            reused.append(item.seed)
        else:
            owned.append(item)
    return SeedClaimPlan(
        reused_seeds=tuple(reused),
        owned=tuple(owned),
        active=tuple(active),
    )


def claimed_seed_from_dict(value: object) -> ClaimedSeed:
    """Parse one coordinator claim record returned through Modal."""
    if not isinstance(value, dict):
        raise TypeError("Claimed seed record must be a dictionary")
    seed = _validate_seed(cast(int, value.get("seed")))
    raw_claim = value.get("claim")
    if not isinstance(raw_claim, dict):
        raise TypeError("Claimed seed record has no claim")
    scope_key = raw_claim.get("scope_key")
    generation_id = raw_claim.get("generation_id")
    owner = raw_claim.get("owner")
    if (
        not isinstance(scope_key, str)
        or not isinstance(generation_id, str)
        or not isinstance(owner, dict)
    ):
        raise TypeError("Claimed seed record contains an invalid claim")
    return ClaimedSeed(
        seed=seed,
        claim=GenerationClaim(
            scope_key=scope_key,
            generation_id=generation_id,
            owner=cast(dict[str, object], owner),
        ),
    )


def seed_claim_plan_from_dict(value: object) -> SeedClaimPlan:
    """Parse a claim plan returned by the lightweight Modal coordinator."""
    if not isinstance(value, dict):
        raise TypeError("Seed claim plan must be a dictionary")
    raw_reused = value.get("reused_seeds")
    raw_owned = value.get("owned")
    raw_active = value.get("active")
    if (
        not isinstance(raw_reused, list)
        or not isinstance(raw_owned, list)
        or not isinstance(raw_active, list)
    ):
        raise TypeError("Seed claim plan fields must be lists")
    reused = tuple(_validate_seed(cast(int, seed)) for seed in raw_reused)
    owned = tuple(claimed_seed_from_dict(item) for item in raw_owned)
    active: list[ActiveSeed] = []
    for item in raw_active:
        if not isinstance(item, dict):
            raise TypeError("Active seed record must be a dictionary")
        seed = _validate_seed(cast(int, item.get("seed")))
        generation_id = item.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            raise TypeError("Active seed generation must be a non-empty string")
        active.append(ActiveSeed(seed=seed, generation_id=generation_id))
    all_seeds = (
        list(reused) + [item.seed for item in owned] + [item.seed for item in active]
    )
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Seed claim plan contains duplicate seeds")
    return SeedClaimPlan(
        reused_seeds=reused,
        owned=owned,
        active=tuple(active),
    )


def partition_claimed_seeds(
    claimed_seeds: tuple[ClaimedSeed, ...],
    max_workers: int,
) -> tuple[tuple[ClaimedSeed, ...], ...]:
    """Partition owned seeds into balanced, disjoint worker assignments."""
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer")
    if not claimed_seeds:
        return ()
    seeds = [item.seed for item in claimed_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("claimed seeds must be unique")
    worker_count = min(max_workers, len(claimed_seeds))
    return tuple(
        tuple(claimed_seeds[index::worker_count]) for index in range(worker_count)
    )


def _worker_id(task: SeedWorkerTask) -> str:
    return hash_sequences(
        task.run_id,
        [
            {
                "seed": item.seed,
                "generation_id": item.claim.generation_id,
            }
            for item in task.claimed_seeds
        ],
    )[:32]


def _required_upstream_job_files(job_root: Path, canonical_name: str) -> None:
    for filename in (
        f"{canonical_name}_data.json",
        f"{canonical_name}_ranking_scores.csv",
        f"{canonical_name}_model.cif",
        f"{canonical_name}_confidences.json",
        f"{canonical_name}_summary_confidences.json",
        "TERMS_OF_USE.md",
    ):
        require_regular_file(job_root / filename)


def _load_worker_rankings(
    job_root: Path,
    canonical_name: str,
    seeds: tuple[int, ...],
    sample_count: int,
) -> dict[int, tuple[RankingRow, ...]]:
    ranking_path = job_root / f"{canonical_name}_ranking_scores.csv"
    require_regular_file(ranking_path)
    frame = pl.read_csv(ranking_path)
    if frame.columns != ["seed", "sample", "ranking_score"]:
        raise RuntimeError(f"Unexpected upstream ranking columns: {frame.columns}")
    expected_rows = len(seeds) * sample_count
    if frame.height != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} upstream ranking rows, found {frame.height}"
        )
    rows_by_seed: dict[int, list[dict[str, object]]] = {seed: [] for seed in seeds}
    for raw_row in frame.iter_rows(named=True):
        seed = raw_row.get("seed")
        sample_index = raw_row.get("sample")
        ranking_score = raw_row.get("ranking_score")
        if seed not in rows_by_seed:
            raise RuntimeError(f"Unexpected upstream ranking seed: {seed!r}")
        rows_by_seed[cast(int, seed)].append({
            "seed": seed,
            "sample_index": sample_index,
            "ranking_score": ranking_score,
        })
    parsed: dict[int, tuple[RankingRow, ...]] = {}
    for seed, raw_rows in rows_by_seed.items():
        rankings = _parse_ranking_rows(
            raw_rows,
            expected_seed=seed,
            expected_sample_count=sample_count,
        )
        if rankings is None:
            raise RuntimeError(f"Invalid upstream rankings for seed {seed}")
        parsed[seed] = rankings
    return parsed


def _validate_seed_output(
    job_root: Path,
    canonical_name: str,
    seed: int,
    sample_count: int,
) -> None:
    for sample_index in range(sample_count):
        sample_root = job_root / f"seed-{seed}_sample-{sample_index}"
        if not sample_root.is_dir() or sample_root.is_symlink():
            raise FileNotFoundError(f"Expected sample directory: {sample_root}")
        prefix = f"{canonical_name}_seed-{seed}_sample-{sample_index}"
        for suffix in CORE_OUTPUT_SUFFIXES:
            require_regular_file(sample_root / f"{prefix}_{suffix}")
    embeddings_root = job_root / f"seed-{seed}_embeddings"
    if embeddings_root.exists():
        if not embeddings_root.is_dir() or embeddings_root.is_symlink():
            raise ValueError(f"Invalid embeddings directory: {embeddings_root}")
        require_regular_file(
            embeddings_root / f"{canonical_name}_seed-{seed}_embeddings.npz"
        )
    distogram_root = job_root / f"seed-{seed}_distogram"
    if distogram_root.exists():
        if not distogram_root.is_dir() or distogram_root.is_symlink():
            raise ValueError(f"Invalid distogram directory: {distogram_root}")
        require_regular_file(
            distogram_root / f"{canonical_name}_seed-{seed}_distogram.npz"
        )


def _replace_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError(f"Invalid existing prediction directory: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _promote_seed_output(
    job_root: Path,
    outputs_root: Path,
    seed: int,
    sample_count: int,
) -> None:
    for sample_index in range(sample_count):
        name = f"seed-{seed}_sample-{sample_index}"
        _replace_directory(job_root / name, outputs_root / name)
    for suffix in ("embeddings", "distogram"):
        name = f"seed-{seed}_{suffix}"
        source = job_root / name
        if source.exists():
            _replace_directory(source, outputs_root / name)


def run_seed_prediction_worker(
    runtime: InferenceRuntime,
    task: SeedWorkerTask,
    execute: PredictionExecutor,
) -> dict[str, object]:
    """Execute, validate, and marker-last publish one disjoint seed group."""
    selected_run = validate_run_id(task.run_id)
    sample_count = _validate_sample_count(task.sample_count)
    if not task.claimed_seeds:
        raise ValueError("Seed worker requires at least one claimed seed")
    claimed_seeds = tuple(sorted(task.claimed_seeds, key=lambda item: item.seed))
    if len({item.seed for item in claimed_seeds}) != len(claimed_seeds):
        raise ValueError("Seed worker claims must be unique")
    for item in claimed_seeds:
        if item.claim.scope_key != _seed_claim_scope(selected_run, item.seed):
            raise ValueError(f"Seed {item.seed} claim scope does not match")
        if item.claim.owner.get("identity") != _seed_claim_identity(
            selected_run, item.seed
        ):
            raise ValueError(f"Seed {item.seed} claim identity does not match")

    run_root = inference_run_root(runtime.output_root, selected_run)
    runtime.volume.reload()
    reused: list[int] = []
    pending: list[ClaimedSeed] = []
    terminal_generations: set[str] = set()
    marker_generations: set[str] = set()
    for item in claimed_seeds:
        marker = load_seed_marker(
            run_root,
            selected_run,
            item.seed,
            sample_count=sample_count,
        )
        if marker is None:
            assert_generation_current(runtime.claims, item.claim)
            pending.append(item)
            continue
        finish_generation_claim(
            runtime.claims,
            item.claim,
            status="complete",
            detail={
                "publication": "raced",
                "marker_sha256": marker.marker_sha256,
            },
        )
        terminal_generations.add(item.claim.generation_id)
        reused.append(item.seed)

    if not pending:
        return {
            "status": "reused",
            "run_id": selected_run,
            "worker_id": _worker_id(task),
            "published_seeds": [],
            "reused_seeds": reused,
        }

    worker_id = _worker_id(
        SeedWorkerTask(
            run_id=selected_run,
            sample_count=sample_count,
            claimed_seeds=tuple(pending),
        )
    )
    canonical_name = canonical_output_name(selected_run)
    worker_root = run_root / "outputs" / ".workers" / worker_id
    if worker_root.exists():
        if worker_root.is_symlink() or not worker_root.is_dir():
            raise ValueError(f"Invalid worker staging path: {worker_root}")
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    job_root = worker_root / canonical_name
    outputs_root = run_root / "outputs"
    published: list[int] = []
    try:
        execute(
            worker_root,
            canonical_name,
            tuple(item.seed for item in pending),
        )
        _required_upstream_job_files(job_root, canonical_name)
        rankings_by_seed = _load_worker_rankings(
            job_root,
            canonical_name,
            tuple(item.seed for item in pending),
            sample_count,
        )
        for item in pending:
            _validate_seed_output(
                job_root,
                canonical_name,
                item.seed,
                sample_count,
            )
            assert_generation_current(runtime.claims, item.claim)
        outputs_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            job_root / "TERMS_OF_USE.md",
            outputs_root / "TERMS_OF_USE.md",
        )

        runtime.volume.commit()
        for item in pending:
            assert_generation_current(runtime.claims, item.claim)
            _promote_seed_output(
                job_root,
                outputs_root,
                item.seed,
                sample_count,
            )
            runtime.volume.commit()
            assert_generation_current(runtime.claims, item.claim)
            marker_path = _seed_marker_path(run_root, item.seed)
            write_json_atomic(
                marker_path,
                {
                    "schema_version": SEED_MARKER_SCHEMA_VERSION,
                    "status": "complete",
                    "completed_at": utc_now(),
                    "run_id": selected_run,
                    "seed": item.seed,
                    "sample_count": sample_count,
                    "generation_id": item.claim.generation_id,
                    "rankings": [row.to_dict() for row in rankings_by_seed[item.seed]],
                },
            )
            runtime.volume.commit()
            marker = load_seed_marker(
                run_root,
                selected_run,
                item.seed,
                sample_count=sample_count,
            )
            if marker is None:
                raise RuntimeError(
                    f"Published seed marker failed validation: {item.seed}"
                )
            marker_generations.add(item.claim.generation_id)
            finish_generation_claim(
                runtime.claims,
                item.claim,
                status="complete",
                detail={
                    "publication": "published",
                    "marker_sha256": marker.marker_sha256,
                },
            )
            terminal_generations.add(item.claim.generation_id)
            published.append(item.seed)

        log_source = worker_root / "run.log"
        logs_root = run_root / "logs" / "workers"
        logs_root.mkdir(parents=True, exist_ok=True)
        if log_source.is_file():
            shutil.copy2(log_source, logs_root / f"{worker_id}.log")
        shutil.rmtree(worker_root, ignore_errors=True)
        runtime.volume.commit()
        return {
            "status": "published",
            "run_id": selected_run,
            "worker_id": worker_id,
            "published_seeds": published,
            "reused_seeds": reused,
        }
    except Exception as exc:
        write_json_atomic(
            worker_root / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
                "run_id": selected_run,
                "worker_id": worker_id,
                "seeds": [item.seed for item in pending],
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        runtime.volume.commit()
        for item in pending:
            if item.claim.generation_id in terminal_generations:
                continue
            terminal_status = (
                "complete"
                if item.claim.generation_id in marker_generations
                else "failed"
            )
            try:
                finish_generation_claim(
                    runtime.claims,
                    item.claim,
                    status=terminal_status,
                    detail={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except Exception as claim_exc:
                write_json_atomic(
                    worker_root / f"claim-{item.seed}-failure.json",
                    {
                        "failed_at": utc_now(),
                        "seed": item.seed,
                        "generation_id": item.claim.generation_id,
                        "error_type": type(claim_exc).__name__,
                        "message": str(claim_exc),
                    },
                )
                runtime.volume.commit()
        raise


def _ranking_from_dict(value: object) -> RankingRow | None:
    rows = _parse_ranking_rows([value])
    return rows[0] if rows is not None else None


def load_summary_entry(run_root: Path, run_id: str) -> SummaryEntry | None:
    """Validate the global summary marker and every declared artifact."""
    selected_run = validate_run_id(run_id)
    marker_path = run_root / ".markers" / "summary.json"
    if not marker_path.is_file():
        return None
    try:
        marker_bytes = marker_path.read_bytes()
        marker = orjson.loads(marker_bytes)
    except (OSError, orjson.JSONDecodeError):
        return None
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != SUMMARY_MARKER_SCHEMA_VERSION
        or marker.get("status") != "complete"
        or marker.get("run_id") != selected_run
    ):
        return None
    raw_seeds = marker.get("included_seeds")
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds
        )
        or raw_seeds != sorted(set(raw_seeds))
    ):
        return None
    best = _ranking_from_dict(marker.get("best"))
    if best is None or best.seed not in raw_seeds:
        return None
    raw_artifacts = marker.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        return None
    artifacts: dict[str, dict[str, object]] = {}
    for role in _SUMMARY_ARTIFACT_FILENAMES:
        artifact = validate_artifact_record(run_root, raw_artifacts.get(role))
        if artifact is None:
            return None
        artifacts[role] = artifact
    return SummaryEntry(
        run_id=selected_run,
        included_seeds=tuple(cast(list[int], raw_seeds)),
        best=best,
        artifacts=artifacts,
        marker_sha256=sha256_bytes(marker_bytes),
    )


def collect_seed_markers(
    run_root: Path,
    run_id: str,
    *,
    sample_count: int,
) -> tuple[SeedMarker, ...]:
    """Collect every valid seed marker currently visible at a run root."""
    selected_run = validate_run_id(run_id)
    selected_samples = _validate_sample_count(sample_count)
    markers_root = run_root / ".markers" / "seeds"
    if not markers_root.is_dir():
        return ()
    markers: list[SeedMarker] = []
    for path in markers_root.iterdir():
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            seed = int(path.stem)
        except ValueError:
            continue
        marker = load_seed_marker(
            run_root,
            selected_run,
            seed,
            sample_count=selected_samples,
        )
        if marker is not None:
            markers.append(marker)
    markers.sort(key=lambda marker: marker.seed)
    return tuple(markers)


def ranked_rows(markers: tuple[SeedMarker, ...]) -> tuple[RankingRow, ...]:
    """Return the deterministic total ranking across seed markers."""
    rows = [row for marker in markers for row in marker.rankings]
    rows.sort(
        key=lambda row: (
            -row.ranking_score,
            row.seed,
            row.sample_index,
        )
    )
    return tuple(rows)


def _summary_claim_scope(run_id: str) -> str:
    return f"summary:{validate_run_id(run_id)}"


def _summary_claim_identity(run_id: str) -> dict[str, object]:
    return {
        "schema_version": SUMMARY_CLAIM_SCHEMA_VERSION,
        "run_id": validate_run_id(run_id),
    }


def copy_best_outputs(
    staging_root: Path,
    outputs_root: Path,
    canonical_name: str,
    best: RankingRow,
) -> None:
    """Copy one ranked sample into upstream's top-level best-file shape."""
    sample_root = outputs_root / f"seed-{best.seed}_sample-{best.sample_index}"
    source_prefix = f"{canonical_name}_seed-{best.seed}_sample-{best.sample_index}"
    for suffix in CORE_OUTPUT_SUFFIXES:
        source = sample_root / f"{source_prefix}_{suffix}"
        require_regular_file(source)
        shutil.copy2(source, staging_root / f"{canonical_name}_{suffix}")


def write_ranking_table(path: Path, rows: tuple[RankingRow, ...]) -> None:
    """Write upstream's stable seed/sample ranking CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "seed": [row.seed for row in rows],
        "sample": [row.sample_index for row in rows],
        "ranking_score": [row.ranking_score for row in rows],
    }).write_csv(path)
    require_regular_file(path)


def finalize_run_summary(
    runtime: InferenceRuntime,
    run_id: str,
    *,
    sample_count: int,
    build_data_json: Callable[[tuple[int, ...]], bytes],
) -> dict[str, object]:
    """Serialize and marker-last publish the accumulated global run summary."""
    selected_run = validate_run_id(run_id)
    selected_samples = _validate_sample_count(sample_count)
    run_root = inference_run_root(runtime.output_root, selected_run)
    deadline = time.monotonic() + float(runtime.wait_timeout_seconds)
    generation_id = uuid.uuid4().hex
    claim: GenerationClaim | None = None
    while claim is None:
        runtime.volume.reload()
        markers = collect_seed_markers(
            run_root,
            selected_run,
            sample_count=selected_samples,
        )
        if not markers:
            raise RuntimeError("Cannot summarize a run with no completed seeds")
        current_seeds = tuple(marker.seed for marker in markers)
        if (
            existing := load_summary_entry(run_root, selected_run)
        ) is not None and existing.included_seeds == current_seeds:
            return existing.summary("reused")
        try:
            claim = acquire_generation_claim(
                runtime.claims,
                scope_key=_summary_claim_scope(selected_run),
                generation_id=generation_id,
                identity=_summary_claim_identity(selected_run),
                container_id=runtime.container_id,
                maximum_age_seconds=runtime.summary_maximum_age_seconds,
            )
        except ActiveGenerationError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                owner = exc.owner["generation_id"]
                raise TimeoutError(
                    f"Timed out waiting for summary owner {owner!r}: {selected_run}"
                ) from exc
            time.sleep(min(runtime.claim_poll_seconds, remaining))

    terminal_status = "failed"
    terminal_detail: dict[str, object] = {}
    try:
        runtime.volume.reload()
        markers = collect_seed_markers(
            run_root,
            selected_run,
            sample_count=selected_samples,
        )
        if not markers:
            raise RuntimeError("Cannot summarize a run with no completed seeds")
        included_seeds = tuple(marker.seed for marker in markers)
        existing = load_summary_entry(run_root, selected_run)
        if existing is not None:
            if existing.included_seeds == included_seeds:
                terminal_status = "complete"
                terminal_detail = {
                    "publication": "raced",
                    "marker_sha256": existing.marker_sha256,
                }
                return existing.summary("reused")
            if not set(existing.included_seeds).issubset(included_seeds):
                raise RuntimeError(
                    "Refusing to regress the accumulated summary seed set"
                )

        rows = ranked_rows(markers)
        if not rows:
            raise RuntimeError("Completed seed markers contain no rankings")
        best = rows[0]
        canonical_name = canonical_output_name(selected_run)
        outputs_root = run_root / "outputs"
        staging_root = outputs_root / ".workers" / "summary" / claim.generation_id
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True)
        write_ranking_table(
            staging_root / f"{canonical_name}_ranking_scores.csv",
            rows,
        )
        data_json = build_data_json(included_seeds)
        if not isinstance(data_json, bytes) or not data_json:
            raise ValueError("build_data_json must return non-empty bytes")
        (staging_root / f"{canonical_name}_data.json").write_bytes(data_json)
        copy_best_outputs(
            staging_root,
            outputs_root,
            canonical_name,
            best,
        )
        require_regular_file(outputs_root / "TERMS_OF_USE.md")
        shutil.copy2(
            outputs_root / "TERMS_OF_USE.md",
            staging_root / "TERMS_OF_USE.md",
        )
        runtime.volume.commit()
        assert_generation_current(runtime.claims, claim)

        role_to_source = {
            "data": staging_root / f"{canonical_name}_data.json",
            "ranking": staging_root / f"{canonical_name}_ranking_scores.csv",
            "model": staging_root / f"{canonical_name}_model.cif",
            "confidences": staging_root / f"{canonical_name}_confidences.json",
            "summary_confidences": (
                staging_root / f"{canonical_name}_summary_confidences.json"
            ),
            "terms": staging_root / "TERMS_OF_USE.md",
        }
        role_to_destination = {
            role: outputs_root
            / (
                "TERMS_OF_USE.md"
                if role == "terms"
                else f"{canonical_name}_{_SUMMARY_ARTIFACT_FILENAMES[role]}"
            )
            for role in role_to_source
        }
        for role, source in role_to_source.items():
            destination = role_to_destination[role]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        runtime.volume.commit()
        artifacts = {
            role: artifact_record(destination, run_root)
            for role, destination in role_to_destination.items()
        }
        assert_generation_current(runtime.claims, claim)
        write_json_atomic(
            run_root / ".markers" / "summary.json",
            {
                "schema_version": SUMMARY_MARKER_SCHEMA_VERSION,
                "status": "complete",
                "completed_at": utc_now(),
                "run_id": selected_run,
                "generation_id": claim.generation_id,
                "included_seeds": list(included_seeds),
                "best": best.to_dict(),
                "artifacts": artifacts,
            },
        )
        runtime.volume.commit()
        entry = load_summary_entry(run_root, selected_run)
        if entry is None:
            raise RuntimeError("Published run summary failed validation")
        shutil.rmtree(staging_root, ignore_errors=True)
        runtime.volume.commit()
        terminal_status = "complete"
        terminal_detail = {
            "publication": "published",
            "marker_sha256": entry.marker_sha256,
            "included_seeds": list(entry.included_seeds),
        }
        return entry.summary("published")
    except Exception as exc:
        terminal_detail = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        finish_generation_claim(
            runtime.claims,
            claim,
            status=terminal_status,
            detail=terminal_detail,
        )
