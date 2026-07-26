"""Build and validate immutable AlphaFold 3 database shard profiles.

The implementation is shared by the temporary scientific-validation app and
the production app. Modal decorators, named objects, and fan-out topology stay
in those app composition roots; this module only receives the mounted roots
and narrow persistence handles it needs at runtime.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import perf_counter, time
from typing import Any, Protocol, cast

import orjson

from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
    COMPOSABLE_MULTISET_RECIPE_VERSION,
    DATABASE_PROFILE_SPECS,
    HMMER_VERSION,
    JACKHMMER_PATCH_SHA256,
    LEGACY_PROFILE_RECIPE_VERSION,
    LEGACY_VALIDATION_RELPATHS,
    MAX_PROFILE_IMBALANCE,
    ORDINAL_SHUFFLER_RECIPE_VERSION,
    ORDINAL_VALIDATION_RELPATHS,
    PROFILE_SCHEMA_VERSION,
    PROFILE_STALE_SECONDS,
    SCRATCH_ROOT,
    SEQKIT_VERSION,
    SHARD_RANDOM_SEED,
    SOURCE_DB_VOLUME_NAME,
    VALIDATION_RELPATHS,
    DatabaseProfileSpec,
    profile_root,
    resolve_database_profile,
    shard_filename,
    shard_names,
    validate_seqkit_threads,
    validate_source_policy,
)
from biomodals.app.fold.alphafold3.sharding import (
    ORDINAL_SHUFFLER_PREFETCH_BYTES,
    ORDINAL_SHUFFLER_PREFETCH_RECORDS,
    ORDINAL_SHUFFLER_SOURCE_SHA256,
    ORDINAL_SHUFFLER_VERSION,
    append_log,
    compile_record_multiset_validator,
    load_json_object,
    record_multiset_identity,
    record_multiset_signature,
    require_executable,
    require_regular_file,
    required_ordinal_shuffler_scratch_bytes,
    scan_record_multiset,
    sha256_file,
    shuffle_fasta_occurrences,
    utc_now,
    verify_file,
    write_json_atomic,
)

_JSON_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
_JSONL_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE


class VolumeHandle(Protocol):
    """Persistence barrier methods required from a mounted Modal Volume."""

    def reload(self) -> None:
        """Reload changes committed by other containers."""
        ...

    def commit(self) -> None:
        """Commit this container's writes."""
        ...


class ClaimStore(Protocol):
    """Minimal Modal Dict surface used for single-profile build claims."""

    def put(
        self,
        key: str,
        value: object,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        """Store a value and report whether an insert-only write succeeded."""
        ...

    def get(self, key: str, default: object = None) -> object:
        """Return a stored value or the supplied default."""
        ...

    def pop(self, key: str, default: object = None) -> object:
        """Remove and return a stored value or the supplied default."""
        ...


@dataclass(frozen=True, slots=True)
class ProfileBuilderRuntime:
    """Mounted paths and persistence handles for one builder container."""

    source_root: Path
    sharded_root: Path
    output_root: Path
    evidence_relpath: str
    source_volume: VolumeHandle
    sharded_volume: VolumeHandle
    output_volume: VolumeHandle
    claims: ClaimStore
    container_id: str


def _json_bytes(value: object) -> bytes:
    return orjson.dumps(value, option=_JSON_OPTIONS)


def run_to_file(argv: list[str], output_path: Path, log_path: Path) -> None:
    """Run a fixed argv command with separate data and diagnostic streams."""
    append_log(log_path, f"Running command: {shlex.join(argv)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output, log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            argv,
            check=False,
            stdout=output,
            stderr=log,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, argv)


def artifact_record(
    path: Path,
    root: Path,
    *,
    forbidden_bytes: bytes | None = None,
) -> dict[str, str | int]:
    """Build one manifest artifact record below a profile root."""
    require_regular_file(path)
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Artifact escapes profile root: {path}")
    return {
        "path": resolved_path.relative_to(resolved_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path, forbidden_bytes=forbidden_bytes),
    }


def finalize_record_multiset_validation(
    source_result: dict[str, object],
    shard_result: dict[str, object],
    output_path: Path,
    *,
    execution: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compare native-helper reports and publish their shared signature."""
    source_report = source_result.get("report")
    shard_report = shard_result.get("report")
    if not isinstance(source_report, dict) or not isinstance(shard_report, dict):
        raise TypeError("Record-multiset helper result lost its report")
    source_signature = record_multiset_signature(cast(dict[str, Any], source_report))
    shard_signature = record_multiset_signature(cast(dict[str, Any], shard_report))
    source_signature_sha256 = hashlib.sha256(_json_bytes(source_signature)).hexdigest()
    shard_signature_sha256 = hashlib.sha256(_json_bytes(shard_signature)).hexdigest()
    match = source_signature == shard_signature
    result: dict[str, object] = {
        "match": match,
        "algorithm": record_multiset_identity(),
        "source_signature_sha256": source_signature_sha256,
        "shard_signature_sha256": shard_signature_sha256,
        "source_signature": source_signature,
        "shard_signature": shard_signature,
        "source": source_result,
        "shards": shard_result,
    }
    if execution is not None:
        result["execution"] = execution
    if match:
        result["signature_sha256"] = source_signature_sha256
        result["signature"] = source_signature
    write_json_atomic(output_path, result)
    if not match:
        raise ValueError("Canonical source and shard record multisets differ")
    return result


def _run_shuffle(
    spec: DatabaseProfileSpec,
    source_path: Path,
    scratch_root: Path,
    validation_dir: Path,
    log_path: Path,
    *,
    expected_records: int,
    seqkit_threads: int,
) -> tuple[Path, Path, dict[str, Any]]:
    """Shuffle every source occurrence into a container-local FASTA."""
    shuffled_path = scratch_root / "shuffled.fasta"
    diagnostics_path = validation_dir / "shuffle-stderr.log"
    metrics_path = validation_dir / "shuffler-metrics.json"
    recovery_report_path = validation_dir / "duplicate-recovery.jsonl"
    validation_dir.mkdir(parents=True, exist_ok=True)
    shuffle_result = shuffle_fasta_occurrences(
        source_path,
        shuffled_path,
        scratch_root,
        diagnostics_path,
        metrics_path,
        log_path,
        expected_records=expected_records,
        seed=SHARD_RANDOM_SEED,
        worker_threads=seqkit_threads,
    )
    recovery_metrics: dict[str, int | str | None] = {
        "recovered_records": 0,
        "recovered_residues": 0,
        "first_byte_offset": None,
        "last_byte_offset": None,
        "temporary_namespace": None,
        "temporary_header_pattern": None,
    }
    with recovery_report_path.open("xb") as report:
        report.write(
            orjson.dumps(
                {
                    "kind": "summary",
                    **recovery_metrics,
                    "warning_source": None,
                    "record_identity": "source-occurrence",
                    "fai_duplicate_omission_possible": False,
                },
                option=_JSONL_OPTIONS,
            )
        )
        report.flush()
        os.fsync(report.fileno())
    append_log(
        log_path,
        "Preserved every FASTA record by source occurrence; FAI duplicate "
        "recovery is not applicable",
    )
    return (
        shuffle_result.staged_source_path,
        shuffled_path,
        recovery_metrics | {"shuffler": shuffle_result.metrics},
    )


def _run_split(
    spec: DatabaseProfileSpec,
    shuffled_path: Path,
    raw_shard_dir: Path,
    shard_dir: Path,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> tuple[Path, ...]:
    """Split one local shuffle and normalize AlphaFold-compatible names."""
    seqkit = require_executable("seqkit")
    require_regular_file(shuffled_path)
    argv = [
        seqkit,
        "split2",
        "-j",
        str(seqkit_threads),
        "--by-part",
        str(spec.shard_count),
        "--out-dir",
        str(raw_shard_dir),
        "--force",
        "--out-prefix",
        "part_",
        str(shuffled_path),
    ]
    append_log(log_path, f"Running command: {shlex.join(argv)}")
    raw_shard_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, argv)

    raw_shards = sorted(
        path
        for path in raw_shard_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if len(raw_shards) != spec.shard_count:
        raise ValueError(
            f"Expected {spec.shard_count} raw shards, found {len(raw_shards)}"
        )

    shuffled_path.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)
    for index, raw_shard in enumerate(raw_shards):
        if raw_shard.stat().st_size <= 0:
            raise ValueError(f"SeqKit produced empty shard: {raw_shard}")
        final_shard = shard_dir / shard_filename(spec, index)
        raw_shard.replace(final_shard)
        require_regular_file(final_shard)
    raw_shard_dir.rmdir()
    return tuple(shard_dir / name for name in shard_names(spec))


def _validate_statistics(
    spec: DatabaseProfileSpec,
    source_stats_path: Path,
    shard_stats_path: Path,
    shard_summary_path: Path,
) -> dict[str, int | float]:
    """Validate source/shard statistics for one fixed specification."""
    import polars as pl

    source_stats = pl.read_csv(source_stats_path, separator="\t")
    shard_stats = pl.read_csv(shard_stats_path, separator="\t")
    required_columns = {"file", "num_seqs", "sum_len"}
    if not required_columns.issubset(source_stats.columns):
        raise ValueError(f"Source stats missing columns: {required_columns}")
    if not required_columns.issubset(shard_stats.columns):
        raise ValueError(f"Shard stats missing columns: {required_columns}")
    if source_stats.height != 1:
        raise ValueError(f"Expected one source stats row, got {source_stats.height}")
    if shard_stats.height != spec.shard_count:
        raise ValueError(
            f"Expected {spec.shard_count} shard stats rows, got {shard_stats.height}"
        )

    shard_stats = shard_stats.with_columns(
        pl
        .col("file")
        .cast(pl.String)
        .str.replace_all(r"\\", "/")
        .str.split("/")
        .list.last()
        .alias("basename")
    ).sort("basename")
    if shard_stats.get_column("basename").to_list() != list(shard_names(spec)):
        raise ValueError("SeqKit stats shard names do not match the profile")

    source_num_seqs = int(source_stats.item(0, "num_seqs"))
    source_sum_len = int(source_stats.item(0, "sum_len"))
    shard_num_seqs = int(shard_stats.get_column("num_seqs").sum())
    shard_sum_len = int(shard_stats.get_column("sum_len").sum())
    if spec.expected_num_seqs is not None and source_num_seqs != spec.expected_num_seqs:
        raise ValueError(
            f"{spec.database_id} sequence count {source_num_seqs} does not "
            f"match expected {spec.expected_num_seqs}"
        )
    if spec.expected_sum_len is not None and source_sum_len != spec.expected_sum_len:
        raise ValueError(
            f"{spec.database_id} residue count {source_sum_len} does not "
            f"match expected {spec.expected_sum_len}"
        )
    if shard_num_seqs != source_num_seqs:
        raise ValueError(
            f"Shard sequence count {shard_num_seqs} != source {source_num_seqs}"
        )
    if shard_sum_len != source_sum_len:
        raise ValueError(
            f"Shard residue count {shard_sum_len} != source {source_sum_len}"
        )

    mean_sum_len = source_sum_len / spec.shard_count
    maximum_imbalance = max(
        abs(int(value) - mean_sum_len) / mean_sum_len
        for value in shard_stats.get_column("sum_len")
    )
    if maximum_imbalance > MAX_PROFILE_IMBALANCE:
        raise ValueError(
            "Shard residue imbalance exceeds "
            f"{MAX_PROFILE_IMBALANCE:.0%}: {maximum_imbalance:.3%}"
        )
    shard_summary_path.parent.mkdir(parents=True, exist_ok=True)
    shard_stats.write_parquet(shard_summary_path)
    return {
        "num_seqs": source_num_seqs,
        "sum_len": source_sum_len,
        "maximum_residue_imbalance": maximum_imbalance,
    }


def _validate_recipe(
    recipe: dict[str, Any],
    spec: DatabaseProfileSpec,
) -> tuple[int, tuple[str, ...]]:
    """Validate one supported immutable sharding recipe."""
    if recipe.get("seqkit_version") != SEQKIT_VERSION:
        raise ValueError("Unexpected profile SeqKit version")
    if recipe.get("random_seed") != SHARD_RANDOM_SEED:
        raise ValueError("Unexpected profile shuffle seed")
    if recipe.get("split") != ["--by-part", spec.shard_count]:
        raise ValueError("Unexpected profile split recipe")
    raw_threads = recipe.get("seqkit_threads")
    if isinstance(raw_threads, bool) or not isinstance(raw_threads, int):
        raise ValueError("Invalid profile SeqKit threads")
    try:
        seqkit_threads = validate_seqkit_threads(raw_threads)
    except ValueError as exc:
        raise ValueError("Invalid profile SeqKit threads") from exc

    recipe_version = recipe.get("version")
    if recipe_version == LEGACY_PROFILE_RECIPE_VERSION:
        if recipe.get("shuffle") != [
            "--two-pass",
            "--update-faidx",
            "--tmp-dir=/tmp",
        ]:
            raise ValueError("Unexpected legacy profile shuffle recipe")
        if recipe.get("duplicate_recovery") != {
            "warning_source": "seqkit-fai-sequence-byte-offset",
            "temporary_header_identity": "generation-unique-uuid",
            "append_after_shuffle": True,
            "strip_after_split": True,
        }:
            raise ValueError("Unexpected legacy duplicate-recovery recipe")
        return recipe_version, LEGACY_VALIDATION_RELPATHS

    if recipe_version not in {
        ORDINAL_SHUFFLER_RECIPE_VERSION,
        COMPOSABLE_MULTISET_RECIPE_VERSION,
    }:
        raise ValueError("Unexpected profile recipe version")
    if recipe.get("shuffle") != [
        "two-pass",
        "first-pass-stage-local-source",
        "source-occurrence-offset-index",
        "splitmix64-fisher-yates-u32",
        "bounded-concurrent-local-pread",
        "ordered-write",
    ]:
        raise ValueError("Unexpected occurrence-indexed shuffle recipe")
    if recipe.get("shuffler") != {
        "version": ORDINAL_SHUFFLER_VERSION,
        "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
        "record_identity": "source-occurrence",
        "offset_index": "uint64-source-occurrence-offsets-v1",
        "permutation": "splitmix64-fisher-yates-u32-v1",
        "staging": "first-pass-tee-to-container-local-v1",
        "read": "bounded-concurrent-local-pread-ordered-write-v2",
        "ordered_output": True,
    }:
        raise ValueError("Unexpected native shuffler identity")
    if recipe.get("execution") != {
        "worker_threads": seqkit_threads,
        "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
        "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
    }:
        raise ValueError("Unexpected native shuffler execution plan")
    if recipe.get("duplicate_recovery") != {
        "warning_source": None,
        "record_identity": "source-occurrence",
        "append_after_shuffle": False,
        "strip_after_split": False,
    }:
        raise ValueError("Unexpected occurrence-indexed duplicate policy")
    if recipe_version == ORDINAL_SHUFFLER_RECIPE_VERSION:
        return recipe_version, ORDINAL_VALIDATION_RELPATHS
    if recipe.get("record_multiset") != (
        record_multiset_identity() | {"shard_threads": seqkit_threads}
    ):
        raise ValueError("Unexpected composable record-multiset validator")
    return recipe_version, VALIDATION_RELPATHS


def validate_profile_manifest(
    manifest: dict[str, Any],
    spec: DatabaseProfileSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one profile manifest without filesystem access."""
    if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unexpected profile schema version")
    if manifest.get("profile_id") != spec.profile_id:
        raise ValueError("Unexpected profile ID")
    if manifest.get("database_id") != spec.database_id:
        raise ValueError("Unexpected database ID")
    if manifest.get("polymer") != spec.polymer:
        raise ValueError("Unexpected profile polymer")
    if manifest.get("shard_count") != spec.shard_count:
        raise ValueError("Unexpected profile shard count")
    if manifest.get("shard_prefix") != f"shards/{spec.source_filename}":
        raise ValueError("Unexpected profile shard prefix")
    if manifest.get("search_space_value") != spec.search_space_value:
        raise ValueError("Unexpected profile search-space value")
    if manifest.get("search_space_unit") != spec.search_space_unit:
        raise ValueError("Unexpected profile search-space unit")

    source = manifest.get("source")
    shards = manifest.get("shards")
    validation = manifest.get("validation")
    recipe = manifest.get("recipe")
    compatibility = manifest.get("compatibility")
    if not isinstance(source, dict):
        raise ValueError("Profile source must be an object")
    if source.get("volume") != SOURCE_DB_VOLUME_NAME:
        raise ValueError("Profile source Volume is invalid")
    if source.get("path") != spec.source_filename:
        raise ValueError("Profile source path is invalid")
    if not isinstance(source.get("size_bytes"), int) or source["size_bytes"] <= 0:
        raise ValueError("Profile source size is invalid")
    if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
        raise ValueError("Profile source SHA-256 is invalid")
    if (
        not isinstance(source.get("num_seqs"), int)
        or source["num_seqs"] <= 0
        or not isinstance(source.get("sum_len"), int)
        or source["sum_len"] <= 0
    ):
        raise ValueError("Profile source statistics are invalid")
    if (
        spec.expected_num_seqs is not None
        and source["num_seqs"] != spec.expected_num_seqs
    ):
        raise ValueError("Profile source sequence count is invalid")
    if spec.expected_sum_len is not None and source["sum_len"] != spec.expected_sum_len:
        raise ValueError("Profile source residue count is invalid")

    if not isinstance(shards, list) or len(shards) != spec.shard_count:
        raise ValueError(f"Profile must declare {spec.shard_count} shards")
    if not isinstance(recipe, dict):
        raise ValueError("Profile recipe must be an object")
    if compatibility != {
        "alphafold_repository": ALPHAFOLD3_REPOSITORY,
        "alphafold_commit": ALPHAFOLD3_COMMIT,
        "hmmer_version": HMMER_VERSION,
        "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
    }:
        raise ValueError("Unexpected profile compatibility pin")
    recipe_version, expected_validation_relpaths = _validate_recipe(recipe, spec)

    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Profile does not declare passed validation")
    if validation.get("temporary_recovery_prefix_absent") is not True:
        raise ValueError("Profile may retain recovery prefixes")
    if validation.get("num_seqs") != source["num_seqs"]:
        raise ValueError("Profile validation sequence count is invalid")
    if validation.get("sum_len") != source["sum_len"]:
        raise ValueError("Profile validation residue count is invalid")
    if recipe_version in {
        ORDINAL_SHUFFLER_RECIPE_VERSION,
        COMPOSABLE_MULTISET_RECIPE_VERSION,
    }:
        if validation.get("record_occurrences_preserved") is not True:
            raise ValueError("Profile does not preserve record occurrences")
        if (
            validation.get("recovered_records") != 0
            or validation.get("recovered_residues") != 0
            or validation.get("first_recovered_byte_offset") is not None
            or validation.get("last_recovered_byte_offset") is not None
        ):
            raise ValueError("Occurrence-indexed profile declares FAI recovery")
    if recipe_version == COMPOSABLE_MULTISET_RECIPE_VERSION:
        if validation.get("canonical_record_multiset_match") is not True:
            raise ValueError("Canonical source and shard record multisets differ")
        signature_sha256 = validation.get("record_multiset_signature_sha256")
        if (
            not isinstance(signature_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", signature_sha256) is None
        ):
            raise ValueError("Invalid canonical record-multiset signature")
        if "seqkit_sum" in validation:
            raise ValueError(
                "Composable-multiset profile unexpectedly declares SeqKit sum"
            )
    validation_artifacts = validation.get("artifacts")
    if not isinstance(validation_artifacts, list):
        raise ValueError("Profile validation artifacts must be a list")

    expected_shard_paths = [f"shards/{name}" for name in shard_names(spec)]
    actual_shard_paths: list[str] = []
    for record in [*shards, *validation_artifacts]:
        if not isinstance(record, dict):
            raise ValueError("Profile artifact record must be an object")
        relative = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Profile artifact path must be relative")
        if ".." in PurePosixPath(relative).parts:
            raise ValueError(f"Profile artifact escapes root: {relative}")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(f"Profile artifact is empty: {relative}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid profile artifact digest: {relative}")
        if relative.startswith("shards/"):
            actual_shard_paths.append(relative)
    if actual_shard_paths != expected_shard_paths:
        raise ValueError("Profile shard order or names are invalid")
    if [str(record["path"]) for record in validation_artifacts] != list(
        expected_validation_relpaths
    ):
        raise ValueError("Profile validation artifact paths are invalid")
    return source, shards, validation_artifacts


def validate_published_profile(
    root: Path,
    spec: DatabaseProfileSpec,
    *,
    verify_digests: bool,
) -> dict[str, Any]:
    """Validate a manifest-last profile publication and its artifacts."""
    manifest_path = root / "manifest.json"
    require_regular_file(manifest_path)
    if (root / "source").exists():
        raise ValueError("Profile must not contain a source copy")
    manifest = load_json_object(manifest_path)
    _, shards, validation_artifacts = validate_profile_manifest(manifest, spec)
    resolved_root = root.resolve()
    for record in [*shards, *validation_artifacts]:
        relative = str(record["path"])
        artifact_path = (root / relative).resolve()
        if not artifact_path.is_relative_to(resolved_root):
            raise ValueError(f"Profile artifact escapes root: {relative}")
        require_regular_file(artifact_path)
        if artifact_path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Profile artifact size mismatch: {relative}")
        if verify_digests and sha256_file(artifact_path) != record["sha256"]:
            raise ValueError(f"Profile artifact digest mismatch: {relative}")
    return manifest


def _profile_claim_key(spec: DatabaseProfileSpec) -> str:
    return f"active:{spec.profile_id}"


def _profile_owner_key(spec: DatabaseProfileSpec, generation_id: str) -> str:
    return f"owner:{spec.profile_id}:{generation_id}"


def _profile_status_key(spec: DatabaseProfileSpec, generation_id: str) -> str:
    return f"status:{spec.profile_id}:{generation_id}"


def _acquire_claim(
    runtime: ProfileBuilderRuntime,
    spec: DatabaseProfileSpec,
    generation_id: str,
) -> dict[str, object]:
    """Elect one builder, failing immediately on a non-stale conflict."""
    owner = {
        "profile_id": spec.profile_id,
        "database_id": spec.database_id,
        "generation_id": generation_id,
        "container_id": runtime.container_id,
        "hostname": socket.gethostname(),
        "started_at": utc_now(),
        "started_at_epoch_seconds": time(),
        "maximum_age_seconds": PROFILE_STALE_SECONDS,
    }
    active_key = _profile_claim_key(spec)
    if runtime.claims.put(active_key, owner, skip_if_exists=True):
        if not runtime.claims.put(
            _profile_owner_key(spec, generation_id),
            owner,
            skip_if_exists=True,
        ):
            runtime.claims.pop(active_key, None)
            raise RuntimeError("Profile generation owner key already exists")
        return owner

    active = runtime.claims.get(active_key, None)
    if not isinstance(active, dict):
        raise RuntimeError(f"Profile {spec.profile_id} has an invalid active claim")
    started_at = active.get("started_at_epoch_seconds")
    if not isinstance(started_at, int | float):
        raise RuntimeError(f"Profile {spec.profile_id} has an invalid claim time")
    age_seconds = time() - float(started_at)
    if age_seconds <= PROFILE_STALE_SECONDS:
        raise RuntimeError(
            f"Profile {spec.profile_id} is already being built by generation "
            f"{active.get('generation_id')!r}"
        )

    removed = runtime.claims.pop(active_key, None)
    if removed != active:
        raise RuntimeError(
            f"Profile {spec.profile_id} claim changed during stale recovery"
        )
    stale_generation = active.get("generation_id")
    if isinstance(stale_generation, str):
        runtime.claims.put(
            _profile_status_key(spec, stale_generation),
            {
                "status": "abandoned",
                "abandoned_at": utc_now(),
                "age_seconds": age_seconds,
                "replaced_by_generation_id": generation_id,
            },
            skip_if_exists=True,
        )
    if not runtime.claims.put(active_key, owner, skip_if_exists=True):
        raise RuntimeError(f"Profile {spec.profile_id} claim was acquired concurrently")
    if not runtime.claims.put(
        _profile_owner_key(spec, generation_id),
        owner,
        skip_if_exists=True,
    ):
        runtime.claims.pop(active_key, None)
        raise RuntimeError("Profile generation owner key already exists")
    return owner


def _finish_claim(
    runtime: ProfileBuilderRuntime,
    spec: DatabaseProfileSpec,
    generation_id: str,
    *,
    status: str,
    detail: dict[str, object],
) -> None:
    """Append terminal claim status and release this generation's slot."""
    runtime.claims.put(
        _profile_status_key(spec, generation_id),
        {
            "status": status,
            "finished_at": utc_now(),
            **detail,
        },
        skip_if_exists=True,
    )
    active_key = _profile_claim_key(spec)
    active = runtime.claims.get(active_key, None)
    if isinstance(active, dict) and active.get("generation_id") == generation_id:
        runtime.claims.pop(active_key, None)


def _hash_decompressed_zstd(
    archive_path: Path,
    log_path: Path,
) -> tuple[str, int]:
    """Stream one zstd archive through SHA-256 without materializing it."""
    zstd = require_executable("zstd")
    argv = [zstd, "--quiet", "--decompress", "--stdout", str(archive_path)]
    append_log(log_path, f"Running command: {shlex.join(argv)}")
    with log_path.open("ab") as log:
        process = subprocess.Popen(  # noqa: S603
            argv,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        if process.stdout is None:
            process.kill()
            raise RuntimeError("zstd did not expose decompressed stdout")
        digest = hashlib.sha256()
        size_bytes = 0
        while chunk := process.stdout.read(8 * 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
        process.stdout.close()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, argv)
    return digest.hexdigest(), size_bytes


def _apply_source_policy(
    runtime: ProfileBuilderRuntime,
    spec: DatabaseProfileSpec,
    manifest: dict[str, Any],
    source_policy: str,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> dict[str, object]:
    """Retire a source only after a valid profile publication exists."""
    policy = validate_source_policy(source_policy)
    source_path = runtime.source_root / spec.source_filename
    archive_path = source_path.with_name(f"{source_path.name}.zst")
    if policy == "keep":
        return {
            "source_policy": policy,
            "source_status": "kept" if source_path.is_file() else "already-retired",
        }

    source_record = manifest.get("source")
    if not isinstance(source_record, dict):
        raise ValueError("Validated manifest lost its source record")
    expected_sha256 = source_record.get("sha256")
    expected_size = source_record.get("size_bytes")
    if not isinstance(expected_sha256, str) or not isinstance(expected_size, int):
        raise ValueError("Validated manifest source identity is invalid")

    if not source_path.is_file():
        if policy == "compress" and archive_path.is_file():
            archive_sha256, archive_size = _hash_decompressed_zstd(
                archive_path,
                log_path,
            )
            if (archive_sha256, archive_size) != (
                expected_sha256,
                expected_size,
            ):
                raise ValueError(
                    f"Existing archive does not reproduce {spec.source_filename}"
                )
            return {
                "source_policy": policy,
                "source_status": "already-compressed",
                "archive_path": str(archive_path),
            }
        if policy == "delete":
            return {
                "source_policy": policy,
                "source_status": "already-deleted",
            }
        raise FileNotFoundError(f"Source FASTA is missing: {source_path}")

    if policy == "delete":
        if (
            source_path.stat().st_size != expected_size
            or sha256_file(source_path) != expected_sha256
        ):
            raise ValueError(
                f"Refusing to delete changed source {spec.source_filename}"
            )
        source_path.unlink()
        runtime.source_volume.commit()
        return {
            "source_policy": policy,
            "source_status": "deleted",
        }

    if archive_path.is_file():
        archive_sha256, archive_size = _hash_decompressed_zstd(
            archive_path,
            log_path,
        )
        if (archive_sha256, archive_size) != (expected_sha256, expected_size):
            raise ValueError(
                f"Existing archive does not reproduce {spec.source_filename}"
            )
    else:
        zstd = require_executable("zstd")
        temporary_archive = archive_path.with_name(
            f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
        )
        argv = [
            zstd,
            f"-T{seqkit_threads}",
            "--quiet",
            "--stdout",
            str(source_path),
        ]
        append_log(log_path, f"Running command: {shlex.join(argv)}")
        try:
            with temporary_archive.open("xb") as archive, log_path.open("ab") as log:
                completed = subprocess.run(  # noqa: S603
                    argv,
                    check=False,
                    stdout=archive,
                    stderr=log,
                )
                archive.flush()
                os.fsync(archive.fileno())
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, argv)
            archive_sha256, archive_size = _hash_decompressed_zstd(
                temporary_archive,
                log_path,
            )
            if (archive_sha256, archive_size) != (
                expected_sha256,
                expected_size,
            ):
                raise ValueError(
                    f"Compressed archive does not reproduce {spec.source_filename}"
                )
            temporary_archive.replace(archive_path)
            runtime.source_volume.commit()
        finally:
            temporary_archive.unlink(missing_ok=True)

    source_path.unlink()
    runtime.source_volume.commit()
    return {
        "source_policy": policy,
        "source_status": "compressed",
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
    }


def _write_success_evidence(
    runtime: ProfileBuilderRuntime,
    evidence_root: Path,
    result: dict[str, object],
) -> None:
    write_json_atomic(evidence_root / "metrics.json", result)
    runtime.output_volume.commit()
    write_json_atomic(
        evidence_root / "done.json",
        result | {"completed_at": utc_now()},
    )
    runtime.output_volume.commit()


def build_profile(
    runtime: ProfileBuilderRuntime,
    database_id: str,
    seqkit_threads: int,
    source_policy: str,
) -> dict[str, object]:
    """Build, publish, deeply validate, and optionally retire one source."""
    spec = resolve_database_profile(database_id)
    threads = validate_seqkit_threads(seqkit_threads)
    policy = validate_source_policy(source_policy)
    generation_id = uuid.uuid4().hex
    source_path = runtime.source_root / spec.source_filename
    published_root = profile_root(runtime.sharded_root, spec)
    evidence_root = (
        runtime.output_root / runtime.evidence_relpath / spec.profile_id / generation_id
    )
    log_path = evidence_root / "run.log"
    evidence_root.mkdir(parents=True, exist_ok=True)
    append_log(log_path, f"Preparing profile {spec.profile_id}")

    runtime.source_volume.reload()
    runtime.sharded_volume.reload()
    runtime.output_volume.reload()
    if (published_root / "manifest.json").is_file():
        manifest = validate_published_profile(
            published_root,
            spec,
            verify_digests=True,
        )
        source_result = _apply_source_policy(
            runtime,
            spec,
            manifest,
            policy,
            log_path,
            seqkit_threads=threads,
        )
        result = {
            "status": "reused",
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_path": str(published_root),
            "manifest_sha256": sha256_file(published_root / "manifest.json"),
            **source_result,
        }
        _write_success_evidence(runtime, evidence_root, result)
        return result

    claim = _acquire_claim(runtime, spec, generation_id)
    write_json_atomic(evidence_root / "claim.json", claim)
    runtime.output_volume.commit()
    staging_root = (
        runtime.sharded_root / ".staging" / f"{spec.profile_id}-{generation_id}"
    )
    raw_shard_dir = staging_root / ".raw-shards"
    shard_dir = staging_root / "shards"
    validation_dir = staging_root / "validation"
    payload_moved = False
    manifest_published = False
    try:
        if published_root.exists():
            orphan_root = runtime.sharded_root / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            published_root.replace(
                orphan_root / f"{spec.profile_id}-{uuid.uuid4().hex}"
            )
            runtime.sharded_volume.commit()

        if not source_path.is_file():
            archive_path = source_path.with_name(f"{source_path.name}.zst")
            if archive_path.is_file():
                raise FileNotFoundError(
                    f"{source_path} is archived as {archive_path}. Restore the "
                    "plain FASTA manually in a Modal Sandbox before rebuilding."
                )
            raise FileNotFoundError(f"Source FASTA is missing: {source_path}")
        require_regular_file(source_path)
        source_size = source_path.stat().st_size

        shard_dir.mkdir(parents=True)
        validation_dir.mkdir(parents=True)
        seqkit = require_executable("seqkit")
        version_output = subprocess.run(  # noqa: S603
            [seqkit, "version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if SEQKIT_VERSION not in version_output:
            raise RuntimeError(
                f"Expected SeqKit {SEQKIT_VERSION}, observed {version_output!r}"
            )
        append_log(log_path, f"Using {version_output}")

        source_stats_path = validation_dir / "source-stats.tsv"
        shard_stats_path = validation_dir / "shard-stats.tsv"
        shard_summary_path = validation_dir / "shard-summary.parquet"
        record_multiset_path = validation_dir / "record-multiset.json"
        run_to_file(
            [
                seqkit,
                "stats",
                "-j",
                str(threads),
                "--all",
                "--tabular",
                str(source_path),
            ],
            source_stats_path,
            log_path,
        )

        import polars as pl

        source_stats = pl.read_csv(source_stats_path, separator="\t")
        if source_stats.height != 1:
            raise ValueError(
                f"Expected one source stats row, got {source_stats.height}"
            )
        source_num_seqs = int(source_stats.item(0, "num_seqs"))
        source_sum_len = int(source_stats.item(0, "sum_len"))
        if (
            spec.expected_num_seqs is not None
            and source_num_seqs != spec.expected_num_seqs
        ):
            raise ValueError(
                f"{spec.database_id} sequence count {source_num_seqs} does not "
                f"match expected {spec.expected_num_seqs}"
            )
        if (
            spec.expected_sum_len is not None
            and source_sum_len != spec.expected_sum_len
        ):
            raise ValueError(
                f"{spec.database_id} residue count {source_sum_len} does not "
                f"match expected {spec.expected_sum_len}"
            )
        scratch_free = shutil.disk_usage(SCRATCH_ROOT).free
        scratch_required = required_ordinal_shuffler_scratch_bytes(
            source_size,
            source_num_seqs,
        )
        if scratch_free < scratch_required:
            raise OSError(
                f"Insufficient /tmp space for {spec.database_id}: need at least "
                f"{scratch_required} bytes, found {scratch_free}"
            )
        append_log(
            log_path,
            f"Reserved scratch budget {scratch_required} bytes for local source "
            "staging, shuffled FASTA, and occurrence index",
        )
        source_sha256 = sha256_file(source_path)

        with tempfile.TemporaryDirectory(
            prefix=f"af3-{spec.database_id}-",
            dir=SCRATCH_ROOT,
        ) as scratch_dir:
            scratch_root = Path(scratch_dir)
            staged_source_path, shuffled_path, recovery_metrics = _run_shuffle(
                spec,
                source_path,
                scratch_root,
                validation_dir,
                log_path,
                expected_records=source_num_seqs,
                seqkit_threads=threads,
            )
            staged_verification = verify_file(
                staged_source_path,
                expected_size=source_size,
                expected_sha256=source_sha256,
            )
            append_log(
                log_path,
                "Verified container-local staged source against the Volume "
                f"source SHA-256 in {staged_verification.wall_seconds:.6f}s",
            )
            validator = compile_record_multiset_validator(scratch_root, log_path)
            parallel_started = perf_counter()
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="af3-source-validator",
            ) as source_validator_pool:
                # The worker waits on native C; Python does not perform the scan.
                source_future = source_validator_pool.submit(
                    scan_record_multiset,
                    validator,
                    (staged_source_path,),
                    scratch_root / "source-record-multiset.json",
                    log_path,
                    threads=1,
                )
                split_started = perf_counter()
                shard_paths = _run_split(
                    spec,
                    shuffled_path,
                    raw_shard_dir,
                    shard_dir,
                    log_path,
                    seqkit_threads=threads,
                )
                split_seconds = perf_counter() - split_started
                source_result = source_future.result() if source_future.done() else None
                shard_result = scan_record_multiset(
                    validator,
                    shard_paths,
                    scratch_root / "shard-record-multiset.json",
                    log_path,
                    threads=threads,
                )
                if source_result is None:
                    source_result = source_future.result()
            parallel_seconds = perf_counter() - parallel_started
            record_multiset = finalize_record_multiset_validation(
                source_result,
                shard_result,
                record_multiset_path,
                execution={
                    "strategy": "native-source-scan-overlaps-split-and-shard-scan-v1",
                    "python_role": "subprocess-orchestration-only",
                    "source_input": "sha256-verified-container-local-staged-copy",
                    "staged_source_size_bytes": staged_verification.size_bytes,
                    "staged_source_sha256": staged_verification.sha256,
                    "staged_source_verification_seconds": (
                        staged_verification.wall_seconds
                    ),
                    "split_seconds": split_seconds,
                    "parallel_stage_wall_seconds": parallel_seconds,
                },
            )

        run_to_file(
            [
                seqkit,
                "stats",
                "-j",
                str(threads),
                "--all",
                "--tabular",
                *(str(path) for path in shard_paths),
            ],
            shard_stats_path,
            log_path,
        )
        statistics = _validate_statistics(
            spec,
            source_stats_path,
            shard_stats_path,
            shard_summary_path,
        )
        multiset_signature = record_multiset.get("signature")
        if not isinstance(multiset_signature, dict):
            raise TypeError("Record-multiset validation lost its signature")
        if multiset_signature.get("records") != statistics["num_seqs"]:
            raise ValueError("Record-multiset count does not match SeqKit statistics")
        if multiset_signature.get("sequence_bytes") != statistics["sum_len"]:
            raise ValueError(
                "Record-multiset sequence bytes do not match SeqKit statistics"
            )
        signature_sha256 = record_multiset.get("signature_sha256")
        if (
            not isinstance(signature_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", signature_sha256) is None
        ):
            raise ValueError("Record-multiset signature SHA-256 is invalid")

        if recovery_metrics.get("temporary_namespace") is not None:
            raise RuntimeError("Occurrence shuffling must not create recovery headers")
        shard_records = [
            artifact_record(shard_path, staging_root) for shard_path in shard_paths
        ]
        validation_records = [
            artifact_record(staging_root / relative, staging_root)
            for relative in VALIDATION_RELPATHS
        ]
        manifest: dict[str, object] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_id": spec.profile_id,
            "database_id": spec.database_id,
            "polymer": spec.polymer,
            "created_at": utc_now(),
            "generation_id": generation_id,
            "source": {
                "volume": SOURCE_DB_VOLUME_NAME,
                "path": spec.source_filename,
                "size_bytes": source_size,
                "sha256": source_sha256,
                "num_seqs": statistics["num_seqs"],
                "sum_len": statistics["sum_len"],
            },
            "shard_count": spec.shard_count,
            "shard_prefix": f"shards/{spec.source_filename}",
            "shards": shard_records,
            "search_space_value": (
                statistics["num_seqs"]
                if spec.polymer == "protein"
                else statistics["sum_len"] / 1_000_000
            ),
            "search_space_unit": spec.search_space_unit,
            "compatibility": {
                "alphafold_repository": ALPHAFOLD3_REPOSITORY,
                "alphafold_commit": ALPHAFOLD3_COMMIT,
                "hmmer_version": HMMER_VERSION,
                "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
            },
            "recipe": {
                "version": COMPOSABLE_MULTISET_RECIPE_VERSION,
                "seqkit_version": SEQKIT_VERSION,
                "seqkit_threads": threads,
                "random_seed": SHARD_RANDOM_SEED,
                "shuffle": [
                    "two-pass",
                    "first-pass-stage-local-source",
                    "source-occurrence-offset-index",
                    "splitmix64-fisher-yates-u32",
                    "bounded-concurrent-local-pread",
                    "ordered-write",
                ],
                "shuffler": {
                    "version": ORDINAL_SHUFFLER_VERSION,
                    "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
                    "record_identity": "source-occurrence",
                    "offset_index": "uint64-source-occurrence-offsets-v1",
                    "permutation": "splitmix64-fisher-yates-u32-v1",
                    "staging": "first-pass-tee-to-container-local-v1",
                    "read": "bounded-concurrent-local-pread-ordered-write-v2",
                    "ordered_output": True,
                },
                "execution": {
                    "worker_threads": threads,
                    "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
                    "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
                },
                "duplicate_recovery": {
                    "warning_source": None,
                    "record_identity": "source-occurrence",
                    "append_after_shuffle": False,
                    "strip_after_split": False,
                },
                "record_multiset": record_multiset_identity()
                | {"shard_threads": threads},
                "split": ["--by-part", spec.shard_count],
            },
            "validation": {
                "passed": True,
                "num_seqs": statistics["num_seqs"],
                "sum_len": statistics["sum_len"],
                "maximum_residue_imbalance": statistics["maximum_residue_imbalance"],
                "maximum_allowed_residue_imbalance": MAX_PROFILE_IMBALANCE,
                "recovered_records": recovery_metrics["recovered_records"],
                "recovered_residues": recovery_metrics["recovered_residues"],
                "first_recovered_byte_offset": recovery_metrics["first_byte_offset"],
                "last_recovered_byte_offset": recovery_metrics["last_byte_offset"],
                "temporary_recovery_prefix_absent": True,
                "record_occurrences_preserved": True,
                "canonical_record_multiset_match": True,
                "record_multiset_signature_sha256": signature_sha256,
                "artifacts": validation_records,
            },
        }
        validate_profile_manifest(cast(dict[str, Any], manifest), spec)

        # Commit the payload before moving it into its immutable public path.
        runtime.sharded_volume.commit()
        published_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.replace(published_root)
        payload_moved = True
        runtime.sharded_volume.commit()
        write_json_atomic(published_root / "manifest.json", manifest)
        runtime.sharded_volume.commit()
        manifest_published = True
        published_manifest = validate_published_profile(
            published_root,
            spec,
            verify_digests=True,
        )
        source_result = _apply_source_policy(
            runtime,
            spec,
            published_manifest,
            policy,
            log_path,
            seqkit_threads=threads,
        )
        result = {
            "status": "published",
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_path": str(published_root),
            "manifest_sha256": sha256_file(published_root / "manifest.json"),
            "source_size_bytes": source_size,
            "source_sha256": source_sha256,
            "num_seqs": statistics["num_seqs"],
            "sum_len": statistics["sum_len"],
            "search_space_value": manifest["search_space_value"],
            "search_space_unit": spec.search_space_unit,
            "maximum_residue_imbalance": statistics["maximum_residue_imbalance"],
            "recovered_records": recovery_metrics["recovered_records"],
            "recovered_residues": recovery_metrics["recovered_residues"],
            **source_result,
        }
        _write_success_evidence(runtime, evidence_root, result)
        _finish_claim(
            runtime,
            spec,
            generation_id,
            status="complete",
            detail={"manifest_sha256": result["manifest_sha256"]},
        )
        return result
    except Exception as exc:
        if not manifest_published:
            if payload_moved and published_root.exists():
                shutil.rmtree(published_root)
            elif staging_root.exists():
                shutil.rmtree(staging_root)
            runtime.sharded_volume.commit()
        failure = {
            "failed_at": utc_now(),
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_published": manifest_published,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        write_json_atomic(evidence_root / "failure.json", failure)
        runtime.output_volume.commit()
        _finish_claim(
            runtime,
            spec,
            generation_id,
            status="failed",
            detail={
                "error_type": type(exc).__name__,
                "profile_published": manifest_published,
            },
        )
        raise


def inspect_profile_registry(sharded_root: Path) -> dict[str, object]:
    """Quickly validate fixed manifests and artifact sizes without rehashing."""
    valid: list[str] = []
    missing: list[str] = []
    invalid: dict[str, dict[str, str]] = {}
    for spec in DATABASE_PROFILE_SPECS:
        root = profile_root(sharded_root, spec)
        if not (root / "manifest.json").is_file():
            missing.append(spec.database_id)
            continue
        try:
            validate_published_profile(root, spec, verify_digests=False)
        except (OSError, TypeError, ValueError) as exc:
            invalid[spec.database_id] = {
                "profile_id": spec.profile_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        else:
            valid.append(spec.database_id)
    profiles_root = sharded_root / "profiles"
    present_profile_ids = (
        sorted(
            path.name
            for path in profiles_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
        if profiles_root.is_dir()
        else []
    )
    selected_profile_ids = [spec.profile_id for spec in DATABASE_PROFILE_SPECS]
    return {
        "schema_version": 1,
        "valid_database_ids": valid,
        "missing_database_ids": missing,
        "invalid_profiles": invalid,
        "selected_profile_ids": selected_profile_ids,
        "present_profile_ids": present_profile_ids,
        "unselected_profile_ids": sorted(
            set(present_profile_ids) - set(selected_profile_ids)
        ),
    }


def plan_missing_profile_builds(
    inventory: dict[str, object],
    seqkit_threads: int,
    source_policy: str,
) -> tuple[tuple[str, int, str], ...]:
    """Return ordered builder inputs for only missing fixed profiles."""
    threads = validate_seqkit_threads(seqkit_threads)
    policy = validate_source_policy(source_policy)
    invalid = inventory.get("invalid_profiles")
    if not isinstance(invalid, dict):
        raise TypeError("Profile inventory has invalid failure details")
    if invalid:
        raise RuntimeError(
            "Existing published profile validation failed; repair it manually "
            f"before setup: {invalid}"
        )
    raw_missing = inventory.get("missing_database_ids")
    if not isinstance(raw_missing, list) or not all(
        isinstance(database_id, str) for database_id in raw_missing
    ):
        raise TypeError("Profile inventory returned invalid missing database IDs")
    missing = {
        database_id for database_id in raw_missing if isinstance(database_id, str)
    }
    selected = {spec.database_id for spec in DATABASE_PROFILE_SPECS}
    unknown = sorted(missing - selected)
    if unknown:
        raise ValueError(f"Profile inventory returned unknown database IDs: {unknown}")
    return tuple(
        (spec.database_id, threads, policy)
        for spec in DATABASE_PROFILE_SPECS
        if spec.database_id in missing
    )


def cleanup_profile_workspace(
    sharded_root: Path,
    claims: ClaimStore,
) -> dict[str, object]:
    """Remove abandoned and unselected profiles after every builder finishes."""
    inventory = inspect_profile_registry(sharded_root)
    if inventory["missing_database_ids"] or inventory["invalid_profiles"]:
        raise RuntimeError(
            "Cannot clean profile workspace before all profiles are valid"
        )
    active = [
        spec.profile_id
        for spec in DATABASE_PROFILE_SPECS
        if claims.get(_profile_claim_key(spec), None) is not None
    ]
    if active:
        raise RuntimeError(f"Cannot clean while profile claims are active: {active}")

    removed_workspace: list[str] = []
    for name in (".staging", ".orphaned"):
        root = sharded_root / name
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Expected profile workspace directory: {root}")
        for child in sorted(root.iterdir()):
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"Unexpected profile workspace entry: {child}")
            shutil.rmtree(child)
            removed_workspace.append(child.relative_to(sharded_root).as_posix())
        root.rmdir()

    removed_profiles: list[str] = []
    profiles_root = sharded_root / "profiles"
    unselected = inventory["unselected_profile_ids"]
    if not isinstance(unselected, list):
        raise TypeError("Profile inventory has invalid unselected profile IDs")
    for profile_id in unselected:
        if not isinstance(profile_id, str) or Path(profile_id).name != profile_id:
            raise ValueError(f"Unsafe unselected profile ID: {profile_id!r}")
        root = profiles_root / profile_id
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Expected unselected profile directory: {root}")
        shutil.rmtree(root)
        removed_profiles.append(root.relative_to(sharded_root).as_posix())
    return {
        "status": "passed",
        "removed_workspace_paths": removed_workspace,
        "removed_unselected_profile_paths": removed_profiles,
        "inventory": inspect_profile_registry(sharded_root),
    }
