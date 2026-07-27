"""Build and validate immutable AlphaFold 3 database shard profiles.

Modal decorators, named objects, and fan-out topology stay in the production
app composition root. This module receives only the mounted roots and narrow
persistence handles it needs at runtime.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, ClassVar, cast

import orjson

from biomodals.app.fold.alphafold3.artifacts import (
    VolumeHandle,
    append_log,
    artifact_record,
    json_bytes,
    require_regular_file,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from biomodals.app.fold.alphafold3.generation_claims import (
    ActiveGenerationError,
    ClaimStore,
    GenerationClaim,
    abandon_generation_claim,
    acquire_generation_claim,
    finish_generation_claim,
    generation_status,
    latest_generation_owner,
)
from biomodals.app.fold.alphafold3.profile_manifest import (
    current_profile_recipe,
    profile_compatibility_identity,
    validate_profile_manifest,
    validate_published_profile,
)
from biomodals.app.fold.alphafold3.profiles import (
    DATABASE_PROFILE_SPECS,
    MAX_PROFILE_IMBALANCE,
    PROFILE_SCHEMA_VERSION,
    PROFILE_STALE_SECONDS,
    SCRATCH_ROOT,
    SEQKIT_VERSION,
    SHARD_RANDOM_SEED,
    SHARDED_DB_VOLUME_NAME,
    SOURCE_DB_VOLUME_NAME,
    VALIDATION_RELPATHS,
    DatabaseProfileSpec,
    SourcePolicy,
    profile_root,
    record_multiset_identity,
    resolve_database_profile,
    shard_filename,
    shard_names,
    validate_seqkit_threads,
    validate_source_policy,
)
from biomodals.app.fold.alphafold3.sharding import (
    compile_record_multiset_validator,
    record_multiset_signature,
    require_executable,
    required_ordinal_shuffler_scratch_bytes,
    scan_record_multiset,
    shuffle_fasta_occurrences,
    verify_file,
)

_JSONL_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE


@dataclass(frozen=True, slots=True)
class ProfileBuilderRuntime:
    """Mounted paths and persistence handles for one builder container."""

    SOURCE_MOUNT: ClassVar[str] = f"/{SOURCE_DB_VOLUME_NAME}"
    SHARDED_MOUNT: ClassVar[str] = f"/{SHARDED_DB_VOLUME_NAME}"
    EVIDENCE_RELPATH: ClassVar[str] = "msa-profile-builds"

    output_root: Path
    source_volume: VolumeHandle
    sharded_volume: VolumeHandle
    output_volume: VolumeHandle
    claims: ClaimStore
    container_id: str
    source_root: Path = Path(SOURCE_MOUNT)
    sharded_root: Path = Path(SHARDED_MOUNT)
    evidence_relpath: str = EVIDENCE_RELPATH


@dataclass(frozen=True, slots=True)
class SourceProfileEvidence:
    """Validated source FASTA identity and SeqKit statistics."""

    size_bytes: int
    sha256: str
    num_seqs: int
    stats_path: Path


@dataclass(frozen=True, slots=True)
class ShardBuildEvidence:
    """Validated shard paths and manifest-relevant scientific evidence."""

    shard_paths: tuple[Path, ...]
    statistics: dict[str, int | float]
    recovery_metrics: dict[str, Any]
    record_multiset_signature_sha256: str


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
    source_signature_sha256 = sha256_bytes(json_bytes(source_signature))
    shard_signature_sha256 = sha256_bytes(json_bytes(shard_signature))
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
    diagnostics_path = scratch_root / "shuffle-stderr.log"
    evidence_path = validation_dir / "shuffler-evidence.json"
    recovery_report_path = validation_dir / "duplicate-recovery.jsonl"
    validation_dir.mkdir(parents=True, exist_ok=True)
    staged_source_path = shuffle_fasta_occurrences(
        source_path,
        shuffled_path,
        scratch_root,
        diagnostics_path,
        evidence_path,
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
        staged_source_path,
        shuffled_path,
        recovery_metrics,
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


def _legacy_profile_claim_key(spec: DatabaseProfileSpec) -> str:
    return f"active:{spec.profile_id}"


def _profile_claim_root_key(spec: DatabaseProfileSpec) -> str:
    return f"claim:{spec.profile_id}:root"


def _adapt_legacy_profile_owner(
    scope_key: str,
    value: object,
) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("profile_id") != scope_key:
        raise RuntimeError(f"Profile {scope_key} has an invalid legacy claim owner")
    database_id = value.get("database_id")
    if not isinstance(database_id, str):
        raise RuntimeError(f"Profile {scope_key} legacy claim identity is invalid")
    spec = resolve_database_profile(database_id)
    if spec.profile_id != scope_key:
        raise RuntimeError(f"Profile {scope_key} legacy claim identity is invalid")
    return {
        "scope_key": scope_key,
        "generation_id": value.get("generation_id"),
        "identity": {
            "profile_id": scope_key,
            "database_id": database_id,
        },
        "container_id": value.get("container_id"),
        "started_at": value.get("started_at"),
        "started_at_epoch_seconds": value.get("started_at_epoch_seconds"),
        "maximum_age_seconds": value.get("maximum_age_seconds"),
    }


def _adopt_legacy_claim(
    claims: ClaimStore,
    spec: DatabaseProfileSpec,
) -> None:
    """Adopt an old active-key owner as the append-only claim root."""
    legacy_claim = claims.get(_legacy_profile_claim_key(spec), None)
    if legacy_claim is None:
        return
    legacy_owner = _adapt_legacy_profile_owner(spec.profile_id, legacy_claim)
    root_key = _profile_claim_root_key(spec)
    claims.put(root_key, legacy_owner, skip_if_exists=True)
    root_owner = claims.get(root_key, None)
    if not isinstance(root_owner, dict):
        raise RuntimeError(f"Profile {spec.profile_id} claim root disappeared")
    if root_owner.get("generation_id") != legacy_owner["generation_id"]:
        raise RuntimeError(
            f"Profile {spec.profile_id} legacy and append-only claims conflict"
        )
    latest_generation_owner(
        claims,
        spec.profile_id,
        owner_adapter=_adapt_legacy_profile_owner,
    )


def _acquire_profile_claim(
    runtime: ProfileBuilderRuntime,
    spec: DatabaseProfileSpec,
    generation_id: str,
) -> GenerationClaim:
    """Append one elected generation after a terminal or stale predecessor."""
    _adopt_legacy_claim(runtime.claims, spec)
    try:
        return acquire_generation_claim(
            runtime.claims,
            scope_key=spec.profile_id,
            generation_id=generation_id,
            identity={
                "profile_id": spec.profile_id,
                "database_id": spec.database_id,
            },
            container_id=runtime.container_id,
            maximum_age_seconds=PROFILE_STALE_SECONDS,
            owner_adapter=_adapt_legacy_profile_owner,
        )
    except ActiveGenerationError as exc:
        raise RuntimeError(
            f"Profile {spec.profile_id} is already being built by generation "
            f"{exc.owner['generation_id']!r}"
        ) from exc


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
    source_policy: SourcePolicy,
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
        if (
            source_path.stat().st_size != expected_size
            or sha256_file(source_path) != expected_sha256
        ):
            raise ValueError(
                f"Refusing to replace changed source {spec.source_filename}"
            )
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
    write_json_atomic(
        evidence_root / "done.json",
        result | {"completed_at": utc_now()},
    )
    runtime.output_volume.commit()


def _reuse_published_profile(
    runtime: ProfileBuilderRuntime,
    spec: DatabaseProfileSpec,
    published_root: Path,
    evidence_root: Path,
    log_path: Path,
    generation_id: str,
    source_policy: SourcePolicy,
    *,
    seqkit_threads: int,
) -> dict[str, object]:
    """Deeply validate and reuse a publication that won a setup race."""
    manifest = validate_published_profile(
        published_root,
        spec,
        verify_digests=True,
    )
    source_result = _apply_source_policy(
        runtime,
        spec,
        manifest,
        source_policy,
        log_path,
        seqkit_threads=seqkit_threads,
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


def _prepare_source_evidence(
    spec: DatabaseProfileSpec,
    source_path: Path,
    validation_dir: Path,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> tuple[SourceProfileEvidence, str]:
    """Validate one source FASTA, its fixed counts, and local scratch budget."""
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
    run_to_file(
        [
            seqkit,
            "stats",
            "-j",
            str(seqkit_threads),
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
        raise ValueError(f"Expected one source stats row, got {source_stats.height}")
    source_num_seqs = int(source_stats.item(0, "num_seqs"))
    source_sum_len = int(source_stats.item(0, "sum_len"))
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
    return (
        SourceProfileEvidence(
            size_bytes=source_size,
            sha256=sha256_file(source_path),
            num_seqs=source_num_seqs,
            stats_path=source_stats_path,
        ),
        seqkit,
    )


def _build_and_validate_shards(
    spec: DatabaseProfileSpec,
    source_path: Path,
    source: SourceProfileEvidence,
    raw_shard_dir: Path,
    shard_dir: Path,
    validation_dir: Path,
    log_path: Path,
    seqkit: str,
    *,
    seqkit_threads: int,
) -> ShardBuildEvidence:
    """Shuffle, split, and prove source/shard record-multiset equivalence."""
    shard_stats_path = validation_dir / "shard-stats.tsv"
    shard_summary_path = validation_dir / "shard-summary.parquet"
    record_multiset_path = validation_dir / "record-multiset.json"
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
            expected_records=source.num_seqs,
            seqkit_threads=seqkit_threads,
        )
        staged_verification = verify_file(
            staged_source_path,
            expected_size=source.size_bytes,
            expected_sha256=source.sha256,
        )
        append_log(
            log_path,
            "Verified container-local staged source against the Volume source SHA-256",
        )
        validator = compile_record_multiset_validator(scratch_root, log_path)
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
            shard_paths = _run_split(
                spec,
                shuffled_path,
                raw_shard_dir,
                shard_dir,
                log_path,
                seqkit_threads=seqkit_threads,
            )
            source_result = source_future.result() if source_future.done() else None
            shard_result = scan_record_multiset(
                validator,
                shard_paths,
                scratch_root / "shard-record-multiset.json",
                log_path,
                threads=seqkit_threads,
            )
            if source_result is None:
                source_result = source_future.result()
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
            },
        )

    run_to_file(
        [
            seqkit,
            "stats",
            "-j",
            str(seqkit_threads),
            "--all",
            "--tabular",
            *(str(path) for path in shard_paths),
        ],
        shard_stats_path,
        log_path,
    )
    statistics = _validate_statistics(
        spec,
        source.stats_path,
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
    return ShardBuildEvidence(
        shard_paths=shard_paths,
        statistics=statistics,
        recovery_metrics=recovery_metrics,
        record_multiset_signature_sha256=signature_sha256,
    )


def build_profile_manifest(
    spec: DatabaseProfileSpec,
    generation_id: str,
    source: SourceProfileEvidence,
    shards: ShardBuildEvidence,
    staging_root: Path,
    *,
    seqkit_threads: int,
) -> dict[str, object]:
    """Construct and validate one immutable profile manifest."""
    statistics = shards.statistics
    recovery_metrics = shards.recovery_metrics
    shard_records = [
        artifact_record(shard_path, staging_root) for shard_path in shards.shard_paths
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
            "size_bytes": source.size_bytes,
            "sha256": source.sha256,
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
        "compatibility": profile_compatibility_identity(),
        "recipe": current_profile_recipe(spec, seqkit_threads),
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
            "record_multiset_signature_sha256": (
                shards.record_multiset_signature_sha256
            ),
            "artifacts": validation_records,
        },
    }
    validate_profile_manifest(cast(dict[str, Any], manifest), spec)
    return manifest


def build_profile(
    runtime: ProfileBuilderRuntime,
    database_id: str,
    seqkit_threads: int,
    source_policy: SourcePolicy,
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
        return _reuse_published_profile(
            runtime,
            spec,
            published_root,
            evidence_root,
            log_path,
            generation_id,
            policy,
            seqkit_threads=threads,
        )

    claim = _acquire_profile_claim(runtime, spec, generation_id)
    staging_root = (
        runtime.sharded_root / ".staging" / f"{spec.profile_id}-{generation_id}"
    )
    raw_shard_dir = staging_root / ".raw-shards"
    shard_dir = staging_root / "shards"
    validation_dir = staging_root / "validation"
    payload_moved = False
    manifest_published = False
    claim_status = "failed"
    claim_detail: dict[str, object] = {
        "error_type": "IncompleteProfileBuild",
        "profile_published": False,
    }
    try:
        write_json_atomic(evidence_root / "claim.json", claim.owner)
        runtime.output_volume.commit()
        runtime.sharded_volume.reload()
        if (published_root / "manifest.json").is_file():
            result = _reuse_published_profile(
                runtime,
                spec,
                published_root,
                evidence_root,
                log_path,
                generation_id,
                policy,
                seqkit_threads=threads,
            )
            claim_status = "complete"
            claim_detail = {"manifest_sha256": result["manifest_sha256"]}
            return result
        if published_root.exists():
            orphan_root = runtime.sharded_root / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            published_root.replace(
                orphan_root / f"{spec.profile_id}-{uuid.uuid4().hex}"
            )
            runtime.sharded_volume.commit()

        source, seqkit = _prepare_source_evidence(
            spec,
            source_path,
            validation_dir,
            log_path,
            seqkit_threads=threads,
        )
        shard_dir.mkdir(parents=True)

        shards = _build_and_validate_shards(
            spec,
            source_path,
            source,
            raw_shard_dir,
            shard_dir,
            validation_dir,
            log_path,
            seqkit,
            seqkit_threads=threads,
        )

        manifest = build_profile_manifest(
            spec,
            generation_id,
            source,
            shards,
            staging_root,
            seqkit_threads=threads,
        )

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
            "source_size_bytes": source.size_bytes,
            "source_sha256": source.sha256,
            "num_seqs": shards.statistics["num_seqs"],
            "sum_len": shards.statistics["sum_len"],
            "search_space_value": manifest["search_space_value"],
            "search_space_unit": spec.search_space_unit,
            "maximum_residue_imbalance": shards.statistics["maximum_residue_imbalance"],
            "recovered_records": shards.recovery_metrics["recovered_records"],
            "recovered_residues": shards.recovery_metrics["recovered_residues"],
            **source_result,
        }
        _write_success_evidence(runtime, evidence_root, result)
        claim_status = "complete"
        claim_detail = {"manifest_sha256": result["manifest_sha256"]}
        return result
    except Exception as exc:
        failure = {
            "failed_at": utc_now(),
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_published": manifest_published,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        evidence_committed = False
        cleanup_completed = manifest_published
        try:
            write_json_atomic(evidence_root / "failure.json", failure)
            runtime.output_volume.commit()
            evidence_committed = True
        except Exception as evidence_exc:
            exc.add_note(
                "Could not commit durable profile-build failure evidence: "
                f"{type(evidence_exc).__name__}: {evidence_exc}"
            )

        if evidence_committed and not manifest_published:
            try:
                if payload_moved and published_root.exists():
                    shutil.rmtree(published_root)
                elif staging_root.exists():
                    shutil.rmtree(staging_root)
                runtime.sharded_volume.commit()
                cleanup_completed = True
            except Exception as cleanup_exc:
                exc.add_note(
                    "Could not clean the failed profile generation: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        claim_detail = {
            "error_type": type(exc).__name__,
            "profile_published": manifest_published,
            "failure_evidence_committed": evidence_committed,
            "cleanup_completed": cleanup_completed,
        }
        raise
    finally:
        finish_generation_claim(
            runtime.claims,
            claim,
            status=claim_status,
            detail=claim_detail,
        )


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
) -> tuple[tuple[str, int, SourcePolicy], ...]:
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
    active: list[str] = []
    for spec in DATABASE_PROFILE_SPECS:
        _adopt_legacy_claim(claims, spec)
        owner = latest_generation_owner(
            claims,
            spec.profile_id,
            owner_adapter=_adapt_legacy_profile_owner,
        )
        if owner is None:
            continue
        generation_id = cast(str, owner["generation_id"])
        status = generation_status(claims, spec.profile_id, generation_id)
        if status is not None:
            continue
        started_at = cast(int | float, owner["started_at_epoch_seconds"])
        age_seconds = time() - float(started_at)
        if age_seconds <= PROFILE_STALE_SECONDS:
            active.append(spec.profile_id)
            continue
        abandon_generation_claim(
            claims,
            GenerationClaim(
                scope_key=spec.profile_id,
                generation_id=generation_id,
                owner=owner,
            ),
            detail={
                "age_seconds": age_seconds,
                "cleanup_recovery": True,
            },
        )
        if generation_status(claims, spec.profile_id, generation_id) is None:
            active.append(spec.profile_id)
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


def finalize_profile_setup(runtime: ProfileBuilderRuntime) -> dict[str, object]:
    """Clean the fixed profile registry and publish durable setup evidence."""
    runtime.sharded_volume.reload()
    runtime.output_volume.reload()
    result = cleanup_profile_workspace(runtime.sharded_root, runtime.claims)
    runtime.sharded_volume.commit()

    setup_id = uuid.uuid4().hex
    evidence_root = runtime.output_root / runtime.evidence_relpath / "setup" / setup_id
    completed = result | {
        "setup_id": setup_id,
        "completed_at": utc_now(),
    }
    write_json_atomic(evidence_root / "inventory.json", completed)
    runtime.output_volume.commit()
    write_json_atomic(
        evidence_root / "done.json",
        {
            "status": "complete",
            "setup_id": setup_id,
            "completed_at": utc_now(),
        },
    )
    runtime.output_volume.commit()
    return completed
