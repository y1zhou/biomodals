"""Benchmark AlphaFold 3 MSA searches against a sharded small-BFD database.

Upstream sources:

- <https://github.com/google-deepmind/alphafold3>
- <https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md>

This is an isolated, temporary benchmark app. It does not mount AlphaFold model
weights or the production MSA cache, and it never imports ``alphafold3_app``.
Commands are plan-only unless ``--submit`` is explicitly supplied.

The first operation copies small BFD from the read-only production database
Volume, creates 64 deterministic SeqKit shards in
``AlphaFold3-msa-db-sharded``, validates them, and publishes ``manifest.json``
last. Benchmark evidence is written to
``AlphaFold3-MSA-Benchmark-outputs``.
"""

from __future__ import annotations

import hashlib
import io
import os
import shlex
import shutil
import socket
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

import modal
import orjson

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper

CAMPAIGN_ID = "small-bfd-phase1-v1"
PROFILE_ID = "small-bfd-64-v1"
PROFILE_SCHEMA_VERSION = 1
SOURCE_DB_FILENAME = "bfd-first_non_consensus_sequences.fasta"
SOURCE_DB_VOLUME_NAME = "AlphaFold3-msa-db"
SHARDED_DB_VOLUME_NAME = "AlphaFold3-msa-db-sharded"
OUTPUT_VOLUME_NAME = "AlphaFold3-MSA-Benchmark-outputs"
DATABASE_ID = "small-bfd"
SHARD_COUNT = 64
SHARD_RANDOM_SEED = 23
SMALL_BFD_Z = 65_984_053
SEQKIT_VERSION = "2.13.0"
DEFAULT_SEQKIT_THREADS = 8
MAX_SEQKIT_THREADS = 32
MAX_PROFILE_IMBALANCE = 0.05
JSON_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE


CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="AlphaFold3-MSA-Benchmark",
    repo_url="https://github.com/y1zhou/alphafold3",
    repo_commit_hash="987ad1cb7d7028b6d35908cf63fe7d951d98d6b6",
    package_name="alphafold3",
    version="3.0.2",
    python_version="3.12",
    cuda_version="cu130",
    timeout=21_600,
)


@dataclass(frozen=True)
class AppInfo:
    """Fixed Volume mount points and profile-relative paths."""

    source_db_dir: str = f"/mnt/{SOURCE_DB_VOLUME_NAME}"
    sharded_db_dir: str = f"/mnt/{SHARDED_DB_VOLUME_NAME}"
    output_dir: str = f"/mnt/{OUTPUT_VOLUME_NAME}"
    profile_relpath: str = f"profiles/{PROFILE_ID}"
    preparation_relpath: str = f"benchmarks/{CAMPAIGN_ID}/preparation"


APP_INFO = AppInfo()

SOURCE_MSA_DB_VOLUME = modal.Volume.from_name(
    SOURCE_DB_VOLUME_NAME,
    version=2,
)
SHARDED_MSA_DB_VOLUME = modal.Volume.from_name(
    SHARDED_DB_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
BENCHMARK_OUTPUT_VOLUME = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)


# Keep the benchmark on the exact AlphaFold/HMMER environment used by the
# production app. SeqKit is an additional preparation-only tool.
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install(
        "git",
        "build-essential",
        "curl",
        "zstd",
        "zlib1g-dev",
        "wget",
    )
    .env(
        CONF.default_env
        | {
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": "0.95",
        }
    )
    .run_commands("curl -L micro.mamba.pm/install.sh | bash")
    .micromamba_install(
        f"seqkit={SEQKIT_VERSION}",
        channels=["conda-forge", "bioconda"],
    )
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            "mkdir /hmmer_build",
            "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz "
            "--directory-prefix /hmmer_build",
            "cd /hmmer_build",
            "echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 "
            "hmmer-3.4.tar.gz' | sha256sum --check",
            "tar zxf hmmer-3.4.tar.gz",
            "rm hmmer-3.4.tar.gz",
            "cd /hmmer_build",
            f"patch -p0 < {CONF.git_clone_dir}/docker/jackhmmer_seq_limit.patch",
            "cd /hmmer_build/hmmer-3.4",
            "./configure --prefix=/hmmer",
            "make -j",
            "make install",
            "cd /hmmer_build/hmmer-3.4/easel",
            "make install",
            "rm -rf /hmmer_build",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(str(CONF.git_clone_dir))
    .run_commands("build_data")
    .env({"PATH": "/hmmer/bin:$PATH"})
    .pipe(patch_image_for_helper)
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


def _utc_now() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _validate_seqkit_threads(seqkit_threads: int) -> int:
    """Validate the SeqKit concurrency argument."""
    if isinstance(seqkit_threads, bool) or not isinstance(seqkit_threads, int):
        raise TypeError("seqkit_threads must be an integer")
    if not 1 <= seqkit_threads <= MAX_SEQKIT_THREADS:
        raise ValueError(
            f"seqkit_threads must be between 1 and {MAX_SEQKIT_THREADS}, "
            f"got {seqkit_threads}"
        )
    return seqkit_threads


def _shard_filename(index: int) -> str:
    """Return the AlphaFold shard filename for a zero-based index."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("shard index must be an integer")
    if not 0 <= index < SHARD_COUNT:
        raise ValueError(f"shard index must be in [0, {SHARD_COUNT}), got {index}")
    return f"{SOURCE_DB_FILENAME}-{index:05d}-of-{SHARD_COUNT:05d}"


def _shard_names() -> tuple[str, ...]:
    """Return every expected shard filename in AlphaFold order."""
    return tuple(_shard_filename(index) for index in range(SHARD_COUNT))


def _json_bytes(value: object) -> bytes:
    """Serialize a JSON value deterministically."""
    return orjson.dumps(value, option=JSON_OPTIONS)


def _write_json_atomic(path: Path, value: object) -> None:
    """Atomically publish one small JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting all other top-level values."""
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_regular_file(path: Path) -> None:
    """Require a non-symlink regular file with at least one byte."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Expected regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Expected nonempty file: {path}")


def _sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute a streaming SHA-256 digest for a regular file."""
    _require_regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file_with_sha256(source: Path, destination: Path) -> tuple[str, int]:
    """Stream one file between Volume mounts while hashing its source bytes."""
    _require_regular_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as source_handle, destination.open("xb") as dest_handle:
        while chunk := source_handle.read(16 * 1024 * 1024):
            dest_handle.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        dest_handle.flush()
        os.fsync(dest_handle.fileno())
    return digest.hexdigest(), byte_count


def _append_log(path: Path, message: str) -> None:
    """Append one timestamped line to a durable operation log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _require_executable(name: str) -> str:
    """Resolve a fixed executable name to an absolute path."""
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"Required executable is not installed: {name}")
    return str(Path(executable).resolve())


def _run_to_file(argv: list[str], output_path: Path, log_path: Path) -> None:
    """Run a fixed argv command with separate data and diagnostic streams."""
    _append_log(log_path, f"Running command: {shlex.join(argv)}")
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


def _run_shuffle_split(
    source_path: Path,
    raw_shard_dir: Path,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> None:
    """Stream deterministic SeqKit shuffle output directly into split2."""
    seqkit = _require_executable("seqkit")
    shuffle_argv = [
        seqkit,
        "shuffle",
        "-j",
        str(seqkit_threads),
        "--two-pass",
        "--update-faidx",
        "--rand-seed",
        str(SHARD_RANDOM_SEED),
        str(source_path),
    ]
    split_argv = [
        seqkit,
        "split2",
        "-j",
        str(seqkit_threads),
        "--by-part",
        str(SHARD_COUNT),
        "--out-dir",
        str(raw_shard_dir),
        "--force",
        "--out-prefix",
        "part_",
        "-",
    ]
    _append_log(log_path, f"Running command: {shlex.join(shuffle_argv)}")
    _append_log(log_path, f"Piping into: {shlex.join(split_argv)}")
    raw_shard_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        shuffle_process = subprocess.Popen(  # noqa: S603
            shuffle_argv,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        if shuffle_process.stdout is None:
            shuffle_process.kill()
            raise RuntimeError("SeqKit shuffle did not expose stdout")
        try:
            split_process = subprocess.run(  # noqa: S603
                split_argv,
                check=False,
                stdin=shuffle_process.stdout,
                stderr=log,
            )
        finally:
            shuffle_process.stdout.close()
        shuffle_returncode = shuffle_process.wait()
    if shuffle_returncode != 0:
        raise subprocess.CalledProcessError(shuffle_returncode, shuffle_argv)
    if split_process.returncode != 0:
        raise subprocess.CalledProcessError(split_process.returncode, split_argv)


def _run_aggregate_seqkit_sum(
    shard_paths: tuple[Path, ...],
    output_path: Path,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> None:
    """Stream every shard into one order-independent SeqKit checksum."""
    cat = _require_executable("cat")
    seqkit = _require_executable("seqkit")
    cat_argv = [cat, *(str(path) for path in shard_paths)]
    sum_argv = [seqkit, "sum", "-j", str(seqkit_threads), "--all", "-"]
    _append_log(log_path, f"Running command: {shlex.join(cat_argv)}")
    _append_log(log_path, f"Piping into: {shlex.join(sum_argv)}")
    with output_path.open("xb") as output, log_path.open("ab") as log:
        cat_process = subprocess.Popen(  # noqa: S603
            cat_argv,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        if cat_process.stdout is None:
            cat_process.kill()
            raise RuntimeError("cat did not expose stdout")
        try:
            sum_process = subprocess.run(  # noqa: S603
                sum_argv,
                check=False,
                stdin=cat_process.stdout,
                stdout=output,
                stderr=log,
            )
        finally:
            cat_process.stdout.close()
        cat_returncode = cat_process.wait()
    if cat_returncode != 0:
        raise subprocess.CalledProcessError(cat_returncode, cat_argv)
    if sum_process.returncode != 0:
        raise subprocess.CalledProcessError(sum_process.returncode, sum_argv)


def _seqkit_sum_digest(path: Path) -> str:
    """Extract the digest token from one SeqKit sum output file."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"Expected one SeqKit sum row in {path}, got {len(lines)}")
    fields = lines[0].split("\t")
    if not fields or not fields[0].startswith("seqkit."):
        raise ValueError(f"Invalid SeqKit sum output in {path}: {lines[0]!r}")
    return fields[0]


def _validate_profile_statistics(
    source_stats_path: Path,
    shard_stats_path: Path,
    shard_summary_path: Path,
) -> dict[str, int | float]:
    """Validate SeqKit statistics and persist the normalized shard table."""
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
    if shard_stats.height != SHARD_COUNT:
        raise ValueError(
            f"Expected {SHARD_COUNT} shard stats rows, got {shard_stats.height}"
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
    expected_names = list(_shard_names())
    actual_names = shard_stats.get_column("basename").to_list()
    if actual_names != expected_names:
        raise ValueError("SeqKit stats shard names do not match the profile")

    source_num_seqs = int(source_stats.item(0, "num_seqs"))
    source_sum_len = int(source_stats.item(0, "sum_len"))
    shard_num_seqs = int(shard_stats.get_column("num_seqs").sum())
    shard_sum_len = int(shard_stats.get_column("sum_len").sum())
    if source_num_seqs != SMALL_BFD_Z:
        raise ValueError(
            f"Expected small-BFD Z={SMALL_BFD_Z}, measured {source_num_seqs}"
        )
    if shard_num_seqs != source_num_seqs:
        raise ValueError(
            f"Shard sequence count {shard_num_seqs} != source {source_num_seqs}"
        )
    if shard_sum_len != source_sum_len:
        raise ValueError(
            f"Shard residue count {shard_sum_len} != source {source_sum_len}"
        )

    mean_sum_len = source_sum_len / SHARD_COUNT
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


def _artifact_record(path: Path, profile_root: Path) -> dict[str, str | int]:
    """Build one manifest artifact record below the profile root."""
    _require_regular_file(path)
    resolved_root = profile_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Artifact escapes profile root: {path}")
    return {
        "path": resolved_path.relative_to(resolved_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_published_profile(
    profile_root: Path,
    *,
    verify_digests: bool,
) -> dict[str, Any]:
    """Validate a published profile manifest and its declared artifacts."""
    manifest_path = profile_root / "manifest.json"
    _require_regular_file(manifest_path)
    manifest = _load_json_object(manifest_path)
    source, shards = _validate_profile_manifest(manifest)

    records = [source, *shards]
    for record in records:
        relative = str(record["path"])
        artifact_path = (profile_root / relative).resolve()
        if not artifact_path.is_relative_to(profile_root.resolve()):
            raise ValueError(f"Profile artifact escapes root: {relative}")
        _require_regular_file(artifact_path)
        if artifact_path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Profile artifact size mismatch: {relative}")
        if verify_digests and _sha256_file(artifact_path) != record["sha256"]:
            raise ValueError(f"Profile artifact digest mismatch: {relative}")
    return manifest


def _validate_profile_manifest(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate profile metadata without requiring a mounted filesystem."""
    if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unexpected profile manifest schema version")
    if manifest.get("profile_id") != PROFILE_ID:
        raise ValueError("Unexpected profile ID")
    if manifest.get("database_id") != DATABASE_ID:
        raise ValueError("Unexpected database ID")
    if manifest.get("shard_count") != SHARD_COUNT:
        raise ValueError("Unexpected shard count")
    if manifest.get("z_value") != SMALL_BFD_Z:
        raise ValueError("Unexpected small-BFD Z value")

    source = manifest.get("source")
    shards = manifest.get("shards")
    validation = manifest.get("validation")
    if not isinstance(source, dict):
        raise ValueError("Profile manifest source must be an object")
    if source.get("path") != f"source/{SOURCE_DB_FILENAME}":
        raise ValueError("Profile manifest source path is invalid")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise ValueError(f"Profile manifest must declare {SHARD_COUNT} shards")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Profile manifest does not declare passed validation")

    expected_paths = [f"shards/{name}" for name in _shard_names()]
    actual_paths: list[str] = []
    records = [source, *shards]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Profile artifact record must be an object")
        relative = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Profile artifact path must be relative")
        if ".." in PurePosixPath(relative).parts:
            raise ValueError(f"Profile artifact path escapes root: {relative}")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError(f"Invalid artifact size: {relative}")
        if size_bytes <= 0:
            raise ValueError(f"Profile artifact is empty: {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid artifact SHA-256: {relative}")
        if relative.startswith("shards/"):
            actual_paths.append(relative)
    if actual_paths != expected_paths:
        raise ValueError("Profile manifest shard order or names are invalid")
    return source, shards


def _build_prepare_plan(seqkit_threads: int) -> dict[str, object]:
    """Build the side-effect-free profile-preparation plan."""
    threads = _validate_seqkit_threads(seqkit_threads)
    return {
        "campaign_id": CAMPAIGN_ID,
        "operation": "prepare",
        "remote_calls": 1,
        "resources": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "timeout_seconds": CONF.timeout,
        },
        "source": {
            "volume": SOURCE_DB_VOLUME_NAME,
            "path": SOURCE_DB_FILENAME,
            "mount": "read-only",
        },
        "destination": {
            "volume": SHARDED_DB_VOLUME_NAME,
            "profile": APP_INFO.profile_relpath,
        },
        "seqkit": {
            "version": SEQKIT_VERSION,
            "threads": threads,
            "random_seed": SHARD_RANDOM_SEED,
            "shards": SHARD_COUNT,
        },
        "existing_profile_policy": "validate-and-reuse",
    }


def _prepare_profile(seqkit_threads: int) -> dict[str, object]:
    """Build, validate, and publish the small-BFD profile."""
    threads = _validate_seqkit_threads(seqkit_threads)
    source_root = Path(APP_INFO.source_db_dir)
    sharded_root = Path(APP_INFO.sharded_db_dir)
    output_root = Path(APP_INFO.output_dir)
    source_path = source_root / SOURCE_DB_FILENAME
    profile_root = sharded_root / APP_INFO.profile_relpath
    evidence_root = output_root / APP_INFO.preparation_relpath
    log_path = evidence_root / "run.log"
    evidence_root.mkdir(parents=True, exist_ok=True)
    _append_log(log_path, f"Preparing profile {PROFILE_ID}")

    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    if (profile_root / "manifest.json").is_file():
        manifest = _validate_published_profile(profile_root, verify_digests=True)
        result = {
            "status": "reused",
            "profile_path": str(profile_root),
            "manifest_sha256": _sha256_file(profile_root / "manifest.json"),
        }
        _write_json_atomic(evidence_root / "metrics.json", result)
        BENCHMARK_OUTPUT_VOLUME.commit()
        _write_json_atomic(
            evidence_root / "done.json",
            result | {"completed_at": _utc_now(), "profile_id": manifest["profile_id"]},
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        return result

    if profile_root.exists():
        orphan_root = sharded_root / ".orphaned"
        orphan_root.mkdir(parents=True, exist_ok=True)
        orphan_path = orphan_root / f"{PROFILE_ID}-{uuid.uuid4().hex}"
        profile_root.replace(orphan_path)
        SHARDED_MSA_DB_VOLUME.commit()
        _append_log(log_path, f"Preserved incomplete profile at {orphan_path}")

    generation_id = uuid.uuid4().hex
    staging_root = sharded_root / ".staging" / f"{PROFILE_ID}-{generation_id}"
    staging_source_dir = staging_root / "source"
    raw_shard_dir = staging_root / ".raw-shards"
    shard_dir = staging_root / "shards"
    validation_dir = staging_root / "validation"
    staging_source_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    copied_source = staging_source_dir / SOURCE_DB_FILENAME
    source_sha256, source_size = _copy_file_with_sha256(source_path, copied_source)
    copied_sha256 = _sha256_file(copied_source)
    if copied_sha256 != source_sha256:
        raise ValueError("Copied small-BFD SHA-256 does not match source")
    if copied_source.stat().st_size != source_size:
        raise ValueError("Copied small-BFD byte size does not match source")
    SHARDED_MSA_DB_VOLUME.commit()

    seqkit = _require_executable("seqkit")
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
    _append_log(log_path, f"Using {version_output}")

    _run_shuffle_split(
        copied_source,
        raw_shard_dir,
        log_path,
        seqkit_threads=threads,
    )
    raw_shards = sorted(
        path
        for path in raw_shard_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if len(raw_shards) != SHARD_COUNT:
        raise ValueError(f"Expected {SHARD_COUNT} raw shards, found {len(raw_shards)}")
    for index, raw_shard in enumerate(raw_shards):
        if raw_shard.stat().st_size <= 0:
            raise ValueError(f"SeqKit produced empty shard: {raw_shard}")
        raw_shard.replace(shard_dir / _shard_filename(index))
    raw_shard_dir.rmdir()
    Path(f"{copied_source}.seqkit.fai").unlink(missing_ok=True)
    shard_paths = tuple(shard_dir / name for name in _shard_names())

    source_stats_path = validation_dir / "source-stats.tsv"
    shard_stats_path = validation_dir / "shard-stats.tsv"
    shard_summary_path = validation_dir / "shard-summary.parquet"
    source_sum_path = validation_dir / "source-sum.tsv"
    shard_sum_path = validation_dir / "shard-sum.tsv"
    _run_to_file(
        [
            seqkit,
            "stats",
            "-j",
            str(threads),
            "--all",
            "--tabular",
            str(copied_source),
        ],
        source_stats_path,
        log_path,
    )
    _run_to_file(
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
    _run_to_file(
        [
            seqkit,
            "sum",
            "-j",
            str(threads),
            "--all",
            str(copied_source),
        ],
        source_sum_path,
        log_path,
    )
    _run_aggregate_seqkit_sum(
        shard_paths,
        shard_sum_path,
        log_path,
        seqkit_threads=threads,
    )
    source_sum = _seqkit_sum_digest(source_sum_path)
    shard_sum = _seqkit_sum_digest(shard_sum_path)
    if source_sum != shard_sum:
        raise ValueError("Aggregate shard SeqKit sum does not match source")
    statistics = _validate_profile_statistics(
        source_stats_path,
        shard_stats_path,
        shard_summary_path,
    )

    source_record = _artifact_record(copied_source, staging_root)
    shard_records = [
        _artifact_record(shard_path, staging_root) for shard_path in shard_paths
    ]
    manifest: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "database_id": DATABASE_ID,
        "created_at": _utc_now(),
        "generation_id": generation_id,
        "source_volume": SOURCE_DB_VOLUME_NAME,
        "source": source_record,
        "shard_count": SHARD_COUNT,
        "shard_prefix": f"shards/{SOURCE_DB_FILENAME}",
        "shards": shard_records,
        "z_value": SMALL_BFD_Z,
        "recipe": {
            "version": 1,
            "seqkit_version": SEQKIT_VERSION,
            "seqkit_threads": threads,
            "random_seed": SHARD_RANDOM_SEED,
            "shuffle": ["--two-pass", "--update-faidx"],
            "split": ["--by-part", SHARD_COUNT],
        },
        "validation": {
            "passed": True,
            "source_sha256_matches_copy": True,
            "seqkit_sum": source_sum,
            "num_seqs": statistics["num_seqs"],
            "sum_len": statistics["sum_len"],
            "maximum_residue_imbalance": statistics["maximum_residue_imbalance"],
            "maximum_allowed_residue_imbalance": MAX_PROFILE_IMBALANCE,
        },
    }

    SHARDED_MSA_DB_VOLUME.commit()
    profile_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.replace(profile_root)
    SHARDED_MSA_DB_VOLUME.commit()
    _write_json_atomic(profile_root / "manifest.json", manifest)
    SHARDED_MSA_DB_VOLUME.commit()
    _validate_published_profile(profile_root, verify_digests=False)

    result = {
        "status": "published",
        "profile_path": str(profile_root),
        "manifest_sha256": _sha256_file(profile_root / "manifest.json"),
        "source_size_bytes": source_size,
        "source_sha256": source_sha256,
        "num_seqs": statistics["num_seqs"],
        "sum_len": statistics["sum_len"],
        "maximum_residue_imbalance": statistics["maximum_residue_imbalance"],
    }
    _append_log(log_path, f"Published profile {PROFILE_ID}")
    _write_json_atomic(evidence_root / "metrics.json", result)
    BENCHMARK_OUTPUT_VOLUME.commit()
    _write_json_atomic(
        evidence_root / "done.json",
        result | {"completed_at": _utc_now(), "profile_id": PROFILE_ID},
    )
    BENCHMARK_OUTPUT_VOLUME.commit()
    return result


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=1,
    volumes={
        APP_INFO.source_db_dir: SOURCE_MSA_DB_VOLUME.with_mount_options(read_only=True),
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME,
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def prepare_small_bfd_profile(
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
) -> dict[str, object]:
    """Prepare and validate the immutable 64-shard small-BFD profile.

    Args:
        seqkit_threads: SeqKit thread count, from 1 through 32.

    Returns:
        Primitive publication status and profile provenance.
    """
    try:
        return _prepare_profile(seqkit_threads)
    except Exception as exc:
        evidence_root = Path(APP_INFO.output_dir) / APP_INFO.preparation_relpath
        evidence_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            evidence_root / "failure.json",
            {
                "failed_at": _utc_now(),
                "profile_id": PROFILE_ID,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        raise


SCAN_BUFFER_SIZE = 8 * 1024 * 1024
DONE_SCHEMA_VERSION = 1
SCAN_PASS_NAMES = ("first-pass", "immediate-repeat")


@dataclass(frozen=True)
class ScanCase:
    """One immutable Volume scan topology."""

    case_id: str
    layout: str
    containers: int
    readers_per_container: int

    def as_dict(self) -> dict[str, str | int]:
        """Return a primitive plan representation."""
        return {
            "case_id": self.case_id,
            "layout": self.layout,
            "containers": self.containers,
            "readers_per_container": self.readers_per_container,
            "aggregate_readers": self.containers * self.readers_per_container,
        }


SCAN_CASES = (
    ScanCase("V0", "monolith", 1, 1),
    ScanCase("V1", "shards", 1, 1),
    ScanCase("V2", "shards", 1, 2),
    ScanCase("V3", "shards", 1, 4),
    ScanCase("V4", "shards", 1, 8),
    ScanCase("V5", "shards", 1, 16),
    ScanCase("V6", "shards", 2, 8),
    ScanCase("V7", "shards", 4, 4),
    ScanCase("V8", "shards", 4, 16),
)


def _scan_case(case_id: str) -> ScanCase:
    """Resolve one fixed case ID, rejecting arbitrary path-like input."""
    for case in SCAN_CASES:
        if case.case_id == case_id:
            return case
    choices = ", ".join(case.case_id for case in SCAN_CASES)
    raise ValueError(f"Unknown scan case {case_id!r}; expected one of {choices}")


def _scan_case_paths(case: ScanCase) -> tuple[str, ...]:
    """Return profile-relative files read by a scan case."""
    if case.layout == "monolith":
        return (f"source/{SOURCE_DB_FILENAME}",)
    if case.layout == "shards":
        return tuple(f"shards/{name}" for name in _shard_names())
    raise ValueError(f"Unsupported scan layout: {case.layout}")


def _scan_partition_paths(case: ScanCase, partition_index: int) -> tuple[str, ...]:
    """Assign a disjoint deterministic subset of files to one container."""
    if isinstance(partition_index, bool) or not isinstance(partition_index, int):
        raise TypeError("partition_index must be an integer")
    if not 0 <= partition_index < case.containers:
        raise ValueError(
            f"partition_index must be in [0, {case.containers}), got {partition_index}"
        )
    return _scan_case_paths(case)[partition_index :: case.containers]


def _profile_artifact_map(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index validated profile artifact records by relative path."""
    source, shards = _validate_profile_manifest(manifest)
    return {str(record["path"]): record for record in [source, *shards]}


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of a small in-memory artifact."""
    return hashlib.sha256(data).hexdigest()


def _scan_partition_identity(
    manifest_sha256: str,
    manifest: dict[str, Any],
    case: ScanCase,
    partition_index: int,
) -> str:
    """Hash the result-affecting identity of one scan partition."""
    records = _profile_artifact_map(manifest)
    files = [
        {
            "path": path,
            "size_bytes": records[path]["size_bytes"],
            "sha256": records[path]["sha256"],
        }
        for path in _scan_partition_paths(case, partition_index)
    ]
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": manifest_sha256,
        "case": case.as_dict(),
        "partition_index": partition_index,
        "passes": list(SCAN_PASS_NAMES),
        "buffer_size": SCAN_BUFFER_SIZE,
        "files": files,
    }
    return _sha256_bytes(_json_bytes(identity))


def _scan_partition_relpath(case_id: str, partition_index: int) -> str:
    """Return the evidence path for one fixed scan partition."""
    case = _scan_case(case_id)
    if not 0 <= partition_index < case.containers:
        raise ValueError("partition index outside case topology")
    return (
        f"benchmarks/{CAMPAIGN_ID}/storage-scans/{case.case_id}/"
        f"partition-{partition_index:02d}"
    )


def _validate_done_marker(
    artifact_root: Path,
    *,
    expected_identity: str,
) -> dict[str, Any]:
    """Validate a local mounted completion marker and every small artifact."""
    marker_path = artifact_root / "done.json"
    _require_regular_file(marker_path)
    marker = _load_json_object(marker_path)
    if marker.get("schema_version") != DONE_SCHEMA_VERSION:
        raise ValueError("Unexpected completion marker schema")
    if marker.get("status") != "complete":
        raise ValueError("Completion marker is not complete")
    if marker.get("identity") != expected_identity:
        raise ValueError("Completion marker identity mismatch")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Completion marker has no artifacts")
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("Completion artifact record must be an object")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("Completion artifact path must be a string")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Completion artifact path escapes its root")
        artifact = (artifact_root / relative).resolve()
        if not artifact.is_relative_to(artifact_root.resolve()):
            raise ValueError("Completion artifact escapes its root")
        _require_regular_file(artifact)
        if artifact.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"Completion artifact size mismatch: {relative}")
        if _sha256_file(artifact) != record.get("sha256"):
            raise ValueError(f"Completion artifact digest mismatch: {relative}")
    return marker


def _scan_one_file(path: Path, relative_path: str) -> dict[str, str | int | float]:
    """Read one complete file and report exact bytes and elapsed time."""
    _require_regular_file(path)
    expected_bytes = path.stat().st_size
    byte_count = 0
    buffer = bytearray(SCAN_BUFFER_SIZE)
    started = perf_counter()
    with path.open("rb", buffering=0) as handle:
        while read_size := handle.readinto(buffer):
            byte_count += read_size
    elapsed = perf_counter() - started
    if byte_count != expected_bytes:
        raise OSError(
            f"Short scan for {relative_path}: read {byte_count}, expected {expected_bytes}"
        )
    return {
        "path": relative_path,
        "bytes": byte_count,
        "wall_seconds": elapsed,
        "throughput_bytes_per_second": byte_count / elapsed if elapsed else 0.0,
    }


def _scan_files(
    profile_root: Path,
    relative_paths: tuple[str, ...],
    *,
    readers: int,
) -> list[dict[str, object]]:
    """Read an assignment twice with one persistent local thread pool."""
    if readers < 1:
        raise ValueError("readers must be positive")
    absolute_paths = tuple(profile_root / path for path in relative_paths)
    pass_metrics: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=readers) as executor:
        for pass_name in SCAN_PASS_NAMES:
            started = perf_counter()
            files = list(executor.map(_scan_one_file, absolute_paths, relative_paths))
            elapsed = perf_counter() - started
            byte_count = sum(int(file["bytes"]) for file in files)
            pass_metrics.append({
                "pass": pass_name,
                "bytes": byte_count,
                "wall_seconds": elapsed,
                "throughput_bytes_per_second": (
                    byte_count / elapsed if elapsed else 0.0
                ),
                "files": files,
            })
    return pass_metrics


def _container_metadata() -> dict[str, object]:
    """Collect portable container placement and CPU-affinity evidence."""
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    load_average: list[float] | None = None
    if hasattr(os, "getloadavg"):
        load_average = list(os.getloadavg())
    return {
        "hostname": socket.gethostname(),
        "cpu_affinity": affinity,
        "cpu_count": os.cpu_count(),
        "load_average_at_finish": load_average,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "modal_region": os.environ.get("MODAL_REGION"),
    }


def _run_scan_partition(case_id: str, partition_index: int) -> dict[str, object]:
    """Execute one two-pass Volume scan assignment."""
    case = _scan_case(case_id)
    relative_paths = _scan_partition_paths(case, partition_index)
    profile_root = Path(APP_INFO.sharded_db_dir) / APP_INFO.profile_relpath
    output_root = Path(APP_INFO.output_dir) / _scan_partition_relpath(
        case_id, partition_index
    )
    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    manifest = _validate_published_profile(profile_root, verify_digests=False)
    manifest_path = profile_root / "manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    identity = _scan_partition_identity(
        manifest_sha256,
        manifest,
        case,
        partition_index,
    )
    try:
        _validate_done_marker(output_root, expected_identity=identity)
    except (FileNotFoundError, ValueError):
        pass
    else:
        metrics = _load_json_object(output_root / "metrics.json")
        return metrics | {"status": "reused"}

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"
    _append_log(
        log_path,
        f"Starting {case.case_id} partition {partition_index:02d} "
        f"with {case.readers_per_container} readers",
    )
    started = perf_counter()
    passes = _scan_files(
        profile_root,
        relative_paths,
        readers=case.readers_per_container,
    )
    sample_wall_seconds = perf_counter() - started
    records = _profile_artifact_map(manifest)
    expected_bytes = sum(int(records[path]["size_bytes"]) for path in relative_paths)
    for result in passes:
        observed_bytes = result.get("bytes")
        if isinstance(observed_bytes, bool) or not isinstance(observed_bytes, int):
            raise ValueError("A Volume scan pass has an invalid byte count")
        if observed_bytes != expected_bytes:
            raise ValueError("A Volume scan pass did not read its complete assignment")
    metadata = _container_metadata()
    metrics: dict[str, object] = {
        "status": "published",
        "campaign_id": CAMPAIGN_ID,
        "case_id": case.case_id,
        "layout": case.layout,
        "containers": case.containers,
        "readers_per_container": case.readers_per_container,
        "partition_index": partition_index,
        "partition_count": case.containers,
        "identity": identity,
        "profile_manifest_sha256": manifest_sha256,
        "relative_paths": list(relative_paths),
        "expected_bytes_per_pass": expected_bytes,
        "passes": passes,
        "sample_wall_seconds": sample_wall_seconds,
        "container": metadata,
    }
    _append_log(
        log_path,
        f"Completed {case.case_id} partition {partition_index:02d}",
    )
    metrics_path = output_root / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    BENCHMARK_OUTPUT_VOLUME.commit()
    marker = {
        "schema_version": DONE_SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "completed_at": _utc_now(),
        "artifacts": [
            _artifact_record(metrics_path, output_root),
            _artifact_record(log_path, output_root),
        ],
    }
    _write_json_atomic(output_root / "done.json", marker)
    BENCHMARK_OUTPUT_VOLUME.commit()
    return metrics


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=4,
    volumes={
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True
        ),
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def scan_volume_partition(case_id: str, partition_index: int) -> dict[str, object]:
    """Read one disjoint part of a fixed Volume scan case twice.

    Args:
        case_id: Fixed scan case ID from V0 through V8.
        partition_index: Zero-based container partition within the case.

    Returns:
        Primitive per-file, per-pass, and container measurements.
    """
    return _run_scan_partition(case_id, partition_index)


def _build_scan_plan() -> dict[str, object]:
    """Build the complete side-effect-free Volume scan plan."""
    function_inputs = sum(case.containers for case in SCAN_CASES)
    return {
        "campaign_id": CAMPAIGN_ID,
        "operation": "scan",
        "cases": [case.as_dict() for case in SCAN_CASES],
        "case_count": len(SCAN_CASES),
        "remote_function_inputs": function_inputs,
        "passes_per_input": list(SCAN_PASS_NAMES),
        "full_dataset_reads_per_case": len(SCAN_PASS_NAMES),
        "execution": "cases-sequential-partitions-concurrent",
        "resources_per_container": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "timeout_seconds": CONF.timeout,
        },
    }


def _read_volume_bytes(
    volume: modal.Volume,
    relative_path: str,
    *,
    maximum_bytes: int = 32 * 1024 * 1024,
) -> bytes:
    """Read one bounded artifact through Modal's local Volume client."""
    data = bytearray()
    for chunk in volume.read_file(relative_path):
        data.extend(chunk)
        if len(data) > maximum_bytes:
            raise ValueError(f"Volume artifact exceeds byte limit: {relative_path}")
    return bytes(data)


def _read_volume_json(
    volume: modal.Volume,
    relative_path: str,
) -> dict[str, Any]:
    """Read one bounded JSON object through the local Volume client."""
    value = orjson.loads(_read_volume_bytes(volume, relative_path))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in Volume path {relative_path}")
    return value


def _upload_volume_bytes(
    volume: modal.Volume,
    relative_path: str,
    data: bytes,
) -> None:
    """Upload one small complete artifact through the local Volume client."""
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(data), relative_path)


def _volume_artifact_record(relative_path: str, data: bytes) -> dict[str, object]:
    """Build a marker record for a client-uploaded artifact."""
    return {
        "path": relative_path,
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _client_done_marker_valid(
    volume: modal.Volume,
    marker_relative_path: str,
    *,
    expected_identity: str,
) -> bool:
    """Validate a completion marker through metadata-only client reads."""
    try:
        marker = _read_volume_json(volume, marker_relative_path)
        if marker.get("schema_version") != DONE_SCHEMA_VERSION:
            return False
        if marker.get("status") != "complete":
            return False
        if marker.get("identity") != expected_identity:
            return False
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return False
        marker_parent = PurePosixPath(marker_relative_path).parent
        for record in artifacts:
            if not isinstance(record, dict):
                return False
            relative = record.get("path")
            if not isinstance(relative, str):
                return False
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                return False
            artifact = _read_volume_bytes(volume, str(marker_parent / path))
            if len(artifact) != record.get("size_bytes"):
                return False
            if _sha256_bytes(artifact) != record.get("sha256"):
                return False
    except (FileNotFoundError, ValueError, orjson.JSONDecodeError):
        return False
    return True


def _scan_results_parquet(results: list[dict[str, Any]]) -> bytes:
    """Serialize one flat row per scan partition and pass."""
    import polars as pl

    rows: list[dict[str, object]] = []
    for result in results:
        container = result.get("container")
        if not isinstance(container, dict):
            raise ValueError("Scan result is missing container metadata")
        passes = result.get("passes")
        if not isinstance(passes, list) or len(passes) != len(SCAN_PASS_NAMES):
            raise ValueError("Scan result has an invalid pass set")
        for pass_result in passes:
            if not isinstance(pass_result, dict):
                raise ValueError("Scan pass result must be an object")
            rows.append({
                "campaign_id": CAMPAIGN_ID,
                "sample_kind": "storage-scan",
                "case_id": result["case_id"],
                "layout": result["layout"],
                "partition_index": result["partition_index"],
                "partition_count": result["partition_count"],
                "readers_per_container": result["readers_per_container"],
                "pass": pass_result["pass"],
                "bytes": pass_result["bytes"],
                "wall_seconds": pass_result["wall_seconds"],
                "throughput_bytes_per_second": pass_result[
                    "throughput_bytes_per_second"
                ],
                "sample_wall_seconds": result["sample_wall_seconds"],
                "remote_call_wall_seconds": result.get("remote_call_wall_seconds"),
                "container_hostname": container.get("hostname"),
                "result_path": result["result_path"],
            })
    buffer = io.BytesIO()
    pl.DataFrame(rows).sort(["case_id", "partition_index", "pass"]).write_parquet(
        buffer
    )
    return buffer.getvalue()


def _scan_operation_identity(manifest_sha256: str) -> str:
    """Hash the immutable campaign scan plan."""
    return _sha256_bytes(
        _json_bytes({
            "campaign_id": CAMPAIGN_ID,
            "profile_manifest_sha256": manifest_sha256,
            "scan_plan": _build_scan_plan(),
        })
    )


def _submit_scan_matrix() -> dict[str, object]:
    """Submit missing scan partitions case by case and publish their index."""
    profile_manifest_relpath = f"{APP_INFO.profile_relpath}/manifest.json"
    manifest_bytes = _read_volume_bytes(
        SHARDED_MSA_DB_VOLUME,
        profile_manifest_relpath,
    )
    manifest = orjson.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Profile manifest must be a JSON object")
    _validate_profile_manifest(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    storage_root = f"benchmarks/{CAMPAIGN_ID}/storage-scans"
    operation_identity = _scan_operation_identity(manifest_sha256)
    operation_marker_path = f"{storage_root}/done.json"
    operation_complete = _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        expected_identity=operation_identity,
    )
    partitions_complete = all(
        _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{_scan_partition_relpath(case.case_id, partition_index)}/done.json",
            expected_identity=_scan_partition_identity(
                manifest_sha256,
                manifest,
                case,
                partition_index,
            ),
        )
        for case in SCAN_CASES
        for partition_index in range(case.containers)
    )
    if operation_complete and partitions_complete:
        return {
            "status": "reused",
            "operation": "scan",
            "remote_function_inputs_submitted": 0,
            "results_path": f"{storage_root}/results.parquet",
        }

    all_results: list[dict[str, Any]] = []
    case_artifacts: list[dict[str, object]] = []
    submitted_inputs = 0
    for case in SCAN_CASES:
        expected_identities = {
            partition_index: _scan_partition_identity(
                manifest_sha256,
                manifest,
                case,
                partition_index,
            )
            for partition_index in range(case.containers)
        }
        missing_partitions = [
            partition_index
            for partition_index, identity in expected_identities.items()
            if not _client_done_marker_valid(
                BENCHMARK_OUTPUT_VOLUME,
                f"{_scan_partition_relpath(case.case_id, partition_index)}/done.json",
                expected_identity=identity,
            )
        ]
        submitted_inputs += len(missing_partitions)
        started = perf_counter()
        new_results: list[dict[str, Any]] = []
        if len(missing_partitions) == 1:
            partition_index = missing_partitions[0]
            new_results.append(
                scan_volume_partition.remote(
                    case_id=case.case_id,
                    partition_index=partition_index,
                )
            )
        elif missing_partitions:
            inputs = [
                (case.case_id, partition_index)
                for partition_index in missing_partitions
            ]
            new_results.extend(scan_volume_partition.starmap(inputs))
        remote_call_wall_seconds = perf_counter() - started

        results_by_partition = {
            int(result["partition_index"]): result for result in new_results
        }
        for partition_index in range(case.containers):
            if partition_index not in results_by_partition:
                result_path = (
                    f"{_scan_partition_relpath(case.case_id, partition_index)}/"
                    "metrics.json"
                )
                results_by_partition[partition_index] = _read_volume_json(
                    BENCHMARK_OUTPUT_VOLUME,
                    result_path,
                )
            result = results_by_partition[partition_index]
            result["remote_call_wall_seconds"] = remote_call_wall_seconds
            result["result_path"] = (
                f"{_scan_partition_relpath(case.case_id, partition_index)}/metrics.json"
            )
            all_results.append(result)

        if len(missing_partitions) == case.containers and case.containers > 1:
            hostnames = {
                str(result["container"]["hostname"])
                for result in results_by_partition.values()
            }
            if len(hostnames) != case.containers:
                raise RuntimeError(
                    f"{case.case_id} used {len(hostnames)} containers, "
                    f"expected {case.containers}"
                )

        case_summary = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "case": case.as_dict(),
            "profile_manifest_sha256": manifest_sha256,
            "partition_identities": expected_identities,
            "remote_call_wall_seconds": remote_call_wall_seconds,
            "submitted_partitions": missing_partitions,
            "completed_at": _utc_now(),
        }
        case_summary_bytes = _json_bytes(case_summary)
        case_summary_path = f"{case.case_id}/case-summary.json"
        _upload_volume_bytes(
            BENCHMARK_OUTPUT_VOLUME,
            f"{storage_root}/{case_summary_path}",
            case_summary_bytes,
        )
        case_artifacts.append(
            _volume_artifact_record(case_summary_path, case_summary_bytes)
        )

    results_bytes = _scan_results_parquet(all_results)
    results_relative_path = "results.parquet"
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{storage_root}/{results_relative_path}",
        results_bytes,
    )
    operation_marker = {
        "schema_version": DONE_SCHEMA_VERSION,
        "status": "complete",
        "identity": operation_identity,
        "completed_at": _utc_now(),
        "artifacts": [
            *case_artifacts,
            _volume_artifact_record(results_relative_path, results_bytes),
        ],
    }
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        _json_bytes(operation_marker),
    )
    return {
        "status": "published",
        "operation": "scan",
        "remote_function_inputs_submitted": submitted_inputs,
        "case_count": len(SCAN_CASES),
        "result_rows": len(all_results) * len(SCAN_PASS_NAMES),
        "results_path": f"{storage_root}/{results_relative_path}",
    }


@app.local_entrypoint()
def submit_alphafold3_msa_task(
    operation: str = "prepare",
    submit: bool = False,
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
) -> None:
    """Plan or submit one isolated AlphaFold 3 MSA benchmark operation.

    Args:
        operation: Operation to plan or run: ``prepare`` or ``scan``.
        submit: Submit the displayed remote work. Defaults to false, which only
            prints the plan and incurs no Modal compute work.
        seqkit_threads: SeqKit thread count for profile preparation, default 8.
    """
    if operation == "prepare":
        plan = _build_prepare_plan(seqkit_threads)
    elif operation == "scan":
        plan = _build_scan_plan()
    else:
        raise ValueError("operation must be 'prepare' or 'scan'")
    print(_json_bytes(plan).decode(), end="")
    if not submit:
        print("🧬 Plan only; no Modal function was submitted.")
        return
    if operation == "prepare":
        print("🧬 Submitting one small-BFD profile preparation function...")
        result = prepare_small_bfd_profile.remote(seqkit_threads=seqkit_threads)
    else:
        print("🧬 Submitting the sequential Volume scan matrix...")
        result = _submit_scan_matrix()
    print(_json_bytes(result).decode(), end="")
