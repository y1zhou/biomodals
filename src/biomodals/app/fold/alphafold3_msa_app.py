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
last. Duplicate full headers omitted by SeqKit's two-pass FASTA index are
recovered from its logged byte offsets before splitting. Benchmark evidence is
written to ``AlphaFold3-MSA-Benchmark-outputs``.
"""

from __future__ import annotations

import hashlib
import io
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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import median
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, BinaryIO

import modal
import orjson

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper

CAMPAIGN_ID = "small-bfd-phase1-v2"
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
EXPECTED_RECOVERED_RECORDS = 55_187
EXPECTED_RECOVERED_RESIDUES = 24_934_582
SEQKIT_VERSION = "2.13.0"
DEFAULT_SEQKIT_THREADS = 8
MAX_SEQKIT_THREADS = 32
MAX_PROFILE_IMBALANCE = 0.05
PROFILE_RECIPE_VERSION = 2
RECOVERED_HEADER_NAMESPACE = "__AF3_RECOVERED_"
MAX_FASTA_HEADER_BYTES = 1024 * 1024
PROFILE_VALIDATION_RELPATHS = (
    "validation/source-stats.tsv",
    "validation/shard-stats.tsv",
    "validation/shard-summary.parquet",
    "validation/source-sum.tsv",
    "validation/shard-sum.tsv",
    "validation/seqkit-sum.json",
    "validation/shuffle-stderr.log",
    "validation/duplicate-recovery.jsonl",
)
HMMER_VERSION = "3.4"
JACKHMMER_PATCH_SHA256 = (
    "df9e3ae35ad1659921d96ebfca67a9616a7a467ddde2be18a56f9bd3edb38c41"
)
JACKHMMER_BINARY_PATH = "/hmmer/bin/jackhmmer"
JACKHMMER_N_ITER = 1
JACKHMMER_E_VALUE = 1e-4
JACKHMMER_MAX_SEQUENCES = 5_000
JACKHMMER_FILTER_F1 = 5e-4
JACKHMMER_FILTER_F2 = 5e-5
JACKHMMER_FILTER_F3 = 5e-7
SCIENTIFIC_COMPARISON_POLICY = "top-target-order-exact-modulo-evalue-bit-score-ties-v2"
RESOURCE_TRACE_INTERVAL_SECONDS = 1.0
MODAL_CPU_USD_PER_CORE_SECOND = 0.0000131
MODAL_MEMORY_USD_PER_GIB_SECOND = 0.00000222
MODAL_PRICING_OBSERVED_DATE = "2026-07-22"
MODAL_PRICING_URL = "https://modal.com/pricing"
JSON_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
JSONL_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
_FAI_DUPLICATE_WARNING = re.compile(
    rb'^\[fai warning\] ignoring duplicate sequence "(?P<name>.*)" '
    rb"at byte offset (?P<offset>[0-9]+)\r?\n?$"
)


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


@dataclass(frozen=True)
class FaiDuplicateWarning:
    """One record omitted by SeqKit's full-header FASTA index."""

    sequence_name: bytes
    sequence_offset: int


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
    .micromamba(python_version=CONF.python_version)
    .apt_install(
        "git",
        "build-essential",
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
    .micromamba_install(
        f"seqkit={SEQKIT_VERSION}",
        channels=["conda-forge", "bioconda"],
    )
    .run_commands("seqkit version")
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

_CONTAINER_INSTANCE_ID = uuid.uuid4().hex
_CONTAINER_SAMPLE_COUNT = 0
_CONTAINER_SAMPLE_LOCK = Lock()


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


def _duplicate_recovery_recipe() -> dict[str, object]:
    """Return the scientific recipe for restoring FAI-omitted records."""
    return {
        "warning_source": "seqkit-fai-sequence-byte-offset",
        "expected_records": EXPECTED_RECOVERED_RECORDS,
        "temporary_header_identity": "unique-uuid",
        "append_after_shuffle": True,
        "strip_after_split": True,
    }


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


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 16 * 1024 * 1024,
    forbidden_bytes: bytes | None = None,
) -> str:
    """Compute a digest and optionally reject a byte marker while streaming."""
    _require_regular_file(path)
    if forbidden_bytes == b"":
        raise ValueError("forbidden_bytes must be nonempty")
    digest = hashlib.sha256()
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            if forbidden_bytes is not None:
                searchable = overlap + chunk
                if forbidden_bytes in searchable:
                    raise ValueError(f"Forbidden byte marker remains in {path}")
                overlap_size = len(forbidden_bytes) - 1
                overlap = searchable[-overlap_size:] if overlap_size else b""
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


def _append_diagnostic_file(source_path: Path, log_path: Path) -> None:
    """Copy one command's raw diagnostics into the durable operation log."""
    _require_regular_file(source_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source, log_path.open("ab") as log:
        shutil.copyfileobj(source, log, length=1024 * 1024)


def _parse_fai_duplicate_warnings(
    diagnostics_path: Path,
) -> tuple[FaiDuplicateWarning, ...]:
    """Parse SeqKit FAI duplicate warnings as ordered sequence offsets."""
    _require_regular_file(diagnostics_path)
    warnings: list[FaiDuplicateWarning] = []
    with diagnostics_path.open("rb") as diagnostics:
        for line_number, line in enumerate(diagnostics, start=1):
            if b"[fai warning]" not in line:
                continue
            match = _FAI_DUPLICATE_WARNING.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"Malformed SeqKit FAI warning at {diagnostics_path}:{line_number}"
                )
            sequence_name = match.group("name")
            sequence_offset = int(match.group("offset"))
            if not sequence_name:
                raise ValueError(
                    f"Empty sequence name in FAI warning at line {line_number}"
                )
            if sequence_offset <= 0:
                raise ValueError(
                    f"Invalid FAI sequence offset at line {line_number}: "
                    f"{sequence_offset}"
                )
            if warnings and sequence_offset <= warnings[-1].sequence_offset:
                raise ValueError(
                    "SeqKit FAI warning offsets must be strictly increasing"
                )
            warnings.append(FaiDuplicateWarning(sequence_name, sequence_offset))
    return tuple(warnings)


def _read_fasta_header_before_sequence_offset(
    source: BinaryIO,
    sequence_offset: int,
) -> bytes:
    """Read the exact FASTA header preceding an FAI sequence-start offset."""
    file_size = os.fstat(source.fileno()).st_size
    if not 1 < sequence_offset <= file_size:
        raise ValueError(
            f"FAI sequence offset {sequence_offset} is outside the source file"
        )
    source.seek(sequence_offset - 1)
    if source.read(1) != b"\n":
        raise ValueError(
            f"FAI sequence offset {sequence_offset} does not follow a header line"
        )

    cursor = sequence_offset - 1
    header_chunks: list[bytes] = []
    scanned_bytes = 0
    header_line: bytes | None = None
    while cursor > 0 and scanned_bytes < MAX_FASTA_HEADER_BYTES:
        read_size = min(4096, cursor, MAX_FASTA_HEADER_BYTES - scanned_bytes)
        chunk_start = cursor - read_size
        source.seek(chunk_start)
        chunk = source.read(read_size)
        previous_newline = chunk.rfind(b"\n")
        if previous_newline >= 0:
            header_chunks.append(chunk[previous_newline + 1 :])
            header_line = b"".join(reversed(header_chunks))
            break
        header_chunks.append(chunk)
        scanned_bytes += read_size
        cursor = chunk_start
    if header_line is None and cursor == 0:
        header_line = b"".join(reversed(header_chunks))
    if header_line is None:
        raise ValueError(
            f"FASTA header before offset {sequence_offset} exceeds "
            f"{MAX_FASTA_HEADER_BYTES} bytes"
        )
    if header_line.endswith(b"\r"):
        header_line = header_line[:-1]
    if not header_line.startswith(b">") or len(header_line) == 1:
        raise ValueError(
            f"FAI sequence offset {sequence_offset} has no valid preceding header"
        )
    return header_line[1:]


def _recovery_header_pattern(temporary_namespace: str) -> str:
    """Return the anchored SeqKit regex for one generation's UUID prefixes."""
    expected = rf"{RECOVERED_HEADER_NAMESPACE}[0-9a-f]{{32}}_"
    if re.fullmatch(expected, temporary_namespace) is None:
        raise ValueError("Invalid temporary recovery namespace")
    return rf"^{temporary_namespace}[0-9a-f]{{32}}__"


def _append_recovered_fasta_records(
    source_path: Path,
    shuffled_path: Path,
    warnings: tuple[FaiDuplicateWarning, ...],
    report_path: Path,
    *,
    temporary_namespace: str,
) -> dict[str, int]:
    """Recover FAI-omitted records and append UUID-prefixed FASTA entries."""
    _require_regular_file(source_path)
    _require_regular_file(shuffled_path)
    _recovery_header_pattern(temporary_namespace)
    if not warnings:
        raise ValueError("SeqKit emitted no duplicate-record byte offsets")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    recovered_residues = 0
    temporary_uuids: set[str] = set()
    previous_offset = 0
    namespace_bytes = temporary_namespace.encode("ascii")
    with (
        source_path.open("rb") as source,
        shuffled_path.open("r+b") as shuffled,
        report_path.open("xb") as report,
    ):
        shuffled.seek(0, os.SEEK_END)
        shuffled_size = shuffled.tell()
        if shuffled_size > 0:
            shuffled.seek(-1, os.SEEK_END)
            if shuffled.read(1) != b"\n":
                shuffled.write(b"\n")
        shuffled.seek(0, os.SEEK_END)

        for warning in warnings:
            if warning.sequence_offset <= previous_offset:
                raise ValueError(
                    "SeqKit FAI warning offsets must be strictly increasing"
                )
            previous_offset = warning.sequence_offset
            original_header = _read_fasta_header_before_sequence_offset(
                source,
                warning.sequence_offset,
            )
            normalized_header = re.sub(rb"\t+", b" ", original_header)
            if normalized_header != warning.sequence_name:
                raise ValueError(
                    "FAI warning name does not match source header at byte offset "
                    f"{warning.sequence_offset}"
                )

            record_uuid = uuid.uuid4().hex
            if not re.fullmatch(r"[0-9a-f]{32}", record_uuid):
                raise RuntimeError(
                    "UUID generator returned an invalid hexadecimal UUID"
                )
            if record_uuid in temporary_uuids:
                raise RuntimeError("UUID generator returned a duplicate recovery UUID")
            temporary_uuids.add(record_uuid)
            temporary_prefix = namespace_bytes + record_uuid.encode("ascii") + b"__"
            shuffled.write(b">" + temporary_prefix + original_header + b"\n")

            sequence_digest = hashlib.sha256()
            sequence_length = 0
            sequence_ends_with_newline = True
            source.seek(warning.sequence_offset)
            while line := source.readline():
                if line.startswith(b">"):
                    break
                shuffled.write(line)
                sequence_bases = line.rstrip(b"\r\n")
                sequence_digest.update(sequence_bases)
                sequence_length += len(sequence_bases)
                sequence_ends_with_newline = line.endswith(b"\n")
            if not sequence_ends_with_newline:
                shuffled.write(b"\n")
            recovered_residues += sequence_length
            report.write(
                orjson.dumps(
                    {
                        "byte_offset": warning.sequence_offset,
                        "header_sha256": hashlib.sha256(original_header).hexdigest(),
                        "sequence_length": sequence_length,
                        "sequence_sha256": sequence_digest.hexdigest(),
                        "temporary_uuid": record_uuid,
                    },
                    option=JSONL_OPTIONS,
                )
            )

        shuffled.flush()
        os.fsync(shuffled.fileno())
        report.flush()
        os.fsync(report.fileno())

    return {
        "recovered_records": len(warnings),
        "recovered_residues": recovered_residues,
        "first_byte_offset": warnings[0].sequence_offset,
        "last_byte_offset": warnings[-1].sequence_offset,
    }


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
    shard_dir: Path,
    validation_dir: Path,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> dict[str, int | str]:
    """Shuffle, recover FAI-omitted records, split, and restore headers."""
    seqkit = _require_executable("seqkit")
    shuffled_path = raw_shard_dir.parent / ".shuffled.fasta"
    shuffle_diagnostics_path = validation_dir / "shuffle-stderr.log"
    recovery_report_path = validation_dir / "duplicate-recovery.jsonl"
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
    _append_log(log_path, f"Running command: {shlex.join(shuffle_argv)}")
    validation_dir.mkdir(parents=True, exist_ok=True)
    with (
        shuffled_path.open("xb") as shuffled,
        shuffle_diagnostics_path.open("xb") as diagnostics,
    ):
        shuffle_process = subprocess.run(  # noqa: S603
            shuffle_argv,
            check=False,
            stdout=shuffled,
            stderr=diagnostics,
        )
    _append_diagnostic_file(shuffle_diagnostics_path, log_path)
    if shuffle_process.returncode != 0:
        raise subprocess.CalledProcessError(shuffle_process.returncode, shuffle_argv)

    warnings = _parse_fai_duplicate_warnings(shuffle_diagnostics_path)
    if len(warnings) != EXPECTED_RECOVERED_RECORDS:
        raise ValueError(
            "Unexpected number of SeqKit FAI duplicate warnings: "
            f"{len(warnings)} != {EXPECTED_RECOVERED_RECORDS}"
        )
    temporary_namespace = f"{RECOVERED_HEADER_NAMESPACE}{uuid.uuid4().hex}_"
    recovery_metrics = _append_recovered_fasta_records(
        source_path,
        shuffled_path,
        warnings,
        recovery_report_path,
        temporary_namespace=temporary_namespace,
    )
    if recovery_metrics["recovered_residues"] != EXPECTED_RECOVERED_RESIDUES:
        raise ValueError(
            "Recovered duplicate residue count does not match the failed-run "
            f"deficit: {recovery_metrics['recovered_residues']} != "
            f"{EXPECTED_RECOVERED_RESIDUES}"
        )
    _append_log(
        log_path,
        "Recovered "
        f"{recovery_metrics['recovered_records']} FAI-omitted records "
        f"({recovery_metrics['recovered_residues']} residues) from byte offsets "
        f"{recovery_metrics['first_byte_offset']} through "
        f"{recovery_metrics['last_byte_offset']}",
    )

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
        str(shuffled_path),
    ]
    _append_log(log_path, f"Running command: {shlex.join(split_argv)}")
    raw_shard_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        split_process = subprocess.run(  # noqa: S603
            split_argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
    if split_process.returncode != 0:
        raise subprocess.CalledProcessError(split_process.returncode, split_argv)

    raw_shards = sorted(
        path
        for path in raw_shard_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if len(raw_shards) != SHARD_COUNT:
        raise ValueError(f"Expected {SHARD_COUNT} raw shards, found {len(raw_shards)}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    recovery_pattern = _recovery_header_pattern(temporary_namespace)
    for index, raw_shard in enumerate(raw_shards):
        if raw_shard.stat().st_size <= 0:
            raise ValueError(f"SeqKit produced empty shard: {raw_shard}")
        _run_to_file(
            [
                seqkit,
                "replace",
                "-j",
                str(seqkit_threads),
                "--pattern",
                recovery_pattern,
                "--replacement",
                "",
                str(raw_shard),
            ],
            shard_dir / _shard_filename(index),
            log_path,
        )

    for raw_shard in raw_shards:
        raw_shard.unlink()
    raw_shard_dir.rmdir()
    shuffled_path.unlink()
    Path(f"{source_path}.seqkit.fai").unlink(missing_ok=True)
    return recovery_metrics | {
        "temporary_namespace": temporary_namespace,
        "temporary_header_pattern": recovery_pattern,
    }


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


def _artifact_record(
    path: Path,
    profile_root: Path,
    *,
    forbidden_bytes: bytes | None = None,
) -> dict[str, str | int]:
    """Build one manifest artifact record below the profile root."""
    _require_regular_file(path)
    resolved_root = profile_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Artifact escapes profile root: {path}")
    return {
        "path": resolved_path.relative_to(resolved_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path, forbidden_bytes=forbidden_bytes),
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
    source, shards, validation_artifacts = _validate_profile_manifest(manifest)

    records = [source, *shards, *validation_artifacts]
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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    recipe = manifest.get("recipe")
    validation = manifest.get("validation")
    if not isinstance(source, dict):
        raise ValueError("Profile manifest source must be an object")
    if source.get("path") != f"source/{SOURCE_DB_FILENAME}":
        raise ValueError("Profile manifest source path is invalid")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise ValueError(f"Profile manifest must declare {SHARD_COUNT} shards")
    if not isinstance(recipe, dict):
        raise ValueError("Profile manifest recipe must be an object")
    if recipe.get("version") != PROFILE_RECIPE_VERSION:
        raise ValueError("Unexpected profile recipe version")
    if recipe.get("seqkit_version") != SEQKIT_VERSION:
        raise ValueError("Unexpected profile SeqKit version")
    try:
        _validate_seqkit_threads(recipe.get("seqkit_threads"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid profile SeqKit thread count") from exc
    if recipe.get("random_seed") != SHARD_RANDOM_SEED:
        raise ValueError("Unexpected profile shuffle seed")
    if recipe.get("shuffle") != ["--two-pass", "--update-faidx"]:
        raise ValueError("Unexpected profile shuffle recipe")
    if recipe.get("duplicate_recovery") != _duplicate_recovery_recipe():
        raise ValueError("Unexpected profile duplicate-recovery recipe")
    if recipe.get("split") != ["--by-part", SHARD_COUNT]:
        raise ValueError("Unexpected profile split recipe")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Profile manifest does not declare passed validation")
    if validation.get("recovered_records") != EXPECTED_RECOVERED_RECORDS:
        raise ValueError("Unexpected recovered duplicate-record count")
    if validation.get("recovered_residues") != EXPECTED_RECOVERED_RESIDUES:
        raise ValueError("Unexpected recovered duplicate-residue count")
    if validation.get("temporary_recovery_prefix_absent") is not True:
        raise ValueError("Profile may retain temporary recovery prefixes")
    validation_artifacts = validation.get("artifacts")
    if not isinstance(validation_artifacts, list):
        raise ValueError("Profile manifest validation artifacts must be a list")

    expected_paths = [f"shards/{name}" for name in _shard_names()]
    actual_paths: list[str] = []
    records = [source, *shards, *validation_artifacts]
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
    actual_validation_paths = [str(record["path"]) for record in validation_artifacts]
    if actual_validation_paths != list(PROFILE_VALIDATION_RELPATHS):
        raise ValueError("Profile manifest validation artifact paths are invalid")
    return source, shards, validation_artifacts


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
            "duplicate_recovery": _duplicate_recovery_recipe(),
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
    _ensure_campaign_plan_mounted()
    BENCHMARK_OUTPUT_VOLUME.commit()
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

    recovery_metrics = _run_shuffle_split(
        copied_source,
        raw_shard_dir,
        shard_dir,
        validation_dir,
        log_path,
        seqkit_threads=threads,
    )
    shuffle_log_path = validation_dir / "shuffle-stderr.log"
    recovery_report_path = validation_dir / "duplicate-recovery.jsonl"
    evidence_shuffle_path = evidence_root / f"{generation_id}-shuffle-stderr.log"
    evidence_recovery_path = evidence_root / f"{generation_id}-duplicate-recovery.jsonl"
    shuffle_log_sha256, shuffle_log_size = _copy_file_with_sha256(
        shuffle_log_path,
        evidence_shuffle_path,
    )
    recovery_report_sha256, recovery_report_size = _copy_file_with_sha256(
        recovery_report_path,
        evidence_recovery_path,
    )
    _write_json_atomic(
        evidence_root / "recovery.json",
        recovery_metrics
        | {
            "generation_id": generation_id,
            "shuffle_diagnostics": {
                "path": evidence_shuffle_path.name,
                "sha256": shuffle_log_sha256,
                "size_bytes": shuffle_log_size,
            },
            "recovery_report": {
                "path": evidence_recovery_path.name,
                "sha256": recovery_report_sha256,
                "size_bytes": recovery_report_size,
            },
        },
    )
    BENCHMARK_OUTPUT_VOLUME.commit()
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
    seqkit_sum_report_path = validation_dir / "seqkit-sum.json"
    _write_json_atomic(
        seqkit_sum_report_path,
        {
            "source": source_sum,
            "aggregate_shards": shard_sum,
            "match": True,
            "seqkit_version": SEQKIT_VERSION,
        },
    )

    source_record = _artifact_record(copied_source, staging_root)
    temporary_namespace = recovery_metrics.get("temporary_namespace")
    if not isinstance(temporary_namespace, str):
        raise RuntimeError("Duplicate recovery did not return its temporary namespace")
    forbidden_recovery_header = b">" + temporary_namespace.encode("ascii")
    shard_records = [
        _artifact_record(
            shard_path,
            staging_root,
            forbidden_bytes=forbidden_recovery_header,
        )
        for shard_path in shard_paths
    ]
    validation_records = [
        _artifact_record(staging_root / relative_path, staging_root)
        for relative_path in PROFILE_VALIDATION_RELPATHS
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
            "version": PROFILE_RECIPE_VERSION,
            "seqkit_version": SEQKIT_VERSION,
            "seqkit_threads": threads,
            "random_seed": SHARD_RANDOM_SEED,
            "shuffle": ["--two-pass", "--update-faidx"],
            "duplicate_recovery": _duplicate_recovery_recipe(),
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
            "recovered_records": recovery_metrics["recovered_records"],
            "recovered_residues": recovery_metrics["recovered_residues"],
            "first_recovered_byte_offset": recovery_metrics["first_byte_offset"],
            "last_recovered_byte_offset": recovery_metrics["last_byte_offset"],
            "temporary_recovery_prefix_absent": True,
            "artifacts": validation_records,
        },
    }

    _write_json_atomic(staging_root / "manifest.json", manifest)
    SHARDED_MSA_DB_VOLUME.commit()
    _validate_published_profile(staging_root, verify_digests=False)

    publication_status = "published"
    profile_root.parent.mkdir(parents=True, exist_ok=True)
    if profile_root.exists():
        try:
            existing_manifest = _validate_published_profile(
                profile_root,
                verify_digests=False,
            )
        except (FileNotFoundError, ValueError):
            orphan_root = sharded_root / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            profile_root.replace(orphan_root / f"{PROFILE_ID}-{uuid.uuid4().hex}")
        else:
            if _profile_scientific_identity(existing_manifest) != (
                _profile_scientific_identity(manifest)
            ):
                raise RuntimeError(
                    "A different valid profile was published concurrently"
                )
            duplicate_root = sharded_root / ".orphaned"
            duplicate_root.mkdir(parents=True, exist_ok=True)
            staging_root.replace(
                duplicate_root / f"{PROFILE_ID}-duplicate-{generation_id}"
            )
            publication_status = "reused-concurrent"
    if publication_status == "published":
        staging_root.replace(profile_root)
    SHARDED_MSA_DB_VOLUME.commit()
    _validate_published_profile(profile_root, verify_digests=False)

    result = {
        "status": publication_status,
        "profile_path": str(profile_root),
        "manifest_sha256": _sha256_file(profile_root / "manifest.json"),
        "source_size_bytes": source_size,
        "source_sha256": source_sha256,
        "num_seqs": statistics["num_seqs"],
        "sum_len": statistics["sum_len"],
        "maximum_residue_imbalance": statistics["maximum_residue_imbalance"],
        "recovered_records": recovery_metrics["recovered_records"],
        "recovered_residues": recovery_metrics["recovered_residues"],
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
    source, shards, _ = _validate_profile_manifest(manifest)
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
    started_at = _utc_now()
    started = perf_counter()
    with path.open("rb", buffering=0) as handle:
        while read_size := handle.readinto(buffer):
            byte_count += read_size
    elapsed = perf_counter() - started
    finished_at = _utc_now()
    if byte_count != expected_bytes:
        raise OSError(
            f"Short scan for {relative_path}: read {byte_count}, expected {expected_bytes}"
        )
    return {
        "path": relative_path,
        "started_at": started_at,
        "finished_at": finished_at,
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


def _container_sample_metadata() -> dict[str, object]:
    """Record whether this interpreter already ran a benchmark sample."""
    global _CONTAINER_SAMPLE_COUNT  # noqa: PLW0603

    with _CONTAINER_SAMPLE_LOCK:
        _CONTAINER_SAMPLE_COUNT += 1
        sample_ordinal = _CONTAINER_SAMPLE_COUNT
    return _container_metadata() | {
        "container_instance_id": _CONTAINER_INSTANCE_ID,
        "container_sample_ordinal": sample_ordinal,
        "container_reused_for_sample": sample_ordinal > 1,
    }


def _run_scan_partition(case_id: str, partition_index: int) -> dict[str, object]:
    """Execute one two-pass Volume scan assignment."""
    case = _scan_case(case_id)
    relative_paths = _scan_partition_paths(case, partition_index)
    profile_root = Path(APP_INFO.sharded_db_dir) / APP_INFO.profile_relpath
    final_output_root = Path(APP_INFO.output_dir) / _scan_partition_relpath(
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
        _validate_done_marker(final_output_root, expected_identity=identity)
    except (FileNotFoundError, ValueError):
        pass
    else:
        metrics = _load_json_object(final_output_root / "metrics.json")
        return metrics | {"status": "reused"}

    output_root = (
        final_output_root.parent
        / ".staging"
        / f"partition-{partition_index:02d}-{uuid.uuid4().hex}"
    )
    output_root.mkdir(parents=True)
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
    if final_output_root.exists():
        try:
            _validate_done_marker(final_output_root, expected_identity=identity)
        except (FileNotFoundError, ValueError):
            orphan_root = final_output_root.parent / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            final_output_root.replace(
                orphan_root / f"partition-{partition_index:02d}-{uuid.uuid4().hex}"
            )
        else:
            duplicate_root = final_output_root.parent / ".orphaned"
            duplicate_root.mkdir(parents=True, exist_ok=True)
            output_root.replace(
                duplicate_root
                / f"partition-{partition_index:02d}-duplicate-{uuid.uuid4().hex}"
            )
            BENCHMARK_OUTPUT_VOLUME.commit()
            existing = _load_json_object(final_output_root / "metrics.json")
            return existing | {"status": "reused-concurrent"}
    output_root.replace(final_output_root)
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
    """Validate a completion marker and every artifact through client reads."""
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
            size_bytes = record.get("size_bytes")
            digest = record.get("sha256")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
                return False
            if not isinstance(digest, str) or len(digest) != 64:
                return False
            artifact_path = (marker_parent / path).as_posix().lstrip("/")
            artifact = _read_volume_bytes(volume, artifact_path)
            if len(artifact) != size_bytes:
                return False
            if _sha256_bytes(artifact) != digest:
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


def _scan_case_pass_summaries(
    results: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Aggregate concurrent partition throughput for each scan pass."""
    summaries: list[dict[str, object]] = []
    for pass_name in SCAN_PASS_NAMES:
        partition_passes: list[dict[str, Any]] = []
        for result in results:
            passes = result.get("passes")
            if not isinstance(passes, list):
                raise ValueError("Scan result has no pass list")
            matches = [
                item
                for item in passes
                if isinstance(item, dict) and item.get("pass") == pass_name
            ]
            if len(matches) != 1:
                raise ValueError(f"Scan result has invalid {pass_name} evidence")
            partition_passes.append(matches[0])
        byte_count = sum(int(item["bytes"]) for item in partition_passes)
        critical_wall_seconds = max(
            float(item["wall_seconds"]) for item in partition_passes
        )
        summaries.append({
            "pass": pass_name,
            "bytes": byte_count,
            "critical_wall_seconds": critical_wall_seconds,
            "aggregate_throughput_bytes_per_second": (
                byte_count / critical_wall_seconds if critical_wall_seconds else 0.0
            ),
        })
    return summaries


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
    _ensure_campaign_plan_client()
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
        _publish_campaign_progress(
            stage="storage scan",
            status="scan complete; search benchmarks pending",
            details=[
                f"The complete V0-V8 scan index is available under `{storage_root}/`.",
                "No search benchmark result is included yet.",
            ],
        )
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
            task_ids = {
                result["container"].get("modal_task_id")
                for result in results_by_partition.values()
            }
            if (
                not all(isinstance(task_id, str) and task_id for task_id in task_ids)
                or len(task_ids) != case.containers
            ):
                raise RuntimeError(
                    f"{case.case_id} used {len(task_ids)} Modal task IDs, "
                    f"expected {case.containers}"
                )

        case_summary = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "case": case.as_dict(),
            "profile_manifest_sha256": manifest_sha256,
            "partition_identities": {
                str(partition_index): identity
                for partition_index, identity in expected_identities.items()
            },
            "pass_summaries": _scan_case_pass_summaries(
                list(results_by_partition.values())
            ),
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
    _publish_campaign_progress(
        stage="storage scan",
        status="scan complete; search benchmarks pending",
        details=[
            f"Completed all {len(SCAN_CASES)} V0-V8 scan cases.",
            f"Submitted {submitted_inputs} remote partition inputs.",
        ],
    )
    return {
        "status": "published",
        "operation": "scan",
        "remote_function_inputs_submitted": submitted_inputs,
        "case_count": len(SCAN_CASES),
        "result_rows": len(all_results) * len(SCAN_PASS_NAMES),
        "results_path": f"{storage_root}/{results_relative_path}",
    }


PEMBROLIZUMAB_VH_SEQUENCE = (
    "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRV"
    "TLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
)
PEMBROLIZUMAB_VH_SHA256 = (
    "5d92fab232244fa55131fc3b8d31b34990aa778623cdd906d58cf920dbdaf28f"
)
ECOLI_K12_GROEL_SEQUENCE = (
    "MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMG"
    "AQMVKEVASKANDAAGDGTTTATVLAQAIITEGLKAVAAGMNPMDLKRGIDKAVTAAVEELKALSVPCSD"
    "SKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLQDELDVVEGMQFDRGYLSPYFINKPET"
    "GAVELESPFILLADKKISNIREMLPVLEAVAKAGKPLLIIAEDVEGEALATLVVNTMRGIVKVAAVKAPG"
    "FGDRRKAMLQDIATLTGGTVISEEIGMELEKATLEDLGQAKRVVINKDTTTIIDGVGEEAAIQGRVAQIR"
    "QQIEEATSDYDREKLQERVAKLAGGVAVIKVGAATEVEMKEKKARVEDALHATRAAVEEGVVAGGGVALI"
    "RVASKLADLRGQNEDQNVGIKVALRAMEAPLRQIVLNCGEEPSVVANTVKGGDGNYGYNAATEEYGNMID"
    "MGILDPTKVTRSALQYAASVAGLMITTECMVTDLPKNDAADLGAAGGMGGMGGMGGMM"
)
ECOLI_K12_GROEL_SHA256 = (
    "40544c6fee0f15b6fe78d6ab7e5e27d8080224fe28dc0d6ca6f2e9a790dd24d4"
)
SMOKE_CASE_IDS = ("B0", "B1", "S3")
MATRIX_CASE_IDS = ("B0", "B1", "S0", "S1", "S2", "S3", "S4", "S5")
FOCUSED_SWEEP_REUSED_CASE_IDS = ("B1", "S3")
FOCUSED_SWEEP_NEW_CASE_IDS = ("S1", "S2", "S4", "S5", "S6")
FOCUSED_SWEEP_CASE_IDS = ("B1", "S1", "S2", "S3", "S4", "S5", "S6")
SCREENING_BLOCK_ORDERS = (
    ("B1", "S3", "S0", "B0", "S5", "S2", "S4", "S1"),
    ("S3", "S0", "B1", "S5", "S2", "B0", "S4", "S1"),
    ("S1", "S4", "B0", "S2", "S5", "B1", "S3", "S0"),
)


@dataclass(frozen=True)
class SearchQuery:
    """One immutable benchmark query sequence."""

    query_id: str
    role: str
    sequence: str
    sequence_sha256: str

    def as_dict(self) -> dict[str, str | int]:
        """Return query provenance without duplicating the full sequence."""
        return {
            "query_id": self.query_id,
            "role": self.role,
            "length": len(self.sequence),
            "sequence_sha256": self.sequence_sha256,
        }


@dataclass(frozen=True)
class SearchCase:
    """One scientific layout and operational Jackhmmer topology."""

    case_id: str
    layout: str
    jackhmmer_n_cpu: int
    active_shards: int
    z_value: int | None

    def as_dict(self) -> dict[str, str | int | None]:
        """Return a primitive plan representation."""
        return {
            "case_id": self.case_id,
            "layout": self.layout,
            "jackhmmer_n_cpu": self.jackhmmer_n_cpu,
            "active_shards": self.active_shards,
            "aggregate_cpu_slots": self.jackhmmer_n_cpu * self.active_shards,
            "z_value": self.z_value,
            "dom_z_value": self.z_value,
        }


SCREENING_QUERY = SearchQuery(
    query_id="pembrolizumab-vh",
    role="screening",
    sequence=PEMBROLIZUMAB_VH_SEQUENCE,
    sequence_sha256=PEMBROLIZUMAB_VH_SHA256,
)
STRESS_QUERY = SearchQuery(
    query_id="ecoli-k12-groel",
    role="stress",
    sequence=ECOLI_K12_GROEL_SEQUENCE,
    sequence_sha256=ECOLI_K12_GROEL_SHA256,
)
SEARCH_CASES = (
    SearchCase("B0", "monolith", 8, 1, None),
    SearchCase("B1", "monolith", 8, 1, SMALL_BFD_Z),
    SearchCase("S0", "shards", 8, 1, SMALL_BFD_Z),
    SearchCase("S1", "shards", 2, 4, SMALL_BFD_Z),
    SearchCase("S2", "shards", 2, 8, SMALL_BFD_Z),
    SearchCase("S3", "shards", 2, 16, SMALL_BFD_Z),
    SearchCase("S4", "shards", 4, 8, SMALL_BFD_Z),
    SearchCase("S5", "shards", 8, 4, SMALL_BFD_Z),
    SearchCase("S6", "shards", 1, 32, SMALL_BFD_Z),
)


def _search_case(case_id: str) -> SearchCase:
    """Resolve one fixed benchmark search case."""
    for case in SEARCH_CASES:
        if case.case_id == case_id:
            return case
    choices = ", ".join(case.case_id for case in SEARCH_CASES)
    raise ValueError(f"Unknown search case {case_id!r}; expected one of {choices}")


def _search_query(query_id: str) -> SearchQuery:
    """Resolve one fixed query and validate its embedded digest."""
    queries = (SCREENING_QUERY, STRESS_QUERY)
    for query in queries:
        if query.query_id == query_id:
            measured_sha256 = _sha256_bytes(query.sequence.encode())
            if measured_sha256 != query.sequence_sha256:
                raise RuntimeError(f"Embedded {query.query_id} SHA-256 is invalid")
            return query
    choices = ", ".join(query.query_id for query in queries)
    raise ValueError(f"Unknown search query {query_id!r}; expected one of {choices}")


def _focused_sweep_sample_id(case_id: str) -> str:
    """Return the fixed reused or new sample ID for one sweep case."""
    _search_case(case_id)
    if case_id in FOCUSED_SWEEP_REUSED_CASE_IDS:
        return f"smoke-{case_id.lower()}"
    if case_id in FOCUSED_SWEEP_NEW_CASE_IDS:
        return f"sweep-{case_id.lower()}"
    choices = ", ".join(FOCUSED_SWEEP_CASE_IDS)
    raise ValueError(f"Focused sweep case must be one of {choices}")


def _validate_sample_id(sample_id: str) -> str:
    """Reject sample identifiers that cannot be safe path components."""
    if not isinstance(sample_id, str) or not 1 <= len(sample_id) <= 64:
        raise ValueError("sample_id must contain between 1 and 64 characters")
    allowed = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
    if sample_id[0] == "-" or any(character not in allowed for character in sample_id):
        raise ValueError("sample_id must use lowercase letters, digits, and hyphens")
    return sample_id


def _scientific_search_config(case: SearchCase) -> dict[str, object]:
    """Return only result-affecting Jackhmmer configuration."""
    return {
        "alphafold_commit": CONF.repo_commit_hash,
        "hmmer_version": HMMER_VERSION,
        "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
        "database_layout": case.layout,
        "n_iter": JACKHMMER_N_ITER,
        "e_value": JACKHMMER_E_VALUE,
        "z_value": case.z_value,
        "dom_z_value": case.z_value,
        "max_sequences": JACKHMMER_MAX_SEQUENCES,
        "filter_f1": JACKHMMER_FILTER_F1,
        "filter_f2": JACKHMMER_FILTER_F2,
        "filter_f3": JACKHMMER_FILTER_F3,
        "seq_limit_patch": True,
    }


def _profile_scientific_identity(manifest: dict[str, Any]) -> str:
    """Hash profile content and recipe while excluding operational thread count."""
    source, shards, _ = _validate_profile_manifest(manifest)
    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Profile manifest recipe must be an object")
    return _sha256_bytes(
        _json_bytes({
            "schema_version": manifest["schema_version"],
            "profile_id": manifest["profile_id"],
            "database_id": manifest["database_id"],
            "source": source,
            "shards": shards,
            "shard_count": manifest["shard_count"],
            "z_value": manifest["z_value"],
            "recipe": {
                "version": recipe.get("version"),
                "seqkit_version": recipe.get("seqkit_version"),
                "random_seed": recipe.get("random_seed"),
                "shuffle": recipe.get("shuffle"),
                "duplicate_recovery": recipe.get("duplicate_recovery"),
                "split": recipe.get("split"),
            },
        })
    )


def _search_identity(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
) -> str:
    """Hash query, database, and result-affecting settings only."""
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "database_id": DATABASE_ID,
            "profile_id": PROFILE_ID,
            "profile_scientific_identity": profile_scientific_identity,
            "query": query.as_dict(),
            "scientific_config": _scientific_search_config(case),
        })
    )


def _search_sample_identity(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
    sample_id: str,
) -> str:
    """Hash both scientific inputs and operational sample settings."""
    validated_sample_id = _validate_sample_id(sample_id)
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "query": query.as_dict(),
            "search_identity": _search_identity(
                profile_scientific_identity,
                query,
                case,
            ),
            "sample_id": validated_sample_id,
            "operational_config": {
                "case_id": case.case_id,
                "jackhmmer_n_cpu": case.jackhmmer_n_cpu,
                "active_shards": case.active_shards,
                "resource_trace_interval_seconds": RESOURCE_TRACE_INTERVAL_SECONDS,
            },
        })
    )


def _search_sample_relpath(
    query: SearchQuery,
    search_identity: str,
    sample_id: str,
) -> str:
    """Return the common sequence-addressed raw-MSA sample path."""
    validated_sample_id = _validate_sample_id(sample_id)
    if len(search_identity) != 64:
        raise ValueError("search_identity must be a SHA-256 digest")
    sequence_sha256 = query.sequence_sha256
    return (
        f"{sequence_sha256[:2]}/{sequence_sha256}/raw-msa/{DATABASE_ID}/"
        f"{search_identity}/samples/{validated_sample_id}"
    )


def _build_search_plan(mode: str) -> dict[str, object]:
    """Build a side-effect-free fixed search plan."""
    common: dict[str, object] = {
        "campaign_id": CAMPAIGN_ID,
        "operation": "search",
        "mode": mode,
        "execution": "sequential",
        "resources_per_container": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "timeout_seconds": CONF.timeout,
        },
        "cache_policy": "validate-done-marker-before-remote-call",
    }
    if mode == "smoke":
        cases = [_search_case(case_id) for case_id in SMOKE_CASE_IDS]
        return common | {
            "query": SCREENING_QUERY.as_dict(),
            "cases": [case.as_dict() for case in cases],
            "remote_function_inputs": len(cases),
            "counted_as_benchmark_samples": False,
            "oracle_case": "B1",
            "scientific_gate_case": "S3",
        }
    if mode == "sweep":
        return common | {
            "query": SCREENING_QUERY.as_dict(),
            "prerequisite": "completed passing smoke gate",
            "oracle_case": "B1",
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "reused_samples": [
                {
                    "case_id": case_id,
                    "sample_id": _focused_sweep_sample_id(case_id),
                }
                for case_id in FOCUSED_SWEEP_REUSED_CASE_IDS
            ],
            "new_samples": [
                {
                    "case": _search_case(case_id).as_dict(),
                    "sample_id": _focused_sweep_sample_id(case_id),
                }
                for case_id in FOCUSED_SWEEP_NEW_CASE_IDS
            ],
            "remote_function_inputs": len(FOCUSED_SWEEP_NEW_CASE_IDS),
            "total_case_results": len(FOCUSED_SWEEP_CASE_IDS),
            "runs_per_new_case": 1,
            "stress_samples": 0,
            "selection_policy": {
                "minimum_search_wall_improvement_vs_B1": 0.20,
                "cost_candidate_maximum_slowdown_vs_fastest": 0.15,
                "close_results_require_review_within": 0.15,
            },
        }
    if mode == "matrix":
        return common | {
            "screening_query": SCREENING_QUERY.as_dict(),
            "stress_query": STRESS_QUERY.as_dict(),
            "screening_block_orders": [list(order) for order in SCREENING_BLOCK_ORDERS],
            "screening_samples": 24,
            "conditional_stress_samples": 12,
            "maximum_remote_function_inputs": 36,
            "stress_cases": "B0, B1, and two promoted sharded layouts",
            "prerequisite": "completed passing smoke gate",
            "scientific_gate": {
                "oracle_case": "B1",
                "top_unique_hits_exact": 100,
                "minimum_full_unique_hit_jaccard": 0.99,
            },
            "performance_gate": {
                "minimum_search_wall_improvement_vs_B1": 0.20,
                "cost_candidate_maximum_slowdown_vs_fastest": 0.15,
                "maximum_three_sample_variation": 0.10,
            },
            "pricing": {
                "cpu_usd_per_core_second": MODAL_CPU_USD_PER_CORE_SECOND,
                "memory_usd_per_gib_second": MODAL_MEMORY_USD_PER_GIB_SECOND,
                "observed_date": MODAL_PRICING_OBSERVED_DATE,
                "source": MODAL_PRICING_URL,
            },
        }
    raise ValueError("search mode must be 'smoke', 'sweep', or 'matrix'")


def _campaign_plan_bytes() -> bytes:
    """Serialize the immutable plan shared by every campaign operation."""
    return _json_bytes({
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "profile_id": PROFILE_ID,
        "prepare": _build_prepare_plan(DEFAULT_SEQKIT_THREADS),
        "storage_scan": _build_scan_plan(),
        "search_smoke": _build_search_plan("smoke"),
        "search_matrix": _build_search_plan("matrix"),
    })


def _ensure_campaign_plan_mounted() -> None:
    """Create or validate the immutable campaign plan through its mount."""
    plan_path = Path(APP_INFO.output_dir) / "benchmarks" / CAMPAIGN_ID / "plan.json"
    expected = _campaign_plan_bytes()
    if plan_path.is_file():
        if plan_path.read_bytes() != expected:
            raise ValueError("Existing campaign plan differs from this app")
        return
    _write_bytes_exclusive(plan_path, expected)


def _ensure_campaign_plan_client() -> None:
    """Create or validate the immutable campaign plan through the client."""
    relative_path = f"benchmarks/{CAMPAIGN_ID}/plan.json"
    expected = _campaign_plan_bytes()
    try:
        existing = _read_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path)
    except FileNotFoundError:
        _upload_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path, expected)
    else:
        if existing != expected:
            raise ValueError("Existing campaign plan differs from this app")


def _parse_a3m_records(a3m: str) -> list[tuple[str, str]]:
    """Parse the small, truncated merged A3M without importing AlphaFold."""
    records: list[tuple[str, str]] = []
    description: str | None = None
    sequence_parts: list[str] = []
    for raw_line in a3m.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if description is not None:
                if not sequence_parts:
                    raise ValueError(f"A3M record {description!r} has no sequence")
                records.append(("".join(sequence_parts), description))
            description = line[1:]
            if not description:
                raise ValueError("A3M record has an empty description")
            sequence_parts = []
        else:
            if description is None:
                raise ValueError("A3M sequence appears before its description")
            sequence_parts.append(line)
    if description is not None:
        if not sequence_parts:
            raise ValueError(f"A3M record {description!r} has no sequence")
        records.append(("".join(sequence_parts), description))
    if not records:
        raise ValueError("Merged A3M is empty")
    return records


def _normalize_a3m_sequence(sequence: str) -> str:
    """Remove A3M insertions and dot gaps for exact aligned-hit comparison."""
    return "".join(
        character
        for character in sequence
        if not character.islower() and character != "."
    )


def _tblout_index(
    raw_tblouts: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Parse raw tblouts using the same target-name key as AlphaFold."""
    latest_by_target: dict[str, dict[str, object]] = {}
    occurrences: dict[str, list[dict[str, object]]] = {}
    for source, tblout in raw_tblouts:
        for line_number, line in enumerate(tblout.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(
                    f"Invalid tblout line {line_number} from {source}: {line!r}"
                )
            entry: dict[str, object] = {
                "target_id": fields[0],
                "e_value_text": fields[4],
                "bit_score_text": fields[5],
                "e_value": float(fields[4]),
                "bit_score": float(fields[5]),
                "source": source,
                "line": line,
            }
            target_id = fields[0]
            occurrences.setdefault(target_id, []).append(entry)
            # This deliberately mirrors AlphaFold's last-tblout-line-wins map.
            latest_by_target[target_id] = entry
    return latest_by_target, occurrences


def _normalized_hit_rows(
    merged_a3m: str,
    raw_tblouts: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Create one normalized evidence row per non-query merged A3M record."""
    records = _parse_a3m_records(merged_a3m)
    latest_by_target, occurrences = _tblout_index(raw_tblouts)
    rows: list[dict[str, object]] = []
    for ordinal, (sequence, description) in enumerate(records[1:], start=1):
        target_id = description.partition(" ")[0].partition("/")[0]
        entry = latest_by_target.get(target_id)
        if entry is None:
            raise ValueError(f"Merged A3M target has no tblout row: {target_id}")
        target_occurrences = occurrences[target_id]
        occurrence_sources = [str(item["source"]) for item in target_occurrences]
        normalized_sequence = _normalize_a3m_sequence(sequence)
        rows.append({
            "ordinal": ordinal,
            "target_id": target_id,
            "description": description,
            "aligned_sequence": sequence,
            "normalized_sequence": normalized_sequence,
            "normalized_sequence_sha256": _sha256_bytes(normalized_sequence.encode()),
            "e_value": entry["e_value"],
            "e_value_text": entry["e_value_text"],
            "bit_score": entry["bit_score"],
            "bit_score_text": entry["bit_score_text"],
            "tblout_source": entry["source"],
            "tblout_line": entry["line"],
            "raw_occurrence_count": len(target_occurrences),
            "raw_occurrence_sources": ",".join(occurrence_sources),
            "cross_shard_duplicate": len(set(occurrence_sources)) > 1,
        })
    return rows


def _normalized_hits_parquet(rows: list[dict[str, object]]) -> bytes:
    """Serialize normalized hit evidence with a stable schema."""
    import polars as pl

    schema = {
        "ordinal": pl.Int64,
        "target_id": pl.String,
        "description": pl.String,
        "aligned_sequence": pl.String,
        "normalized_sequence": pl.String,
        "normalized_sequence_sha256": pl.String,
        "e_value": pl.Float64,
        "e_value_text": pl.String,
        "bit_score": pl.Float64,
        "bit_score_text": pl.String,
        "tblout_source": pl.String,
        "tblout_line": pl.String,
        "raw_occurrence_count": pl.Int64,
        "raw_occurrence_sources": pl.String,
        "cross_shard_duplicate": pl.Boolean,
    }
    table = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    buffer = io.BytesIO()
    table.write_parquet(buffer)
    return buffer.getvalue()


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    """Write one evidence artifact without replacing an existing sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _read_optional_integer(path: Path) -> int | None:
    """Read a Linux counter if exposed by the current cgroup."""
    try:
        value = path.read_text(encoding="utf-8").strip()
        return None if value == "max" else int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _cgroup_cpu_stats() -> dict[str, int | None]:
    """Read cumulative cgroup CPU use and throttling counters."""
    stats: dict[str, int | None] = {
        "usage_usec": None,
        "nr_periods": None,
        "nr_throttled": None,
        "throttled_usec": None,
    }
    try:
        lines = Path("/sys/fs/cgroup/cpu.stat").read_text(encoding="utf-8")
        for line in lines.splitlines():
            key, _, value = line.partition(" ")
            if key in stats:
                stats[key] = int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    if stats["usage_usec"] is None:
        usage_ns = _read_optional_integer(Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"))
        stats["usage_usec"] = None if usage_ns is None else usage_ns // 1_000
    return stats


def _mark_resource_phase(
    phase_state: dict[str, Any],
    phase: str,
    started: float,
) -> None:
    """Record an exact phase boundary for the next trace observation."""
    phase_state["current"] = phase
    events = phase_state.setdefault("events", [])
    if not isinstance(events, list):
        raise TypeError("Resource phase events must be a list")
    events.append({
        "phase": phase,
        "observed_at": _utc_now(),
        "elapsed_seconds": perf_counter() - started,
    })


def _resource_snapshot(
    started: float,
    phase_state: dict[str, Any],
    context: dict[str, object],
) -> dict[str, object]:
    """Capture one portable one-second resource observation."""
    import resource

    load_average: list[float] | None = None
    if hasattr(os, "getloadavg"):
        load_average = list(os.getloadavg())
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_stats = _cgroup_cpu_stats()
    events = phase_state.get("events")
    if not isinstance(events, list):
        events = []
    return context | {
        "observed_at": _utc_now(),
        "elapsed_seconds": perf_counter() - started,
        "phase": phase_state.get("current"),
        "phase_events": list(events),
        "cpu_usage_usec": cpu_stats["usage_usec"],
        "cpu_nr_periods": cpu_stats["nr_periods"],
        "cpu_nr_throttled": cpu_stats["nr_throttled"],
        "cpu_throttled_usec": cpu_stats["throttled_usec"],
        "memory_current_bytes": _read_optional_integer(
            Path("/sys/fs/cgroup/memory.current")
        ),
        "memory_peak_bytes": _read_optional_integer(Path("/sys/fs/cgroup/memory.peak")),
        "load_average": load_average,
        "cpu_affinity": affinity,
        "children_user_seconds": children.ru_utime,
        "children_system_seconds": children.ru_stime,
    }


def _trace_resources(
    trace_path: Path,
    stop: Event,
    started: float,
    phase_state: dict[str, Any],
    context: dict[str, object],
) -> None:
    """Persist cgroup measurements once a second until the sample finishes."""
    with trace_path.open("xb") as handle:
        while True:
            snapshot = _resource_snapshot(started, phase_state, context)
            handle.write(orjson.dumps(snapshot) + b"\n")
            handle.flush()
            if stop.wait(RESOURCE_TRACE_INTERVAL_SECONDS):
                snapshot = _resource_snapshot(started, phase_state, context)
                handle.write(orjson.dumps(snapshot) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                return


def _summarize_resource_trace(trace_path: Path) -> dict[str, int | float | None]:
    """Reduce the raw trace to actual CPU, throttling, and memory signals."""
    snapshots = [
        orjson.loads(line)
        for line in trace_path.read_bytes().splitlines()
        if line.strip()
    ]
    if not snapshots:
        raise ValueError("Resource trace is empty")
    cpu_rates: list[float] = []
    memory_gib_seconds = 0.0
    has_memory_integral = False
    for previous, current in zip(snapshots, snapshots[1:], strict=False):
        elapsed = float(current["elapsed_seconds"]) - float(previous["elapsed_seconds"])
        previous_cpu = previous.get("cpu_usage_usec")
        current_cpu = current.get("cpu_usage_usec")
        if (
            isinstance(previous_cpu, int)
            and isinstance(current_cpu, int)
            and elapsed > 0
        ):
            cpu_rates.append((current_cpu - previous_cpu) / 1_000_000 / elapsed)
        previous_memory = previous.get("memory_current_bytes")
        current_memory = current.get("memory_current_bytes")
        if (
            isinstance(previous_memory, int)
            and isinstance(current_memory, int)
            and elapsed > 0
        ):
            mean_memory_bytes = (previous_memory + current_memory) / 2
            memory_gib_seconds += mean_memory_bytes / (1024**3) * elapsed
            has_memory_integral = True

    first_cpu = snapshots[0].get("cpu_usage_usec")
    last_cpu = snapshots[-1].get("cpu_usage_usec")
    cpu_core_seconds: float | None = None
    if isinstance(first_cpu, int) and isinstance(last_cpu, int):
        cpu_core_seconds = (last_cpu - first_cpu) / 1_000_000
    memory_values = [
        value
        for snapshot in snapshots
        for value in (snapshot.get("memory_current_bytes"),)
        if isinstance(value, int)
    ]
    peak_values = [
        value
        for snapshot in snapshots
        for value in (snapshot.get("memory_peak_bytes"),)
        if isinstance(value, int)
    ]
    first_throttled = snapshots[0].get("cpu_throttled_usec")
    last_throttled = snapshots[-1].get("cpu_throttled_usec")
    throttled_seconds: float | None = None
    if isinstance(first_throttled, int) and isinstance(last_throttled, int):
        throttled_seconds = (last_throttled - first_throttled) / 1_000_000
    first_nr_throttled = snapshots[0].get("cpu_nr_throttled")
    last_nr_throttled = snapshots[-1].get("cpu_nr_throttled")
    throttled_periods: int | None = None
    if isinstance(first_nr_throttled, int) and isinstance(last_nr_throttled, int):
        throttled_periods = last_nr_throttled - first_nr_throttled
    first_child_user = snapshots[0].get("children_user_seconds")
    last_child_user = snapshots[-1].get("children_user_seconds")
    first_child_system = snapshots[0].get("children_system_seconds")
    last_child_system = snapshots[-1].get("children_system_seconds")
    child_cpu_values = (
        first_child_user,
        last_child_user,
        first_child_system,
        last_child_system,
    )
    child_cpu_seconds: float | None = None
    if all(isinstance(value, int | float) for value in child_cpu_values):
        child_cpu_seconds = (
            float(last_child_user)
            - float(first_child_user)
            + float(last_child_system)
            - float(first_child_system)
        )
    return {
        "observations": len(snapshots),
        "elapsed_seconds": float(snapshots[-1]["elapsed_seconds"]),
        "cpu_core_seconds": cpu_core_seconds,
        "child_process_cpu_seconds": child_cpu_seconds,
        "peak_interval_cpu_cores": max(cpu_rates) if cpu_rates else None,
        "cpu_throttled_periods": throttled_periods,
        "cpu_throttled_seconds": throttled_seconds,
        "memory_gib_seconds": memory_gib_seconds if has_memory_integral else None,
        "peak_memory_current_bytes": max(memory_values) if memory_values else None,
        "cgroup_memory_peak_bytes": max(peak_values) if peak_values else None,
    }


def _estimate_compute_cost(
    resource_summary: dict[str, int | float | None],
    sample_wall_seconds: float,
) -> dict[str, float | str]:
    """Estimate billed CPU and memory using Modal's published Function rates."""
    cpu_core_seconds = resource_summary.get("cpu_core_seconds")
    memory_gib_seconds = resource_summary.get("memory_gib_seconds")
    observed_cpu = (
        float(cpu_core_seconds) if isinstance(cpu_core_seconds, int | float) else 0.0
    )
    observed_memory = (
        float(memory_gib_seconds)
        if isinstance(memory_gib_seconds, int | float)
        else 0.0
    )
    billed_cpu = max(0.125 * sample_wall_seconds, observed_cpu)
    billed_memory = max(1.0 * sample_wall_seconds, observed_memory)
    cpu_cost = billed_cpu * MODAL_CPU_USD_PER_CORE_SECOND
    memory_cost = billed_memory * MODAL_MEMORY_USD_PER_GIB_SECOND
    return {
        "estimated_billed_cpu_core_seconds": billed_cpu,
        "estimated_billed_memory_gib_seconds": billed_memory,
        "estimated_cpu_cost_usd": cpu_cost,
        "estimated_memory_cost_usd": memory_cost,
        "estimated_compute_cost_usd": cpu_cost + memory_cost,
        "pricing_observed_date": MODAL_PRICING_OBSERVED_DATE,
        "pricing_source": MODAL_PRICING_URL,
    }


def _execute_jackhmmer_search(
    profile_root: Path,
    query: SearchQuery,
    case: SearchCase,
    phase_state: dict[str, Any],
    sample_started: float,
) -> tuple[str, list[tuple[str, str]], dict[str, object]]:
    """Run the pinned wrapper while retaining raw tblout and phase timings."""
    from importlib import import_module

    jackhmmer = import_module("alphafold3.data.tools.jackhmmer")

    if case.layout == "monolith":
        database_path = str(profile_root / "source" / SOURCE_DB_FILENAME)
    else:
        database_path = f"{profile_root / 'shards' / SOURCE_DB_FILENAME}@{SHARD_COUNT}"
    tool = jackhmmer.Jackhmmer(
        binary_path=JACKHMMER_BINARY_PATH,
        database_path=database_path,
        n_cpu=case.jackhmmer_n_cpu,
        n_iter=JACKHMMER_N_ITER,
        e_value=JACKHMMER_E_VALUE,
        z_value=case.z_value,
        dom_z_value=case.z_value,
        max_sequences=JACKHMMER_MAX_SEQUENCES,
        filter_f1=JACKHMMER_FILTER_F1,
        filter_f2=JACKHMMER_FILTER_F2,
        filter_f3=JACKHMMER_FILTER_F3,
        max_threads=case.active_shards,
    )
    _mark_resource_phase(phase_state, "query", sample_started)
    search_started = perf_counter()
    if case.layout == "monolith":
        shard_started = perf_counter()
        result = tool._query_db_shard(  # noqa: SLF001
            target_sequence=query.sequence,
            db_shard_path=database_path,
            get_tblout=True,
        )
        shard_finished = perf_counter()
        if result.tblout is None:
            raise ValueError("Monolith Jackhmmer result did not contain tblout")
        search_wall_seconds = perf_counter() - search_started
        return (
            result.a3m,
            [("monolith", result.tblout)],
            {
                "search_wall_seconds": search_wall_seconds,
                "merge_wall_seconds": 0.0,
                "shards": [
                    {
                        "source": "monolith",
                        "started_seconds": shard_started - search_started,
                        "finished_seconds": shard_finished - search_started,
                        "wall_seconds": shard_finished - shard_started,
                    }
                ],
            },
        )

    shard_paths = tuple(profile_root / "shards" / name for name in _shard_names())
    global_temp_dir = tempfile.mkdtemp(prefix="af3-msa-search-")

    def query_shard(shard_path: Path) -> tuple[Any, dict[str, object]]:
        shard_started = perf_counter()
        result = tool._query_db_shard(  # noqa: SLF001
            target_sequence=query.sequence,
            db_shard_path=str(shard_path),
            get_tblout=True,
            global_temp_dir=global_temp_dir,
        )
        shard_finished = perf_counter()
        return result, {
            "source": shard_path.name,
            "started_seconds": shard_started - search_started,
            "finished_seconds": shard_finished - search_started,
            "wall_seconds": shard_finished - shard_started,
        }

    try:
        with ThreadPoolExecutor(max_workers=case.active_shards) as executor:
            outputs = tuple(executor.map(query_shard, shard_paths))
    finally:
        shutil.rmtree(global_temp_dir, ignore_errors=True)
    results = [result for result, unused_timing in outputs]
    raw_tblouts: list[tuple[str, str]] = []
    for result, timing in outputs:
        if result.tblout is None:
            raise ValueError(f"Shard {timing['source']} did not contain tblout")
        raw_tblouts.append((str(timing["source"]), result.tblout))
    _mark_resource_phase(phase_state, "merge", sample_started)
    merge_started = perf_counter()
    merged = jackhmmer._merge_jackhmmer_results(  # noqa: SLF001
        results,
        JACKHMMER_MAX_SEQUENCES,
    )
    merge_wall_seconds = perf_counter() - merge_started
    search_wall_seconds = perf_counter() - search_started
    return (
        merged.a3m,
        raw_tblouts,
        {
            "search_wall_seconds": search_wall_seconds,
            "merge_wall_seconds": merge_wall_seconds,
            "shards": [timing for unused_result, timing in outputs],
        },
    )


def _run_search_sample(
    query_id: str,
    case_id: str,
    sample_id: str,
    expected_search_identity: str,
    expected_sample_identity: str,
) -> dict[str, object]:
    """Run one immutable sample and publish its completion marker last."""
    sample_started = perf_counter()
    query = _search_query(query_id)
    case = _search_case(case_id)
    validated_sample_id = _validate_sample_id(sample_id)
    profile_root = Path(APP_INFO.sharded_db_dir) / APP_INFO.profile_relpath
    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    manifest = _validate_published_profile(profile_root, verify_digests=False)
    manifest_path = profile_root / "manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    profile_scientific_identity = _profile_scientific_identity(manifest)
    search_identity = _search_identity(profile_scientific_identity, query, case)
    sample_identity = _search_sample_identity(
        profile_scientific_identity,
        query,
        case,
        validated_sample_id,
    )
    if search_identity != expected_search_identity:
        raise ValueError("Client and worker search identities differ")
    if sample_identity != expected_sample_identity:
        raise ValueError("Client and worker sample identities differ")

    sample_relpath = _search_sample_relpath(
        query,
        search_identity,
        validated_sample_id,
    )
    final_output_root = Path(APP_INFO.output_dir) / sample_relpath
    try:
        _validate_done_marker(final_output_root, expected_identity=sample_identity)
    except (FileNotFoundError, ValueError):
        pass
    else:
        metrics = _load_json_object(final_output_root / "metrics.json")
        return metrics | {"status": "reused"}

    output_root = (
        final_output_root.parent
        / ".staging"
        / f"{validated_sample_id}-{uuid.uuid4().hex}"
    )
    output_root.mkdir(parents=True)
    log_path = output_root / "run.log"
    trace_path = output_root / "trace.jsonl"
    container = _container_sample_metadata()
    phase_state: dict[str, Any] = {"current": None, "events": []}
    _mark_resource_phase(phase_state, "warmup", sample_started)
    _append_log(
        log_path,
        f"Starting {query.query_id} {case.case_id} sample {validated_sample_id}",
    )
    trace_stop = Event()
    trace_thread = Thread(
        target=_trace_resources,
        args=(
            trace_path,
            trace_stop,
            sample_started,
            phase_state,
            {
                "query_id": query.query_id,
                "case_id": case.case_id,
                "jackhmmer_n_cpu": case.jackhmmer_n_cpu,
                "active_shards": case.active_shards,
                "container_instance_id": container["container_instance_id"],
            },
        ),
        daemon=True,
    )
    trace_thread.start()
    try:
        merged_a3m, raw_tblouts, timings = _execute_jackhmmer_search(
            profile_root,
            query,
            case,
            phase_state,
            sample_started,
        )
        _mark_resource_phase(phase_state, "publish", sample_started)
        merged_a3m_path = output_root / "result.a3m"
        _write_bytes_exclusive(merged_a3m_path, merged_a3m.encode())
        raw_tblout_paths: list[Path] = []
        for source, tblout in raw_tblouts:
            if source == "monolith":
                tblout_path = output_root / "result.tblout"
            else:
                tblout_path = output_root / "shards" / f"{source}.tblout"
            _write_bytes_exclusive(tblout_path, tblout.encode())
            raw_tblout_paths.append(tblout_path)
        hit_rows = _normalized_hit_rows(merged_a3m, raw_tblouts)
        normalized_hits_path = output_root / "hits.parquet"
        _write_bytes_exclusive(
            normalized_hits_path,
            _normalized_hits_parquet(hit_rows),
        )
        _append_log(
            log_path,
            f"Completed search with {len(hit_rows)} merged hit rows",
        )
    except Exception as exc:
        trace_stop.set()
        trace_thread.join()
        _append_log(log_path, f"Failed with {type(exc).__name__}: {exc}")
        _write_json_atomic(
            output_root / "failure.json",
            {
                "failed_at": _utc_now(),
                "query": query.as_dict(),
                "case": case.as_dict(),
                "sample_id": validated_sample_id,
                "sample_identity": sample_identity,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        raise
    _mark_resource_phase(phase_state, "complete", sample_started)
    trace_stop.set()
    trace_thread.join()
    resource_summary = _summarize_resource_trace(trace_path)
    core_artifacts = [
        _artifact_record(merged_a3m_path, output_root),
        _artifact_record(normalized_hits_path, output_root),
        _artifact_record(trace_path, output_root),
        _artifact_record(log_path, output_root),
        *(_artifact_record(path, output_root) for path in raw_tblout_paths),
    ]
    BENCHMARK_OUTPUT_VOLUME.commit()
    sample_wall_seconds = perf_counter() - sample_started
    cost_estimate = _estimate_compute_cost(resource_summary, sample_wall_seconds)
    unique_hit_count = len({str(row["target_id"]) for row in hit_rows})
    cross_shard_duplicate_count = len({
        str(row["target_id"])
        for row in hit_rows
        if row["cross_shard_duplicate"] is True
    })
    metrics: dict[str, object] = {
        "status": "published",
        "campaign_id": CAMPAIGN_ID,
        "database_id": DATABASE_ID,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": manifest_sha256,
        "profile_scientific_identity": profile_scientific_identity,
        "query": query.as_dict(),
        "case": case.as_dict(),
        "scientific_config": _scientific_search_config(case),
        "search_identity": search_identity,
        "sample_id": validated_sample_id,
        "sample_identity": sample_identity,
        "result_path": sample_relpath,
        "search_wall_seconds": timings["search_wall_seconds"],
        "merge_wall_seconds": timings["merge_wall_seconds"],
        "sample_wall_seconds": sample_wall_seconds,
        "sample_wall_endpoint": "durable-core-evidence-commit",
        "shard_timings": timings["shards"],
        "hit_rows": len(hit_rows),
        "unique_hits": unique_hit_count,
        "duplicate_hit_rows": len(hit_rows) - unique_hit_count,
        "cross_shard_duplicate_targets": cross_shard_duplicate_count,
        "resource_summary": resource_summary,
        "cost_estimate": cost_estimate,
        "container": container,
    }
    metrics_path = output_root / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    BENCHMARK_OUTPUT_VOLUME.commit()
    artifacts = [*core_artifacts, _artifact_record(metrics_path, output_root)]
    _write_json_atomic(
        output_root / "done.json",
        {
            "schema_version": DONE_SCHEMA_VERSION,
            "status": "complete",
            "identity": sample_identity,
            "completed_at": _utc_now(),
            "artifacts": artifacts,
        },
    )
    BENCHMARK_OUTPUT_VOLUME.commit()
    if final_output_root.exists():
        try:
            _validate_done_marker(
                final_output_root,
                expected_identity=sample_identity,
            )
        except (FileNotFoundError, ValueError):
            orphan_root = final_output_root.parent / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            final_output_root.replace(
                orphan_root / f"{validated_sample_id}-{uuid.uuid4().hex}"
            )
        else:
            duplicate_root = final_output_root.parent / ".orphaned"
            duplicate_root.mkdir(parents=True, exist_ok=True)
            output_root.replace(
                duplicate_root / f"{validated_sample_id}-duplicate-{uuid.uuid4().hex}"
            )
            BENCHMARK_OUTPUT_VOLUME.commit()
            existing = _load_json_object(final_output_root / "metrics.json")
            return existing | {"status": "reused-concurrent"}
    output_root.replace(final_output_root)
    BENCHMARK_OUTPUT_VOLUME.commit()
    return metrics


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=1,
    volumes={
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True
        ),
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def benchmark_small_bfd_search(
    query_id: str,
    case_id: str,
    sample_id: str,
    expected_search_identity: str,
    expected_sample_identity: str,
) -> dict[str, object]:
    """Run one cache-aware small-BFD Jackhmmer benchmark sample."""
    return _run_search_sample(
        query_id,
        case_id,
        sample_id,
        expected_search_identity,
        expected_sample_identity,
    )


def _unique_hit_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep the first merged occurrence of each AlphaFold target name."""
    unique_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        target_id = str(row["target_id"])
        if target_id not in seen:
            seen.add(target_id)
            unique_rows.append(row)
    return unique_rows


def _top_hits_tie_equivalent(
    oracle_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> bool:
    """Compare ranked hits while ignoring order inside exact score ties."""
    if len(oracle_rows) != len(candidate_rows):
        return False

    def tie_blocks(
        rows: list[dict[str, object]],
    ) -> list[tuple[tuple[str, str], tuple[str, ...]]]:
        blocks: list[tuple[tuple[str, str], tuple[str, ...]]] = []
        current_key: tuple[str, str] | None = None
        current_ids: list[str] = []
        for row in rows:
            key = (str(row["e_value_text"]), str(row["bit_score_text"]))
            if current_key is not None and key != current_key:
                blocks.append((current_key, tuple(sorted(current_ids))))
                current_ids = []
            current_key = key
            current_ids.append(str(row["target_id"]))
        if current_key is not None:
            blocks.append((current_key, tuple(sorted(current_ids))))
        return blocks

    return tie_blocks(oracle_rows) == tie_blocks(candidate_rows)


def _compare_normalized_hits(
    oracle_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the fixed smoke scientific-equivalence gate."""
    oracle_unique = _unique_hit_rows(oracle_rows)
    candidate_unique = _unique_hit_rows(candidate_rows)
    oracle_ids = [str(row["target_id"]) for row in oracle_unique]
    candidate_ids = [str(row["target_id"]) for row in candidate_unique]
    top_width = min(100, max(len(oracle_ids), len(candidate_ids)))
    top_oracle = oracle_ids[:top_width]
    top_candidate = candidate_ids[:top_width]
    top_hits_exact = top_oracle == top_candidate
    top_hits_tie_equivalent = _top_hits_tie_equivalent(
        oracle_unique[:top_width],
        candidate_unique[:top_width],
    )

    oracle_by_id = {str(row["target_id"]): row for row in oracle_unique}
    candidate_by_id = {str(row["target_id"]): row for row in candidate_unique}
    shared_ids = set(oracle_by_id) & set(candidate_by_id)
    score_mismatches = sorted(
        target_id
        for target_id in shared_ids
        if (
            oracle_by_id[target_id]["e_value_text"],
            oracle_by_id[target_id]["bit_score_text"],
        )
        != (
            candidate_by_id[target_id]["e_value_text"],
            candidate_by_id[target_id]["bit_score_text"],
        )
    )
    sequence_mismatches = sorted(
        target_id
        for target_id in shared_ids
        if oracle_by_id[target_id]["normalized_sequence_sha256"]
        != candidate_by_id[target_id]["normalized_sequence_sha256"]
    )
    oracle_set = set(oracle_ids)
    candidate_set = set(candidate_ids)
    union = oracle_set | candidate_set
    overlap = len(oracle_set & candidate_set) / len(union) if union else 1.0
    oracle_only = [
        target_id for target_id in oracle_ids if target_id not in candidate_set
    ]
    candidate_only = [
        target_id for target_id in candidate_ids if target_id not in oracle_set
    ]
    oracle_positions = {
        target_id: position for position, target_id in enumerate(oracle_ids, start=1)
    }
    candidate_positions = {
        target_id: position for position, target_id in enumerate(candidate_ids, start=1)
    }
    difference_positions = [
        *(oracle_positions[target_id] for target_id in oracle_only),
        *(candidate_positions[target_id] for target_id in candidate_only),
    ]
    differences_are_below_top_100 = all(
        position > 100 for position in difference_positions
    )
    duplicate_targets = sorted({
        str(row["target_id"])
        for row in candidate_rows
        if row.get("cross_shard_duplicate") is True
    })
    has_set_differences = bool(oracle_only or candidate_only)
    both_results_reached_hit_row_limit = (
        len(oracle_rows) == JACKHMMER_MAX_SEQUENCES - 1
        and len(candidate_rows) == JACKHMMER_MAX_SEQUENCES - 1
    )
    candidate_duplicate_hit_rows = len(candidate_rows) - len(candidate_unique)
    oracle_tail_start = len(oracle_ids) - candidate_duplicate_hit_rows + 1
    oracle_only_is_displaced_tail = (
        not candidate_only
        and len(oracle_only) <= candidate_duplicate_hit_rows <= SHARD_COUNT
        and all(
            oracle_positions[target_id] >= oracle_tail_start
            for target_id in oracle_only
        )
    )
    differences_characterized = not has_set_differences or (
        differences_are_below_top_100
        and bool(duplicate_targets)
        and both_results_reached_hit_row_limit
        and oracle_only_is_displaced_tail
    )
    passed = (
        top_hits_tie_equivalent
        and not score_mismatches
        and not sequence_mismatches
        and overlap >= 0.99
        and differences_characterized
    )
    return {
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "passed": passed,
        "top_comparison_width": top_width,
        "top_hits_exact": top_hits_exact,
        "top_hits_tie_equivalent": top_hits_tie_equivalent,
        "top_order_differs_only_within_ties": (
            top_hits_tie_equivalent and not top_hits_exact
        ),
        "oracle_top_ids": top_oracle,
        "candidate_top_ids": top_candidate,
        "oracle_unique_hits": len(oracle_ids),
        "candidate_unique_hits": len(candidate_ids),
        "shared_unique_hits": len(shared_ids),
        "full_unique_hit_jaccard": overlap,
        "required_full_unique_hit_jaccard": 0.99,
        "score_mismatch_count": len(score_mismatches),
        "score_mismatch_ids": score_mismatches,
        "sequence_mismatch_count": len(sequence_mismatches),
        "sequence_mismatch_ids": sequence_mismatches,
        "oracle_only_ids": oracle_only,
        "candidate_only_ids": candidate_only,
        "differences_are_below_top_100": differences_are_below_top_100,
        "both_results_reached_hit_row_limit": both_results_reached_hit_row_limit,
        "candidate_duplicate_hit_rows": candidate_duplicate_hit_rows,
        "oracle_tail_start": oracle_tail_start,
        "oracle_only_is_displaced_tail": oracle_only_is_displaced_tail,
        "candidate_cross_shard_duplicate_targets": duplicate_targets,
        "differences_characterized_as_duplicate_tail": differences_characterized,
    }


def _read_normalized_hits(sample_relpath: str) -> list[dict[str, object]]:
    """Read one sample's normalized hit table through the Volume client."""
    import polars as pl

    data = _read_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{sample_relpath}/hits.parquet",
    )
    return pl.read_parquet(io.BytesIO(data)).to_dicts()


def _search_results_parquet(results: list[dict[str, Any]]) -> bytes:
    """Serialize one flat timing and resource row per search sample."""
    import polars as pl

    rows: list[dict[str, object]] = []
    for result in results:
        case = result.get("case")
        query = result.get("query")
        resource = result.get("resource_summary")
        cost = result.get("cost_estimate")
        container = result.get("container")
        if not isinstance(case, dict):
            raise ValueError("Search result is missing case metadata")
        if not isinstance(query, dict):
            raise ValueError("Search result is missing query metadata")
        if not isinstance(resource, dict):
            raise ValueError("Search result is missing resource metadata")
        if not isinstance(cost, dict):
            raise ValueError("Search result is missing cost metadata")
        if not isinstance(container, dict):
            raise ValueError("Search result is missing nested metadata")
        rows.append({
            "campaign_id": CAMPAIGN_ID,
            "sample_kind": result["sample_kind"],
            "block_index": result.get("block_index"),
            "query_id": query["query_id"],
            "query_length": query["length"],
            "case_id": case["case_id"],
            "layout": case["layout"],
            "jackhmmer_n_cpu": case["jackhmmer_n_cpu"],
            "active_shards": case["active_shards"],
            "aggregate_cpu_slots": case["aggregate_cpu_slots"],
            "search_identity": result["search_identity"],
            "sample_id": result["sample_id"],
            "sample_identity": result["sample_identity"],
            "search_wall_seconds": result["search_wall_seconds"],
            "merge_wall_seconds": result["merge_wall_seconds"],
            "sample_wall_seconds": result["sample_wall_seconds"],
            "remote_call_wall_seconds": result["remote_call_wall_seconds"],
            "hit_rows": result["hit_rows"],
            "unique_hits": result["unique_hits"],
            "duplicate_hit_rows": result["duplicate_hit_rows"],
            "cross_shard_duplicate_targets": result["cross_shard_duplicate_targets"],
            "cpu_core_seconds": resource.get("cpu_core_seconds"),
            "peak_interval_cpu_cores": resource.get("peak_interval_cpu_cores"),
            "cpu_throttled_periods": resource.get("cpu_throttled_periods"),
            "cpu_throttled_seconds": resource.get("cpu_throttled_seconds"),
            "peak_memory_current_bytes": resource.get("peak_memory_current_bytes"),
            "estimated_compute_cost_usd": cost.get("estimated_compute_cost_usd"),
            "container_hostname": container.get("hostname"),
            "container_instance_id": container.get("container_instance_id"),
            "container_sample_ordinal": container.get("container_sample_ordinal"),
            "container_reused_for_sample": container.get("container_reused_for_sample"),
            "result_path": result["result_path"],
            "reused": result["reused"],
        })
    buffer = io.BytesIO()
    pl.DataFrame(rows).sort(["query_id", "case_id", "sample_id"]).write_parquet(buffer)
    return buffer.getvalue()


def _smoke_summary_markdown(
    results: list[dict[str, Any]],
    comparisons: dict[str, dict[str, object]],
) -> str:
    """Render a compact durable smoke-test report."""
    gate_passed = comparisons["S3_vs_B1"]["passed"] is True
    jaccard = comparisons["S3_vs_B1"]["full_unique_hit_jaccard"]
    if not isinstance(jaccard, int | float):
        raise ValueError("Smoke comparison has an invalid Jaccard value")
    lines = [
        f"# {CAMPAIGN_ID} search smoke",
        "",
        f"Scientific gate: **{'PASS' if gate_passed else 'FAIL'}**",
        "",
        "| Case | Search wall (s) | Sample wall (s) | Unique hits | Reused |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in sorted(results, key=lambda item: str(item["case"]["case_id"])):
        case = result["case"]
        lines.append(
            f"| {case['case_id']} | {float(result['search_wall_seconds']):.3f} | "
            f"{float(result['sample_wall_seconds']):.3f} | "
            f"{int(result['unique_hits'])} | {bool(result['reused'])} |"
        )
    lines.extend([
        "",
        "B1 is the explicit-Z oracle. B0 is descriptive only; S3 is the "
        "required sharded scientific gate.",
        "",
        f"Scientific comparison policy: {SCIENTIFIC_COMPARISON_POLICY}",
        f"S3/B1 top hits exact: {comparisons['S3_vs_B1']['top_hits_exact']}",
        "S3/B1 top hits equivalent modulo exact score ties: "
        f"{comparisons['S3_vs_B1']['top_hits_tie_equivalent']}",
        f"S3/B1 full unique-hit Jaccard: {float(jaccard):.6f}",
        "",
    ])
    return "\n".join(lines)


def _smoke_operation_identity(
    manifest_sha256: str,
    sample_identities: dict[str, str],
) -> str:
    """Hash the complete smoke operation and its immutable sample identities."""
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "plan": _build_search_plan("smoke"),
            "sample_identities": sample_identities,
        })
    )


def _submit_search_smoke() -> dict[str, object]:
    """Run only missing smoke samples, then publish the scientific gate."""
    _ensure_campaign_plan_client()
    manifest_relpath = f"{APP_INFO.profile_relpath}/manifest.json"
    manifest_bytes = _read_volume_bytes(SHARDED_MSA_DB_VOLUME, manifest_relpath)
    manifest = orjson.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Profile manifest must be a JSON object")
    _validate_profile_manifest(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    profile_scientific_identity = _profile_scientific_identity(manifest)
    cases = [_search_case(case_id) for case_id in SMOKE_CASE_IDS]
    sample_ids = {case.case_id: f"smoke-{case.case_id.lower()}" for case in cases}
    search_identities = {
        case.case_id: _search_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
        )
        for case in cases
    }
    sample_identities = {
        case.case_id: _search_sample_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
            sample_ids[case.case_id],
        )
        for case in cases
    }
    sample_relpaths = {
        case.case_id: _search_sample_relpath(
            SCREENING_QUERY,
            search_identities[case.case_id],
            sample_ids[case.case_id],
        )
        for case in cases
    }
    operation_root = f"benchmarks/{CAMPAIGN_ID}/search/smoke"
    operation_identity = _smoke_operation_identity(
        manifest_sha256,
        sample_identities,
    )
    operation_marker_path = f"{operation_root}/done.json"
    samples_complete = all(
        _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{sample_relpaths[case.case_id]}/done.json",
            expected_identity=sample_identities[case.case_id],
        )
        for case in cases
    )
    if samples_complete and _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        expected_identity=operation_identity,
    ):
        summary = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/summary.json",
        )
        gate_status = (
            "smoke gate passed; measured matrix pending"
            if summary.get("scientific_gate_passed") is True
            else "blocked by smoke scientific gate"
        )
        _publish_campaign_progress(
            stage="search smoke",
            status=gate_status,
            details=[
                "B0, B1, and S3 smoke evidence is available under "
                f"`{operation_root}/`.",
            ],
        )
        return summary | {
            "status": "reused",
            "remote_function_inputs_submitted": 0,
        }

    results: list[dict[str, Any]] = []
    submitted_inputs = 0
    for case in cases:
        marker_path = f"{sample_relpaths[case.case_id]}/done.json"
        reused = _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            marker_path,
            expected_identity=sample_identities[case.case_id],
        )
        if reused:
            result = _read_volume_json(
                BENCHMARK_OUTPUT_VOLUME,
                f"{sample_relpaths[case.case_id]}/metrics.json",
            )
            remote_call_wall_seconds = 0.0
        else:
            submitted_inputs += 1
            remote_started = perf_counter()
            result = benchmark_small_bfd_search.remote(
                query_id=SCREENING_QUERY.query_id,
                case_id=case.case_id,
                sample_id=sample_ids[case.case_id],
                expected_search_identity=search_identities[case.case_id],
                expected_sample_identity=sample_identities[case.case_id],
            )
            remote_call_wall_seconds = perf_counter() - remote_started
        results.append(
            result
            | {
                "sample_kind": "smoke",
                "remote_call_wall_seconds": remote_call_wall_seconds,
                "reused": reused,
            }
        )

    hits_by_case = {
        case.case_id: _read_normalized_hits(sample_relpaths[case.case_id])
        for case in cases
    }
    comparisons = {
        "B0_vs_B1": _compare_normalized_hits(hits_by_case["B1"], hits_by_case["B0"])
        | {"oracle_case": "B1", "candidate_case": "B0", "is_gate": False},
        "S3_vs_B1": _compare_normalized_hits(hits_by_case["B1"], hits_by_case["S3"])
        | {"oracle_case": "B1", "candidate_case": "S3", "is_gate": True},
    }
    scientific_gate_passed = comparisons["S3_vs_B1"]["passed"] is True
    results_bytes = _search_results_parquet(results)
    comparisons_bytes = _json_bytes(comparisons)
    summary_markdown = _smoke_summary_markdown(results, comparisons).encode()
    summary = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "operation": "search",
        "mode": "smoke",
        "operation_identity": operation_identity,
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "profile_manifest_sha256": manifest_sha256,
        "scientific_gate_passed": scientific_gate_passed,
        "oracle_case": "B1",
        "gate_case": "S3",
        "sample_paths": sample_relpaths,
        "remote_function_inputs_submitted": submitted_inputs,
        "completed_at": _utc_now(),
        "results_path": f"{operation_root}/results.parquet",
        "comparisons_path": f"{operation_root}/comparisons.json",
    }
    summary_bytes = _json_bytes(summary)
    operation_artifacts = {
        "results.parquet": results_bytes,
        "comparisons.json": comparisons_bytes,
        "summary.md": summary_markdown,
        "summary.json": summary_bytes,
    }
    for relative_path, data in operation_artifacts.items():
        _upload_volume_bytes(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/{relative_path}",
            data,
        )
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        _json_bytes({
            "schema_version": DONE_SCHEMA_VERSION,
            "status": "complete",
            "identity": operation_identity,
            "completed_at": _utc_now(),
            "artifacts": [
                _volume_artifact_record(relative_path, data)
                for relative_path, data in operation_artifacts.items()
            ],
        }),
    )
    _publish_campaign_progress(
        stage="search smoke",
        status=(
            "smoke gate passed; measured matrix pending"
            if scientific_gate_passed
            else "blocked by smoke scientific gate"
        ),
        details=[
            f"Scientific gate: {'PASS' if scientific_gate_passed else 'FAIL'}.",
            f"Submitted {submitted_inputs} remote search inputs.",
        ],
    )
    return summary


def _validate_screening_block_orders() -> None:
    """Require three distinct permutations of every fixed search case."""
    expected = set(MATRIX_CASE_IDS)
    if len(SCREENING_BLOCK_ORDERS) != 3:
        raise ValueError("The screening matrix must contain three blocks")
    if len(set(SCREENING_BLOCK_ORDERS)) != len(SCREENING_BLOCK_ORDERS):
        raise ValueError("Screening block orders must be distinct")
    for order in SCREENING_BLOCK_ORDERS:
        if len(order) != len(expected) or set(order) != expected:
            raise ValueError("Each screening block must contain every case once")


def _matrix_operation_identity(manifest_sha256: str) -> str:
    """Hash the fixed one-shot matrix plan and profile identity."""
    _validate_screening_block_orders()
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "plan": _build_search_plan("matrix"),
        })
    )


def _matrix_sample_id(kind: str, case_id: str, block_index: int) -> str:
    """Return one deterministic screening or stress sample ID."""
    if kind not in {"screen", "stress"}:
        raise ValueError("Matrix sample kind must be 'screen' or 'stress'")
    _search_case(case_id)
    if block_index not in {1, 2, 3}:
        raise ValueError("Matrix block index must be 1, 2, or 3")
    return f"{kind}-{case_id.lower()}-block-{block_index:02d}"


def _matrix_sample_spec(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
    sample_id: str,
) -> dict[str, str]:
    """Build all immutable identifiers and paths for one matrix sample."""
    search_identity = _search_identity(profile_scientific_identity, query, case)
    sample_identity = _search_sample_identity(
        profile_scientific_identity,
        query,
        case,
        sample_id,
    )
    return {
        "search_identity": search_identity,
        "sample_identity": sample_identity,
        "sample_relpath": _search_sample_relpath(
            query,
            search_identity,
            sample_id,
        ),
    }


def _run_or_reuse_matrix_sample(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
    sample_id: str,
    *,
    sample_kind: str,
    block_index: int,
) -> tuple[dict[str, Any], bool]:
    """Check durable evidence before submitting exactly one remote sample."""
    spec = _matrix_sample_spec(
        profile_scientific_identity,
        query,
        case,
        sample_id,
    )
    marker_path = f"{spec['sample_relpath']}/done.json"
    reused = _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        marker_path,
        expected_identity=spec["sample_identity"],
    )
    if reused:
        result = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{spec['sample_relpath']}/metrics.json",
        )
        remote_call_wall_seconds = 0.0
    else:
        remote_started = perf_counter()
        result = benchmark_small_bfd_search.remote(
            query_id=query.query_id,
            case_id=case.case_id,
            sample_id=sample_id,
            expected_search_identity=spec["search_identity"],
            expected_sample_identity=spec["sample_identity"],
        )
        remote_call_wall_seconds = perf_counter() - remote_started
    return (
        result
        | {
            "sample_kind": sample_kind,
            "block_index": block_index,
            "remote_call_wall_seconds": remote_call_wall_seconds,
            "reused": reused,
        },
        not reused,
    )


def _require_passing_smoke_gate(
    manifest_sha256: str,
    profile_scientific_identity: str,
) -> None:
    """Block matrix submission unless the exact current smoke gate passed."""
    cases = [_search_case(case_id) for case_id in SMOKE_CASE_IDS]
    sample_identities = {
        case.case_id: _search_sample_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
            f"smoke-{case.case_id.lower()}",
        )
        for case in cases
    }
    operation_identity = _smoke_operation_identity(
        manifest_sha256,
        sample_identities,
    )
    smoke_root = f"benchmarks/{CAMPAIGN_ID}/search/smoke"
    if not _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        f"{smoke_root}/done.json",
        expected_identity=operation_identity,
    ):
        raise RuntimeError("The current profile does not have a complete smoke gate")
    for case in cases:
        search_identity = _search_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
        )
        sample_relpath = _search_sample_relpath(
            SCREENING_QUERY,
            search_identity,
            f"smoke-{case.case_id.lower()}",
        )
        if not _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{sample_relpath}/done.json",
            expected_identity=sample_identities[case.case_id],
        ):
            raise RuntimeError(f"Smoke sample {case.case_id} is incomplete")
    summary = _read_volume_json(
        BENCHMARK_OUTPUT_VOLUME,
        f"{smoke_root}/summary.json",
    )
    if summary.get("scientific_gate_passed") is not True:
        raise RuntimeError("The current profile's smoke scientific gate did not pass")


def _metric_float(result: dict[str, Any], key: str) -> float:
    """Read one finite, nonnegative numeric sample metric."""
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Search result has invalid numeric metric {key!r}")
    numeric = float(value)
    if numeric < 0 or numeric == float("inf") or numeric != numeric:
        raise ValueError(f"Search result has non-finite metric {key!r}")
    return numeric


def _sample_cost(result: dict[str, Any]) -> float:
    """Read the pinned compute-cost estimate from one sample."""
    estimate = result.get("cost_estimate")
    if not isinstance(estimate, dict):
        raise ValueError("Search result is missing its cost estimate")
    value = estimate.get("estimated_compute_cost_usd")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Search result has an invalid compute-cost estimate")
    return float(value)


def _median_summary(values: list[float]) -> dict[str, float]:
    """Report median, range, MAD, and three-sample variation."""
    if not values:
        raise ValueError("Cannot summarize an empty metric sample")
    center = median(values)
    absolute_deviations = [abs(value - center) for value in values]
    variation = (max(values) - min(values)) / center if center else 0.0
    return {
        "median": center,
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
        "median_absolute_deviation": median(absolute_deviations),
        "relative_range": variation,
    }


def _case_performance_statistics(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    """Aggregate three measured samples for every represented case."""
    case_ids = sorted({str(result["case"]["case_id"]) for result in results})
    statistics: dict[str, dict[str, object]] = {}
    for case_id in case_ids:
        case_results = [
            result for result in results if result["case"]["case_id"] == case_id
        ]
        if len(case_results) != 3:
            raise ValueError(
                f"Expected three samples for {case_id}, got {len(case_results)}"
            )
        statistics[case_id] = {
            "search_wall_seconds": _median_summary([
                _metric_float(result, "search_wall_seconds") for result in case_results
            ]),
            "sample_wall_seconds": _median_summary([
                _metric_float(result, "sample_wall_seconds") for result in case_results
            ]),
            "remote_call_wall_seconds": _median_summary([
                _metric_float(result, "remote_call_wall_seconds")
                for result in case_results
            ]),
            "estimated_compute_cost_usd": _median_summary([
                _sample_cost(result) for result in case_results
            ]),
            "new_container_samples": sum(
                result["container"].get("container_reused_for_sample") is False
                for result in case_results
            ),
            "reused_container_samples": sum(
                result["container"].get("container_reused_for_sample") is True
                for result in case_results
            ),
        }
    return statistics


def _case_statistic_float(
    statistics: dict[str, dict[str, object]],
    case_id: str,
    metric: str,
    statistic: str,
) -> float:
    """Read one checked numeric value from nested case statistics."""
    summary = statistics[case_id].get(metric)
    if not isinstance(summary, dict):
        raise ValueError(f"Missing {metric} statistics for {case_id}")
    value = summary.get(statistic)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Invalid {metric}.{statistic} for {case_id}")
    return float(value)


def _ranking_float(row: dict[str, object], key: str) -> float:
    """Read one checked numeric candidate-ranking value."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Invalid candidate ranking value: {key}")
    return float(value)


def _matrix_comparisons(
    results: list[dict[str, Any]],
    candidate_case_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Compare every candidate and B0 with the same-block B1 oracle."""
    indexed = {
        (int(result["block_index"]), str(result["case"]["case_id"])): result
        for result in results
    }
    hit_tables = {
        key: _read_normalized_hits(str(result["result_path"]))
        for key, result in indexed.items()
    }
    comparisons: dict[str, dict[str, object]] = {}
    for block_index in (1, 2, 3):
        oracle = hit_tables[(block_index, "B1")]
        for case_id in ("B0", *candidate_case_ids):
            comparison_id = f"block-{block_index:02d}-{case_id}-vs-B1"
            comparisons[comparison_id] = _compare_normalized_hits(
                oracle,
                hit_tables[(block_index, case_id)],
            ) | {
                "block_index": block_index,
                "oracle_case": "B1",
                "candidate_case": case_id,
                "is_gate": case_id != "B0",
            }
    return comparisons


def _rank_screening_cases(
    results: list[dict[str, Any]],
    comparisons: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Apply scientific, stability, speed, overhead, and cost promotion rules."""
    statistics = _case_performance_statistics(results)
    b1_search = _case_statistic_float(statistics, "B1", "search_wall_seconds", "median")
    b1_sample = _case_statistic_float(statistics, "B1", "sample_wall_seconds", "median")
    candidate_rows: list[dict[str, object]] = []
    for case_id in MATRIX_CASE_IDS:
        case = _search_case(case_id)
        if not case.case_id.startswith("S"):
            continue
        scientific_valid = all(
            comparisons[f"block-{block_index:02d}-{case.case_id}-vs-B1"]["passed"]
            is True
            for block_index in (1, 2, 3)
        )
        search_median = _case_statistic_float(
            statistics, case.case_id, "search_wall_seconds", "median"
        )
        sample_median = _case_statistic_float(
            statistics, case.case_id, "sample_wall_seconds", "median"
        )
        cost_median = _case_statistic_float(
            statistics, case.case_id, "estimated_compute_cost_usd", "median"
        )
        search_improvement = 1.0 - search_median / b1_search
        sample_improvement = 1.0 - sample_median / b1_sample
        stable = (
            _case_statistic_float(
                statistics,
                case.case_id,
                "search_wall_seconds",
                "relative_range",
            )
            <= 0.10
        )
        operational_overhead_preserves_improvement = sample_improvement > 0
        meaningful = (
            scientific_valid
            and stable
            and search_improvement >= 0.20
            and operational_overhead_preserves_improvement
        )
        candidate_rows.append({
            "case_id": case.case_id,
            "scientific_valid": scientific_valid,
            "stable_within_10_percent": stable,
            "search_improvement_vs_B1": search_improvement,
            "sample_improvement_vs_B1": sample_improvement,
            "operational_overhead_preserves_improvement": (
                operational_overhead_preserves_improvement
            ),
            "meaningful_20_percent_success": meaningful,
            "median_search_wall_seconds": search_median,
            "median_sample_wall_seconds": sample_median,
            "median_estimated_compute_cost_usd": cost_median,
        })

    invalid_cases = [
        str(row["case_id"])
        for row in candidate_rows
        if row["scientific_valid"] is False
    ]
    unstable_cases = [
        str(row["case_id"])
        for row in candidate_rows
        if row["scientific_valid"] is True and row["stable_within_10_percent"] is False
    ]
    meaningful = [
        row for row in candidate_rows if row["meaningful_20_percent_success"] is True
    ]
    status = "promoted"
    selected: list[str] = []
    if invalid_cases:
        status = "blocked_scientific_review"
    elif unstable_cases:
        status = "blocked_high_variation"
    elif len(meaningful) < 2:
        status = "complete_insufficient_meaningful_layouts"
    else:
        fastest = min(
            meaningful,
            key=lambda row: (
                _ranking_float(row, "median_search_wall_seconds"),
                _ranking_float(row, "median_estimated_compute_cost_usd"),
                str(row["case_id"]),
            ),
        )
        cost_pool = [
            row
            for row in meaningful
            if row["case_id"] != fastest["case_id"]
            and _ranking_float(row, "median_search_wall_seconds")
            <= _ranking_float(fastest, "median_search_wall_seconds") * 1.15
        ]
        if not cost_pool:
            status = "complete_no_distinct_cost_candidate_within_15_percent"
        else:
            lowest_cost = min(
                cost_pool,
                key=lambda row: (
                    _ranking_float(row, "median_estimated_compute_cost_usd"),
                    _ranking_float(row, "median_search_wall_seconds"),
                    str(row["case_id"]),
                ),
            )
            selected = [str(fastest["case_id"]), str(lowest_cost["case_id"])]
    return {
        "status": status,
        "selected_case_ids": selected,
        "invalid_scientific_case_ids": invalid_cases,
        "unstable_case_ids": unstable_cases,
        "candidate_rankings": candidate_rows,
        "case_statistics": statistics,
        "rules": _build_search_plan("matrix")["performance_gate"],
    }


def _stress_block_orders(
    selected_case_ids: tuple[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Return three distinct deterministic stress permutations."""
    first, second = selected_case_ids
    if first == second or not all(
        case_id.startswith("S") for case_id in (first, second)
    ):
        raise ValueError("Stress promotion requires two distinct sharded cases")
    _search_case(first)
    _search_case(second)
    return (
        ("B0", "B1", first, second),
        (second, "B0", first, "B1"),
        ("B1", first, "B0", second),
    )


def _matrix_sample_records(results: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Build marker-validation references for every measured sample."""
    return [
        {
            "query_id": result["query"]["query_id"],
            "case_id": result["case"]["case_id"],
            "sample_id": result["sample_id"],
            "sample_identity": result["sample_identity"],
            "result_path": result["result_path"],
        }
        for result in results
    ]


def _matrix_summary_markdown(
    summary: dict[str, object],
    rankings: dict[str, object],
) -> str:
    """Render the matrix outcome, gates, variability, and candidate ranks."""
    selected = summary.get("selected_case_ids")
    if not isinstance(selected, list):
        raise ValueError("Matrix summary has invalid selected cases")
    selected_ids = {str(case_id) for case_id in selected}
    selected_text = ", ".join(sorted(selected_ids)) or "none"
    candidate_rows = rankings.get("candidate_rankings")
    statistics = rankings.get("case_statistics")
    if not isinstance(candidate_rows, list) or not isinstance(statistics, dict):
        raise ValueError("Matrix rankings are missing candidate statistics")
    typed_candidates: list[dict[str, object]] = []
    for candidate in candidate_rows:
        if not isinstance(candidate, dict):
            raise ValueError("Candidate ranking row must be an object")
        typed_candidates.append({str(key): value for key, value in candidate.items()})
    typed_statistics: dict[str, dict[str, object]] = {}
    for case_id, case_statistics in statistics.items():
        if not isinstance(case_statistics, dict):
            raise ValueError(f"Invalid matrix statistics for {case_id}")
        typed_statistics[str(case_id)] = {
            str(key): value for key, value in case_statistics.items()
        }

    ordered_candidates = sorted(
        typed_candidates,
        key=lambda row: (
            not bool(row["scientific_valid"]),
            _ranking_float(row, "median_search_wall_seconds"),
            _ranking_float(row, "median_estimated_compute_cost_usd"),
            str(row["case_id"]),
        ),
    )
    campaign_complete = not str(summary["status"]).startswith("blocked_")
    lines = [
        f"# {CAMPAIGN_ID} measured matrix",
        "",
        f"Status: **{summary['status']}**",
        f"Campaign complete: **{'YES' if campaign_complete else 'NO'}**",
        f"Screening samples: {summary['screening_samples']}",
        f"Stress samples: {summary['stress_samples']}",
        f"Submitted remote samples: {summary['remote_function_inputs_submitted']}",
        f"Promoted sharded cases: {selected_text}",
        "",
        "B1 is the scientific and performance oracle. B0 remains descriptive.",
        "No additional diagnostic samples are submitted automatically.",
        "",
        "## Screening candidate ranking",
        "",
        "| Rank | Case | Scientific | Stable | Search median (s) | "
        "Search min-max (s) | Search MAD (s) | vs B1 | Sample median (s) | "
        "Cost median (USD) | 20% gate | Selected |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- | --- |",
    ]
    for rank, candidate in enumerate(ordered_candidates, start=1):
        case_id = str(candidate["case_id"])
        minimum = _case_statistic_float(
            typed_statistics, case_id, "search_wall_seconds", "minimum"
        )
        maximum = _case_statistic_float(
            typed_statistics, case_id, "search_wall_seconds", "maximum"
        )
        mad = _case_statistic_float(
            typed_statistics,
            case_id,
            "search_wall_seconds",
            "median_absolute_deviation",
        )
        lines.append(
            f"| {rank} | {case_id} | "
            f"{'PASS' if candidate['scientific_valid'] is True else 'FAIL'} | "
            f"{'PASS' if candidate['stable_within_10_percent'] is True else 'FAIL'} | "
            f"{_ranking_float(candidate, 'median_search_wall_seconds'):.3f} | "
            f"{minimum:.3f}-{maximum:.3f} | {mad:.3f} | "
            f"{_ranking_float(candidate, 'search_improvement_vs_B1'):.1%} | "
            f"{_ranking_float(candidate, 'median_sample_wall_seconds'):.3f} | "
            f"{_ranking_float(candidate, 'median_estimated_compute_cost_usd'):.6f} | "
            f"{'PASS' if candidate['meaningful_20_percent_success'] is True else 'FAIL'} | "
            f"{'yes' if case_id in selected_ids else 'no'} |"
        )

    b1_search = _case_statistic_float(
        typed_statistics, "B1", "search_wall_seconds", "median"
    )
    b1_sample = _case_statistic_float(
        typed_statistics, "B1", "sample_wall_seconds", "median"
    )
    b1_remote = _case_statistic_float(
        typed_statistics, "B1", "remote_call_wall_seconds", "median"
    )
    lines.extend([
        "",
        "## Oracle timing",
        "",
        f"B1 median Search Wall Time: {b1_search:.3f} s",
        f"B1 median Sample Wall Time: {b1_sample:.3f} s",
        f"B1 median Remote Call Wall Time: {b1_remote:.3f} s",
        "",
        "The ranking uses median Search Wall Time. Variation is the three-sample "
        "range divided by the median; MAD is reported instead of p95.",
        "",
    ])
    return "\n".join(lines)


def _campaign_results_parquet(
    matrix_results: list[dict[str, Any]] | None = None,
) -> bytes:
    """Combine available scan and search sample indexes at campaign root."""
    import polars as pl

    tables: list[pl.DataFrame] = []
    operation_paths = (
        (
            "storage-scan",
            f"benchmarks/{CAMPAIGN_ID}/storage-scans/results.parquet",
        ),
        ("smoke", f"benchmarks/{CAMPAIGN_ID}/search/smoke/results.parquet"),
    )
    for campaign_stage, relative_path in operation_paths:
        try:
            data = _read_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path)
        except FileNotFoundError:
            continue
        tables.append(
            pl.read_parquet(io.BytesIO(data)).with_columns(
                pl.lit(campaign_stage).alias("campaign_stage")
            )
        )

    if matrix_results is None:
        matrix_paths = (
            f"benchmarks/{CAMPAIGN_ID}/search/matrix/all-results.parquet",
            f"benchmarks/{CAMPAIGN_ID}/search/matrix/screening-results.parquet",
        )
        for relative_path in matrix_paths:
            try:
                data = _read_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path)
            except FileNotFoundError:
                continue
            tables.append(
                pl.read_parquet(io.BytesIO(data)).with_columns(
                    pl.lit("matrix").alias("campaign_stage")
                )
            )
            break
    else:
        tables.append(
            pl.read_parquet(
                io.BytesIO(_search_results_parquet(matrix_results))
            ).with_columns(pl.lit("matrix").alias("campaign_stage"))
        )

    if not tables:
        raise ValueError("No completed scan or search samples are available")
    combined = pl.concat(tables, how="diagonal_relaxed")
    sort_columns = [
        column
        for column in (
            "campaign_stage",
            "sample_kind",
            "query_id",
            "case_id",
            "block_index",
            "partition_index",
            "pass",
        )
        if column in combined.columns
    ]
    buffer = io.BytesIO()
    combined.sort(sort_columns).write_parquet(buffer)
    return buffer.getvalue()


def _publish_campaign_progress(
    *,
    stage: str,
    status: str,
    details: list[str],
) -> None:
    """Publish a clearly incomplete campaign snapshot after an operation."""
    campaign_root = f"benchmarks/{CAMPAIGN_ID}"
    try:
        matrix_summary = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{campaign_root}/search/matrix/summary.json",
        )
    except FileNotFoundError:
        pass
    else:
        if matrix_summary.get("campaign_id") == CAMPAIGN_ID:
            return
    results = _campaign_results_parquet()
    summary = "\n".join([
        f"# {CAMPAIGN_ID}",
        "",
        f"Current stage: **{stage}**",
        f"Status: **{status}**",
        "Campaign complete: **NO**",
        "",
        *details,
        "",
        "This is a progress snapshot, not a final sharding recommendation.",
        "",
    ]).encode()
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/results.parquet",
        results,
    )
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/summary.md",
        summary,
    )


def _publish_campaign_matrix_snapshot(
    summary: dict[str, object],
    rankings: dict[str, object],
    matrix_results: list[dict[str, Any]] | None = None,
) -> None:
    """Publish the campaign-wide sample index and measured matrix report."""
    campaign_root = f"benchmarks/{CAMPAIGN_ID}"
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/results.parquet",
        _campaign_results_parquet(matrix_results),
    )
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/summary.md",
        _matrix_summary_markdown(summary, rankings).encode(),
    )


def _publish_matrix_artifacts(
    operation_root: str,
    artifacts: dict[str, bytes],
) -> None:
    """Upload a complete set of small matrix index artifacts."""
    for relative_path, data in artifacts.items():
        _upload_volume_bytes(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/{relative_path}",
            data,
        )


def _finalize_matrix_operation(
    operation_root: str,
    operation_identity: str,
    artifacts: dict[str, bytes],
) -> None:
    """Publish all matrix indexes, then its completion marker last."""
    _publish_matrix_artifacts(operation_root, artifacts)
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{operation_root}/done.json",
        _json_bytes({
            "schema_version": DONE_SCHEMA_VERSION,
            "status": "complete",
            "identity": operation_identity,
            "completed_at": _utc_now(),
            "artifacts": [
                _volume_artifact_record(relative_path, data)
                for relative_path, data in artifacts.items()
            ],
        }),
    )


def _completed_matrix_is_valid(
    operation_root: str,
    operation_identity: str,
) -> dict[str, Any] | None:
    """Validate a complete matrix index and each referenced sample marker."""
    if not _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        f"{operation_root}/done.json",
        expected_identity=operation_identity,
    ):
        return None
    summary = _read_volume_json(
        BENCHMARK_OUTPUT_VOLUME,
        f"{operation_root}/summary.json",
    )
    samples = summary.get("samples")
    if not isinstance(samples, list):
        return None
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        result_path = sample.get("result_path")
        sample_identity = sample.get("sample_identity")
        if not isinstance(result_path, str) or not isinstance(sample_identity, str):
            return None
        if not _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{result_path}/done.json",
            expected_identity=sample_identity,
        ):
            return None
    return summary


def _submit_search_matrix() -> dict[str, object]:
    """Run the fixed screening matrix and conditionally run its stress matrix."""
    _ensure_campaign_plan_client()
    _validate_screening_block_orders()
    manifest_relpath = f"{APP_INFO.profile_relpath}/manifest.json"
    manifest_bytes = _read_volume_bytes(SHARDED_MSA_DB_VOLUME, manifest_relpath)
    manifest = orjson.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Profile manifest must be a JSON object")
    _validate_profile_manifest(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    profile_scientific_identity = _profile_scientific_identity(manifest)
    _require_passing_smoke_gate(
        manifest_sha256,
        profile_scientific_identity,
    )
    operation_root = f"benchmarks/{CAMPAIGN_ID}/search/matrix"
    operation_identity = _matrix_operation_identity(manifest_sha256)
    completed = _completed_matrix_is_valid(operation_root, operation_identity)
    if completed is not None:
        completed_rankings = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/rankings.json",
        )
        _publish_campaign_matrix_snapshot(completed, completed_rankings)
        return completed | {
            "status": "reused",
            "remote_function_inputs_submitted": 0,
        }

    screening_results: list[dict[str, Any]] = []
    submitted_inputs = 0
    for block_index, order in enumerate(SCREENING_BLOCK_ORDERS, start=1):
        for case_id in order:
            result, submitted = _run_or_reuse_matrix_sample(
                profile_scientific_identity,
                SCREENING_QUERY,
                _search_case(case_id),
                _matrix_sample_id("screen", case_id, block_index),
                sample_kind="screening",
                block_index=block_index,
            )
            screening_results.append(result)
            submitted_inputs += submitted

    sharded_case_ids = tuple(
        case_id for case_id in MATRIX_CASE_IDS if case_id.startswith("S")
    )
    screening_comparisons = _matrix_comparisons(
        screening_results,
        sharded_case_ids,
    )
    rankings = _rank_screening_cases(screening_results, screening_comparisons)
    screening_summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "operation_identity": operation_identity,
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "status": rankings["status"],
        "sample_count": len(screening_results),
        "remote_function_inputs_submitted": submitted_inputs,
        "samples": _matrix_sample_records(screening_results),
        "completed_at": _utc_now(),
    }
    screening_artifacts = {
        "screening-results.parquet": _search_results_parquet(screening_results),
        "screening-comparisons.json": _json_bytes(screening_comparisons),
        "rankings.json": _json_bytes(rankings),
        "screening-summary.json": _json_bytes(screening_summary),
    }
    _publish_matrix_artifacts(operation_root, screening_artifacts)

    ranking_status = str(rankings["status"])
    if ranking_status.startswith("blocked_"):
        summary: dict[str, object] = {
            "schema_version": 1,
            "status": ranking_status,
            "campaign_id": CAMPAIGN_ID,
            "operation": "search",
            "mode": "matrix",
            "operation_identity": operation_identity,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "screening_samples": len(screening_results),
            "stress_samples": 0,
            "selected_case_ids": [],
            "samples": _matrix_sample_records(screening_results),
            "remote_function_inputs_submitted": submitted_inputs,
            "completed_at": _utc_now(),
            "requires_human_review": True,
        }
        blocked_artifacts = screening_artifacts | {
            "summary.json": _json_bytes(summary),
            "summary.md": _matrix_summary_markdown(summary, rankings).encode(),
        }
        _publish_matrix_artifacts(operation_root, blocked_artifacts)
        _publish_campaign_matrix_snapshot(summary, rankings, screening_results)
        return summary

    selected = rankings.get("selected_case_ids")
    if not isinstance(selected, list):
        raise ValueError("Screening rankings have invalid selected cases")
    if len(selected) != 2:
        summary = {
            "schema_version": 1,
            "status": ranking_status,
            "campaign_id": CAMPAIGN_ID,
            "operation": "search",
            "mode": "matrix",
            "operation_identity": operation_identity,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "screening_samples": len(screening_results),
            "stress_samples": 0,
            "selected_case_ids": selected,
            "samples": _matrix_sample_records(screening_results),
            "remote_function_inputs_submitted": submitted_inputs,
            "completed_at": _utc_now(),
            "requires_human_review": False,
        }
        final_artifacts = screening_artifacts | {
            "summary.json": _json_bytes(summary),
            "summary.md": _matrix_summary_markdown(summary, rankings).encode(),
        }
        _finalize_matrix_operation(
            operation_root,
            operation_identity,
            final_artifacts,
        )
        _publish_campaign_matrix_snapshot(summary, rankings, screening_results)
        return summary

    selected_pair = (str(selected[0]), str(selected[1]))
    stress_results: list[dict[str, Any]] = []
    for block_index, order in enumerate(
        _stress_block_orders(selected_pair),
        start=1,
    ):
        for case_id in order:
            result, submitted = _run_or_reuse_matrix_sample(
                profile_scientific_identity,
                STRESS_QUERY,
                _search_case(case_id),
                _matrix_sample_id("stress", case_id, block_index),
                sample_kind="stress",
                block_index=block_index,
            )
            stress_results.append(result)
            submitted_inputs += submitted

    stress_comparisons = _matrix_comparisons(stress_results, selected_pair)
    stress_gate_passed = all(
        comparison["passed"] is True
        for comparison in stress_comparisons.values()
        if comparison["is_gate"] is True
    )
    stress_statistics = _case_performance_statistics(stress_results)
    final_status = "complete" if stress_gate_passed else "complete_stress_gate_failed"
    all_results = [*screening_results, *stress_results]
    summary = {
        "schema_version": 1,
        "status": final_status,
        "campaign_id": CAMPAIGN_ID,
        "operation": "search",
        "mode": "matrix",
        "operation_identity": operation_identity,
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "profile_manifest_sha256": manifest_sha256,
        "screening_samples": len(screening_results),
        "stress_samples": len(stress_results),
        "selected_case_ids": list(selected_pair),
        "stress_scientific_gate_passed": stress_gate_passed,
        "stress_statistics": stress_statistics,
        "samples": _matrix_sample_records(all_results),
        "remote_function_inputs_submitted": submitted_inputs,
        "completed_at": _utc_now(),
        "requires_human_review": not stress_gate_passed,
    }
    final_artifacts = screening_artifacts | {
        "stress-results.parquet": _search_results_parquet(stress_results),
        "stress-comparisons.json": _json_bytes(stress_comparisons),
        "stress-statistics.json": _json_bytes(stress_statistics),
        "all-results.parquet": _search_results_parquet(all_results),
        "summary.json": _json_bytes(summary),
        "summary.md": _matrix_summary_markdown(summary, rankings).encode(),
    }
    _finalize_matrix_operation(
        operation_root,
        operation_identity,
        final_artifacts,
    )
    _publish_campaign_matrix_snapshot(summary, rankings, all_results)
    return summary


@app.local_entrypoint()
def submit_alphafold3_msa_task(
    operation: str = "prepare",
    submit: bool = False,
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
    search_mode: str = "smoke",
) -> None:
    """Plan or submit one isolated AlphaFold 3 MSA benchmark operation.

    Args:
        operation: Operation to plan or run: ``prepare``, ``scan``, or ``search``.
        submit: Submit the displayed remote work. Defaults to false, which only
            prints the plan and incurs no Modal compute work.
        seqkit_threads: SeqKit thread count for profile preparation, default 8.
        search_mode: Fixed search workload: ``smoke`` or ``matrix``.
    """
    if operation == "prepare":
        plan = _build_prepare_plan(seqkit_threads)
    elif operation == "scan":
        plan = _build_scan_plan()
    elif operation == "search":
        plan = _build_search_plan(search_mode)
    else:
        raise ValueError("operation must be 'prepare', 'scan', or 'search'")
    print(_json_bytes(plan).decode(), end="")
    if not submit:
        print("🧬 Plan only; no Modal function was submitted.")
        return
    if operation == "prepare":
        print("🧬 Submitting one small-BFD profile preparation function...")
        result = prepare_small_bfd_profile.remote(seqkit_threads=seqkit_threads)
    elif operation == "scan":
        print("🧬 Submitting the sequential Volume scan matrix...")
        result = _submit_scan_matrix()
    else:
        if search_mode == "smoke":
            print("🧬 Submitting the sequential small-BFD search smoke...")
            result = _submit_search_smoke()
        else:
            print("🧬 Submitting the one-shot measured small-BFD matrix...")
            result = _submit_search_matrix()
    print(_json_bytes(result).decode(), end="")
