"""Native, Modal-independent primitives for deterministic FASTA sharding."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from biomodals.app.fold.alphafold3.artifacts import (
    append_log,
    load_json_object,
    require_regular_file,
    sha256_file,
    write_json_atomic,
)
from biomodals.app.fold.alphafold3.profiles import (
    ORDINAL_SHUFFLER_PREFETCH_BYTES,
    ORDINAL_SHUFFLER_PREFETCH_RECORDS,
    ORDINAL_SHUFFLER_SOURCE_SHA256,
    ORDINAL_SHUFFLER_VERSION,
    RECORD_MULTISET_SOURCE_SHA256,
    RECORD_MULTISET_VERSION,
    record_multiset_identity,
)

SHUFFLER_SCRATCH_HEADROOM_BYTES = 1024 * 1024 * 1024

NATIVE_SOURCE_DIR_ENV = "BIOMODALS_AF3_NATIVE_SOURCE_DIR"
CONTAINER_NATIVE_SOURCE_DIR = Path("/opt/biomodals/alphafold3/native")
_LOCAL_NATIVE_SOURCE_DIR = Path(__file__).parent / "native"


@dataclass(frozen=True)
class NativeTool:
    """Identity and source asset for one pinned native helper."""

    version: str
    source_filename: str
    source_sha256: str


@dataclass(frozen=True)
class FileVerification:
    """Evidence that a file matches an expected byte identity."""

    size_bytes: int
    sha256: str


ORDINAL_SHUFFLER = NativeTool(
    version=ORDINAL_SHUFFLER_VERSION,
    source_filename="af3-fasta-two-pass.c",
    source_sha256=ORDINAL_SHUFFLER_SOURCE_SHA256,
)
RECORD_MULTISET_VALIDATOR = NativeTool(
    version=RECORD_MULTISET_VERSION,
    source_filename="af3-fasta-record-multiset.c",
    source_sha256=RECORD_MULTISET_SOURCE_SHA256,
)


def verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> FileVerification:
    """Verify one file's size and digest and return measured evidence."""
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise ValueError("expected_size must be a positive integer")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    require_regular_file(path)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"File size mismatch for {path}: {actual_size} != {expected_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"File SHA-256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
        )
    return FileVerification(
        size_bytes=actual_size,
        sha256=actual_sha256,
    )


def _native_source_path(tool: NativeTool) -> Path:
    source_dir = Path(os.environ.get(NATIVE_SOURCE_DIR_ENV, _LOCAL_NATIVE_SOURCE_DIR))
    source_path = source_dir / tool.source_filename
    require_regular_file(source_path)
    if sha256_file(source_path) != tool.source_sha256:
        raise RuntimeError(
            f"Pinned native source digest does not match for {source_path}"
        )
    return source_path


def require_executable(name: str) -> str:
    """Resolve a fixed executable name to an absolute path."""
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"Required executable is not installed: {name}")
    return str(Path(executable).resolve())


def _compile_native_tool(
    tool: NativeTool,
    scratch_root: Path,
    log_path: Path,
) -> Path:
    compiler = require_executable("cc")
    source_path = _native_source_path(tool)
    executable_path = scratch_root / tool.version
    compile_argv = [
        compiler,
        "-std=c11",
        "-O3",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source_path),
        "-o",
        str(executable_path),
    ]
    append_log(log_path, f"Running command: {shlex.join(compile_argv)}")
    with log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            compile_argv,
            check=False,
            stdout=log,
            stderr=log,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, compile_argv)
    require_regular_file(executable_path)
    if not os.access(executable_path, os.X_OK):
        raise PermissionError(f"Native helper is not executable: {executable_path}")
    version = subprocess.run(  # noqa: S603
        [str(executable_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != tool.version:
        raise RuntimeError(f"Expected native helper {tool.version}, got {version!r}")
    return executable_path


def compile_ordinal_shuffler(scratch_root: Path, log_path: Path) -> Path:
    """Compile the pinned occurrence-preserving FASTA shuffler."""
    return _compile_native_tool(ORDINAL_SHUFFLER, scratch_root, log_path)


def compile_record_multiset_validator(
    scratch_root: Path,
    log_path: Path,
) -> Path:
    """Compile the pinned composable FASTA-record validator."""
    return _compile_native_tool(RECORD_MULTISET_VALIDATOR, scratch_root, log_path)


def record_multiset_signature(report: dict[str, Any]) -> dict[str, object]:
    """Validate one native report and return its composable signature."""
    expected_strings = record_multiset_identity()
    for field in ("version", "canonicalization", "digest", "aggregate"):
        if report.get(field) != expected_strings[field]:
            raise ValueError(f"Unexpected record-multiset {field}")
    for field in ("files", "threads", "records", "header_bytes", "sequence_bytes"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Record-multiset {field} must be an integer")
        if value < (1 if field in {"files", "threads", "records"} else 0):
            raise ValueError(f"Record-multiset {field} is outside its range")
    if report["threads"] > report["files"]:
        raise ValueError("Record-multiset threads exceed input files")
    signature: dict[str, object] = {
        "records": report["records"],
        "header_bytes": report["header_bytes"],
        "sequence_bytes": report["sequence_bytes"],
    }
    for field in (
        "sum_sha256_lanes",
        "xor_sha256_lanes",
        "sum_square_sha256_lanes",
    ):
        values = report.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{16}", value) is None
                for value in values
            )
        ):
            raise ValueError(f"Invalid record-multiset {field}")
        signature[field] = values
    return signature


def scan_record_multiset(
    executable: Path,
    input_paths: tuple[Path, ...],
    output_path: Path,
    log_path: Path,
    *,
    threads: int,
) -> dict[str, object]:
    """Scan one file set with the native composable multiset validator."""
    if not input_paths:
        raise ValueError("Record-multiset validation requires at least one input")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")
    selected_threads = min(threads, len(input_paths))
    input_bytes = 0
    for path in input_paths:
        require_regular_file(path)
        input_bytes += path.stat().st_size
    argv = [
        str(executable),
        str(selected_threads),
        str(output_path),
        *(str(path) for path in input_paths),
    ]
    start_message = (
        "Running record-multiset helper with "
        f"{selected_threads} threads over {len(input_paths)} files "
        f"({input_bytes} bytes)"
    )
    append_log(log_path, start_message)
    print(f"🧬 validator {start_message}", flush=True)
    with log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            argv,
            check=False,
            stdout=log,
            stderr=log,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, argv)
    require_regular_file(output_path)
    report = load_json_object(output_path)
    if report.get("files") != len(input_paths):
        raise ValueError("Record-multiset helper reported the wrong file count")
    if report.get("threads") != selected_threads:
        raise ValueError("Record-multiset helper reported the wrong thread count")
    record_multiset_signature(report)
    completed_message = "Completed record-multiset helper"
    append_log(log_path, completed_message)
    print(f"🧬 validator {completed_message}", flush=True)
    return {
        "input_bytes": input_bytes,
        "report": report,
    }


def required_ordinal_shuffler_scratch_bytes(
    source_size: int,
    record_count: int,
    *,
    headroom_bytes: int = SHUFFLER_SCRATCH_HEADROOM_BYTES,
) -> int:
    """Return bytes needed for the staged source, shuffle, index, and headroom."""
    for name, value in (
        ("source_size", source_size),
        ("record_count", record_count),
        ("headroom_bytes", headroom_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    index_size = 48 + (record_count + 1) * 8
    return source_size + source_size + 1 + index_size + headroom_bytes


def append_diagnostic_file(source_path: Path, log_path: Path) -> None:
    """Copy one command's raw diagnostics into a durable operation log."""
    require_regular_file(source_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source, log_path.open("ab") as log:
        shutil.copyfileobj(source, log, length=1024 * 1024)


def shuffle_fasta_occurrences(
    source_path: Path,
    shuffled_path: Path,
    scratch_root: Path,
    diagnostics_path: Path,
    evidence_path: Path,
    log_path: Path,
    *,
    expected_records: int,
    seed: int,
    worker_threads: int,
) -> Path:
    """Shuffle all FASTA occurrences through a verified local source copy."""
    for name, value in (
        ("expected_records", expected_records),
        ("worker_threads", worker_threads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= (2**64 - 1)
    ):
        raise ValueError("seed must be an unsigned 64-bit integer")
    require_regular_file(source_path)
    executable = compile_ordinal_shuffler(scratch_root, log_path)
    staged_source_path = scratch_root / "source.fasta"
    index_path = scratch_root / "occurrence-offsets.bin"
    argv = [
        str(executable),
        "--source",
        str(source_path),
        "--staged-source",
        str(staged_source_path),
        "--output",
        str(shuffled_path),
        "--index",
        str(index_path),
        "--expected-records",
        str(expected_records),
        "--seed",
        str(seed),
        "--threads",
        str(worker_threads),
        "--prefetch-records",
        str(ORDINAL_SHUFFLER_PREFETCH_RECORDS),
        "--prefetch-bytes",
        str(ORDINAL_SHUFFLER_PREFETCH_BYTES),
    ]
    append_log(log_path, f"Running command: {shlex.join(argv)}")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("xb") as diagnostics:
        process = subprocess.Popen(  # noqa: S603
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("Native shuffler did not expose output streams")
        for line in iter(process.stderr.readline, b""):
            diagnostics.write(line)
            diagnostics.flush()
            print(f"🧬 shuffler {line.decode(errors='replace')}", end="", flush=True)
        process.stderr.close()
        report_bytes = process.stdout.read()
        process.stdout.close()
        returncode = process.wait()
    if returncode != 0:
        append_diagnostic_file(diagnostics_path, log_path)
        raise subprocess.CalledProcessError(returncode, argv)
    diagnostics_path.unlink()
    try:
        report = orjson.loads(report_bytes)
    except orjson.JSONDecodeError as exc:
        raise ValueError("Native shuffler returned invalid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("Native shuffler report must be a JSON object")
    source_size = source_path.stat().st_size
    with source_path.open("rb") as source:
        source.seek(-1, os.SEEK_END)
        output_size = source_size + (source.read(1) != b"\n")
    expected_evidence = {
        "schema_version": 1,
        "version": ORDINAL_SHUFFLER_VERSION,
        "record_count": expected_records,
        "source_size_bytes": source_size,
        "staged_source_size_bytes": source_size,
        "output_size_bytes": output_size,
        "seed": seed,
        "threads": worker_threads,
        "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
        "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
        "random_read_source": "container-local-staged-copy",
    }
    for key, expected in expected_evidence.items():
        if report.get(key) != expected:
            raise ValueError(
                f"Native shuffler field {key!r} is {report.get(key)!r}, "
                f"expected {expected!r}"
            )
    offset_index_size = 48 + (expected_records + 1) * 8
    permutation_size = expected_records * 4
    if report.get("offset_index_size_bytes") != offset_index_size:
        raise ValueError("Native shuffler offset index has an unexpected size")
    if report.get("permutation_size_bytes") != permutation_size:
        raise ValueError("Native shuffler permutation has an unexpected size")
    peak_batch_bytes = report.get("peak_batch_bytes")
    if (
        isinstance(peak_batch_bytes, bool)
        or not isinstance(peak_batch_bytes, int)
        or peak_batch_bytes < 0
    ):
        raise ValueError("Native shuffler peak batch size is invalid")
    require_regular_file(shuffled_path)
    if shuffled_path.stat().st_size != output_size:
        raise ValueError("Native shuffler output size is not normalized")
    require_regular_file(staged_source_path)
    if staged_source_path.stat().st_size != source_size:
        raise ValueError("Native shuffler staged source size does not match source")
    published_evidence = expected_evidence | {
        "offset_index_size_bytes": offset_index_size,
        "permutation_size_bytes": permutation_size,
        "peak_batch_bytes": peak_batch_bytes,
        "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
        "index_identity": "uint64-source-occurrence-offsets-v1",
        "permutation_identity": "splitmix64-fisher-yates-u32-v1",
        "staging_identity": "first-pass-tee-to-container-local-v1",
        "read_identity": "bounded-concurrent-local-pread-ordered-write-v2",
    }
    write_json_atomic(evidence_path, published_evidence)
    return staged_source_path
