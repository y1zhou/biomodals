"""BoltzGen-owned publication validation and cross-run output claims."""

from __future__ import annotations

import time
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import orjson

from biomodals.helper.app_run import AppRunLayout

COLLECTION_PUBLICATION_SCHEMA_VERSION = 1
TASK_PUBLICATION_SCHEMA_VERSION = 1
_TASK_PUBLICATION_PATH = PurePosixPath(".biomodals") / "task.json"
_FINAL_PDF_PATH = PurePosixPath("final_ranked_designs") / "results_overview.pdf"
_HASH_CHUNK_BYTES = 1024 * 1024
_OUTPUT_CLAIM_PREFIX = "boltzgen-output"


def boltzgen_run_root(
    output_root: str | Path,
    run_name: str,
    run_id: str,
) -> Path:
    """Return one independently reusable stochastic run directory."""
    return AppRunLayout.from_run_root(Path(output_root) / run_name).outputs_dir / run_id


def is_boltzgen_run_complete(
    run_dir: Path,
    *,
    task_fingerprint: str | None = None,
) -> bool:
    """Return whether a BoltzGen run has its established final publication."""
    final_pdf = run_dir.joinpath(*_FINAL_PDF_PATH.parts)
    if not run_dir.is_dir() or not final_pdf.is_file():
        return False
    if task_fingerprint is None:
        return True
    marker = run_dir.joinpath(*_TASK_PUBLICATION_PATH.parts)
    if not marker.is_file():
        return False
    try:
        value: Any = orjson.loads(marker.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return False
    if not (
        isinstance(value, dict)
        and value.get("schema_version") == TASK_PUBLICATION_SCHEMA_VERSION
        and value.get("status") == "complete"
        and value.get("task_fingerprint") == task_fingerprint
        and value.get("artifact_path") == _FINAL_PDF_PATH.as_posix()
        and isinstance(value.get("artifact_size_bytes"), int)
        and not isinstance(value.get("artifact_size_bytes"), bool)
        and isinstance(value.get("artifact_sha256"), str)
    ):
        return False
    size_bytes, digest = _hash_file(final_pdf)
    return (
        size_bytes == value["artifact_size_bytes"]
        and digest == value["artifact_sha256"]
    )


def write_boltzgen_task_publication(
    run_dir: Path,
    *,
    task_fingerprint: str,
) -> None:
    """Publish fingerprint-bound evidence for one completed design Task."""
    if len(task_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in task_fingerprint
    ):
        raise ValueError("BoltzGen Task fingerprint must be lowercase SHA-256")
    final_pdf = run_dir.joinpath(*_FINAL_PDF_PATH.parts)
    if not final_pdf.is_file():
        raise RuntimeError("BoltzGen returned without its final publication")
    size_bytes, digest = _hash_file(final_pdf)
    marker = run_dir.joinpath(*_TASK_PUBLICATION_PATH.parts)
    marker.parent.mkdir(parents=True, exist_ok=True)
    content = orjson.dumps(
        {
            "schema_version": TASK_PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "task_fingerprint": task_fingerprint,
            "artifact_path": _FINAL_PDF_PATH.as_posix(),
            "artifact_size_bytes": size_bytes,
            "artifact_sha256": digest,
        },
        option=orjson.OPT_SORT_KEYS,
    )
    temporary = marker.with_suffix(f".{time.time_ns()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(marker)


def boltzgen_output_claim_key(
    run_dir: str | Path,
    *,
    output_root: str | Path,
) -> str:
    """Return one stable workload-owned claim key for an output directory."""
    root = Path(output_root).resolve()
    relative = Path(run_dir).resolve().relative_to(root).as_posix()
    digest = sha256(relative.encode()).hexdigest()
    return f"{_OUTPUT_CLAIM_PREFIX}:{digest}"


def collection_publication_path(
    *,
    run_name: str,
    workload_plan_fingerprint: str,
) -> PurePosixPath:
    """Return the stable result marker shared by compatible Successor Runs."""
    return PurePosixPath(run_name) / "results" / f"{workload_plan_fingerprint}.json"


def write_collection_publication(
    output_root: str | Path,
    relative_path: str | PurePosixPath,
    publication: dict[str, object],
) -> dict[str, object]:
    """Atomically publish the final collection record."""
    path = _contained_path(output_root, relative_path)
    record = {
        "schema_version": COLLECTION_PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        **publication,
    }
    content = orjson.dumps(record, option=orjson.OPT_SORT_KEYS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{time.time_ns()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return record


def load_collection_publication(
    output_root: str | Path,
    relative_path: str | PurePosixPath,
) -> dict[str, object] | None:
    """Load one structurally valid final collection publication."""
    path = _contained_path(output_root, relative_path)
    if not path.is_file():
        return None
    try:
        value: Any = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != COLLECTION_PUBLICATION_SCHEMA_VERSION
        or value.get("status") != "complete"
        or not isinstance(value.get("run_name"), str)
        or not isinstance(value.get("run_ids"), list)
        or not all(isinstance(item, str) and item for item in value["run_ids"])
    ):
        return None
    archive_path = value.get("archive_path")
    if archive_path is not None:
        if not isinstance(archive_path, str):
            return None
        archive = _contained_path(output_root, archive_path)
        if not archive.is_file() or archive.stat().st_size != value.get(
            "archive_size_bytes"
        ):
            return None
    return value


def _contained_path(
    root: str | Path,
    relative_path: str | PurePosixPath,
) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("BoltzGen publication path must be relative and contained")
    root_path = Path(root).resolve()
    path = root_path.joinpath(*relative.parts).resolve()
    path.relative_to(root_path)
    return path


def _hash_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()
