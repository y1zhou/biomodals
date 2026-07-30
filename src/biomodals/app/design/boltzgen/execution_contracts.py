"""BoltzGen-owned publication validation and cross-run output claims."""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath
from typing import Any

import orjson

from biomodals.helper.app_run import AppRunLayout
from biomodals.helper.shell import warmup_directory

COLLECTION_PUBLICATION_SCHEMA_VERSION = 1
_CLAIM_OWNER_FILENAME = "owner"


def boltzgen_run_root(
    output_root: str | Path,
    run_name: str,
    run_id: str,
) -> Path:
    """Return one independently reusable stochastic run directory."""
    return AppRunLayout.from_run_root(Path(output_root) / run_name).outputs_dir / run_id


def is_boltzgen_run_complete(run_dir: Path) -> bool:
    """Return whether a BoltzGen run directory contains the final outputs."""
    final_dir = run_dir / "final_ranked_designs"
    return (
        run_dir.exists()
        and final_dir.exists()
        and (final_dir / "results_overview.pdf").is_file()
    )


def acquire_output_claim(
    run_dir: Path,
    *,
    owner: str,
    replace_owner: str | None = None,
) -> Path:
    """Acquire one filesystem publication claim without using it as a queue."""
    if not owner:
        raise ValueError("BoltzGen output claim owner cannot be empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    claim_dir = run_dir / ".lock"
    owner_path = claim_dir / _CLAIM_OWNER_FILENAME
    try:
        claim_dir.mkdir()
        owner_path.write_text(owner, encoding="utf-8")
        return claim_dir
    except FileExistsError:
        try:
            existing_owner = owner_path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(
                f"BoltzGen output is claimed without readable ownership: {run_dir}"
            ) from error
        if existing_owner == owner:
            warmup_directory(run_dir)
            return claim_dir
        if replace_owner is None or existing_owner != replace_owner:
            raise RuntimeError(
                f"BoltzGen output is already claimed by another Provider Call: "
                f"{run_dir}"
            ) from None
        temporary_owner = owner_path.with_suffix(".next")
        temporary_owner.write_text(owner, encoding="utf-8")
        temporary_owner.replace(owner_path)
        warmup_directory(run_dir)
        return claim_dir


def release_output_claim(claim_dir: Path, *, owner: str) -> None:
    """Remove a claim only after its owner has published complete output."""
    owner_path = claim_dir / _CLAIM_OWNER_FILENAME
    if owner_path.read_text(encoding="utf-8") != owner:
        raise RuntimeError("BoltzGen output claim ownership changed")
    owner_path.unlink()
    claim_dir.rmdir()


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
