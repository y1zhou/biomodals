"""PPIFlow-local candidate manifest helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath

import orjson
import polars as pl

from biomodals.helper.app_run import volume_app_output
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import AppOutput, ArtifactFile, ArtifactKind, VolumePath

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "candidate_manifest.parquet"
MANIFEST_OUTPUT_NAME = "candidate_manifest"
MANIFEST_MEDIA_TYPE = "application/vnd.apache.parquet"
MANIFEST_FILE_ROLE = "candidate_manifest"

REQUIRED_COLUMNS = {
    "candidate_id",
    "parent_candidate_id",
    "stage_name",
    "stage_role",
    "operation_mode",
    "candidate_status",
    "source_artifact_id",
    "source_path",
    "derived_path",
    "status_reason",
    "error",
    "files",
}


def initial_candidate_id(
    *,
    stage_name: str,
    source_artifact_id: str,
    source_path: str,
    basename: str | None = None,
) -> str:
    """Return a deterministic id for a source candidate."""
    return _candidate_id({
        "kind": "initial",
        "stage_name": stage_name,
        "source_artifact_id": source_artifact_id,
        "source_path": _posix_path(source_path),
        "basename": _normalized_basename(basename or source_path),
    })


def derived_candidate_id(
    *,
    parent_candidate_id: str,
    stage_name: str,
    operation_mode: str,
    derived_basename: str,
) -> str:
    """Return a deterministic id for a candidate derived from another row."""
    return _candidate_id({
        "kind": "derived",
        "parent_candidate_id": parent_candidate_id,
        "stage_name": stage_name,
        "operation_mode": operation_mode,
        "derived_basename": _normalized_basename(derived_basename),
    })


def stage2_input_candidate_id(index: int) -> str:
    """Return the sequential convenience id used by Stage2Input rows."""
    if index < 1:
        raise ValueError("Stage2Input candidate indices are 1-based")
    return f"stage2_input_{index:06d}"


def candidate_file_record(
    *,
    role: str,
    workflow_path: str | None = None,
    volume_name: str | None = None,
    app_volume_path: str | None = None,
    path: str | None = None,
    media_type: str | None = None,
    size_bytes: int | None = None,
    content_sha256: str | None = None,
    expected: bool = True,
) -> dict[str, object]:
    """Build one nested file record for a candidate manifest row."""
    return {
        "role": role,
        "workflow_path": _optional_posix_path(workflow_path),
        "volume_name": volume_name,
        "app_volume_path": _optional_posix_path(app_volume_path),
        "path": _optional_posix_path(path),
        "media_type": media_type,
        "size_bytes": size_bytes,
        "content_sha256": content_sha256,
        "expected": expected,
    }


def candidate_manifest_row(
    *,
    candidate_id: str,
    stage_name: str,
    stage_role: str,
    operation_mode: str,
    candidate_status: str,
    parent_candidate_id: str | None = None,
    source_artifact_id: str | None = None,
    source_path: str | None = None,
    derived_path: str | None = None,
    status_reason: str | None = None,
    error: str | None = None,
    files: Sequence[Mapping[str, object]] | None = None,
    summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one candidate manifest record."""
    return {
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_candidate_id,
        "stage_name": stage_name,
        "stage_role": stage_role,
        "operation_mode": operation_mode,
        "candidate_status": candidate_status,
        "source_artifact_id": source_artifact_id,
        "source_path": _optional_posix_path(source_path),
        "derived_path": _optional_posix_path(derived_path),
        "status_reason": status_reason,
        "error": error,
        "summary_json": (
            orjson.dumps(summary, option=orjson.OPT_SORT_KEYS).decode("utf-8")
            if summary is not None
            else None
        ),
        "files": [dict(file_record) for file_record in files or ()],
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
    }


def write_manifest(records: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    """Write candidate rows to a Parquet manifest and return the path."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame([dict(record) for record in records])
    validate_manifest_frame(frame)
    frame.write_parquet(manifest_path)
    return manifest_path


def read_manifest(path: str | Path) -> pl.DataFrame:
    """Read and validate a candidate manifest."""
    frame = pl.read_parquet(path)
    validate_manifest_frame(frame)
    return frame


def validate_manifest_frame(frame: pl.DataFrame) -> None:
    """Raise if a frame does not satisfy the PPIFlow manifest minimum shape."""
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"PPIFlow candidate manifest is missing columns: {missing}")
    duplicate_ids = (
        frame
        .group_by("candidate_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("candidate_id")
        .to_list()
    )
    if duplicate_ids:
        raise ValueError(
            f"PPIFlow candidate manifest has duplicate ids: {duplicate_ids}"
        )


def manifest_artifact_output(
    *,
    manifest_path: str | Path,
    mount_root: str,
    volume_name: str,
    stage_name: str,
    row_count: int,
    name: str = MANIFEST_OUTPUT_NAME,
) -> AppOutput:
    """Build a workflow-compatible Parquet manifest output."""
    manifest_path = Path(manifest_path)
    return volume_app_output(
        name=name,
        kind=ArtifactKind.TABLE,
        remote_path=str(manifest_path),
        mount_root=mount_root,
        volume_name=volume_name,
        media_type=MANIFEST_MEDIA_TYPE,
        metadata={
            "stage_name": stage_name,
            "rows": row_count,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        },
        files=[
            ArtifactFile(
                path=manifest_path.name,
                role=MANIFEST_FILE_ROLE,
                media_type=MANIFEST_MEDIA_TYPE,
                size_bytes=manifest_path.stat().st_size,
            )
        ],
    )


def read_manifest_volume_path(
    *,
    storage: VolumePath,
    volume_roots: Mapping[str, str | Path],
) -> pl.DataFrame:
    """Read a manifest from a mounted volume path."""
    volume_root = volume_roots.get(storage.volume_name)
    if volume_root is None:
        raise ValueError(f"Missing mounted volume root for {storage.volume_name!r}")
    return read_manifest(Path(volume_root) / storage.path)


def strict_candidate_join(
    required: pl.DataFrame,
    available: pl.DataFrame,
    *,
    on: str = "candidate_id",
    allow_missing_candidates: bool = False,
) -> pl.DataFrame:
    """Join candidate-keyed tables while failing on missing required rows."""
    if on not in required.columns:
        raise ValueError(f"Required candidate table is missing {on!r}")
    if on not in available.columns:
        raise ValueError(f"Available candidate table is missing {on!r}")

    available_counts = available.group_by(on).len().filter(pl.col("len") > 1)
    if available_counts.height:
        duplicate_ids = available_counts.get_column(on).to_list()
        raise ValueError(
            f"Available candidate table has duplicate ids: {duplicate_ids}"
        )

    required_ids = set(required.get_column(on).to_list())
    available_ids = set(available.get_column(on).to_list())
    missing = sorted(required_ids.difference(available_ids))
    if missing and not allow_missing_candidates:
        raise ValueError(f"Missing required candidate ids: {missing}")

    how = "inner" if allow_missing_candidates else "left"
    return required.join(available, on=on, how=how)


def expected_file_errors(
    records_or_frame: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    volume_roots: Mapping[str, str | Path],
    workflow_volume_name: str,
) -> list[str]:
    """Return missing-file errors for expected files recorded in manifest rows."""
    if isinstance(records_or_frame, pl.DataFrame):
        records = records_or_frame.iter_rows(named=True)
    else:
        records = iter(records_or_frame)

    errors: list[str] = []
    roots = {volume_name: Path(root) for volume_name, root in volume_roots.items()}
    for record in records:
        candidate_id = str(record.get("candidate_id") or "<unknown>")
        for file_record in _iter_file_records(record.get("files")):
            if not bool(file_record.get("expected", True)):
                continue

            volume_name = file_record.get("volume_name")
            path = None
            if file_record.get("workflow_path"):
                volume_name = workflow_volume_name
                path = str(file_record["workflow_path"])
            elif file_record.get("app_volume_path"):
                path = str(file_record["app_volume_path"])
            elif file_record.get("path"):
                path = str(file_record["path"])

            if not volume_name or not path:
                errors.append(f"{candidate_id}: expected file is missing volume/path")
                continue

            volume_root = roots.get(str(volume_name))
            if volume_root is None:
                errors.append(
                    f"{candidate_id}: missing mounted volume root for {volume_name!r}"
                )
                continue

            file_path = _safe_volume_child(volume_root, path)
            if not file_path.is_file():
                errors.append(
                    f"{candidate_id}: missing expected file {volume_name}:{path}"
                )
                continue

            expected_size = file_record.get("size_bytes")
            if expected_size is not None and file_path.stat().st_size != expected_size:
                errors.append(
                    f"{candidate_id}: expected file {volume_name}:{path} has size "
                    f"{file_path.stat().st_size}, expected {expected_size}"
                )
    return errors


def reusable_completed_candidate_ids(
    records_or_frame: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    volume_roots: Mapping[str, str | Path],
    workflow_volume_name: str,
) -> set[str]:
    """Return completed candidate ids whose expected files are still available."""
    if isinstance(records_or_frame, pl.DataFrame):
        records = list(records_or_frame.iter_rows(named=True))
    else:
        records = [dict(record) for record in records_or_frame]

    reusable = set()
    for record in records:
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id or record.get("candidate_status") != "succeeded":
            continue
        if expected_file_errors(
            [record],
            volume_roots=volume_roots,
            workflow_volume_name=workflow_volume_name,
        ):
            continue
        reusable.add(candidate_id)
    return reusable


def _candidate_id(payload: Mapping[str, object]) -> str:
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return f"cand_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _normalized_basename(path: str) -> str:
    name = PurePosixPath(_posix_path(path)).name
    return sanitize_filename(name)


def _optional_posix_path(path: str | None) -> str | None:
    return _posix_path(path) if path else None


def _posix_path(path: str) -> str:
    return PurePosixPath(str(path).replace("\\", "/")).as_posix()


def _iter_file_records(value: object) -> Iterable[Mapping[str, object]]:
    if value is None:
        return ()
    if isinstance(value, pl.Series):
        value = value.to_list()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Manifest row files must be a list of records")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _safe_volume_child(root: Path, path: str) -> Path:
    relative = PurePosixPath(path)
    if (
        path == ""
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"Invalid manifest file path: {path!r}")
    return root / Path(*relative.parts)
