"""Content-bound AF3Score publication records."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import orjson

from biomodals.helper.artifacts import replace_bytes_atomic, sha256_file

COMPLETION_SAMPLE_SUBDIR = "seed-10_sample-0"
COMPLETION_REQUIRED_FILES = (
    "summary_confidences.json",
    "confidences.json",
)
METRICS_FILENAME = "af3score_metrics.csv"
_METRICS_PUBLICATION_SCHEMA_VERSION = 1
_INPUT_PUBLICATION_SCHEMA_VERSION = 2


def _metrics_publication_path(run_root: str | Path) -> Path:
    return Path(run_root) / ".biomodals" / "af3score-metrics.json"


def _input_output_dir(output_dir: str | Path, input_id: str) -> Path:
    if not input_id or Path(input_id).name != input_id or input_id in {".", ".."}:
        raise ValueError("AF3Score input ID must be a safe path component")
    return Path(output_dir) / input_id


def _input_publication_path(output_dir: str | Path, input_id: str) -> Path:
    return (
        _input_output_dir(output_dir, input_id) / ".biomodals" / "af3score-input.json"
    )


def _input_output_records(
    output_dir: str | Path,
    input_id: str,
) -> dict[str, dict[str, int | str]]:
    sample_dir = _input_output_dir(output_dir, input_id) / COMPLETION_SAMPLE_SUBDIR
    records: dict[str, dict[str, int | str]] = {}
    for filename in COMPLETION_REQUIRED_FILES:
        path = sample_dir / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"AF3Score output is incomplete for '{input_id}'")
        records[filename] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def _write_input_publication(
    output_dir: str | Path,
    input_id: str,
    *,
    publication_key: str,
    input_sha256: str,
) -> None:
    """Atomically bind one complete AF3Score output to its scientific input."""
    outputs = _input_output_records(output_dir, input_id)
    marker = _input_publication_path(output_dir, input_id)
    replace_bytes_atomic(
        marker,
        orjson.dumps(
            {
                "schema_version": _INPUT_PUBLICATION_SCHEMA_VERSION,
                "publication_key": publication_key,
                "input_sha256": input_sha256,
                "outputs": outputs,
            },
            option=orjson.OPT_SORT_KEYS,
        ),
    )


def _input_publication_ready(
    output_dir: str | Path,
    input_id: str,
    *,
    publication_key: str,
    input_sha256: str,
) -> bool:
    """Validate one scored output against the current scientific request."""
    marker = _input_publication_record(
        output_dir,
        input_id,
        publication_key=publication_key,
        input_sha256=input_sha256,
    )
    if marker is None:
        return False
    try:
        return marker["outputs"] == _input_output_records(output_dir, input_id)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, RuntimeError):
        return False


def _input_publication_record(
    output_dir: str | Path,
    input_id: str,
    *,
    publication_key: str,
    input_sha256: str,
) -> dict[str, object] | None:
    """Load one identity-matched AF3Score publication marker."""
    try:
        marker = orjson.loads(
            _input_publication_path(output_dir, input_id).read_bytes()
        )
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return None
    if not (
        isinstance(marker, dict)
        and marker.get("schema_version") == _INPUT_PUBLICATION_SCHEMA_VERSION
        and marker.get("publication_key") == publication_key
        and marker.get("input_sha256") == input_sha256
        and isinstance(marker.get("outputs"), dict)
    ):
        return None
    return marker


def _input_summary_publication_ready(
    output_dir: str | Path,
    input_id: str,
    *,
    publication_key: str,
    input_sha256: str,
) -> bool:
    """Validate only the scored summary consumed by postprocessing."""
    marker = _input_publication_record(
        output_dir,
        input_id,
        publication_key=publication_key,
        input_sha256=input_sha256,
    )
    if marker is None:
        return False
    outputs = cast(dict[str, object], marker["outputs"])
    expected = outputs.get("summary_confidences.json")
    if not isinstance(expected, dict):
        return False
    path = (
        _input_output_dir(output_dir, input_id)
        / COMPLETION_SAMPLE_SUBDIR
        / "summary_confidences.json"
    )
    try:
        return path.stat().st_size == expected.get("size") and sha256_file(
            path
        ) == expected.get("sha256")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, RuntimeError):
        return False


def _invalidate_input_publications(
    output_dir: str | Path,
    input_ids: Sequence[str],
) -> bool:
    """Remove batch markers before any corresponding output is rewritten."""
    invalidated = False
    for input_id in input_ids:
        marker = _input_publication_path(output_dir, input_id)
        try:
            marker.unlink()
        except FileNotFoundError:
            continue
        invalidated = True
    return invalidated


def _write_metrics_publication(
    run_root: str | Path,
    publication_key: str,
    metrics_path: Path,
) -> None:
    """Atomically bind the metrics artifact to one scientific request."""
    size = metrics_path.stat().st_size
    if size < 1:
        raise RuntimeError("AF3Score metrics publication is empty")
    marker = _metrics_publication_path(run_root)
    replace_bytes_atomic(
        marker,
        orjson.dumps(
            {
                "schema_version": _METRICS_PUBLICATION_SCHEMA_VERSION,
                "publication_key": publication_key,
                "metrics_filename": metrics_path.name,
                "size": size,
                "sha256": sha256_file(metrics_path),
            },
            option=orjson.OPT_SORT_KEYS,
        ),
    )


def _metrics_publication_ready(
    run_root: str | Path,
    publication_key: str,
) -> bool:
    """Validate fingerprint-bound metrics without hiding unreadable state."""
    marker_path = _metrics_publication_path(run_root)
    try:
        marker = orjson.loads(marker_path.read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    if not (
        isinstance(marker, dict)
        and marker.get("schema_version") == _METRICS_PUBLICATION_SCHEMA_VERSION
        and marker.get("publication_key") == publication_key
        and marker.get("metrics_filename") == METRICS_FILENAME
        and isinstance(marker.get("size"), int)
        and not isinstance(marker.get("size"), bool)
        and marker["size"] > 0
        and isinstance(marker.get("sha256"), str)
    ):
        return False
    metrics = Path(run_root) / METRICS_FILENAME
    try:
        return (
            not metrics.is_symlink()
            and metrics.stat().st_size == marker["size"]
            and sha256_file(metrics) == marker["sha256"]
        )
    except (FileNotFoundError, NotADirectoryError):
        return False
