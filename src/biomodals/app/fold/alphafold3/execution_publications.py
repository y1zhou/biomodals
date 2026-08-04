"""Small references to app-owned AlphaFold3 coordinator result publications."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import orjson

from biomodals.app.fold.alphafold3.artifacts import (
    VolumeHandle,
    artifact_record,
    load_artifact_bytes,
    write_json_atomic,
)

MAX_EXECUTION_RESULT_BYTES = 1024 * 1024 * 1024


def execution_result_path(
    workload_plan_fingerprint: str,
    node_key: str,
    task_fingerprint: str,
) -> PurePosixPath:
    """Return one successor-reusable result path outside kernel state."""
    if (
        len(workload_plan_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in workload_plan_fingerprint
        )
        or not node_key
        or "/" in node_key
        or node_key in {".", ".."}
        or len(task_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in task_fingerprint)
    ):
        raise ValueError("Invalid AlphaFold3 execution result identity")
    return (
        PurePosixPath("execution-publications")
        / workload_plan_fingerprint
        / node_key
        / f"{task_fingerprint}.json"
    )


def publish_execution_result(
    output_root: Path,
    output_volume: VolumeHandle,
    relative_path: str,
    result: dict[str, object],
) -> dict[str, object]:
    """Atomically publish one provider result and return its small reference."""
    selected = PurePosixPath(relative_path)
    if selected.is_absolute() or ".." in selected.parts:
        raise ValueError("Execution result path must be Volume-relative")
    path = output_root.joinpath(*selected.parts)
    content = orjson.dumps(result, option=orjson.OPT_SORT_KEYS)
    if not 0 < len(content) <= MAX_EXECUTION_RESULT_BYTES:
        raise ValueError("AlphaFold3 execution result exceeds its byte limit")
    write_json_atomic(path, result)
    output_volume.commit()
    return artifact_record(path, output_root)


def load_execution_result(
    output_root: Path,
    reference: object,
    *,
    expected_path: PurePosixPath,
) -> dict[str, object] | None:
    """Load one exact provider result when its reference still validates."""
    content = load_artifact_bytes(
        output_root,
        reference,
        expected_path.as_posix(),
    )
    if content is None or len(content) > MAX_EXECUTION_RESULT_BYTES:
        return None
    try:
        value = orjson.loads(content)
    except orjson.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_execution_result_path(
    output_root: Path,
    expected_path: PurePosixPath,
) -> dict[str, object] | None:
    """Load one atomic result publication without a Run-local envelope."""
    path = output_root.joinpath(*expected_path.parts)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    if not 0 < len(content) <= MAX_EXECUTION_RESULT_BYTES:
        return None
    try:
        value = orjson.loads(content)
    except orjson.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
