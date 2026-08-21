"""Generic artifact helpers used by focused PPIFlow task modules."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    ExecutionArtifact,
    InlineBytes,
)
from biomodals.workflow.ppiflow import manifests, tables
from biomodals.workflow.ppiflow.runtime_context import (
    SOURCE_VOLUME_ROOTS,
    WORKFLOW_OUTPUT_MOUNTPOINT,
    WORKFLOW_OUTPUT_VOLUME,
    WORKFLOW_OUTPUT_VOLUME_NAME,
)


def result_with_output_kind(
    result: AppRunResult,
    kind: ArtifactKind,
    metadata: Mapping[str, object],
) -> AppRunResult:
    """Replace output kinds while preserving workload result content."""
    return result.model_copy(
        update={
            "outputs": [
                output.model_copy(
                    update={
                        "kind": kind,
                        "metadata": dict(output.metadata) | dict(metadata),
                    }
                )
                for output in result.outputs
            ]
        }
    )


def inline_output_file_records(
    outputs: Sequence[AppOutput],
) -> list[dict[str, object]]:
    """Describe inline candidate outputs with one durable content digest."""
    return [
        manifests.candidate_file_record(
            role=(
                "structure"
                if output.kind == ArtifactKind.STRUCTURES
                else output.kind.value
            ),
            path=output.storage.filename,
            media_type=output.storage.media_type,
            size_bytes=len(output.storage.data),
            content_sha256=hashlib.sha256(output.storage.data).hexdigest(),
        )
        for output in outputs
        if isinstance(output.storage, InlineBytes)
    ]


def inline_csv_table_output(
    *,
    name: str,
    filename: str,
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> AppOutput | None:
    """Encode table rows as one inline CSV output."""
    if not rows:
        return None
    csv_text = pl.DataFrame([dict(row) for row in rows]).write_csv()
    return AppOutput(
        name=name,
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=csv_text.encode("utf-8"),
            filename=filename,
            media_type="text/csv",
        ),
        metadata={"rows": len(rows)} | dict(metadata),
    )


def write_candidate_manifest_output(
    *,
    run_id: str,
    node_id: str,
    step_name: str,
    rows: Sequence[Mapping[str, object]],
) -> AppOutput:
    """Publish one content-bound candidate manifest."""
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / manifests.MANIFEST_OUTPUT_NAME
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    manifest_path = output_dir / manifests.MANIFEST_FILENAME
    manifests.write_manifest(rows, manifest_path)
    WORKFLOW_OUTPUT_VOLUME.commit()
    return manifests.manifest_artifact_output(
        manifest_path=manifest_path,
        mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
        volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
        stage_name=step_name,
        row_count=len(rows),
    )


def candidate_manifest_frame_from_inputs(
    candidate_manifests: Sequence[ExecutionArtifact],
    selected_structures: Sequence[tuple[str, bytes]],
    *,
    step_name: str,
) -> pl.DataFrame:
    """Read canonical candidate identities or derive legacy filename keys."""
    if candidate_manifests:
        frames = read_candidate_manifest_artifacts(candidate_manifests)
        frame = pl.concat(frames, how="diagonal") if len(frames) > 1 else frames[0]
        if frame.is_empty():
            raise ValueError("Supplied candidate manifests produced no usable rows")
        return frame

    return pl.DataFrame([
        manifests.candidate_manifest_row(
            candidate_id=tables.candidate_key(name),
            stage_name=step_name,
            stage_role="structure_selection",
            operation_mode="legacy_structure_keys",
            candidate_status=AppRunStatus.SUCCEEDED.value,
            source_path=name,
            derived_path=name,
            files=[
                manifests.candidate_file_record(
                    role="structure",
                    path=name,
                    size_bytes=len(data),
                    content_sha256=hashlib.sha256(data).hexdigest(),
                )
            ],
        )
        for name, data in selected_structures
    ])


def read_candidate_manifest_artifacts(
    artifacts: Sequence[ExecutionArtifact],
) -> list[pl.DataFrame]:
    """Read and validate candidate-manifest artifacts."""
    frames = []
    for artifact in artifacts:
        if artifact.kind != ArtifactKind.TABLE:
            raise ValueError(
                f"Candidate manifest {artifact.artifact_id!r} is not a table"
            )
        try:
            frames.append(
                manifests.read_manifest_volume_path(
                    storage=artifact.storage,
                    volume_roots=SOURCE_VOLUME_ROOTS,
                )
            )
        except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(
                f"Could not read candidate manifest {artifact.artifact_id!r}: {error}"
            ) from error
    return frames


def bytes_payload(value: object, label: str) -> bytes:
    """Require a bytes-valued task payload field."""
    if not isinstance(value, bytes):
        raise TypeError(f"{label} must be bytes")
    return value


def config_int(config: Mapping[str, object], key: str, default: int) -> int:
    """Read one integer configuration value."""
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{key} must be an integer")


def optional_config_int(config: Mapping[str, object], key: str) -> int | None:
    """Read one optional integer configuration value."""
    value = config.get(key)
    if value is None:
        return None
    return config_int(config, key, 0)


def patterns_from_config(
    config: Mapping[str, object],
    *,
    default: Sequence[str] | None = None,
) -> tuple[str, ...] | None:
    """Read optional structure-selection patterns."""
    value = config.get("structure_patterns") or config.get("patterns")
    if value is None:
        return tuple(default) if default is not None else None
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(pattern) for pattern in value)


def parse_seed_values(value: object) -> list[int]:
    """Normalize a scalar or sequence of seed values."""
    if isinstance(value, str):
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        seeds = [value]
    elif isinstance(value, Sequence):
        seeds = [int(seed) for seed in value]
    else:
        raise TypeError("seeds must be an integer, comma-separated string, or sequence")
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    return seeds


def file_sha256(path: Path) -> str:
    """Hash a local file without loading it into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
