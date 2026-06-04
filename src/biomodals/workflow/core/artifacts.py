"""Local helpers for materializing app outputs into workflow artifacts."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import orjson
from pydantic import BaseModel

from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppRunResult,
    ArtifactFile,
    ArtifactKind,
    InlineBytes,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


def _artifact_id(producing_node_id: str, output_name: str) -> str:
    return sanitize_filename(f"{producing_node_id}-{output_name}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    if isinstance(payload, BaseModel):
        tmp_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    else:
        tmp_path.write_bytes(
            orjson.dumps(
                payload,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            )
        )
    tmp_path.replace(path)


def _artifact_files(root: Path) -> list[ArtifactFile]:
    if root.is_file():
        return [
            ArtifactFile(
                path=root.name,
                size_bytes=root.stat().st_size,
            )
        ]
    return [
        ArtifactFile(
            path=str(path.relative_to(root)),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _validate_inline_text_bytes(
    storage: InlineBytes, output_kind: ArtifactKind
) -> None:
    if storage.media_type == ZSTD_MEDIA_TYPE:
        return
    if output_kind == ArtifactKind.ARCHIVE or getattr(storage, "archive_format", None):
        raise ValueError(
            f"InlineBytes archive outputs must use media_type='{ZSTD_MEDIA_TYPE}' "
            "or VolumePath storage"
        )
    try:
        storage.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("InlineBytes outputs must contain UTF-8 text bytes") from exc


def _materialize_inline_bytes(
    *,
    storage: InlineBytes,
    output_name: str,
    output_kind: ArtifactKind,
    workflow_volume_name: str,
    attempt_dir: Path,
    volume_root: Path | None,
    producing_node_id: str,
    metadata: dict[str, Any] | None = None,
    artifact_output_name: str | None = None,
    source_app_output_name: str | None = None,
    raw_dir: Path | None = None,
    materialized_parent: Path | None = None,
) -> WorkflowArtifact:
    artifact_id = _artifact_id(
        producing_node_id,
        artifact_output_name or output_name,
    )
    _validate_inline_text_bytes(storage, output_kind)
    safe_filename = sanitize_filename(storage.filename)
    raw_dir = raw_dir or attempt_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / safe_filename
    raw_path.write_bytes(storage.data)

    materialized_parent = materialized_parent or attempt_dir / "materialized_outputs"
    materialized_dir = materialized_parent / artifact_id
    materialized_dir.mkdir(parents=True, exist_ok=True)
    materialized_dir.joinpath(safe_filename).write_bytes(storage.data)

    return WorkflowArtifact(
        artifact_id=artifact_id,
        producing_node_id=producing_node_id,
        kind=output_kind,
        storage=VolumePath(
            volume_name=workflow_volume_name,
            path=_volume_path(materialized_dir, volume_root),
            media_type=storage.media_type,
        ),
        files=_artifact_files(materialized_dir),
        source_app_output_name=source_app_output_name or output_name,
        metadata=metadata or {},
    )


def _volume_path(path: Path, volume_root: Path | None) -> str:
    if volume_root is None:
        return str(path)
    return path.relative_to(volume_root).as_posix()


def _resolve_volume_child(root: Path, path: str) -> Path:
    relative = PurePosixPath(path)
    if path == "" or path == ".":
        raise ValueError("VolumePath.path must be a non-empty relative path")
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("VolumePath.path must be relative and must not traverse")
    if "\\" in path:
        raise ValueError("VolumePath.path must use POSIX separators")

    resolved_root = root.resolve()
    raw_path = resolved_root / Path(*relative.parts)
    current = resolved_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("VolumePath.path must not contain symlinks")

    resolved_path = raw_path.resolve()
    try:
        # Validate-only: reject paths that resolve outside the mounted volume.
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("VolumePath.path escapes the mounted volume root") from exc
    return resolved_path


def _copy_volume_path_tree(
    *,
    source_path: Path,
    materialized_dir: Path,
    source_root: Path,
) -> None:
    resolved_source_root = source_root.resolve()
    if source_path.is_symlink():
        raise ValueError("VolumePath copy source must not be a symlink")
    if source_path.is_file():
        materialized_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            source_path, materialized_dir / source_path.name, follow_symlinks=False
        )
        return

    materialized_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_path.rglob("*")):
        if child.is_symlink():
            raise ValueError("VolumePath copy source tree must not contain symlinks")
        try:
            child.resolve().relative_to(resolved_source_root)
        except ValueError as exc:
            raise ValueError(
                "VolumePath copy source tree escapes the mounted volume root"
            ) from exc

        destination = materialized_dir / child.relative_to(source_path)
        if child.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination, follow_symlinks=False)


def _materialize_volume_path_copy(
    *,
    storage: VolumePath,
    output_name: str,
    output_kind: ArtifactKind,
    workflow_volume_name: str,
    attempt_dir: Path,
    volume_root: Path | None,
    producing_node_id: str,
    metadata: dict[str, Any],
    volume_roots: Mapping[str, Path],
    artifact_output_name: str | None = None,
    source_app_output_name: str | None = None,
    materialized_parent: Path | None = None,
) -> WorkflowArtifact:
    artifact_id = _artifact_id(
        producing_node_id,
        artifact_output_name or output_name,
    )
    source_root = volume_roots.get(storage.volume_name)
    if source_root is None:
        raise ValueError(
            f"Missing mounted volume root for output volume {storage.volume_name!r}"
        )
    source_path = _resolve_volume_child(source_root, storage.path)
    if not source_path.exists():
        raise FileNotFoundError(f"Volume output path not found: {source_path}")

    materialized_parent = materialized_parent or attempt_dir / "materialized_outputs"
    materialized_dir = materialized_parent / artifact_id
    _copy_volume_path_tree(
        source_path=source_path,
        materialized_dir=materialized_dir,
        source_root=source_root,
    )

    return WorkflowArtifact(
        artifact_id=artifact_id,
        producing_node_id=producing_node_id,
        kind=output_kind,
        storage=VolumePath(
            volume_name=workflow_volume_name,
            path=_volume_path(materialized_dir, volume_root),
            media_type=storage.media_type,
        ),
        files=_artifact_files(materialized_dir),
        source_app_output_name=source_app_output_name or output_name,
        metadata=metadata,
    )


def materialize_app_run_result(
    *,
    result: AppRunResult,
    workflow_volume_name: str,
    attempt_dir: Path,
    artifact_dir: Path,
    producing_node_id: str,
    volume_root: Path | None = None,
    volume_path_mode: Literal["reference", "copy"] = "reference",
    volume_roots: Mapping[str, Path] | None = None,
) -> list[WorkflowArtifact]:
    """Write app outputs into local workflow volume paths and return manifests."""
    artifacts: list[WorkflowArtifact] = []

    def materialize_output(
        output,
        *,
        artifact_output_name: str | None = None,
        source_app_output_name: str | None = None,
        raw_dir: Path | None = None,
        materialized_parent: Path | None = None,
    ) -> WorkflowArtifact:
        artifact_id = _artifact_id(
            producing_node_id,
            artifact_output_name or output.name,
        )
        if isinstance(output.storage, InlineBytes):
            return _materialize_inline_bytes(
                storage=output.storage,
                output_name=output.name,
                output_kind=output.kind,
                workflow_volume_name=workflow_volume_name,
                attempt_dir=attempt_dir,
                volume_root=volume_root,
                producing_node_id=producing_node_id,
                metadata=output.metadata,
                artifact_output_name=artifact_output_name,
                source_app_output_name=source_app_output_name,
                raw_dir=raw_dir,
                materialized_parent=materialized_parent,
            )

        if volume_path_mode == "copy":
            return _materialize_volume_path_copy(
                storage=output.storage,
                output_name=output.name,
                output_kind=output.kind,
                workflow_volume_name=workflow_volume_name,
                attempt_dir=attempt_dir,
                volume_root=volume_root,
                producing_node_id=producing_node_id,
                metadata=output.metadata,
                volume_roots=volume_roots or {},
                artifact_output_name=artifact_output_name,
                source_app_output_name=source_app_output_name,
                materialized_parent=materialized_parent,
            )
        return WorkflowArtifact(
            artifact_id=artifact_id,
            producing_node_id=producing_node_id,
            kind=output.kind,
            storage=output.storage,
            source_app_output_name=source_app_output_name or output.name,
            metadata=output.metadata,
        )

    for output in result.outputs:
        artifact = materialize_output(output)
        _write_json(artifact_dir / f"{artifact.artifact_id}.json", artifact)
        artifacts.append(artifact)

    for log_output in result.logs:
        artifact = materialize_output(
            log_output,
            artifact_output_name=f"logs-{log_output.name}",
            source_app_output_name=log_output.name,
            raw_dir=attempt_dir / "logs" / "raw_outputs",
            materialized_parent=attempt_dir / "logs",
        )
        _write_json(artifact_dir / f"{artifact.artifact_id}.json", artifact)
        artifacts.append(artifact)
    return artifacts
