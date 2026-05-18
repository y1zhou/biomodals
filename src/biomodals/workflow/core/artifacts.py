"""Local helpers for materializing app outputs into workflow artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

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


def _artifact_id(producing_node_id: str, output_name: str) -> str:
    return sanitize_filename(f"{producing_node_id}-{output_name}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    data: Any
    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _extract_tar_zst_bytes(data: bytes, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_bin = shutil.which("tar")
    if tar_bin is None:
        raise FileNotFoundError("tar is not available in PATH")
    with TemporaryDirectory(prefix="biomodals_artifact_extract_") as tmpdir:
        archive_path = Path(tmpdir) / "bundle.tar.zst"
        archive_path.write_bytes(data)
        subprocess.run(  # noqa: S603
            [tar_bin, "-I", "zstd", "-xf", str(archive_path), "-C", str(out_dir)],
            check=True,
        )


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
) -> WorkflowArtifact:
    artifact_id = _artifact_id(producing_node_id, output_name)
    safe_filename = sanitize_filename(storage.filename)
    raw_dir = attempt_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / safe_filename
    raw_path.write_bytes(storage.data)

    materialized_dir = attempt_dir / "materialized_outputs" / artifact_id
    materialized_dir.mkdir(parents=True, exist_ok=True)
    if storage.archive_format == "tar.zst":
        _extract_tar_zst_bytes(storage.data, materialized_dir)
    else:
        materialized_dir.joinpath(safe_filename).write_bytes(storage.data)

    return WorkflowArtifact(
        artifact_id=artifact_id,
        producing_node_id=producing_node_id,
        kind=output_kind,
        storage=VolumePath(
            volume_name=workflow_volume_name,
            path=_volume_path(materialized_dir, volume_root),
        ),
        files=_artifact_files(materialized_dir),
        source_app_output_name=output_name,
        metadata=metadata or {},
    )


def _volume_path(path: Path, volume_root: Path | None) -> str:
    if volume_root is None:
        return str(path)
    return path.relative_to(volume_root).as_posix()


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
) -> WorkflowArtifact:
    artifact_id = _artifact_id(producing_node_id, output_name)
    source_root = volume_roots.get(storage.volume_name)
    if source_root is None:
        raise ValueError(
            f"Missing mounted volume root for output volume {storage.volume_name!r}"
        )
    source_path = source_root / storage.path
    if not source_path.exists():
        raise FileNotFoundError(f"Volume output path not found: {source_path}")

    materialized_dir = attempt_dir / "materialized_outputs" / artifact_id
    if source_path.is_dir():
        shutil.copytree(source_path, materialized_dir, dirs_exist_ok=True)
    else:
        materialized_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, materialized_dir / source_path.name)

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
        source_app_output_name=output_name,
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
    for output in result.outputs:
        artifact_id = _artifact_id(producing_node_id, output.name)
        if isinstance(output.storage, InlineBytes):
            artifact = _materialize_inline_bytes(
                storage=output.storage,
                output_name=output.name,
                output_kind=output.kind,
                workflow_volume_name=workflow_volume_name,
                attempt_dir=attempt_dir,
                volume_root=volume_root,
                producing_node_id=producing_node_id,
                metadata=output.metadata,
            )
        else:
            if volume_path_mode == "copy":
                artifact = _materialize_volume_path_copy(
                    storage=output.storage,
                    output_name=output.name,
                    output_kind=output.kind,
                    workflow_volume_name=workflow_volume_name,
                    attempt_dir=attempt_dir,
                    volume_root=volume_root,
                    producing_node_id=producing_node_id,
                    metadata=output.metadata,
                    volume_roots=volume_roots or {},
                )
            else:
                artifact = WorkflowArtifact(
                    artifact_id=artifact_id,
                    producing_node_id=producing_node_id,
                    kind=output.kind,
                    storage=output.storage,
                    source_app_output_name=output.name,
                    metadata=output.metadata,
                )
        _write_json(artifact_dir / f"{artifact.artifact_id}.json", artifact)
        artifacts.append(artifact)
    return artifacts
