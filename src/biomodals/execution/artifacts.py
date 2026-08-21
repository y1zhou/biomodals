"""Materialize workload outputs into durable execution artifacts."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import orjson
from pydantic import BaseModel

from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    ExecutionArtifact,
    InlineBytes,
    VolumePath,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

_MAX_ARTIFACT_ID_BYTES = 200
_MAX_FILENAME_BYTES = 255


def _artifact_id(producing_node_id: str, output_name: str) -> str:
    artifact_id = sanitize_filename(f"{producing_node_id}-{output_name}")
    if len(artifact_id.encode("utf-8")) <= _MAX_ARTIFACT_ID_BYTES:
        return artifact_id
    return f"artifact-{hashlib.sha256(artifact_id.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class MaterializedAppRunResult:
    """Execution artifacts plus the ledger-safe app result that produced them."""

    artifacts: list[ExecutionArtifact]
    result: AppRunResult


@dataclass(frozen=True)
class ContentBoundFileSet:
    """Persist and validate one exact workload-owned file publication."""

    root: Path
    marker_path: Path
    expected_paths: tuple[str, ...]
    identity: Mapping[str, Any]

    def load(self) -> tuple[ArtifactFile, ...] | None:
        """Return exact file records only while identity and bytes still match."""
        try:
            marker = orjson.loads(self.marker_path.read_bytes())
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            orjson.JSONDecodeError,
        ):
            return None
        if not (
            isinstance(marker, dict)
            and marker.get("schema_version") == 1
            and marker.get("identity") == dict(self.identity)
            and isinstance(marker.get("files"), list)
        ):
            return None
        try:
            files = tuple(ArtifactFile.model_validate(item) for item in marker["files"])
        except (TypeError, ValueError):
            return None
        if tuple(file.path for file in files) != self.expected_paths or any(
            file.size_bytes is None or file.content_sha256 is None for file in files
        ):
            return None
        for file in files:
            path = _resolve_artifact_file(self.root, file.path)
            try:
                if (
                    not path.is_file()
                    or path.stat().st_size != file.size_bytes
                    or _file_sha256(path) != file.content_sha256
                ):
                    return None
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return None
        return files

    def write(self, files: tuple[ArtifactFile, ...]) -> None:
        """Write a marker only from a complete content-bound manifest."""
        if tuple(file.path for file in files) != self.expected_paths or any(
            file.size_bytes is None or file.content_sha256 is None for file in files
        ):
            raise ValueError("Publication files do not match the expected file set")
        _write_json(
            self.marker_path,
            {
                "schema_version": 1,
                "identity": dict(self.identity),
                "files": [
                    file.model_dump(mode="json", exclude_none=True) for file in files
                ],
            },
        )


def republish_execution_artifact(artifact: ExecutionArtifact) -> AppOutput:
    """Return an App output that keeps one upstream publication authoritative."""
    metadata = dict(artifact.metadata)
    metadata["source_artifact_id"] = artifact.artifact_id
    if artifact.files:
        metadata["files"] = [
            file.model_dump(exclude_defaults=True, exclude_none=True)
            for file in artifact.files
        ]
    return AppOutput(
        name=f"artifact-{hashlib.sha256(artifact.artifact_id.encode()).hexdigest()[:16]}",
        kind=artifact.kind,
        storage=artifact.storage,
        metadata=metadata,
    )


def inline_json_result(
    *,
    name: str,
    value: object,
    filename: str,
    kind: ArtifactKind = ArtifactKind.TABLE,
) -> AppRunResult:
    """Publish one small JSON value as an execution-owned artifact."""
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name=name,
                kind=kind,
                storage=InlineBytes(
                    data=orjson.dumps(value, option=orjson.OPT_SORT_KEYS),
                    filename=filename,
                    media_type="application/json",
                ),
            )
        ],
    )


def _write_json(path: Path, payload: object) -> None:
    from tempfile import TemporaryDirectory

    path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=path.parent) as tmp_dir:
        tmp_path = Path(tmp_dir) / path.name
        if isinstance(payload, BaseModel):
            tmp_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        else:
            tmp_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

        tmp_path.replace(path)


def _artifact_files(root: Path) -> list[ArtifactFile]:
    if root.is_symlink():
        raise ValueError("Execution artifact paths must not be symlinks")
    if root.is_file():
        return [
            ArtifactFile(
                path=root.name,
                size_bytes=root.stat().st_size,
                content_sha256=_file_sha256(root),
            )
        ]
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Execution artifact trees must not contain symlinks")
    return [
        ArtifactFile(
            path=str(path.relative_to(root)),
            size_bytes=path.stat().st_size,
            content_sha256=_file_sha256(path),
        )
        for path in paths
        if path.is_file()
    ]


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _declared_artifact_files(metadata: Mapping[str, Any]) -> list[ArtifactFile]:
    raw_files = metadata.get("files")
    if raw_files is None:
        return []
    if not isinstance(raw_files, list):
        raise ValueError("AppOutput.metadata['files'] must be a list")

    files: list[ArtifactFile] = []
    for raw_file in raw_files:
        if isinstance(raw_file, ArtifactFile):
            files.append(raw_file)
        elif isinstance(raw_file, str):
            files.append(ArtifactFile(path=raw_file))
        elif isinstance(raw_file, Mapping):
            files.append(ArtifactFile.model_validate(raw_file))
        else:
            raise ValueError(
                "AppOutput.metadata['files'] entries must be paths or "
                "ArtifactFile-compatible mappings"
            )
    return files


def _reference_artifact_files(
    *,
    storage: VolumePath,
    metadata: Mapping[str, Any],
    artifact_volume_name: str,
    volume_root: Path | None,
) -> tuple[list[ArtifactFile], bool]:
    """Bind execution-owned references to their current file contents."""
    declared = _declared_artifact_files(metadata)
    if storage.volume_name != artifact_volume_name or volume_root is None:
        return declared, False

    artifact_path = _resolve_volume_child(volume_root, storage.path)
    if not artifact_path.exists():
        return declared, False
    if declared and all(
        file.size_bytes is not None and file.content_sha256 is not None
        for file in declared
    ):
        return declared, False
    if not declared:
        return _artifact_files(artifact_path), True

    if artifact_path.is_file():
        actual_files = (
            _artifact_files(artifact_path)
            if declared[0].path == artifact_path.name
            else []
        )
    else:
        actual_files = []
        for file in declared:
            file_path = _resolve_artifact_file(artifact_path, file.path)
            if file_path.is_file():
                actual_files.append(
                    ArtifactFile(
                        path=file.path,
                        size_bytes=file_path.stat().st_size,
                        content_sha256=_file_sha256(file_path),
                    )
                )
    actual_by_path = {file.path: file for file in actual_files}

    if any(file.path not in actual_by_path for file in declared):
        return declared, False

    files = []
    for file in declared:
        actual = actual_by_path[file.path]
        if file.size_bytes is not None and file.size_bytes != actual.size_bytes:
            raise FileNotFoundError(
                f"Execution artifact file size does not match: {file.path}"
            )
        if (
            file.content_sha256 is not None
            and file.content_sha256 != actual.content_sha256
        ):
            raise FileNotFoundError(
                f"Execution artifact file SHA-256 does not match: {file.path}"
            )
        files.append(
            file.model_copy(
                update={
                    "size_bytes": actual.size_bytes,
                    "content_sha256": actual.content_sha256,
                }
            )
        )
    return files, True


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
    artifact_volume_name: str,
    result_dir: Path,
    volume_root: Path | None,
    producing_node_id: str,
    metadata: dict[str, Any] | None = None,
    artifact_output_name: str | None = None,
    source_app_output_name: str | None = None,
    artifact_parent: Path | None = None,
) -> ExecutionArtifact:
    artifact_id = _artifact_id(
        producing_node_id,
        artifact_output_name or output_name,
    )
    _validate_inline_text_bytes(storage, output_kind)
    safe_filename = sanitize_filename(storage.filename)
    if len(safe_filename.encode("utf-8")) > _MAX_FILENAME_BYTES:
        raise ValueError("InlineBytes filename exceeds the filesystem limit")

    artifact_parent = artifact_parent or result_dir
    materialized_dir = artifact_parent / artifact_id
    materialized_dir.mkdir(parents=True, exist_ok=True)
    materialized_file = materialized_dir.joinpath(safe_filename)
    materialized_file.write_bytes(storage.data)

    return ExecutionArtifact(
        artifact_id=artifact_id,
        producing_node_id=producing_node_id,
        kind=output_kind,
        storage=VolumePath(
            volume_name=artifact_volume_name,
            path=_volume_path(materialized_file, volume_root),
            media_type=storage.media_type,
        ),
        files=_artifact_files(materialized_file),
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


def _resolve_artifact_file(root: Path, path: str) -> Path:
    relative = PurePosixPath(path)
    if path == "" or path == ".":
        raise ValueError("ArtifactFile.path must be a non-empty relative path")
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("ArtifactFile.path must be relative and must not traverse")
    if "\\" in path:
        raise ValueError("ArtifactFile.path must use POSIX separators")

    resolved_root = root.resolve()
    raw_path = resolved_root / Path(*relative.parts)
    current = resolved_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("ArtifactFile.path must not contain symlinks")

    resolved_path = raw_path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("ArtifactFile.path escapes artifact root") from exc
    return resolved_path


def execution_artifact_availability_errors(
    artifact: ExecutionArtifact,
    *,
    artifact_volume_name: str,
    volume_root: Path,
) -> list[str]:
    """Return missing-file errors for workflow-volume artifacts.

    Artifacts stored in app-owned volumes are intentionally treated as unknown:
    the workflow runtime cannot validate volumes it has not mounted.
    """
    if artifact.storage.volume_name != artifact_volume_name:
        return []

    try:
        artifact_path = _resolve_volume_child(volume_root, artifact.storage.path)
    except ValueError as exc:
        return [
            f"{artifact.artifact_id}: invalid execution artifact path "
            f"{artifact.storage.path!r}: {exc}"
        ]

    if not artifact_path.exists():
        return [
            f"{artifact.artifact_id}: missing execution artifact path "
            f"{artifact.storage.path}"
        ]
    if not artifact.files:
        return []

    if artifact_path.is_file():
        if len(artifact.files) != 1 or artifact.files[0].path != artifact_path.name:
            return [
                f"{artifact.artifact_id}: artifact storage path is a file but "
                "manifest file entries do not match it"
            ]
        return _artifact_file_metadata_errors(
            artifact_id=artifact.artifact_id,
            file_path=artifact_path,
            file=artifact.files[0],
        )

    if not artifact_path.is_dir():
        return [
            f"{artifact.artifact_id}: execution artifact path is not a file or "
            f"directory: {artifact.storage.path}"
        ]

    errors: list[str] = []
    for file in artifact.files:
        try:
            file_path = _resolve_artifact_file(artifact_path, file.path)
        except ValueError as exc:
            errors.append(
                f"{artifact.artifact_id}: invalid execution artifact file "
                f"{file.path!r}: {exc}"
            )
            continue
        if not file_path.is_file():
            errors.append(
                f"{artifact.artifact_id}: missing execution artifact file "
                f"{artifact.storage.path}/{file.path}"
            )
            continue
        errors.extend(
            _artifact_file_metadata_errors(
                artifact_id=artifact.artifact_id,
                file_path=file_path,
                file=file,
            )
        )
    return errors


def _artifact_file_metadata_errors(
    *,
    artifact_id: str,
    file_path: Path,
    file: ArtifactFile,
) -> list[str]:
    actual_size = file_path.stat().st_size
    if actual_size < 1 and file.size_bytes != 0:
        return [f"{artifact_id}: execution artifact file {file.path} is empty"]
    if file.size_bytes is not None and actual_size != file.size_bytes:
        return [
            f"{artifact_id}: execution artifact file {file.path} has size "
            f"{actual_size}, expected {file.size_bytes}"
        ]
    if (
        file.content_sha256 is not None
        and _file_sha256(file_path) != file.content_sha256
    ):
        return [
            f"{artifact_id}: execution artifact file {file.path} does not match "
            "its SHA-256"
        ]
    return []


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
    artifact_volume_name: str,
    result_dir: Path,
    volume_root: Path | None,
    producing_node_id: str,
    metadata: dict[str, Any],
    volume_roots: Mapping[str, Path],
    artifact_output_name: str | None = None,
    source_app_output_name: str | None = None,
    artifact_parent: Path | None = None,
) -> ExecutionArtifact:
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

    artifact_parent = artifact_parent or result_dir
    materialized_dir = artifact_parent / artifact_id
    _copy_volume_path_tree(
        source_path=source_path,
        materialized_dir=materialized_dir,
        source_root=source_root,
    )

    return ExecutionArtifact(
        artifact_id=artifact_id,
        producing_node_id=producing_node_id,
        kind=output_kind,
        storage=VolumePath(
            volume_name=artifact_volume_name,
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
    artifact_volume_name: str,
    result_dir: Path,
    artifact_dir: Path,
    producing_node_id: str,
    artifact_id_scope: str | None = None,
    volume_root: Path | None = None,
    volume_path_mode: Literal["reference", "copy"] = "reference",
    volume_roots: Mapping[str, Path] | None = None,
) -> MaterializedAppRunResult:
    """Write app outputs into local artifact volume paths and return manifests."""
    artifacts: list[ExecutionArtifact] = []
    persisted_outputs: list[AppOutput] = []
    persisted_logs: list[AppOutput] = []

    def scoped_output_name(output_name: str) -> str:
        if artifact_id_scope is None:
            return output_name
        return f"{artifact_id_scope}-{output_name}"

    def materialize_output(
        output,
        *,
        artifact_output_name: str | None = None,
        source_app_output_name: str | None = None,
        artifact_parent: Path | None = None,
    ) -> tuple[ExecutionArtifact, AppOutput]:
        artifact_output_name = scoped_output_name(artifact_output_name or output.name)
        artifact_id = _artifact_id(
            producing_node_id,
            artifact_output_name,
        )
        if isinstance(output.storage, InlineBytes):
            artifact = _materialize_inline_bytes(
                storage=output.storage,
                output_name=output.name,
                output_kind=output.kind,
                artifact_volume_name=artifact_volume_name,
                result_dir=result_dir,
                volume_root=volume_root,
                producing_node_id=producing_node_id,
                metadata=output.metadata,
                artifact_output_name=artifact_output_name,
                source_app_output_name=source_app_output_name,
                artifact_parent=artifact_parent,
            )
            return artifact, _persisted_output(output, artifact.storage)

        if volume_path_mode == "copy":
            artifact = _materialize_volume_path_copy(
                storage=output.storage,
                output_name=output.name,
                output_kind=output.kind,
                artifact_volume_name=artifact_volume_name,
                result_dir=result_dir,
                volume_root=volume_root,
                producing_node_id=producing_node_id,
                metadata=output.metadata,
                volume_roots=volume_roots or {},
                artifact_output_name=artifact_output_name,
                source_app_output_name=source_app_output_name,
                artifact_parent=artifact_parent,
            )
            return artifact, _persisted_output(output, artifact.storage)
        reference_files, execution_reference_validated = _reference_artifact_files(
            storage=output.storage,
            metadata=output.metadata,
            artifact_volume_name=artifact_volume_name,
            volume_root=volume_root,
        )
        artifact = ExecutionArtifact(
            artifact_id=artifact_id,
            producing_node_id=producing_node_id,
            kind=output.kind,
            storage=output.storage,
            files=reference_files,
            source_app_output_name=source_app_output_name or output.name,
            metadata=output.metadata,
        )
        if not execution_reference_validated:
            raise_for_unavailable_execution_artifact(artifact)
        return artifact, _persisted_output(output, artifact.storage)

    def raise_for_unavailable_execution_artifact(artifact: ExecutionArtifact) -> None:
        if volume_root is None:
            return
        errors = execution_artifact_availability_errors(
            artifact,
            artifact_volume_name=artifact_volume_name,
            volume_root=volume_root,
        )
        if not errors:
            return
        raise FileNotFoundError(
            "Execution artifact is unavailable:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    for output in result.outputs:
        artifact, persisted_output = materialize_output(output)
        _write_json(artifact_dir / f"{artifact.artifact_id}.json", artifact)
        artifacts.append(artifact)
        persisted_outputs.append(persisted_output)

    for log_output in result.logs:
        artifact, persisted_log = materialize_output(
            log_output,
            artifact_output_name=f"logs-{log_output.name}",
            source_app_output_name=log_output.name,
            artifact_parent=result_dir / "logs",
        )
        _write_json(artifact_dir / f"{artifact.artifact_id}.json", artifact)
        artifacts.append(artifact)
        persisted_logs.append(persisted_log)
    return MaterializedAppRunResult(
        artifacts=artifacts,
        result=result.model_copy(
            update={
                "outputs": persisted_outputs,
                "logs": persisted_logs,
            },
        ),
    )


def _persisted_output(output: AppOutput, storage: VolumePath) -> AppOutput:
    """Return an app output with durable workflow-volume storage."""
    return output.model_copy(update={"storage": storage})
