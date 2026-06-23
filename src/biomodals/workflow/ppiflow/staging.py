"""Artifact staging and file-selection helpers for the PPIFlow workflow."""

from __future__ import annotations

import fnmatch
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from biomodals.helper.shell import sanitize_filename
from biomodals.schema import ArtifactKind, WorkflowArtifact
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

STRUCTURE_SUFFIXES = {".pdb", ".cif"}


def artifact_mount_path(
    artifact: WorkflowArtifact,
    volume_roots: Mapping[str, str],
) -> Path:
    """Return an artifact path resolved under its mounted volume root."""
    mountpoint = volume_roots.get(artifact.storage.volume_name)
    if mountpoint is None:
        raise ValueError(
            "PPIFlow workflow cannot read artifact volume "
            f"{artifact.storage.volume_name!r}"
        )
    return artifact.storage.at_mountpoint(mountpoint)


def matches_structure_pattern(path: str, patterns: Sequence[str] | None) -> bool:
    """Return whether a path looks like a selected structure file."""
    suffix = Path(path).suffix.lower()
    if suffix not in STRUCTURE_SUFFIXES:
        return False
    if patterns is None:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def structure_patterns_from_metadata(
    artifact: WorkflowArtifact,
    patterns: Sequence[str] | None,
) -> Sequence[str] | None:
    """Resolve explicit or artifact-provided structure selection patterns."""
    if patterns is not None:
        return patterns
    metadata_patterns = artifact.metadata.get("structure_patterns")
    if isinstance(metadata_patterns, str):
        return tuple(
            pattern.strip()
            for pattern in metadata_patterns.split(",")
            if pattern.strip()
        )
    if isinstance(metadata_patterns, Sequence):
        return tuple(str(pattern) for pattern in metadata_patterns)
    return None


def safe_selected_file_name(artifact_id: str, member_name: str) -> str:
    """Return a collision-resistant selected file name."""
    parts = [sanitize_filename(part) for part in Path(member_name).parts if part]
    return sanitize_filename("__".join([artifact_id, *parts]))


def artifact_is_zstd_archive(artifact: WorkflowArtifact, path: Path) -> bool:
    """Return whether an artifact should be read as a tar.zst archive."""
    return (
        artifact.kind == ArtifactKind.ARCHIVE
        or artifact.storage.media_type == ZSTD_MEDIA_TYPE
        or artifact.metadata.get("archive_format") == "tar.zst"
        or path.name.endswith(".tar.zst")
    )


def structure_files_from_tar_zst(
    artifact: WorkflowArtifact,
    archive_path: Path,
    patterns: Sequence[str] | None,
) -> list[tuple[str, bytes]]:
    """Read selected structure files from a tar.zst artifact."""
    import zstandard as zstd

    selected: list[tuple[str, bytes]] = []
    with archive_path.open("rb") as compressed:
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        with reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if not matches_structure_pattern(member.name, patterns):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                selected.append((
                    safe_selected_file_name(artifact.artifact_id, member.name),
                    extracted.read(),
                ))
    return selected


def structure_files_from_artifact(
    artifact: WorkflowArtifact,
    patterns: Sequence[str] | None,
    volume_roots: Mapping[str, str],
) -> list[tuple[str, bytes]]:
    """Read selected structure files from one workflow artifact."""
    patterns = structure_patterns_from_metadata(artifact, patterns)
    root = artifact_mount_path(artifact, volume_roots)
    if not root.exists():
        raise FileNotFoundError(f"PPIFlow input artifact path not found: {root}")
    if root.is_file():
        if artifact_is_zstd_archive(artifact, root):
            return structure_files_from_tar_zst(artifact, root, patterns)
        if matches_structure_pattern(root.name, patterns):
            return [
                (
                    safe_selected_file_name(artifact.artifact_id, root.name),
                    root.read_bytes(),
                )
            ]
        return []

    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if matches_structure_pattern(relative, patterns):
            files.append((
                safe_selected_file_name(artifact.artifact_id, relative),
                path.read_bytes(),
            ))
    return files


def csv_files_from_artifact(
    artifact: WorkflowArtifact,
    volume_roots: Mapping[str, str],
) -> list[tuple[str, bytes]]:
    """Read CSV files from one workflow artifact."""
    root = artifact_mount_path(artifact, volume_roots)
    if not root.exists():
        raise FileNotFoundError(f"PPIFlow tabular artifact path not found: {root}")
    if root.is_file() and artifact_is_zstd_archive(artifact, root):
        selected = []
        import zstandard as zstd

        with root.open("rb") as compressed:
            reader = zstd.ZstdDecompressor().stream_reader(compressed)
            with reader, tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or Path(member.name).suffix != ".csv":
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        selected.append((member.name, extracted.read()))
        return selected
    if root.is_file():
        return [(root.name, root.read_bytes())] if root.suffix == ".csv" else []
    return [
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*.csv"))
    ]


def select_structure_files_from_artifacts(
    artifacts: Sequence[WorkflowArtifact],
    volume_roots: Mapping[str, str],
    *,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[tuple[str, bytes]]:
    """Read and sort selected structure files from workflow artifacts."""
    selected = [
        structure_file
        for artifact in artifacts
        for structure_file in structure_files_from_artifact(
            artifact, patterns, volume_roots
        )
    ]
    selected.sort(key=lambda item: item[0])
    if max_files is not None:
        selected = selected[:max_files]
    if not selected:
        raise FileNotFoundError("No PPIFlow structure files were found in inputs")
    return selected
