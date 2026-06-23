"""Artifact staging and file-selection helpers for the PPIFlow workflow."""

from __future__ import annotations

import fnmatch
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import polars as pl

from biomodals.helper.shell import sanitize_filename
from biomodals.schema import AppRunStatus, ArtifactKind, WorkflowArtifact
from biomodals.schema.storage import ZSTD_MEDIA_TYPE
from biomodals.workflow.ppiflow import manifests, tables

STRUCTURE_SUFFIXES = {".pdb", ".cif"}


@dataclass(frozen=True)
class SelectedStructureFile:
    """One selected structure file with enough provenance for manifest rows."""

    artifact_id: str
    file_name: str
    artifact_file_path: str
    app_volume_path: str
    volume_name: str
    size_bytes: int | None = None
    media_type: str | None = None


@dataclass(frozen=True)
class CandidateStructureFile:
    """One selected structure keyed to a PPIFlow candidate id."""

    candidate_id: str
    file_name: str
    data: bytes
    source_path: str | None = None


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


def files_from_tar_zst_bytes(
    data: bytes,
    *,
    suffixes: Sequence[str] | None = None,
) -> list[tuple[str, bytes]]:
    """Read selected files from tar.zst bytes."""
    import zstandard as zstd

    suffix_set = {suffix.lower() for suffix in suffixes or ()}
    selected = []
    with BytesIO(data) as compressed:
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        with reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if suffix_set and Path(member.name).suffix.lower() not in suffix_set:
                    continue
                extracted = tar.extractfile(member)
                if extracted is not None:
                    selected.append((member.name, extracted.read()))
    return selected


def selected_structure_file_records_from_artifact(
    artifact: WorkflowArtifact,
    patterns: Sequence[str] | None,
    volume_roots: Mapping[str, str],
) -> list[SelectedStructureFile]:
    """Return selected structure files without loading their bytes."""
    patterns = structure_patterns_from_metadata(artifact, patterns)
    root = artifact_mount_path(artifact, volume_roots)
    if not root.exists():
        raise FileNotFoundError(f"PPIFlow input artifact path not found: {root}")

    if root.is_file():
        if artifact_is_zstd_archive(artifact, root):
            return _selected_structure_file_records_from_tar_zst(
                artifact, root, patterns
            )
        if not matches_structure_pattern(root.name, patterns):
            return []
        return [
            SelectedStructureFile(
                artifact_id=artifact.artifact_id,
                file_name=safe_selected_file_name(artifact.artifact_id, root.name),
                artifact_file_path=root.name,
                app_volume_path=artifact.storage.path,
                volume_name=artifact.storage.volume_name,
                size_bytes=root.stat().st_size,
                media_type=artifact.storage.media_type,
            )
        ]

    selected = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if not matches_structure_pattern(relative, patterns):
            continue
        selected.append(
            SelectedStructureFile(
                artifact_id=artifact.artifact_id,
                file_name=safe_selected_file_name(artifact.artifact_id, relative),
                artifact_file_path=relative,
                app_volume_path=str(Path(artifact.storage.path) / relative),
                volume_name=artifact.storage.volume_name,
                size_bytes=path.stat().st_size,
                media_type=artifact.storage.media_type,
            )
        )
    return selected


def _selected_structure_file_records_from_tar_zst(
    artifact: WorkflowArtifact,
    archive_path: Path,
    patterns: Sequence[str] | None,
) -> list[SelectedStructureFile]:
    import zstandard as zstd

    selected = []
    archive_size = archive_path.stat().st_size
    with archive_path.open("rb") as compressed:
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        with reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not matches_structure_pattern(
                    member.name, patterns
                ):
                    continue
                selected.append(
                    SelectedStructureFile(
                        artifact_id=artifact.artifact_id,
                        file_name=safe_selected_file_name(
                            artifact.artifact_id, member.name
                        ),
                        artifact_file_path=member.name,
                        app_volume_path=artifact.storage.path,
                        volume_name=artifact.storage.volume_name,
                        size_bytes=archive_size,
                        media_type=artifact.storage.media_type or ZSTD_MEDIA_TYPE,
                    )
                )
    return selected


def stage2_input_manifest_rows(
    artifact: WorkflowArtifact,
    volume_roots: Mapping[str, str],
    *,
    patterns: Sequence[str] | None = None,
    stage_name: str = "Stage2Input",
) -> list[dict[str, object]]:
    """Build synthetic Stage2Input candidate rows from a structure location."""
    selected = selected_structure_file_records_from_artifact(
        artifact,
        patterns,
        volume_roots,
    )
    if not selected:
        raise FileNotFoundError("Stage2Input did not find any structure files")

    rows = []
    for index, structure in enumerate(
        sorted(selected, key=lambda item: item.app_volume_path),
        start=1,
    ):
        rows.append(
            manifests.candidate_manifest_row(
                candidate_id=manifests.stage2_input_candidate_id(index),
                stage_name=stage_name,
                stage_role="stage2_input",
                operation_mode="existing_structures",
                candidate_status=AppRunStatus.SUCCEEDED.value,
                source_artifact_id=structure.artifact_id,
                source_path=structure.app_volume_path,
                derived_path=structure.app_volume_path,
                files=[
                    manifests.candidate_file_record(
                        role="structure",
                        volume_name=structure.volume_name,
                        app_volume_path=structure.app_volume_path,
                        path=structure.artifact_file_path,
                        media_type=structure.media_type,
                        size_bytes=structure.size_bytes,
                        expected=True,
                    )
                ],
                summary={"file_name": structure.file_name},
            )
        )
    return rows


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


def candidate_structure_files_from_selected(
    selected: Sequence[tuple[str, bytes]],
    *,
    manifest_frame: pl.DataFrame | None = None,
) -> list[CandidateStructureFile]:
    """Attach candidate ids to selected structure bytes."""
    lookup = _candidate_key_lookup(manifest_frame)
    keyed = []
    for file_name, data in selected:
        key = tables.candidate_key(file_name)
        keyed.append(
            CandidateStructureFile(
                candidate_id=lookup.get(key, key),
                file_name=file_name,
                data=data,
                source_path=file_name,
            )
        )
    keyed.sort(key=lambda item: (item.candidate_id, item.file_name))
    return keyed


def candidate_structure_files_from_artifacts(
    artifacts: Sequence[WorkflowArtifact],
    volume_roots: Mapping[str, str],
    *,
    manifest_frame: pl.DataFrame | None = None,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[CandidateStructureFile]:
    """Read selected structures and return candidate-keyed records."""
    return candidate_structure_files_from_selected(
        select_structure_files_from_artifacts(
            artifacts,
            volume_roots,
            patterns=patterns,
            max_files=max_files,
        ),
        manifest_frame=manifest_frame,
    )


def prepare_dockq_pairs_by_candidate(
    *,
    references: Sequence[CandidateStructureFile],
    models: Sequence[CandidateStructureFile],
    mapping: object = None,
) -> list[dict[str, object]]:
    """Pair DockQ reference/model structures by candidate id."""
    references_by_id = _unique_candidate_structures(references, "reference")
    models_by_id = _unique_candidate_structures(models, "model")
    missing_models = sorted(set(references_by_id).difference(models_by_id))
    missing_references = sorted(set(models_by_id).difference(references_by_id))
    if missing_models or missing_references:
        raise ValueError(
            "DockQ candidate pairing mismatch: "
            f"missing models={missing_models}, missing references={missing_references}"
        )
    pairs = []
    for candidate_id in sorted(references_by_id):
        reference = references_by_id[candidate_id]
        model = models_by_id[candidate_id]
        pairs.append({
            "id": candidate_id,
            "candidate_id": candidate_id,
            "model_name": model.file_name,
            "model_bytes": model.data,
            "reference_name": reference.file_name,
            "reference_bytes": reference.data,
            "mapping": mapping,
        })
    return pairs


def discover_partial_sample_dirs(root: str | Path) -> list[Path]:
    """Return PPIFlow partial sample directories below a run root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"PPIFlow partial root was not found: {root}")
    return sorted({
        path.parent
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in STRUCTURE_SUFFIXES
        and ("sample" in path.parent.name.lower() or "partial" in path.parts)
    })


def rosetta_job_manifest_rows(
    structures: Sequence[CandidateStructureFile],
    *,
    rosetta_binary: str,
    rosetta_script: str | None = None,
    flags_file: str | None = None,
) -> list[dict[str, object]]:
    """Build PPIFlow-owned Rosetta queue/job manifest rows."""
    rows = []
    for index, structure in enumerate(structures, start=1):
        input_pdb = f"inputs/{index}/{sanitize_filename(structure.file_name)}"
        output_dir = f"outputs/{index}"
        rows.append({
            "candidate_id": structure.candidate_id,
            "index": index,
            "status": "pending",
            "binary": rosetta_binary,
            "pdb": input_pdb,
            "rosetta_script": rosetta_script,
            "flags_file": flags_file,
            "expected_output_dir": output_dir,
            "expected_score_file": f"{output_dir}/score.sc",
            "worker_log": f"logs/{index}.log",
        })
    return rows


def write_rosetta_job_manifest(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
) -> Path:
    """Write Rosetta job manifest rows as a small CSV table."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([dict(row) for row in rows]).write_csv(manifest_path)
    return manifest_path


def _candidate_key_lookup(manifest_frame: pl.DataFrame | None) -> dict[str, str]:
    if manifest_frame is None or manifest_frame.is_empty():
        return {}
    lookup = {}
    for row in manifest_frame.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        for value in (row.get("source_path"), row.get("derived_path")):
            if value:
                lookup[tables.candidate_key(str(value))] = candidate_id
        for file_record in row.get("files") or []:
            if isinstance(file_record, Mapping):
                for field_name in ("path", "app_volume_path", "workflow_path"):
                    if file_record.get(field_name):
                        lookup[tables.candidate_key(str(file_record[field_name]))] = (
                            candidate_id
                        )
    return lookup


def _unique_candidate_structures(
    structures: Sequence[CandidateStructureFile],
    role: str,
) -> dict[str, CandidateStructureFile]:
    by_id = {}
    for structure in structures:
        if structure.candidate_id in by_id:
            raise ValueError(
                f"Duplicate {role} structure for candidate {structure.candidate_id!r}"
            )
        by_id[structure.candidate_id] = structure
    return by_id
