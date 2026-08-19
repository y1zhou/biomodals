"""CPU artifact transforms for the PPIFlow workflow."""

from __future__ import annotations

import ast
import hashlib
import shutil
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import cast

import polars as pl

from biomodals.execution.artifact_availability import (
    ArtifactAvailability,
    check_external_artifact_status,
)
from biomodals.helper.app_run import volume_app_output, volume_path_from_mount_path
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    ExecutionArtifact,
    VolumePath,
)
from biomodals.workflow.ppiflow import manifests as ppiflow_manifests
from biomodals.workflow.ppiflow import staging as ppiflow_staging
from biomodals.workflow.ppiflow import tables as ppiflow_tables
from biomodals.workflow.ppiflow.runtime_context import (
    SOURCE_VOLUME_ROOTS as PPI_FLOW_SOURCE_VOLUME_ROOTS,
)
from biomodals.workflow.ppiflow.runtime_context import (
    WORKFLOW_OUTPUT_MOUNTPOINT,
    WORKFLOW_OUTPUT_VOLUME,
    WORKFLOW_OUTPUT_VOLUME_NAME,
)
from biomodals.workflow.ppiflow.runtime_context import (
    reload_source_volumes as _reload_ppiflow_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    candidate_manifest_frame_from_inputs as _candidate_manifest_frame_from_inputs,
)
from biomodals.workflow.ppiflow.runtime_support import file_sha256 as _file_sha256
from biomodals.workflow.ppiflow.runtime_support import (
    patterns_from_config as _patterns_from_config,
)

_STAGE2_INPUT_SNAPSHOT_KEY = "_input_snapshot"


def _streamed_volume_file_record(
    *,
    volume_name: str,
    path: str,
    chunks: Iterable[bytes],
) -> dict[str, object]:
    digest = hashlib.sha256()
    size_bytes = 0
    for chunk in chunks:
        digest.update(chunk)
        size_bytes += len(chunk)
    return {
        "volume_name": volume_name,
        "path": path,
        "size_bytes": size_bytes,
        "content_sha256": digest.hexdigest(),
    }


def _mounted_volume_file_record(storage: VolumePath) -> dict[str, object]:
    root = PPI_FLOW_SOURCE_VOLUME_ROOTS.get(storage.volume_name)
    if root is None:
        raise ValueError(f"Unknown Stage2Input volume {storage.volume_name!r}")
    path = storage.at_mountpoint(root)
    if not path.is_file():
        raise FileNotFoundError(f"Stage2Input file was not found: {storage}")
    with path.open("rb") as handle:
        return _streamed_volume_file_record(
            volume_name=storage.volume_name,
            path=storage.path,
            chunks=iter(lambda: handle.read(1024 * 1024), b""),
        )


def _validate_stage2_input_snapshot(
    *, storage: VolumePath, config: Mapping[str, object]
) -> None:
    snapshot = config.get(_STAGE2_INPUT_SNAPSHOT_KEY)
    if not isinstance(snapshot, Mapping):
        raise ValueError("Stage2Input is missing its content snapshot")
    manifest = snapshot.get("manifest")
    if manifest is not None:
        if not isinstance(manifest, Mapping):
            raise ValueError("Stage2Input manifest snapshot is invalid")
        manifest = cast(Mapping[str, object], manifest)
        manifest_storage = VolumePath(
            volume_name=str(manifest["volume_name"]),
            path=str(manifest["path"]),
        )
        if _mounted_volume_file_record(manifest_storage) != dict(manifest):
            raise ValueError("Stage2Input manifest changed after submission")
    raw_structures = snapshot.get("structures")
    if not isinstance(raw_structures, list) or not raw_structures:
        raise ValueError("Stage2Input snapshot contains no structures")
    if any(not isinstance(record, Mapping) for record in raw_structures):
        raise ValueError("Stage2Input structure snapshot is invalid")
    expected_records = [
        dict(cast(Mapping[str, object], record)) for record in raw_structures
    ]
    actual_records = [
        _mounted_volume_file_record(
            VolumePath(
                volume_name=str(record["volume_name"]),
                path=str(record["path"]),
            )
        )
        for record in expected_records
    ]
    if actual_records != expected_records:
        raise ValueError("Stage2Input structures changed after submission")

    if manifest is None:
        root = storage.at_mountpoint(PPI_FLOW_SOURCE_VOLUME_ROOTS[storage.volume_name])
        if root.is_file():
            current_paths = [storage.path]
        else:
            patterns = _patterns_from_config(config)
            current_paths = sorted(
                str(PurePosixPath(storage.path) / path.relative_to(root).as_posix())
                for path in root.rglob("*")
                if path.is_file()
                and ppiflow_staging.matches_structure_pattern(
                    path.relative_to(root).as_posix(), patterns
                )
            )
        if current_paths != [str(record["path"]) for record in expected_records]:
            raise ValueError("Stage2Input structure set changed after submission")


def _stage2_manifest_storage_from_config(
    config: Mapping[str, object],
    *,
    default_volume_name: str,
) -> VolumePath | None:
    raw_path = config.get("manifest_path")
    if raw_path is None:
        return None
    return _volume_path_from_stage_config(
        str(raw_path),
        volume_name=str(
            config.get("manifest_volume_name")
            or config.get("volume_name")
            or default_volume_name
        ),
    )


def _volume_path_from_stage_config(path: str, *, volume_name: str) -> VolumePath:
    if path.startswith("/"):
        for known_volume, mountpoint in PPI_FLOW_SOURCE_VOLUME_ROOTS.items():
            try:
                return volume_path_from_mount_path(path, mountpoint, known_volume)
            except ValueError:
                continue
        raise ValueError(f"Stage2Input path is not under a known mountpoint: {path}")
    return VolumePath(volume_name=volume_name, path=path)


def check_ppiflow_external_artifact(
    artifact: ExecutionArtifact,
) -> ArtifactAvailability:
    """Validate app-owned artifacts referenced by the PPIFlow workflow."""
    _reload_ppiflow_source_volumes()
    return check_external_artifact_status(
        artifact,
        artifact_volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )


def normalize_ppiflow_stage2_input(
    *,
    storage: VolumePath,
    config: dict[str, object],
    run_id: str,
    node_id: str,
    step_name: str,
) -> AppRunResult:
    """Normalize Stage2Input structures into a workflow-owned manifest."""
    _reload_ppiflow_source_volumes()
    _validate_stage2_input_snapshot(storage=storage, config=config)
    structure_artifact = ExecutionArtifact(
        artifact_id=f"{sanitize_filename(node_id)}-stage2-input-structures",
        producing_node_id=node_id,
        kind=ArtifactKind.STRUCTURES,
        storage=storage,
        metadata={
            "step_name": step_name,
            "run_name": config.get("run_name", run_id),
        },
    )
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / "stage2_input"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    manifest_path = output_dir / ppiflow_manifests.MANIFEST_FILENAME

    manifest_storage = _stage2_manifest_storage_from_config(
        config,
        default_volume_name=storage.volume_name,
    )
    if manifest_storage is not None:
        frame = ppiflow_manifests.read_manifest_volume_path(
            storage=manifest_storage,
            volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
        )
        ppiflow_manifests.write_manifest(frame.to_dicts(), manifest_path)
        row_count = frame.height
    else:
        rows = ppiflow_staging.stage2_input_manifest_rows(
            structure_artifact,
            PPI_FLOW_SOURCE_VOLUME_ROOTS,
            patterns=_patterns_from_config(config),
            stage_name=step_name,
        )
        ppiflow_manifests.write_manifest(rows, manifest_path)
        row_count = len(rows)

    structure_path = storage.at_mountpoint(
        PPI_FLOW_SOURCE_VOLUME_ROOTS[storage.volume_name]
    )
    if structure_path.is_file():
        structure_files = [structure_path.name]
    else:
        manifest_frame = ppiflow_manifests.read_manifest(manifest_path)
        structure_files = sorted({
            str(file_record["path"])
            for row in manifest_frame.iter_rows(named=True)
            for file_record in row["files"]
            if file_record.get("path")
        })

    WORKFLOW_OUTPUT_VOLUME.commit()
    structure_metadata = {
        "step_name": step_name,
        "run_name": config.get("run_name", run_id),
        "structure_count": row_count,
        "files": structure_files,
    }
    patterns = _patterns_from_config(config)
    if patterns is not None:
        structure_metadata["structure_patterns"] = patterns
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="stage2_input_structures",
                kind=ArtifactKind.STRUCTURES,
                storage=storage,
                metadata=structure_metadata,
            ),
            ppiflow_manifests.manifest_artifact_output(
                manifest_path=manifest_path,
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                stage_name=step_name,
                row_count=row_count,
            ),
        ],
    )


def filter_ppiflow_artifacts(
    *,
    structures: list[ExecutionArtifact],
    scores: list[ExecutionArtifact],
    candidate_manifests: list[ExecutionArtifact] | None = None,
    config: dict[str, object],
    run_id: str,
    node_id: str,
    step_name: str,
) -> AppRunResult:
    """Filter structure artifacts using an AF3Score-compatible CSV."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        structures,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )
    score_files = [
        score_file
        for artifact in scores
        for score_file in ppiflow_staging.csv_files_from_artifact(
            artifact,
            PPI_FLOW_SOURCE_VOLUME_ROOTS,
        )
    ]
    if not score_files:
        raise FileNotFoundError(f"{step_name} did not find a score CSV")
    preferred_name = str(config.get("score_csv") or "af3score_metrics.csv")
    _, score_bytes = next(
        (
            score_file
            for score_file in score_files
            if Path(score_file[0]).name == preferred_name
        ),
        score_files[0],
    )
    score_frame = pl.read_csv(BytesIO(score_bytes), infer_schema_length=0)
    filename_col = str(config.get("filename_col") or "description")
    if filename_col not in score_frame.columns:
        for fallback in ("filename", "pdb", "name"):
            if fallback in score_frame.columns:
                filename_col = fallback
                break
        else:
            raise ValueError(
                f"{step_name} score CSV is missing candidate column {filename_col!r}"
            )

    raw_filters = config.get("filters")
    if raw_filters is None and config.get("score_column") is not None:
        raw_filters = {
            str(config["score_column"]): (
                f"{config.get('operator', '>=')} {config.get('threshold', 0)}"
            )
        }
    if raw_filters is None:
        raw_filters = {"iptm": "> 0.7" if "stage1" in step_name.lower() else "> 0.8"}
    if not isinstance(raw_filters, Mapping) or not raw_filters:
        raise ValueError(f"{step_name} filters must be a non-empty mapping")

    manifest_frame = _candidate_manifest_frame_from_inputs(
        candidate_manifests or [],
        selected,
        step_name=step_name,
    )
    retained_manifest, retained_scores, audit_frame = ppiflow_tables.filter_candidates(
        manifest_frame=manifest_frame,
        score_frame=score_frame,
        filters=raw_filters,
        filename_col=filename_col,
        stage_name=step_name,
    )
    structures_by_key = {}
    for name, data in selected:
        key = ppiflow_tables.candidate_key(name)
        if key in structures_by_key:
            raise ValueError(f"Duplicate PPIFlow candidate identity: {key!r}")
        structures_by_key[key] = (name, data)
    retained = []
    for row in retained_scores.iter_rows(named=True):
        key = ppiflow_tables.candidate_key(str(row.get(filename_col) or ""))
        structure = structures_by_key.get(key)
        if structure is not None:
            retained.append(structure)
    if not retained:
        raise ValueError(f"{step_name} filters rejected every available structure")

    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / "filtered"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True)
    for file_name, file_bytes in retained:
        (structures_dir / sanitize_filename(file_name)).write_bytes(file_bytes)
    filtered_csv = output_dir / "filtered_scores.csv"
    retained_scores.write_csv(filtered_csv)
    audit_csv = output_dir / "filter_audit.csv"
    audit_frame.write_csv(audit_csv)
    manifest_path = output_dir / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(retained_manifest.to_dicts(), manifest_path)
    WORKFLOW_OUTPUT_VOLUME.commit()
    metadata = {
        "step_name": step_name,
        "input_count": len(selected),
        "retained_count": len(retained),
        "files": [name for name, _ in retained],
        "structure_patterns": ("*.pdb", "*.cif"),
    }
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="filtered_structures",
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(structures_dir),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                metadata=metadata,
            ),
            volume_app_output(
                name="filtered_scores",
                kind=ArtifactKind.TABLE,
                remote_path=str(filtered_csv),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={"step_name": step_name, "rows": retained_scores.height},
            ),
            ppiflow_manifests.manifest_artifact_output(
                manifest_path=manifest_path,
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                stage_name=step_name,
                row_count=retained_manifest.height,
                name="retained_candidate_manifest",
            ),
            volume_app_output(
                name="filter_audit",
                kind=ArtifactKind.TABLE,
                remote_path=str(audit_csv),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={"step_name": step_name, "rows": audit_frame.height},
            ),
        ],
    )


def derive_ppiflow_fixed_positions(
    *,
    artifacts: list[ExecutionArtifact],
    candidate_manifests: list[ExecutionArtifact] | None = None,
    config: dict[str, object],
    run_id: str,
    node_id: str,
    step_name: str,
) -> AppRunResult:
    """Derive per-structure fixed positions from Rosetta residue energies."""
    _reload_ppiflow_source_volumes()
    structures = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )
    explicit = str(config.get("fixed_positions") or "").strip()
    fixed_by_structure: dict[str, str] = {}
    if explicit:
        fixed_by_structure = {
            ppiflow_tables.candidate_key(name): explicit for name, _ in structures
        }
    else:
        energy_threshold = float(config.get("energy_threshold", -5))
        gentype = str(config.get("gentype") or "binder")
        expected_chains = {
            "binder": {"interface_energy_A_B": "A"},
            "nanobody": {"interface_energy_A_C": "A"},
            "antibody": {
                "interface_energy_A_C": "A",
                "interface_energy_B_C": "B",
            },
        }.get(gentype)
        if expected_chains is None:
            raise ValueError(f"Unsupported PPIFlow gentype: {gentype!r}")
        energies: dict[str, dict[str, dict[int, float]]] = {}
        energy_files = [
            csv_file
            for artifact in artifacts
            for csv_file in ppiflow_staging.csv_files_from_artifact(
                artifact,
                PPI_FLOW_SOURCE_VOLUME_ROOTS,
            )
            if Path(csv_file[0]).name == "residue_energy.csv"
        ]
        if not energy_files:
            raise FileNotFoundError(
                f"{step_name} did not find Rosetta residue_energy.csv outputs"
            )
        for csv_name, csv_bytes in energy_files:
            chain = next(
                (
                    chain
                    for directory, chain in expected_chains.items()
                    if directory in Path(csv_name).parts
                ),
                None,
            )
            if chain is None:
                continue
            frame = pl.read_csv(BytesIO(csv_bytes), infer_schema_length=0)
            if "binder_energy" not in frame.columns:
                raise ValueError(f"{csv_name} is missing binder_energy")
            for row in frame.iter_rows(named=True):
                pdb_name = str(row.get("pdbname") or row.get("pdbpath") or "")
                structure_name = Path(pdb_name).stem
                parts = structure_name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    structure_name = parts[0]
                try:
                    binder_energy = ast.literal_eval(str(row["binder_energy"]))
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        f"Could not parse binder_energy for {pdb_name!r}"
                    ) from exc
                if not isinstance(binder_energy, Mapping):
                    raise ValueError(f"binder_energy for {pdb_name!r} is not a mapping")
                chain_energies = energies.setdefault(
                    structure_name.lower(), {}
                ).setdefault(chain, {})
                for residue, energy in binder_energy.items():
                    residue_id = int(residue)
                    energy_value = float(energy)
                    chain_energies[residue_id] = min(
                        chain_energies.get(residue_id, energy_value), energy_value
                    )
        for structure_name, chain_energies in energies.items():
            positions = [
                f"{chain}{residue}"
                for chain in sorted(chain_energies)
                for residue, energy in sorted(chain_energies[chain].items())
                if energy < energy_threshold
            ]
            fixed_by_structure[structure_name] = ",".join(positions) or "NONE"

    rows = [
        {
            "filename": name,
            "fixed_positions": fixed_by_structure.get(
                ppiflow_tables.candidate_key(name), "NONE"
            ),
        }
        for name, _ in structures
    ]
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / "fixed_positions"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True)
    for file_name, file_bytes in structures:
        (structures_dir / sanitize_filename(file_name)).write_bytes(file_bytes)
    positions_csv = output_dir / "fixed_positions.csv"
    pl.DataFrame(rows).write_csv(positions_csv)
    manifest_frame = _candidate_manifest_frame_from_inputs(
        candidate_manifests or [],
        structures,
        step_name=step_name,
    )
    manifest_path = output_dir / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(manifest_frame.to_dicts(), manifest_path)
    WORKFLOW_OUTPUT_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="fixed_position_structures",
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(structures_dir),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                metadata={
                    "step_name": step_name,
                    "fixed_positions": rows[0]["fixed_positions"] if rows else "NONE",
                    "fixed_positions_by_structure": fixed_by_structure,
                    "structure_patterns": ("*.pdb", "*.cif"),
                },
            ),
            volume_app_output(
                name="fixed_positions",
                kind=ArtifactKind.TABLE,
                remote_path=str(positions_csv),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={"step_name": step_name, "rows": len(rows)},
            ),
            ppiflow_manifests.manifest_artifact_output(
                manifest_path=manifest_path,
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                stage_name=step_name,
                row_count=manifest_frame.height,
            ),
        ],
    )


def rank_ppiflow_artifacts(
    *,
    structures: list[ExecutionArtifact],
    candidate_manifests: list[ExecutionArtifact],
    score_artifacts: list[ExecutionArtifact],
    config: dict[str, object],
    run_id: str,
    node_id: str,
    step_name: str,
) -> AppRunResult:
    """Merge available score tables and rank final PPIFlow structures."""
    _reload_ppiflow_source_volumes()
    selected_structures = ppiflow_staging.select_structure_files_from_artifacts(
        structures,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )
    csv_frames = [
        (name, pl.read_csv(BytesIO(data), infer_schema_length=0))
        for artifact in [*structures, *score_artifacts]
        for name, data in ppiflow_staging.csv_files_from_artifact(
            artifact,
            PPI_FLOW_SOURCE_VOLUME_ROOTS,
        )
    ]
    score_frames = [frame for _, frame in csv_frames]
    if not score_frames:
        raise ValueError(f"{step_name} did not find any supported score tables")

    manifest_frame = _candidate_manifest_frame_from_inputs(
        candidate_manifests,
        selected_structures,
        step_name=step_name,
    )
    candidate_structures = ppiflow_staging.candidate_structure_files_from_selected(
        selected_structures,
        manifest_frame=manifest_frame,
    )
    structure_by_key = {}
    candidate_ids_by_filename = {}
    for structure in candidate_structures:
        if structure.candidate_id in structure_by_key:
            raise ValueError(
                f"Duplicate PPIFlow candidate identity: {structure.candidate_id!r}"
            )
        structure_by_key[structure.candidate_id] = (
            structure.file_name,
            structure.data,
        )
        candidate_ids_by_filename[structure.file_name] = structure.candidate_id
    raw_dockq_threshold = config.get("dockq_threshold", 0.49)
    if not isinstance(raw_dockq_threshold, str | int | float):
        raise TypeError("dockq_threshold must be a string or number")
    ranked = ppiflow_tables.ranked_design_rows(
        structures=selected_structures,
        score_frames=score_frames,
        gentype=str(config.get("gentype") or "binder"),
        dockq_threshold=float(raw_dockq_threshold),
        candidate_ids_by_filename=candidate_ids_by_filename,
    )
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / "ranked"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True)
    ranked_structure_files = []
    for row in ranked:
        file_name, file_bytes = structure_by_key[str(row["design"])]
        output_path = structures_dir / sanitize_filename(file_name)
        output_path.write_bytes(file_bytes)
        ranked_structure_files.append(
            ArtifactFile(
                path=output_path.name,
                role="structure",
                media_type=(
                    "chemical/x-pdb"
                    if output_path.suffix.lower() == ".pdb"
                    else "chemical/x-mmcif"
                ),
                size_bytes=len(file_bytes),
                content_sha256=hashlib.sha256(file_bytes).hexdigest(),
            )
        )
    ranked_csv = output_dir / str(config.get("output_csv_name") or "ranked_designs.csv")
    if ranked:
        pl.DataFrame(ranked).write_csv(ranked_csv)
        warnings = []
    else:
        pl.DataFrame(
            schema={
                "design": pl.String,
                "filename": pl.String,
                "rank_score": pl.Float64,
                "dockq": pl.Float64,
                "iptm": pl.Float64,
                "interface_score": pl.Float64,
            }
        ).write_csv(ranked_csv)
        warnings = [f"{step_name} found no structures with usable ranking metrics"]
    WORKFLOW_OUTPUT_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="ranked_structures",
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(structures_dir),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                metadata={
                    "step_name": step_name,
                    "structure_count": len(ranked),
                    "structure_patterns": ("*.pdb", "*.cif"),
                },
                files=ranked_structure_files,
            ),
            volume_app_output(
                name="ranked_designs",
                kind=ArtifactKind.TABLE,
                remote_path=str(ranked_csv),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={"step_name": step_name, "rows": len(ranked)},
                files=[
                    ArtifactFile(
                        path=ranked_csv.name,
                        role="ranked_designs",
                        media_type="text/csv",
                        size_bytes=ranked_csv.stat().st_size,
                        content_sha256=_file_sha256(ranked_csv),
                    )
                ],
            ),
        ],
        warnings=warnings,
    )
