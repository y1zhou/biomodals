"""PPIFlow workflow definition built on the reusable workflow runtime."""

from __future__ import annotations

import ast
import os
import shlex
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import modal
import polars as pl
import yaml

from biomodals.app.bioinfo import rosetta_app
from biomodals.app.design import ligandmpnn_app, ppiflow_app
from biomodals.app.fold import alphafold3_app, flowpacker_app
from biomodals.app.score import af3score_app, dockq_app
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import (
    AppRunLayout,
    volume_app_output,
    volume_path_from_mount_path,
)
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppConfig,
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    NodeExecutionPolicy,
    NodePlacement,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE
from biomodals.workflow.core import (
    AppBackedNode,
    NodeRunContext,
    RemoteNodeSubmission,
    Workflow,
    WorkflowNativeNode,
    orchestrator,
    print_workflow_dag,
)
from biomodals.workflow.core.artifact_availability import (
    ArtifactAvailability,
    check_external_artifact_status,
)
from biomodals.workflow.ppiflow import coordinators as ppiflow_coordinators
from biomodals.workflow.ppiflow import manifests as ppiflow_manifests
from biomodals.workflow.ppiflow import staging as ppiflow_staging
from biomodals.workflow.ppiflow import tables as ppiflow_tables

PPI_FLOW_OUTPUT_LAYOUT = (
    "stage1/",
    "stage2/",
    "design_output/",
    "design_output/ranked_designs.csv",
    "design_output/design_report.md",
    "design_output/design_report.html",
)
PPI_FLOW_APP_STEPS = ("PPIFlowStep", "PartialStep")
PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS = (
    "outputs/*.pdb",
    "outputs/**/*.pdb",
    "outputs/*.cif",
    "outputs/**/*.cif",
)
APP_RUN_OUTPUT_STRUCTURE_PATTERNS = PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS

DEPENDENCY_APPS = (
    "ppiflow",
    "rosetta",
    "flowpacker",
    "ligandmpnn",
    "dockq",
    "af3score",
    "alphafold3",
)
CONF = AppConfig(
    tags={"depends_on": "-".join(DEPENDENCY_APPS)},
    depends_on_apps=DEPENDENCY_APPS,
    name="PPIFlowWorkflow",
    package_name="biomodals-ppiflow-workflow",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)
PPI_FLOW_OUTPUT_VOLUME = ppiflow_app.CONF.output_volume
PPI_FLOW_OUTPUT_VOLUME_NAME = ppiflow_app.CONF.output_volume_name
PPI_FLOW_OUTPUT_MOUNTPOINT = ppiflow_app.CONF.output_volume_mountpoint
FLOWPACKER_OUTPUT_VOLUME = flowpacker_app.CONF.output_volume
FLOWPACKER_OUTPUT_VOLUME_NAME = flowpacker_app.CONF.output_volume_name
FLOWPACKER_OUTPUT_MOUNTPOINT = flowpacker_app.CONF.output_volume_mountpoint
AF3SCORE_OUTPUT_VOLUME = af3score_app.CONF.output_volume
AF3SCORE_OUTPUT_VOLUME_NAME = af3score_app.CONF.output_volume_name
AF3SCORE_OUTPUT_MOUNTPOINT = af3score_app.CONF.output_volume_mountpoint
ROSETTA_OUTPUT_VOLUME = rosetta_app.CONF.output_volume
ROSETTA_OUTPUT_VOLUME_NAME = rosetta_app.CONF.output_volume_name
ROSETTA_OUTPUT_MOUNTPOINT = rosetta_app.CONF.output_volume_mountpoint
WORKFLOW_OUTPUT_VOLUME = orchestrator.OUT_VOLUME
WORKFLOW_OUTPUT_VOLUME_NAME = orchestrator.OUT_VOLUME_NAME
WORKFLOW_OUTPUT_MOUNTPOINT = orchestrator.CONF.output_volume_mountpoint
PPI_FLOW_SOURCE_VOLUME_ROOTS = {
    PPI_FLOW_OUTPUT_VOLUME_NAME: PPI_FLOW_OUTPUT_MOUNTPOINT,
    FLOWPACKER_OUTPUT_VOLUME_NAME: FLOWPACKER_OUTPUT_MOUNTPOINT,
    AF3SCORE_OUTPUT_VOLUME_NAME: AF3SCORE_OUTPUT_MOUNTPOINT,
    ROSETTA_OUTPUT_VOLUME_NAME: ROSETTA_OUTPUT_MOUNTPOINT,
    WORKFLOW_OUTPUT_VOLUME_NAME: WORKFLOW_OUTPUT_MOUNTPOINT,
}
PPI_FLOW_SOURCE_VOLUME_MOUNTS = {
    PPI_FLOW_OUTPUT_MOUNTPOINT: PPI_FLOW_OUTPUT_VOLUME,
    FLOWPACKER_OUTPUT_MOUNTPOINT: FLOWPACKER_OUTPUT_VOLUME,
    AF3SCORE_OUTPUT_MOUNTPOINT: AF3SCORE_OUTPUT_VOLUME,
    ROSETTA_OUTPUT_MOUNTPOINT: ROSETTA_OUTPUT_VOLUME,
    WORKFLOW_OUTPUT_MOUNTPOINT: WORKFLOW_OUTPUT_VOLUME,
}


def _reload_ppiflow_source_volumes() -> None:
    PPI_FLOW_OUTPUT_VOLUME.reload()
    FLOWPACKER_OUTPUT_VOLUME.reload()
    AF3SCORE_OUTPUT_VOLUME.reload()
    ROSETTA_OUTPUT_VOLUME.reload()
    WORKFLOW_OUTPUT_VOLUME.reload()


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def select_ppiflow_structure_files(
    *,
    artifacts: list[WorkflowArtifact],
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[tuple[str, bytes]]:
    """Read structure files from mounted PPIFlow workflow artifacts."""
    _reload_ppiflow_source_volumes()
    return ppiflow_staging.select_structure_files_from_artifacts(
        artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=patterns,
        max_files=max_files,
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def check_ppiflow_external_artifact(
    artifact: WorkflowArtifact,
) -> ArtifactAvailability:
    """Validate app-owned artifacts referenced by the PPIFlow workflow."""
    _reload_ppiflow_source_volumes()
    return check_external_artifact_status(
        artifact,
        workflow_volume_name=orchestrator.OUT_VOLUME_NAME,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def copy_ppiflow_structure_artifacts(
    *,
    artifacts: list[WorkflowArtifact],
    run_id: str,
    node_id: str,
    attempt_id: str,
    output_name: str,
    metadata: dict[str, object] | None = None,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> AppRunResult:
    """Copy selected structure artifacts into the workflow output volume."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts=artifacts,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=patterns,
        max_files=max_files,
    )
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / sanitize_filename(attempt_id)
        / sanitize_filename(output_name)
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, file_bytes in selected:
        (output_dir / sanitize_filename(file_name)).write_bytes(file_bytes)
    WORKFLOW_OUTPUT_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name=output_name,
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(output_dir),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                metadata={
                    "structure_count": len(selected),
                    "files": [file_name for file_name, _ in selected],
                }
                | dict(metadata or {}),
            )
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def normalize_ppiflow_stage2_input(
    *,
    storage: VolumePath,
    config: dict[str, object],
    run_id: str,
    node_id: str,
    attempt_id: str,
    step_name: str,
) -> AppRunResult:
    """Normalize Stage2Input structures into a workflow-owned manifest."""
    _reload_ppiflow_source_volumes()
    structure_artifact = WorkflowArtifact(
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
        / sanitize_filename(attempt_id)
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


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def filter_ppiflow_artifacts(
    *,
    structures: list[WorkflowArtifact],
    scores: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None = None,
    config: dict[str, object],
    run_id: str,
    node_id: str,
    attempt_id: str,
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
        / sanitize_filename(attempt_id)
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


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def derive_ppiflow_fixed_positions(
    *,
    artifacts: list[WorkflowArtifact],
    config: dict[str, object],
    run_id: str,
    node_id: str,
    attempt_id: str,
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
        / sanitize_filename(attempt_id)
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
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def rank_ppiflow_artifacts(
    *,
    structures: list[WorkflowArtifact],
    score_artifacts: list[WorkflowArtifact],
    config: dict[str, object],
    run_id: str,
    node_id: str,
    attempt_id: str,
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

    structure_by_key = {}
    for name, data in selected_structures:
        key = ppiflow_tables.candidate_key(name)
        if key in structure_by_key:
            raise ValueError(f"Duplicate PPIFlow candidate identity: {key!r}")
        structure_by_key[key] = (name, data)
    ranked = ppiflow_tables.ranked_design_rows(
        structures=selected_structures,
        score_frames=score_frames,
        gentype=str(config.get("gentype") or "binder"),
        dockq_threshold=float(config.get("dockq_threshold", 0.49)),
    )
    if not ranked:
        raise ValueError(f"{step_name} found no structures with usable ranking metrics")

    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / sanitize_filename(attempt_id)
        / "ranked"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True)
    for row in ranked:
        file_name, file_bytes = structure_by_key[str(row["design"])]
        (structures_dir / sanitize_filename(file_name)).write_bytes(file_bytes)
    ranked_csv = output_dir / str(config.get("output_csv_name") or "ranked_designs.csv")
    pl.DataFrame(ranked).write_csv(ranked_csv)
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
            ),
            volume_app_output(
                name="ranked_designs",
                kind=ArtifactKind.TABLE,
                remote_path=str(ranked_csv),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={"step_name": step_name, "rows": len(ranked)},
            ),
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def stage_af3score_inputs(
    *,
    artifacts: list[WorkflowArtifact],
    run_name: str,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Stage selected PDB files into AF3Score's input directory."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts=artifacts,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=patterns,
        max_files=max_files,
    )
    layout = AppRunLayout.from_run_root(
        Path(AF3SCORE_OUTPUT_MOUNTPOINT) / sanitize_filename(run_name)
    )
    if layout.inputs_dir.exists():
        shutil.rmtree(layout.inputs_dir)
    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    input_names = []
    for file_name, file_bytes in selected:
        pdb_name = f"{sanitize_filename(ppiflow_tables.candidate_key(file_name))}.pdb"
        if pdb_name in input_names:
            raise ValueError(f"Duplicate AF3Score staged input name: {pdb_name}")
        (layout.inputs_dir / pdb_name).write_bytes(file_bytes)
        input_names.append(pdb_name)
    AF3SCORE_OUTPUT_VOLUME.commit()
    return input_names


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def stage_rosetta_inputs(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None = None,
    run_name: str,
    run_id: str,
    rosetta_binary: str,
    rosetta_script: str | None = None,
    flags_file: str | None = None,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> dict[str, object]:
    """Stage selected structures and enqueue Rosetta jobs."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts=artifacts,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=patterns,
        max_files=max_files,
    )
    candidate_structures = ppiflow_staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=_candidate_manifest_frame_from_inputs(
            candidate_manifests or [],
            selected,
            step_name="RosettaInput",
        ),
    )
    safe_run_name = sanitize_filename(run_name)
    safe_run_id = sanitize_filename(run_id)
    layout = AppRunLayout.from_run_root(
        Path(ROSETTA_OUTPUT_MOUNTPOINT) / f"{safe_run_name}-{safe_run_id}"
    )
    if layout.run_root.exists():
        shutil.rmtree(layout.run_root)
    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    remote_script = None
    if rosetta_script:
        remote_script = "inputs/_script/workflow.xml"
        script_path = layout.run_root / remote_script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            _resolve_rosetta_config_text(rosetta_script, "rosetta_script"),
            encoding="utf-8",
        )
    remote_flags = None
    if flags_file:
        remote_flags = "inputs/_flags/workflow.flags"
        flags_path = layout.run_root / remote_flags
        flags_path.parent.mkdir(parents=True, exist_ok=True)
        flags_path.write_text(
            _resolve_rosetta_config_text(flags_file, "flags_file"),
            encoding="utf-8",
        )

    queue = modal.Queue.from_name(
        f"{rosetta_app.CONF.name}-queue-{safe_run_id}",
        create_if_missing=True,
    )
    rosetta_rows = ppiflow_staging.rosetta_job_manifest_rows(
        candidate_structures,
        rosetta_binary=rosetta_binary,
        rosetta_script=remote_script,
        flags_file=remote_flags,
    )
    for row, structure in zip(rosetta_rows, candidate_structures, strict=True):
        remote_pdb = str(row["pdb"])
        pdb_path = layout.run_root / remote_pdb
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_bytes(structure.data)
        queue.put({
            "index": int(row["index"]),
            "candidate_id": str(row["candidate_id"]),
            "binary": str(row["binary"]),
            "pdb": remote_pdb,
            "rosetta_script": remote_script,
            "flags_file": remote_flags,
        })
    job_manifest = ppiflow_staging.write_rosetta_job_manifest(
        rosetta_rows,
        layout.run_root / "rosetta_job_manifest.csv",
    )
    ROSETTA_OUTPUT_VOLUME.commit()
    return {
        "run_name": safe_run_name,
        "run_id": safe_run_id,
        "run_root": str(layout.run_root),
        "num_jobs": len(selected),
        "job_manifest": str(job_manifest),
    }


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def run_ppiflow_af3score_stage(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None = None,
    config: dict[str, object],
    step_name: str,
    run_name: str,
    run_id: str,
    node_id: str,
    attempt_id: str,
) -> AppRunResult:
    """Run AF3Score prepare/GPU/postprocess as one workflow stage call."""
    # TODO: tune CPU/memory/timeout once AF3Score candidate-stage telemetry exists.
    input_names = stage_af3score_inputs.get_raw_f()(
        artifacts=artifacts,
        run_name=run_name,
        patterns=_patterns_from_config(config, default=("*.pdb",)),
        max_files=config.get("max_structures"),
    )
    if not input_names:
        raise ValueError(f"{step_name} requires at least one AF3Score input")
    manifest_frame = _candidate_manifest_frame_from_inputs(
        candidate_manifests or [],
        [(input_name, b"") for input_name in input_names],
        step_name=step_name,
    )
    input_candidates = ppiflow_staging.candidate_structure_files_from_selected(
        [(input_name, b"") for input_name in input_names],
        manifest_frame=manifest_frame,
    )

    af3score_app.af3score_manage_lock.remote(run_name=run_name, acquire=True)
    try:
        task_spec = af3score_app.af3score_prepare.remote(
            run_name=run_name,
            input_files=input_names,
            num_jobs=int(config.get("num_jobs", config.get("max_batches", 10))),
            prepare_workers=int(config.get("prepare_workers", 8)),
        )
        chunk_specs = (
            task_spec.get("chunk_specs", [])
            if isinstance(task_spec, Mapping)
            else getattr(task_spec, "chunk_specs", [])
        )
        calls = []
        for chunk in chunk_specs:
            batch_name = (
                chunk["batch_name"] if isinstance(chunk, Mapping) else chunk.batch_name
            )
            batch_json_dir = (
                chunk["batch_json_dir"]
                if isinstance(chunk, Mapping)
                else chunk.batch_json_dir
            )
            batch_pdb_dir = (
                chunk["batch_pdb_dir"]
                if isinstance(chunk, Mapping)
                else chunk.batch_pdb_dir
            )
            calls.append(
                af3score_app.af3score_run.spawn(
                    run_name=run_name,
                    batch_name=batch_name,
                    batch_json_dir=batch_json_dir,
                    batch_pdb_dir=batch_pdb_dir,
                )
            )
        for call in calls:
            call.get()
        metrics = af3score_app.af3score_postprocess.remote(
            run_name=run_name,
            input_files=input_names,
        )
    finally:
        af3score_app.af3score_manage_lock.remote(run_name=run_name, acquire=False)

    metrics_csv = str(metrics["metrics_csv"])
    status = ppiflow_tables.score_table_status(
        requested_count=len(input_names),
        usable_rows=int(metrics.get("metrics_rows", 0)),
        failed_count=int(metrics.get("failed", 0)),
    )
    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        attempt_id=attempt_id,
        step_name=step_name,
        rows=[
            ppiflow_manifests.candidate_manifest_row(
                candidate_id=input_candidate.candidate_id,
                stage_name=step_name,
                stage_role="score",
                operation_mode="af3score",
                candidate_status=status.value,
                source_path=input_candidate.file_name,
                derived_path=metrics_csv,
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="scores",
                        volume_name=AF3SCORE_OUTPUT_VOLUME_NAME,
                        app_volume_path=volume_path_from_mount_path(
                            metrics_csv,
                            AF3SCORE_OUTPUT_MOUNTPOINT,
                            AF3SCORE_OUTPUT_VOLUME_NAME,
                        ).path,
                    )
                ],
                summary=metrics,
            )
            for input_candidate in input_candidates
        ],
    )
    return AppRunResult(
        status=status,
        outputs=[
            AppOutput(
                name="af3score_metrics",
                kind=ArtifactKind.SCORES,
                storage=volume_path_from_mount_path(
                    metrics_csv,
                    AF3SCORE_OUTPUT_MOUNTPOINT,
                    AF3SCORE_OUTPUT_VOLUME_NAME,
                ),
                metadata={
                    "step_name": step_name,
                    "run_name": run_name,
                }
                | dict(metrics),
            ),
            manifest_output,
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes={WORKFLOW_OUTPUT_MOUNTPOINT: WORKFLOW_OUTPUT_VOLUME},
)
def run_ppiflow_ligandmpnn_stage(
    *,
    selected_structures: list[dict[str, object]],
    config: dict[str, object],
    step_name: str,
    run_name: str,
    run_id: str,
    node_id: str,
    attempt_id: str,
    script_mode: str,
    model_type: str,
    cli_args: dict[str, str | int | float | bool],
) -> AppRunResult:
    """Run LigandMPNN for every selected PPIFlow candidate."""

    # TODO: tune CPU/memory/timeout once real candidate fan-out telemetry exists.
    def submit(task: ppiflow_coordinators.CandidateTask):
        structure = task.payload
        candidate_run_name = sanitize_filename(f"{run_name}-{task.candidate_id}")
        try:
            call = ligandmpnn_app.ligandmpnn_run.spawn(
                run_name=candidate_run_name,
                script_mode=script_mode,
                struct_bytes=structure["data"],
                seeds=_parse_seed_values(config.get("seeds", [0])),
                cli_args=cli_args,
                bias_aa_per_residue_bytes=config.get("bias_aa_per_residue_bytes"),
                omit_aa_per_residue_bytes=config.get("omit_aa_per_residue_bytes"),
            )
            result = AppRunResult.model_validate(call.get())
            status = result.status
            outputs = _ligandmpnn_stage_outputs(
                result,
                candidate_id=task.candidate_id,
                step_name=step_name,
                selected_structure=str(structure["file_name"]),
            )
            return ppiflow_coordinators.CandidateOutcome(
                candidate_id=task.candidate_id,
                status=status,
                outputs={"app_outputs": outputs},
            )
        except Exception as exc:  # noqa: BLE001
            return ppiflow_coordinators.CandidateOutcome(
                candidate_id=task.candidate_id,
                status=AppRunStatus.FAILED,
                error=str(exc),
            )

    tasks = [
        ppiflow_coordinators.CandidateTask(
            candidate_id=str(structure["candidate_id"]),
            payload=structure,
        )
        for structure in selected_structures
    ]
    outcomes = ppiflow_coordinators.run_candidate_tasks(
        tasks,
        submit,
        candidate_concurrency=int(
            config.get(
                "candidate_concurrency",
                ppiflow_coordinators.DEFAULT_CANDIDATE_CONCURRENCY,
            )
        ),
    )
    outputs = [
        output
        for outcome in outcomes
        for output in outcome.outputs.get("app_outputs", [])
        if isinstance(output, AppOutput)
    ]
    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        attempt_id=attempt_id,
        step_name=step_name,
        rows=ppiflow_coordinators.outcome_manifest_rows(
            stage_name=step_name,
            stage_role="sequence_design",
            operation_mode=model_type,
            outcomes=outcomes,
        ),
    )
    return AppRunResult(
        status=ppiflow_coordinators.status_from_candidate_outcomes(outcomes),
        outputs=[*outputs, manifest_output],
        warnings=[
            f"{outcome.candidate_id}: {outcome.error}"
            for outcome in outcomes
            if outcome.error
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes={
        PPI_FLOW_OUTPUT_MOUNTPOINT: PPI_FLOW_OUTPUT_VOLUME,
        WORKFLOW_OUTPUT_MOUNTPOINT: WORKFLOW_OUTPUT_VOLUME,
    },
)
def run_ppiflow_partial_stage(
    *,
    selected_structures: list[dict[str, object]],
    config: dict[str, object],
    step_name: str,
    run_name: str,
    run_id: str,
    node_id: str,
    attempt_id: str,
    fixed_positions_by_candidate: dict[str, str] | None = None,
) -> AppRunResult:
    """Run PPIFlow partial design for every selected candidate."""
    # TODO: tune CPU/memory/timeout once PPIFlow partial fan-out telemetry exists.
    PPI_FLOW_OUTPUT_VOLUME.reload()
    raw_args_template = deepcopy(config.get("args", config))
    if not isinstance(raw_args_template, dict):
        raise ValueError(f"PPIFlow step {step_name!r} args must be a mapping")
    staged_paths: dict[str, str] = {}
    field_name = "complex_pdb" if "complex_pdb" in raw_args_template else "input_pdb"
    for structure in selected_structures:
        candidate_id = str(structure["candidate_id"])
        staged_path = (
            Path(PPI_FLOW_OUTPUT_MOUNTPOINT)
            / sanitize_filename(run_name)
            / sanitize_filename(step_name)
            / sanitize_filename(candidate_id)
            / sanitize_filename(field_name)
            / sanitize_filename(str(structure["file_name"]))
        )
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_bytes(_bytes_payload(structure["data"], "structure data"))
        staged_paths[candidate_id] = str(staged_path)
    PPI_FLOW_OUTPUT_VOLUME.commit()

    def submit(task: ppiflow_coordinators.CandidateTask):
        try:
            raw_args = deepcopy(raw_args_template)
            raw_args[field_name] = staged_paths[task.candidate_id]
            if "fixed_positions" not in raw_args and fixed_positions_by_candidate:
                fixed_positions = fixed_positions_by_candidate.get(task.candidate_id)
                if fixed_positions:
                    raw_args["fixed_positions"] = fixed_positions
            app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
            call = ppiflow_app.ppiflow_run_workflow.spawn(
                args=app_args,
                run_name=sanitize_filename(f"{run_name}-{task.candidate_id}"),
            )
            result = AppRunResult.model_validate(call.get())
            return ppiflow_coordinators.CandidateOutcome(
                candidate_id=task.candidate_id,
                status=result.status,
                outputs={"app_outputs": result.outputs},
            )
        except Exception as exc:  # noqa: BLE001
            return ppiflow_coordinators.CandidateOutcome(
                candidate_id=task.candidate_id,
                status=AppRunStatus.FAILED,
                error=str(exc),
            )

    tasks = [
        ppiflow_coordinators.CandidateTask(
            candidate_id=str(structure["candidate_id"]),
            payload=structure,
        )
        for structure in selected_structures
    ]
    outcomes = ppiflow_coordinators.run_candidate_tasks(
        tasks,
        submit,
        candidate_concurrency=int(
            config.get(
                "candidate_concurrency",
                ppiflow_coordinators.DEFAULT_CANDIDATE_CONCURRENCY,
            )
        ),
    )
    outputs = [
        output
        for outcome in outcomes
        for output in outcome.outputs.get("app_outputs", [])
        if isinstance(output, AppOutput)
    ]
    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        attempt_id=attempt_id,
        step_name=step_name,
        rows=ppiflow_coordinators.outcome_manifest_rows(
            stage_name=step_name,
            stage_role="partial_design",
            operation_mode="ppiflow_partial",
            outcomes=outcomes,
        ),
    )
    return AppRunResult(
        status=ppiflow_coordinators.status_from_candidate_outcomes(outcomes),
        outputs=[*outputs, manifest_output],
        warnings=[
            f"{outcome.candidate_id}: {outcome.error}"
            for outcome in outcomes
            if outcome.error
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes={WORKFLOW_OUTPUT_MOUNTPOINT: WORKFLOW_OUTPUT_VOLUME},
)
def run_ppiflow_refold_stage(
    *,
    selected_structures: list[dict[str, object]],
    config: dict[str, object],
    step_name: str,
    run_name: str,
    run_id: str,
    node_id: str,
    attempt_id: str,
) -> AppRunResult:
    """Run AlphaFold3 refolding for every selected candidate."""
    # TODO: tune CPU/memory/timeout/GPU once ReFold candidate-stage telemetry exists.
    outcomes = []
    outputs = []
    for structure in selected_structures:
        candidate_id = str(structure["candidate_id"])
        candidate_run_name = sanitize_filename(f"{run_name}-{candidate_id}")
        try:
            candidate_outputs = _run_one_refold_candidate(
                structure_name=str(structure["file_name"]),
                structure_bytes=_bytes_payload(structure["data"], "structure data"),
                candidate_id=candidate_id,
                run_name=candidate_run_name,
                step_name=step_name,
                config=config,
            )
            outputs.extend(candidate_outputs)
            outcomes.append(
                ppiflow_coordinators.CandidateOutcome(
                    candidate_id,
                    AppRunStatus.SUCCEEDED,
                )
            )
        except Exception as exc:  # noqa: BLE001
            outcomes.append(
                ppiflow_coordinators.CandidateOutcome(
                    candidate_id,
                    AppRunStatus.FAILED,
                    error=str(exc),
                )
            )
    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        attempt_id=attempt_id,
        step_name=step_name,
        rows=ppiflow_coordinators.outcome_manifest_rows(
            stage_name=step_name,
            stage_role="refold",
            operation_mode="alphafold3",
            outcomes=outcomes,
        ),
    )
    return AppRunResult(
        status=ppiflow_coordinators.status_from_candidate_outcomes(outcomes),
        outputs=[*outputs, manifest_output],
        warnings=[
            f"{outcome.candidate_id}: {outcome.error}"
            for outcome in outcomes
            if outcome.error
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def run_ppiflow_rosetta_stage(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None = None,
    config: dict[str, object],
    step_name: str,
    run_name: str,
    run_id: str,
    node_id: str,
    attempt_id: str,
) -> AppRunResult:
    """Stage Rosetta candidates, run worker pods, and classify outputs."""
    # TODO: tune CPU/memory/timeout and pod sizing once Rosetta stage telemetry exists.
    staged = stage_rosetta_inputs.get_raw_f()(
        artifacts=artifacts,
        candidate_manifests=candidate_manifests,
        run_name=run_name,
        run_id=sanitize_filename(f"{run_id}-{node_id}-{attempt_id}"),
        rosetta_binary=str(config.get("rosetta_binary", "relax")),
        rosetta_script=config.get("rosetta_script"),
        flags_file=config.get("flags_file"),
        patterns=None,
        max_files=config.get("max_structures"),
    )
    num_jobs = int(staged["num_jobs"])
    if num_jobs < 1:
        raise ValueError(f"{step_name} requires at least one Rosetta input")

    num_cpu_per_pod = min(30, max(1, num_jobs))
    max_num_pods = max(1, int(config.get("max_num_pods", 1)))
    num_pods = min(max_num_pods, (num_jobs + num_cpu_per_pod - 1) // num_cpu_per_pod)
    calls = [
        rosetta_app.run_rosetta.spawn(
            str(staged["run_name"]),
            str(staged["run_id"]),
            num_cpu_per_pod,
        )
        for _ in range(num_pods)
    ]
    worker_errors = []
    for call in calls:
        try:
            call.get()
        except Exception as exc:  # noqa: BLE001
            worker_errors.append(str(exc))
    ROSETTA_OUTPUT_VOLUME.reload()
    cleanup_warnings = []
    try:
        modal.Queue.objects.delete(f"{rosetta_app.CONF.name}-queue-{staged['run_id']}")
    except Exception as exc:  # noqa: BLE001
        cleanup_warnings.append(f"queue cleanup failed: {exc}")

    run_root = Path(str(staged["run_root"]))
    job_manifest = Path(str(staged["job_manifest"]))
    row_frame = pl.read_csv(job_manifest, infer_schema_length=0)
    rows = []
    outcomes = []
    for row in row_frame.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        expected_score = run_root / str(row["expected_score_file"])
        log_path = run_root / str(row["worker_log"])
        status = (
            AppRunStatus.SUCCEEDED
            if expected_score.is_file() and not worker_errors
            else AppRunStatus.FAILED
        )
        error = None if status == AppRunStatus.SUCCEEDED else "; ".join(worker_errors)
        outcomes.append(
            ppiflow_coordinators.CandidateOutcome(
                candidate_id,
                status,
                error=error or None,
            )
        )
        rows.append(
            ppiflow_manifests.candidate_manifest_row(
                candidate_id=candidate_id,
                stage_name=step_name,
                stage_role="rosetta",
                operation_mode=str(config.get("rosetta_binary", "relax")),
                candidate_status=status.value,
                source_path=str(row["pdb"]),
                derived_path=str(row["expected_output_dir"]),
                error=error or None,
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="score",
                        volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                        app_volume_path=str(
                            Path(str(staged["run_name"]) + "-" + str(staged["run_id"]))
                            / str(row["expected_score_file"])
                        ),
                        expected=True,
                    ),
                    ppiflow_manifests.candidate_file_record(
                        role="worker_log",
                        volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                        app_volume_path=str(
                            Path(str(staged["run_name"]) + "-" + str(staged["run_id"]))
                            / str(row["worker_log"])
                        ),
                        expected=log_path.is_file(),
                    ),
                ],
                summary={"index": row["index"], "num_pods": num_pods},
            )
        )

    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        attempt_id=attempt_id,
        step_name=step_name,
        rows=rows,
    )
    status = ppiflow_coordinators.status_from_candidate_outcomes(outcomes)
    warnings = [
        f"{outcome.candidate_id}: {outcome.error}"
        for outcome in outcomes
        if outcome.error
    ] + cleanup_warnings
    return AppRunResult(
        status=status,
        outputs=[
            volume_app_output(
                name="rosetta_outputs",
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(staged["run_root"]),
                mount_root=ROSETTA_OUTPUT_MOUNTPOINT,
                volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                metadata={
                    "step_name": step_name,
                    "run_name": str(staged["run_name"]),
                    "run_id": str(staged["run_id"]),
                    "num_jobs": num_jobs,
                    "num_pods": num_pods,
                    "structure_patterns": APP_RUN_OUTPUT_STRUCTURE_PATTERNS,
                },
            ),
            volume_app_output(
                name="rosetta_job_manifest",
                kind=ArtifactKind.TABLE,
                remote_path=str(staged["job_manifest"]),
                mount_root=ROSETTA_OUTPUT_MOUNTPOINT,
                volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={"step_name": step_name, "rows": num_jobs},
            ),
            manifest_output,
        ],
        warnings=warnings,
    )


@dataclass(frozen=True)
class PPIFlowModalNamespace:
    """Hydrated Modal objects carried across the orchestrator boundary."""

    ppiflow_run: modal.Function
    ppiflow_partial_stage: modal.Function
    ligandmpnn_stage: modal.Function
    flowpacker_run: modal.Function
    af3score_stage: modal.Function
    dockq_run: modal.Function
    rosetta_stage: modal.Function
    refold_stage: modal.Function
    select_structures: modal.Function
    copy_structures: modal.Function
    filter_artifacts: modal.Function
    derive_fixed_positions: modal.Function
    rank_artifacts: modal.Function
    stage2_input_manifest: modal.Function


@dataclass
class _ConfiguredAppStepNode(AppBackedNode):
    """Base class for configured PPIFlow app-backed workflow nodes."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def _run_name(self, context: NodeRunContext) -> str:
        run_name = sanitize_filename(
            str(self.config.get("run_name") or f"{context.run_id}-{self.step_name}")
        )
        return run_name

    def _select_structures(
        self,
        context: NodeRunContext,
        *,
        max_files: int | None = None,
        default_patterns: Sequence[str] | None = None,
    ) -> list[tuple[str, bytes]]:
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(
                f"PPIFlow workflow step {self.step_name!r} requires structure inputs"
            )
        patterns = _patterns_from_config(
            self.config,
            default=default_patterns,
        )
        return self.modal_namespace.select_structures.remote(
            artifacts=artifacts,
            patterns=patterns,
            max_files=max_files,
        )

    def _select_one_structure(
        self,
        context: NodeRunContext,
        *,
        default_patterns: Sequence[str] | None = None,
    ) -> tuple[str, bytes]:
        selected = self._select_structures(
            context,
            max_files=self.config.get("max_structures"),
            default_patterns=default_patterns,
        )
        explicit_index = "structure_index" in self.config
        if len(selected) > 1 and not explicit_index:
            raise ValueError(
                f"{self.step_name} selected {len(selected)} structures; set an "
                "explicit structure_index or structure_patterns/max_structures "
                "to avoid silently discarding candidates"
            )
        structure_index = int(self.config.get("structure_index", 0))
        if structure_index < 0 or structure_index >= len(selected):
            raise IndexError(
                f"{self.step_name} structure_index {structure_index} is out of "
                f"range for {len(selected)} selected structure(s)"
            )
        return selected[structure_index]


@dataclass
class _PPIFlowRunNode(_ConfiguredAppStepNode):
    """App-backed node implemented by the PPIFlow workflow app function."""

    def _app_kwargs(self, context: NodeRunContext) -> dict[str, object]:
        raw_args = self.config.get("args", self.config)
        if not isinstance(raw_args, dict):
            raise ValueError(f"PPIFlow step {self.step_name!r} args must be a mapping")

        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        return {"args": app_args, "run_name": self._run_name(context)}

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the PPIFlow app function directly from the orchestrator."""
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.ppiflow_run.spawn(
                **self._app_kwargs(context)
            ),
            function_name="ppiflow_run",
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Expose PPIFlow app output directories as structure artifacts."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.STRUCTURES,
            {
                "step_name": self.step_name,
                "structure_patterns": PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS,
            }
            | dict(metadata),
        )


@dataclass
class PPIFlowDesignNode(_PPIFlowRunNode):
    """Initial PPIFlow design step."""


@dataclass
class PPIFlowPartialNode(_PPIFlowRunNode):
    """PPIFlow partial-design step for stage 2."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit candidate-wide PPIFlow partial design."""
        selected_structures = ppiflow_staging.candidate_structure_files_from_selected(
            self._select_structures(
                context,
                max_files=self.config.get("max_structures"),
            ),
            manifest_frame=_candidate_manifest_frame_from_context(context),
        )
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.ppiflow_partial_stage.spawn(
                selected_structures=[
                    asdict(structure) for structure in selected_structures
                ],
                config=self.config,
                step_name=self.step_name,
                run_name=self._run_name(context),
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                fixed_positions_by_candidate=_fixed_positions_by_candidate(
                    context.inputs.get("structures") or [],
                    selected_structures,
                ),
            ),
            function_name="run_ppiflow_partial_stage",
            metadata={"structure_count": len(selected_structures)},
        )


@dataclass
class LigandMPNNNode(_ConfiguredAppStepNode):
    """LigandMPNN or AbMPNN design step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the LigandMPNN app function."""
        selected_structures = ppiflow_staging.candidate_structure_files_from_selected(
            self._select_structures(
                context,
                max_files=self.config.get("max_structures"),
            ),
            manifest_frame=_candidate_manifest_frame_from_context(context),
        )
        script_mode = str(self.config.get("script_mode", "run"))
        model_type = str(
            self.config.get(
                "model_type",
                "abmpnn" if self.step_name.startswith("AbMPNN") else "protein_mpnn",
            )
        )
        cli_kwargs = _ligandmpnn_cli_kwargs(
            self.config,
            script_mode=script_mode,
            model_type=model_type,
        )
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.ligandmpnn_stage.spawn(
                selected_structures=[
                    asdict(structure) for structure in selected_structures
                ],
                config=self.config,
                run_name=self._run_name(context),
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                script_mode=script_mode,
                model_type=model_type,
                cli_args=ligandmpnn_app.build_ligandmpnn_cli_args(**cli_kwargs),
                step_name=self.step_name,
            ),
            function_name="run_ppiflow_ligandmpnn_stage",
            metadata={"structure_count": len(selected_structures)},
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Expose LigandMPNN archives as structure artifacts."""
        result = AppRunResult.model_validate(result)
        if any(output.kind == ArtifactKind.TABLE for output in result.outputs):
            return result
        return _result_with_output_kind(
            result,
            ArtifactKind.STRUCTURES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class FlowPackerNode(_ConfiguredAppStepNode):
    """FlowPacker side-chain packing step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the FlowPacker app function."""
        selected_structures = self._select_structures(
            context,
            max_files=self.config.get("max_structures"),
        )
        kwargs = {
            key: self.config[key]
            for key in (
                "model_name",
                "use_confidence",
                "n_samples",
                "num_steps",
                "sample_coeff",
                "use_gt_masks",
                "inpaint",
                "save_traj",
                "seed",
            )
            if key in self.config
        }
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.flowpacker_run.spawn(
                input_files=selected_structures,
                run_name=self._run_name(context),
                **kwargs,
            ),
            function_name="run_flowpacker_workflow",
            metadata={"structure_count": len(selected_structures)},
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Expose FlowPacker archives as structure artifacts."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.STRUCTURES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class AF3ScoreNode(_ConfiguredAppStepNode):
    """AF3Score structure scoring step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit AF3Score as one recoverable workflow stage call."""
        structures = context.inputs.get("structures") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.af3score_stage.spawn(
                artifacts=structures,
                candidate_manifests=context.inputs.get("candidate_manifest") or [],
                config=self.config,
                step_name=self.step_name,
                run_name=self._run_name(context),
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
            ),
            function_name="run_ppiflow_af3score_stage",
        )


@dataclass
class _RosettaNode(_ConfiguredAppStepNode):
    """Base class for PPIFlow Rosetta steps."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit Rosetta staging and workers as one workflow stage call."""
        structures = context.inputs.get("structures") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.rosetta_stage.spawn(
                artifacts=structures,
                candidate_manifests=context.inputs.get("candidate_manifest") or [],
                config=self.config,
                step_name=self.step_name,
                run_name=self._run_name(context),
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
            ),
            function_name="run_ppiflow_rosetta_stage",
        )


@dataclass
class RosettaFixNode(_RosettaNode):
    """Rosetta fixed-position analysis step."""


@dataclass
class RosettaRelaxNode(_RosettaNode):
    """Rosetta relaxation step."""


@dataclass
class ReFoldNode(_ConfiguredAppStepNode):
    """AlphaFold3 refolding step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit AlphaFold3 refolding as one workflow stage call."""
        selected_structures = ppiflow_staging.candidate_structure_files_from_selected(
            self._select_structures(
                context,
                max_files=self.config.get("max_structures"),
                default_patterns=("*.pdb",),
            ),
            manifest_frame=_candidate_manifest_frame_from_context(context),
        )
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.refold_stage.spawn(
                selected_structures=[
                    asdict(structure) for structure in selected_structures
                ],
                config=self.config,
                step_name=self.step_name,
                run_name=self._run_name(context),
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
            ),
            function_name="run_ppiflow_refold_stage",
            metadata={"structure_count": len(selected_structures)},
        )


@dataclass
class DockQNode(_ConfiguredAppStepNode):
    """DockQ model/reference scoring step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the DockQ app function."""
        references = ppiflow_staging.candidate_structure_files_from_selected(
            self._select_structures(context),
            manifest_frame=_candidate_manifest_frame_from_context(context),
        )
        model_artifacts = context.inputs.get("models") or []
        if not model_artifacts:
            raise ValueError(f"{self.step_name} requires model structure inputs")
        models = ppiflow_staging.candidate_structure_files_from_selected(
            self.modal_namespace.select_structures.remote(
                artifacts=model_artifacts,
                patterns=None,
                max_files=self.config.get("max_models"),
            ),
            manifest_frame=_candidate_manifest_frame_from_context(context),
        )
        pairs = ppiflow_staging.prepare_dockq_pairs_by_candidate(
            references=references,
            models=models,
            mapping=self.config.get("mapping"),
        )
        if not pairs:
            raise ValueError(f"{self.step_name} did not find any DockQ pairs")
        dockq_args = self.config.get("dockq_args", "--short")
        if isinstance(dockq_args, str):
            dockq_args = shlex.split(dockq_args)
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.dockq_run.spawn(
                pairs=pairs,
                run_name=self._run_name(context),
                dockq_args=dockq_args,
            ),
            function_name="run_dockq_workflow",
            metadata={"pair_count": len(pairs)},
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Attach DockQ workflow metadata to score outputs."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.SCORES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class ExistingStructuresNode(WorkflowNativeNode):
    """Reference existing structures for stage-2-only PPIFlow runs."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    storage: VolumePath
    config: dict[str, Any] = field(default_factory=dict)
    placement: NodePlacement = NodePlacement.REMOTE

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit Stage2Input normalization as a remote workflow adapter."""
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.stage2_input_manifest.spawn(
                storage=self.storage,
                config=self.config,
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                step_name=self.step_name,
            ),
            function_name="normalize_ppiflow_stage2_input",
        )


@dataclass
class FilterStructuresNode(WorkflowNativeNode):
    """Filter structures using score artifacts."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute filtering logic."""
        structures = context.inputs.get("structures") or []
        scores = context.inputs.get("scores") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        if not scores:
            raise ValueError(f"{self.step_name} requires score inputs")
        return AppRunResult.model_validate(
            self.modal_namespace.filter_artifacts.remote(
                structures=structures,
                scores=scores,
                candidate_manifests=context.inputs.get("candidate_manifest") or [],
                config=self.config,
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                step_name=self.step_name,
            )
        )


@dataclass
class FixedPositionsNode(WorkflowNativeNode):
    """Convert Rosetta residue energies into fixed-position constraints."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute fixed-position conversion logic."""
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return AppRunResult.model_validate(
            self.modal_namespace.derive_fixed_positions.remote(
                artifacts=artifacts,
                config=self.config,
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                step_name=self.step_name,
            )
        )


@dataclass
class RankNode(WorkflowNativeNode):
    """Rank final designs."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute ranking logic."""
        structures = context.inputs.get("structures") or []
        score_artifacts = [
            artifact
            for input_name, artifact_list in context.inputs.items()
            if input_name != "structures"
            for artifact in artifact_list
        ]
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return AppRunResult.model_validate(
            self.modal_namespace.rank_artifacts.remote(
                structures=structures,
                score_artifacts=score_artifacts,
                config=self.config,
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                step_name=self.step_name,
            )
        )


@dataclass
class ReportNode(WorkflowNativeNode):
    """Write the final design report."""

    step_name: str
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute report generation logic."""
        artifacts = [
            artifact
            for artifact_list in context.inputs.values()
            for artifact in artifact_list
        ]
        ranked_rows: list[dict[str, object]] = []
        attrition_rows: list[dict[str, object]] = []
        for artifact in context.inputs.get("rank", []):
            path = ppiflow_staging.artifact_mount_path(
                artifact,
                PPI_FLOW_SOURCE_VOLUME_ROOTS,
            )
            if path.is_file() and path.suffix == ".csv":
                ranked_rows.extend(
                    pl.read_csv(path, infer_schema_length=0).iter_rows(named=True)
                )
        manifest_frames = []
        audit_frames = []
        for artifact in artifacts:
            path = ppiflow_staging.artifact_mount_path(
                artifact,
                PPI_FLOW_SOURCE_VOLUME_ROOTS,
            )
            if artifact.kind != ArtifactKind.TABLE or not path.is_file():
                continue
            if path.suffix == ".parquet":
                try:
                    manifest_frames.append(ppiflow_manifests.read_manifest(path))
                except ValueError:
                    continue
            elif path.name == "filter_audit.csv":
                audit_frames.append(pl.read_csv(path, infer_schema_length=0))
        for index, manifest_frame in enumerate(manifest_frames):
            audit_frame = audit_frames[index] if index < len(audit_frames) else None
            attrition_rows.extend(
                ppiflow_tables.candidate_attrition_rows(
                    stage_name=str(
                        manifest_frame.get_column("stage_name").item(0)
                        if manifest_frame.height
                        else "unknown"
                    ),
                    manifest_frame=manifest_frame,
                    audit_frame=audit_frame,
                )
            )
        markdown = ppiflow_tables.render_report_markdown(
            step_name=self.step_name,
            artifact_count=len(artifacts),
            ranked_rows=ranked_rows,
            attrition_rows=attrition_rows,
            max_rows=int(self.config.get("max_rows", 25)),
        )
        report_filename = str(
            self.config.get("report_filename") or "design_report.html"
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="design_report",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=markdown.encode("utf-8"),
                        filename="design_report.md",
                        media_type="text/markdown",
                    ),
                    metadata={"step_name": self.step_name},
                ),
                AppOutput(
                    name="design_report_html",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=ppiflow_tables.render_report_html(markdown).encode(
                            "utf-8"
                        ),
                        filename=report_filename,
                        media_type="text/html",
                    ),
                    metadata={"step_name": self.step_name},
                ),
            ],
        )


def _result_with_output_kind(
    result: AppRunResult,
    kind: ArtifactKind,
    metadata: Mapping[str, object],
) -> AppRunResult:
    outputs = [
        output.model_copy(
            update={
                "kind": kind,
                "metadata": dict(output.metadata) | dict(metadata),
            }
        )
        for output in result.outputs
    ]
    return result.model_copy(update={"outputs": outputs})


def _ligandmpnn_stage_outputs(
    result: AppRunResult,
    *,
    candidate_id: str,
    step_name: str,
    selected_structure: str,
) -> list[AppOutput]:
    adapted = _result_with_output_kind(
        result,
        ArtifactKind.STRUCTURES,
        {
            "candidate_id": candidate_id,
            "step_name": step_name,
            "selected_structure": selected_structure,
        },
    )
    sequence_rows = []
    for output in adapted.outputs:
        if not isinstance(output.storage, InlineBytes):
            continue
        if output.storage.media_type != ZSTD_MEDIA_TYPE:
            continue
        sequence_rows.extend(
            ppiflow_tables.mpnn_sequence_rows_from_fasta_files(
                ppiflow_staging.files_from_tar_zst_bytes(
                    output.storage.data,
                    suffixes=(".fa", ".faa", ".fasta"),
                ),
                stage_name=step_name,
                parent_candidate_id=candidate_id,
            )
        )
    sequence_output = _inline_csv_table_output(
        name="mpnn_seqs",
        filename=f"{sanitize_filename(candidate_id)}_mpnn_seqs.csv",
        rows=sequence_rows,
        metadata={"candidate_id": candidate_id, "step_name": step_name},
    )
    return [*adapted.outputs, *([sequence_output] if sequence_output else [])]


def _inline_csv_table_output(
    *,
    name: str,
    filename: str,
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> AppOutput | None:
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


def _write_candidate_manifest_output(
    *,
    run_id: str,
    node_id: str,
    attempt_id: str,
    step_name: str,
    rows: Sequence[Mapping[str, object]],
) -> AppOutput:
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / sanitize_filename(attempt_id)
        / ppiflow_manifests.MANIFEST_OUTPUT_NAME
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    manifest_path = output_dir / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(rows, manifest_path)
    WORKFLOW_OUTPUT_VOLUME.commit()
    return ppiflow_manifests.manifest_artifact_output(
        manifest_path=manifest_path,
        mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
        volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
        stage_name=step_name,
        row_count=len(rows),
    )


def _run_one_refold_candidate(
    *,
    structure_name: str,
    structure_bytes: bytes,
    candidate_id: str,
    run_name: str,
    step_name: str,
    config: Mapping[str, object],
) -> list[AppOutput]:
    conf = _af3_config_for_refold(
        structure_name=structure_name,
        structure_bytes=structure_bytes,
        run_name=run_name,
        config=config,
    )
    json_bytes = conf.model_dump_json().encode("utf-8")
    if bool(config.get("search_msa", False)):
        json_bytes = alphafold3_app.run_data_pipeline.remote(
            json_bytes=json_bytes,
            copy_msa_to_ssd=True,
        )
    function_call = alphafold3_app.run_inference_pipeline.spawn(
        json_bytes=json_bytes,
        recycle=int(config.get("recycle", 10)),
        sample=int(config.get("sample", 5)),
        model_seeds=list(conf.modelSeeds),
    )
    tarball_bytes = function_call.get()
    metric_rows = ppiflow_tables.refold_metric_rows_from_json_files(
        ppiflow_staging.files_from_tar_zst_bytes(
            tarball_bytes,
            suffixes=(".json",),
        ),
        stage_name=step_name,
    )
    metrics_output = _inline_csv_table_output(
        name=f"refold_quality_metrics_{sanitize_filename(candidate_id)}",
        filename=f"{sanitize_filename(candidate_id)}_refold_quality_metrics.csv",
        rows=metric_rows,
        metadata={
            "candidate_id": candidate_id,
            "step_name": step_name,
            "source_structure": structure_name,
        },
    )
    outputs = [
        AppOutput(
            name=f"alphafold3_refolded_structures_{sanitize_filename(candidate_id)}",
            kind=ArtifactKind.STRUCTURES,
            storage=InlineBytes(
                data=tarball_bytes,
                filename=f"{run_name}_alphafold3.tar.zst",
                media_type=ZSTD_MEDIA_TYPE,
            ),
            metadata={
                "step_name": step_name,
                "run_name": run_name,
                "candidate_id": candidate_id,
                "source_structure": structure_name,
                "archive_format": "tar.zst",
            },
        )
    ]
    if metrics_output is not None:
        outputs.append(metrics_output)
    return outputs


def _bytes_payload(value: object, label: str) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{label} must be bytes")
    return value


def _candidate_manifest_frame_from_inputs(
    candidate_manifests: Sequence[WorkflowArtifact],
    selected_structures: Sequence[tuple[str, bytes]],
    *,
    step_name: str,
) -> pl.DataFrame:
    frames = _read_candidate_manifest_artifacts(candidate_manifests)
    if frames:
        return pl.concat(frames, how="diagonal") if len(frames) > 1 else frames[0]

    rows = [
        ppiflow_manifests.candidate_manifest_row(
            candidate_id=ppiflow_tables.candidate_key(name),
            stage_name=step_name,
            stage_role="structure_selection",
            operation_mode="legacy_structure_keys",
            candidate_status=AppRunStatus.SUCCEEDED.value,
            source_path=name,
            derived_path=name,
            files=[
                ppiflow_manifests.candidate_file_record(
                    role="structure",
                    path=name,
                    size_bytes=len(data),
                )
            ],
        )
        for name, data in selected_structures
    ]
    return pl.DataFrame(rows)


def _candidate_manifest_frame_from_context(
    context: NodeRunContext,
) -> pl.DataFrame | None:
    frames = _read_candidate_manifest_artifacts(
        context.inputs.get("candidate_manifest") or []
    )
    if not frames:
        return None
    return pl.concat(frames, how="diagonal") if len(frames) > 1 else frames[0]


def _read_candidate_manifest_artifacts(
    artifacts: Sequence[WorkflowArtifact],
) -> list[pl.DataFrame]:
    frames = []
    for artifact in artifacts:
        if artifact.kind != ArtifactKind.TABLE:
            continue
        try:
            frames.append(
                ppiflow_manifests.read_manifest_volume_path(
                    storage=artifact.storage,
                    volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
                )
            )
        except (FileNotFoundError, ValueError, pl.exceptions.PolarsError):
            continue
    return frames


def _fixed_positions_by_candidate(
    artifacts: Sequence[WorkflowArtifact],
    selected_structures: Sequence[ppiflow_staging.CandidateStructureFile],
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for artifact in artifacts:
        raw_mapping = artifact.metadata.get("fixed_positions_by_structure")
        if isinstance(raw_mapping, Mapping):
            lookup.update({
                str(candidate_key): str(fixed_positions)
                for candidate_key, fixed_positions in raw_mapping.items()
                if fixed_positions
            })
        fixed_positions = artifact.metadata.get("fixed_positions")
        if fixed_positions:
            for structure in selected_structures:
                lookup.setdefault(structure.candidate_id, str(fixed_positions))

    by_candidate = {}
    for structure in selected_structures:
        keys = {
            structure.candidate_id,
            ppiflow_tables.candidate_key(structure.file_name),
        }
        if structure.source_path:
            keys.add(ppiflow_tables.candidate_key(structure.source_path))
        for key in keys:
            if key in lookup:
                by_candidate[structure.candidate_id] = lookup[key]
                break
    return by_candidate


def _parse_seed_values(value: object) -> list[int]:
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


def _patterns_from_config(
    config: Mapping[str, object],
    *,
    default: Sequence[str] | None = None,
) -> tuple[str, ...] | None:
    raw_patterns = config.get("structure_patterns") or config.get("patterns")
    if raw_patterns is None:
        return tuple(default) if default is not None else None
    if isinstance(raw_patterns, str):
        return tuple(
            pattern.strip() for pattern in raw_patterns.split(",") if pattern.strip()
        )
    return tuple(str(pattern) for pattern in raw_patterns)


def _resolve_rosetta_config_text(value: str, field_name: str) -> str:
    config_path = Path(value).expanduser()
    has_newline = "\n" in value
    looks_like_path = (
        config_path.suffix in {".xml", ".flags"} or "/" in value or "\\" in value
    )
    if looks_like_path and not has_newline and config_path.exists():
        return config_path.read_text(encoding="utf-8")
    if looks_like_path and not has_newline:
        raise FileNotFoundError(
            f"Rosetta {field_name} path was not found locally or in the mounted "
            f"container filesystem: {value}"
        )
    return value


def _inline_rosetta_config_files(steps_doc: dict[str, Any]) -> dict[str, Any]:
    staged_steps = deepcopy(steps_doc)
    for step_name in ("RosettaFixStep", "RosettaRelaxStep"):
        if step_name not in staged_steps:
            continue
        cfg = _step_cfg(staged_steps, step_name)
        for field_name in ("rosetta_script", "flags_file"):
            value = cfg.get(field_name)
            if isinstance(value, str):
                cfg[field_name] = _resolve_rosetta_config_text(value, field_name)
    return staged_steps


def _ligandmpnn_cli_kwargs(
    config: Mapping[str, object],
    *,
    script_mode: str,
    model_type: str,
) -> dict[str, object]:
    excluded = {
        "run_name",
        "seeds",
        "script_mode",
        "structure_index",
        "max_structures",
        "structure_patterns",
        "patterns",
        "bias_aa_per_residue_bytes",
        "omit_aa_per_residue_bytes",
    }
    allowed = set(ligandmpnn_app.build_ligandmpnn_cli_args.__annotations__)
    allowed.discard("return")
    kwargs = {
        key: value
        for key, value in config.items()
        if key in allowed and key not in excluded
    }
    kwargs["script_mode"] = script_mode
    kwargs["model_type"] = model_type
    return kwargs


def _af3_config_for_refold(
    *,
    structure_name: str,
    structure_bytes: bytes,
    run_name: str,
    config: Mapping[str, object],
):
    if config.get("af3_config_json") is not None:
        conf = alphafold3_app.AF3Config.model_validate_json(
            str(config["af3_config_json"])
        )
        conf.name = run_name
        return conf

    residue_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for line in structure_bytes.decode("utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain_id = line[21].strip() or "A"
        residue_id = line[22:27].strip()
        residue_key = (chain_id, residue_id)
        if residue_key in seen:
            continue
        seen.add(residue_key)
        residue_name = line[17:20].strip().upper()
        chains.setdefault(chain_id, []).append(residue_map.get(residue_name, "X"))
    if not chains:
        raise ValueError(f"Could not derive AlphaFold3 sequence from {structure_name}")

    return alphafold3_app.AF3Config(
        name=run_name,
        modelSeeds=[
            int(seed) for seed in _parse_seed_values(config.get("model_seeds", [1]))
        ],
        sequences=[
            alphafold3_app.AF3SequenceEntry(
                protein=alphafold3_app.AF3Protein(
                    id=chain_id,
                    sequence="".join(sequence),
                )
            )
            for chain_id, sequence in sorted(chains.items())
        ],
    )


def build_ppiflow_workflow(
    *,
    task_yaml_bytes: bytes,
    steps_yaml_bytes: bytes,
    stage: int | None = None,
    modal_namespace: PPIFlowModalNamespace | None = None,
) -> Workflow:
    """Build a PPIFlow workflow DAG from upstream-style YAML files."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    if modal_namespace is None:
        modal_namespace = PPIFlowModalNamespace(
            ppiflow_run=ppiflow_app.ppiflow_run_workflow,
            ppiflow_partial_stage=run_ppiflow_partial_stage,
            ligandmpnn_stage=run_ppiflow_ligandmpnn_stage,
            flowpacker_run=flowpacker_app.run_flowpacker_workflow,
            af3score_stage=run_ppiflow_af3score_stage,
            dockq_run=dockq_app.run_dockq_workflow,
            rosetta_stage=run_ppiflow_rosetta_stage,
            refold_stage=run_ppiflow_refold_stage,
            select_structures=select_ppiflow_structure_files,
            copy_structures=copy_ppiflow_structure_artifacts,
            filter_artifacts=filter_ppiflow_artifacts,
            derive_fixed_positions=derive_ppiflow_fixed_positions,
            rank_artifacts=rank_ppiflow_artifacts,
            stage2_input_manifest=normalize_ppiflow_stage2_input,
        )

    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _load_yaml_bytes(steps_yaml_bytes)
    task = _task_section(task_doc)
    enabled = _enabled_section(task_doc)
    gentype = str(task.get("gentype") or task.get("design_mode") or "binder")
    candidate_concurrency = ppiflow_coordinators.candidate_concurrency_from_config(
        task,
        steps_doc,
    )
    workflow = Workflow("ppiflow-v2")
    report_table_inputs: dict[str, Any] = {}

    stage1_tail = None
    if stage in {None, 1}:
        stage1_tail = _add_stage1_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            modal_namespace=modal_namespace,
            report_table_inputs=report_table_inputs,
            candidate_concurrency=candidate_concurrency,
        )

    if stage in {None, 2}:
        stage2_upstream = stage1_tail
        if stage == 2:
            stage2_upstream = workflow.add_node(
                _stage2_input_node(task, steps_doc, modal_namespace),
                id="stage2-existing-input",
            )
        _add_stage2_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            upstream=stage2_upstream,
            modal_namespace=modal_namespace,
            report_table_inputs=report_table_inputs,
            candidate_concurrency=candidate_concurrency,
        )

    return workflow


def _add_stage1_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    modal_namespace: PPIFlowModalNamespace,
    report_table_inputs: dict[str, Any],
    candidate_concurrency: int,
):
    tail = None
    if _step_enabled(enabled, "PPIFlowStep"):
        tail = workflow.add_node(
            PPIFlowDesignNode(
                "PPIFlowStep",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "PPIFlowStep",
                    candidate_concurrency,
                ),
            ),
            id="stage1-ppiflow-design",
        )

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage1"):
        mpnn_step = ("stage1-ligandmpnn", "MPNNStep_stage1")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage1"
    ):
        mpnn_step = ("stage1-abmpnn", "AbMPNNStep_stage1")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            LigandMPNNNode(
                step_name,
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    step_name,
                    candidate_concurrency,
                ),
            ),
            id=node_id,
            inputs=_structure_inputs(tail),
        )
        report_table_inputs["stage1_mpnn_seqs"] = tail.outputs(kind=ArtifactKind.TABLE)

    if _step_enabled(enabled, "FlowpackerStep_stage1"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage1",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FlowpackerStep_stage1",
                    candidate_concurrency,
                ),
            ),
            id="stage1-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage1"):
        score = workflow.add_node(
            AF3ScoreNode(
                "AF3scoreStep_stage1",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "AF3scoreStep_stage1",
                    candidate_concurrency,
                ),
            ),
            id="stage1-af3score",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FilterStep_stage1"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        tail = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage1",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FilterStep_stage1",
                    candidate_concurrency,
                ),
            ),
            id="stage1-filter",
            inputs=inputs,
        )
        report_table_inputs["stage1_filter_tables"] = tail.outputs(
            kind=ArtifactKind.TABLE
        )
    return tail


def _add_stage2_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    upstream,
    modal_namespace: PPIFlowModalNamespace,
    report_table_inputs: dict[str, Any],
    candidate_concurrency: int,
) -> None:
    tail = upstream
    if _step_enabled(enabled, "RosettaFixStep"):
        tail = workflow.add_node(
            RosettaFixNode(
                "RosettaFixStep",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "RosettaFixStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-rosetta-fix",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "RosettaFixStep") and _step_enabled(
        enabled, "PartialStep"
    ):
        tail = workflow.add_node(
            FixedPositionsNode(
                "FixedPositions",
                modal_namespace,
                {"gentype": gentype} | _step_cfg(steps, "FixedPositions"),
            ),
            id="stage2-fixed-positions",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "PartialStep"):
        tail = workflow.add_node(
            PPIFlowPartialNode(
                "PartialStep",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "PartialStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-partial-ppiflow",
            inputs=_structure_inputs(tail),
        )

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage2"):
        mpnn_step = ("stage2-ligandmpnn", "MPNNStep_stage2")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage2"
    ):
        mpnn_step = ("stage2-abmpnn", "AbMPNNStep_stage2")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            LigandMPNNNode(
                step_name,
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    step_name,
                    candidate_concurrency,
                ),
            ),
            id=node_id,
            inputs=_structure_inputs(tail),
        )
        report_table_inputs["mpnn_seqs"] = tail.outputs(kind=ArtifactKind.TABLE)

    if _step_enabled(enabled, "FlowpackerStep_stage2"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage2",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FlowpackerStep_stage2",
                    candidate_concurrency,
                ),
            ),
            id="stage2-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage2"):
        score = workflow.add_node(
            AF3ScoreNode(
                "AF3scoreStep_stage2",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "AF3scoreStep_stage2",
                    candidate_concurrency,
                ),
            ),
            id="stage2-af3score",
            inputs=_structure_inputs(tail),
        )

    filtered = tail
    if _step_enabled(enabled, "FilterStep_stage2"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        filtered = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage2",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FilterStep_stage2",
                    candidate_concurrency,
                ),
            ),
            id="stage2-filter",
            inputs=inputs,
        )
        report_table_inputs["filter_tables"] = filtered.outputs(kind=ArtifactKind.TABLE)

    refold = None
    if _step_enabled(enabled, "ReFoldStep"):
        refold = workflow.add_node(
            ReFoldNode(
                "ReFoldStep",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "ReFoldStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-alphafold3-refold",
            inputs=_structure_inputs(filtered),
        )
        report_table_inputs["refold_metrics"] = refold.outputs(kind=ArtifactKind.TABLE)

    dockq = None
    if _step_enabled(enabled, "DockQStep"):
        inputs = _structure_inputs(filtered)
        if refold is not None:
            inputs["models"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
        dockq = workflow.add_node(
            DockQNode(
                "DockQStep",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "DockQStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-dockq",
            inputs=inputs,
        )

    relaxed = None
    if _step_enabled(enabled, "RosettaRelaxStep"):
        inputs = _structure_inputs(filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        relaxed = workflow.add_node(
            RosettaRelaxNode(
                "RosettaRelaxStep",
                modal_namespace,
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "RosettaRelaxStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-rosetta-relax",
            inputs=inputs,
        )

    rank = None
    if _step_enabled(enabled, "RankStep"):
        inputs = _structure_inputs(relaxed or filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        if refold is not None:
            inputs["refold"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
            inputs["refold_metrics"] = refold.outputs(kind=ArtifactKind.TABLE)
        if score is not None:
            inputs["af3scores"] = score.outputs(kind=ArtifactKind.SCORES)
        rank = workflow.add_node(
            RankNode(
                "RankStep",
                modal_namespace,
                {"gentype": gentype} | _step_cfg(steps, "RankStep"),
            ),
            id="stage2-rank",
            inputs=inputs,
        )

    if _step_enabled(enabled, "ReportStep"):
        inputs = (
            {"rank": rank.outputs(kind=ArtifactKind.TABLE)}
            if rank is not None
            else _structure_inputs(filtered)
        )
        inputs.update(report_table_inputs)
        workflow.add_node(
            ReportNode("ReportStep", _step_cfg(steps, "ReportStep")),
            id="stage2-report",
            inputs=inputs,
        )


def _structure_inputs(upstream) -> dict[str, Any]:
    if upstream is None:
        return {}
    return {
        "structures": upstream.outputs(kind=ArtifactKind.STRUCTURES),
        "candidate_manifest": upstream.outputs(
            kind=ArtifactKind.TABLE,
            role=ppiflow_manifests.MANIFEST_FILE_ROLE,
        ),
    }


def _stage2_input_node(
    task: Mapping[str, Any],
    steps: Mapping[str, Any],
    modal_namespace: PPIFlowModalNamespace,
) -> ExistingStructuresNode:
    raw_cfg = steps.get("Stage2Input") or task.get("stage2_input")
    if not isinstance(raw_cfg, Mapping):
        raise ValueError(
            "stage=2 PPIFlow runs require a Stage2Input step config or "
            "task.stage2_input mapping with an existing structure path"
        )
    raw_path = raw_cfg.get("path")
    if raw_path is None:
        raise ValueError("Stage2Input requires a 'path' value")
    volume_name = str(raw_cfg.get("volume_name", PPI_FLOW_OUTPUT_VOLUME_NAME))
    storage = _volume_path_from_stage_config(str(raw_path), volume_name=volume_name)
    return ExistingStructuresNode(
        "Stage2Input",
        modal_namespace,
        storage,
        dict(raw_cfg),
    )


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


def _load_yaml_bytes(data: bytes) -> dict[str, Any]:
    loaded = yaml.safe_load(data.decode("utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML root must be a mapping")
    return loaded


def _task_section(task_doc: dict[str, Any]) -> dict[str, Any]:
    section = task_doc.get("task", task_doc)
    if not isinstance(section, dict):
        raise ValueError("task.yaml must contain a mapping under 'task'")
    return section


def _enabled_section(task_doc: dict[str, Any]) -> dict[str, bool]:
    enabled = task_doc.get("steps", {})
    if not isinstance(enabled, dict):
        raise ValueError("task.yaml 'steps' section must be a mapping")
    return {str(key): bool(value) for key, value in enabled.items()}


def _step_enabled(enabled: dict[str, bool], step_name: str) -> bool:
    return bool(enabled.get(step_name, False))


def _step_cfg(steps: dict[str, Any], step_name: str) -> dict[str, Any]:
    cfg = steps.get(step_name, {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"steps.yaml entry {step_name!r} must be a mapping")
    return cfg


def _step_cfg_with_candidate_concurrency(
    steps: dict[str, Any],
    step_name: str,
    candidate_concurrency: int,
) -> dict[str, Any]:
    cfg = dict(_step_cfg(steps, step_name))
    cfg.setdefault("candidate_concurrency", candidate_concurrency)
    return cfg


def _ppiflow_input_fields(args: object) -> tuple[str, ...]:
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyConfig):
        return ("antigen_pdb", "framework_pdb")
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyPartialConfig):
        return ("complex_pdb",)
    if isinstance(
        args,
        (ppiflow_app.SampleBinderConfig, ppiflow_app.SampleBinderPartialConfig),
    ):
        return ("input_pdb",)
    raise TypeError(f"Unsupported PPIFlow args type: {type(args).__name__}")


def _active_ppiflow_app_steps(
    task_doc: dict[str, Any], stage: int | None
) -> tuple[str, ...]:
    """Return PPIFlow app steps that should be staged for the selected run."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    enabled = _enabled_section(task_doc)
    active_steps: list[str] = []
    if stage in {None, 1} and _step_enabled(enabled, "PPIFlowStep"):
        active_steps.append("PPIFlowStep")
    return tuple(active_steps)


def _stage_ppiflow_app_inputs(
    *,
    steps_doc: dict[str, Any],
    run_id: str,
    app_steps: tuple[str, ...],
    force: bool = False,
) -> dict[str, Any]:
    """Upload local PPIFlow app inputs and rewrite step args to mounted paths."""
    staged_steps = deepcopy(steps_doc)
    uploads: list[tuple[Path, str]] = []
    volume_root = Path(ppiflow_app.CONF.output_volume_mountpoint)

    for step_name in app_steps:
        if step_name not in staged_steps:
            continue
        cfg = _step_cfg(staged_steps, step_name)
        raw_args = cfg.get("args", cfg)
        if not isinstance(raw_args, dict):
            continue

        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        for field_name in _ppiflow_input_fields(app_args.args):
            current_value = getattr(app_args.args, field_name)
            current_path = Path(current_value)
            if current_path.is_absolute() and current_path.is_relative_to(volume_root):
                continue

            local_path = current_path.expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(
                    f"PPIFlow {step_name} input {field_name!r} was not found "
                    f"locally or in the mounted output volume: {current_value}"
                )

            remote_rel = (
                Path(run_id)
                / sanitize_filename(step_name)
                / sanitize_filename(field_name)
                / sanitize_filename(local_path.name)
            )
            raw_args[field_name] = str(volume_root / remote_rel)
            uploads.append((local_path, remote_rel.as_posix()))

    if uploads:
        with ppiflow_app.CONF.output_volume.batch_upload(force=force) as batch:
            for local_path, remote_rel in uploads:
                remote_storage = volume_path_from_mount_path(
                    str(volume_root / remote_rel),
                    str(volume_root),
                    ppiflow_app.CONF.output_volume_name,
                )
                print(
                    f"Uploading PPIFlow input '{local_path}' to {remote_storage}",
                    flush=True,
                )
                batch.put_file(local_path, f"/{remote_storage.path}")
    return staged_steps


@app.local_entrypoint()
def submit_ppiflow_workflow(
    task_yaml: str,
    steps_yaml: str,
    run_id: str | None = None,
    stage: int | None = None,
    force: bool = False,
    wait: bool = True,
    max_parallel: int = 16,
    dry_run: bool = False,
    strict_artifact_checks: bool = False,
) -> None:
    """Build and submit a PPIFlow workflow from task and step YAML files.

    Args:
        task_yaml: Path to the PPIFlow task YAML declaring enabled workflow
            steps and design mode.
        steps_yaml: Path to the YAML file containing per-step app arguments.
        run_id: Stable workflow run id for durable ledger state. Defaults to
            the task YAML filename stem.
        stage: Optional stage selector. Use 1 for stage 1 only, 2 for stage 2
            only, or omit to build both stages.
        force: Replace an existing workflow run ledger before running and
            overwrite staged PPIFlow input files in the app output volume.
        wait: Wait locally for the remote workflow result. Disable to print the
            Modal function call id for asynchronous collection.
        max_parallel: Maximum number of ready workflow nodes to execute
            concurrently in one scheduler wave.
        dry_run: Print the workflow DAG graph and skip orchestrator execution.
        strict_artifact_checks: Validate referenced app-owned volume artifacts
            before reusing completed workflow nodes.
    """
    task_yaml_path = Path(task_yaml).expanduser().resolve()
    steps_yaml_path = Path(steps_yaml).expanduser().resolve()
    resolved_run_id = sanitize_filename(run_id or task_yaml_path.stem)
    task_yaml_bytes = task_yaml_path.read_bytes()
    steps_yaml_bytes = steps_yaml_path.read_bytes()
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    if dry_run:
        workflow = build_ppiflow_workflow(
            task_yaml_bytes=task_yaml_bytes,
            steps_yaml_bytes=steps_yaml_bytes,
            stage=stage,
        )
        print_workflow_dag(workflow.validate())
        return

    steps_doc = _stage_ppiflow_app_inputs(
        steps_doc=_load_yaml_bytes(steps_yaml_bytes),
        run_id=resolved_run_id,
        app_steps=_active_ppiflow_app_steps(task_doc, stage),
        force=force,
    )
    steps_doc = _inline_rosetta_config_files(steps_doc)
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=task_yaml_bytes,
        steps_yaml_bytes=yaml.safe_dump(steps_doc).encode("utf-8"),
        stage=stage,
    )

    orchestrator_handle = orchestrator.WorkflowOrchestrator()
    orchestrator_kwargs = {
        "workflow": workflow,
        "run_id": resolved_run_id,
        "force": force,
        "max_ready_workers": max_parallel,
    }
    if strict_artifact_checks:
        orchestrator_kwargs["strict_external_artifact_checks"] = True
        orchestrator_kwargs["external_artifact_checker"] = (
            check_ppiflow_external_artifact.remote
        )
    print(
        f"Submitting PPIFlow workflow '{resolved_run_id}' with "
        f"{len(workflow.validate().nodes)} node(s)",
        flush=True,
    )
    if wait:
        result: AppRunResult | str = AppRunResult.model_validate(
            orchestrator_handle.run.remote(**orchestrator_kwargs)
        )
    else:
        function_call = orchestrator_handle.run.spawn(**orchestrator_kwargs)
        result = str(getattr(function_call, "object_id", function_call))
    if isinstance(result, AppRunResult):
        print(f"PPIFlow workflow run finished with status: {result.status}", flush=True)
    else:
        print(f"PPIFlow workflow run submitted. FunctionCall id: {result}", flush=True)
