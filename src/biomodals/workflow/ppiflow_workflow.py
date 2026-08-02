"""PPIFlow workflow definition built on the reusable workflow runtime."""

from __future__ import annotations

import ast
import hashlib
import os
import shlex
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, cast
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl
import yaml
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.bioinfo import rosetta_app
from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    execute_rosetta_task,
    validate_task_publication_from_volume,
)
from biomodals.app.design import ligandmpnn_app, ppiflow_app
from biomodals.app.fold import alphafold3_app, flowpacker_app
from biomodals.app.fold.alphafold3.inference_inputs import prepare_inference_run
from biomodals.app.fold.alphafold3.inference_pipeline import (
    coordinate_seed_predictions,
)
from biomodals.app.fold.alphafold3.modal_adapters import (
    InProcessInferenceExecutor,
    stage_inference_run,
)
from biomodals.app.fold.alphafold3.request_results import (
    RequestPublication,
    create_request_archive,
    load_request_manifest,
    request_manifest_from_result,
)
from biomodals.app.fold.alphafold3.search_pipeline import (
    resolve_msa_and_templates,
)
from biomodals.app.score import af3score_app, dockq_app
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    NodeAggregationPolicy,
    WorkerAssignmentRecord,
)
from biomodals.execution.pull_worker import drive_pull_worker
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
    VolumePath,
    WorkflowArtifact,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE
from biomodals.workflow.core import (
    AppBackedNode,
    NodeRunContext,
    RemoteNodeCall,
    RemotePullTaskWorkflowNode,
    RemotePullWorkerCall,
    RemoteTaskWorkflowNode,
    RemoteWorkflowNode,
    RemoteWorkflowTask,
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
_ROSETTA_PLAN_SCHEMA_VERSION = 1

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
ppiflow_task_image = ppiflow_app.runtime_image.add_local_python_source(
    "biomodals.workflow"
)
ligandmpnn_task_image = ligandmpnn_app.runtime_image.add_local_python_source(
    "biomodals.workflow"
)
flowpacker_task_image = flowpacker_app.runtime_image.add_local_python_source(
    "biomodals.workflow"
)
rosetta_task_image = rosetta_app.runtime_image.add_local_python_source(
    "biomodals.workflow"
)
dockq_task_image = dockq_app.runtime_image.add_local_python_source("biomodals.workflow")
af3score_task_image = af3score_app.runtime_image.add_local_python_source(
    "biomodals.workflow"
)
alphafold3_task_image = alphafold3_app.runtime_image.add_local_python_source(
    "biomodals.workflow"
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
PPI_FLOW_SOURCE_VOLUME_MOUNTS: dict[
    str | PurePosixPath, modal.Volume | modal.CloudBucketMount
] = {
    PPI_FLOW_OUTPUT_MOUNTPOINT: PPI_FLOW_OUTPUT_VOLUME,
    FLOWPACKER_OUTPUT_MOUNTPOINT: FLOWPACKER_OUTPUT_VOLUME,
    AF3SCORE_OUTPUT_MOUNTPOINT: AF3SCORE_OUTPUT_VOLUME,
    ROSETTA_OUTPUT_MOUNTPOINT: ROSETTA_OUTPUT_VOLUME,
    WORKFLOW_OUTPUT_MOUNTPOINT: WORKFLOW_OUTPUT_VOLUME,
}
PPI_FLOW_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **ppiflow_app.CONF.mounts(output_volume=True, model_volume=True),
}
LIGANDMPNN_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **ligandmpnn_app.CONF.mounts(model_volume=True),
}
FLOWPACKER_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **flowpacker_app.CONF.mounts(
        output_volume=True,
        model_volume=True,
        model_ro=False,
    ),
}
AF3SCORE_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **af3score_app.CONF.mounts(
        output_volume=True,
        model_volume=True,
        model_mount_subdir=False,
    ),
}
ALPHAFOLD3_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **alphafold3_app.CONF.mounts(
        output_volume=True,
        model_volume=True,
        model_ro=True,
    ),
    alphafold3_app.JAX_CACHE_MOUNTPOINT: alphafold3_app.JAX_CACHE_VOLUME,
}
ROSETTA_TASK_VOLUME_MOUNTS = rosetta_app.CONF.mounts(output_volume=True)


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
    image=ppiflow_task_image,
    gpu=ppiflow_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_TASK_VOLUME_MOUNTS,
)
def run_ppiflow_design_stage(
    *,
    args: ppiflow_app.PPIFlowArgs,
    run_name: str,
    run_id: str,
    node_id: str,
    step_name: str,
) -> AppRunResult:
    """Run initial PPIFlow design and publish candidate identities."""
    _reload_ppiflow_source_volumes()
    result = AppRunResult.model_validate(
        ppiflow_app.ppiflow_run_workflow.get_raw_f()(
            args=args,
            run_name=run_name,
        )
    )
    adapted = _result_with_output_kind(
        result,
        ArtifactKind.STRUCTURES,
        {
            "step_name": step_name,
            "structure_patterns": PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS,
        },
    )
    rows = _initial_ppiflow_candidate_rows(adapted, step_name=step_name)
    return adapted.model_copy(
        update={
            "outputs": [
                *adapted.outputs,
                _write_candidate_manifest_output(
                    run_id=run_id,
                    node_id=node_id,
                    step_name=step_name,
                    rows=rows,
                ),
            ]
        }
    )


@app.function(
    image=flowpacker_task_image,
    gpu=flowpacker_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=FLOWPACKER_TASK_VOLUME_MOUNTS,
)
def run_ppiflow_flowpacker_stage(
    *,
    artifacts: list[WorkflowArtifact],
    config: dict[str, object],
    run_name: str,
) -> AppRunResult:
    """Select PPIFlow structures and invoke the FlowPacker app."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=_patterns_from_config(config),
        max_files=_optional_config_int(config, "max_structures"),
    )
    kwargs = {
        key: config[key]
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
        if key in config
    }
    return AppRunResult.model_validate(
        flowpacker_app.run_flowpacker_workflow.get_raw_f()(
            input_files=selected,
            run_name=run_name,
            **kwargs,
        )
    )


@app.function(
    image=dockq_task_image,
    cpu=(0.125, 16.125),
    memory=(512, 16384),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def run_ppiflow_dockq_stage(
    *,
    reference_artifacts: list[WorkflowArtifact],
    model_artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None,
    config: dict[str, object],
    run_name: str,
) -> AppRunResult:
    """Select and pair candidate structures before invoking DockQ."""
    _reload_ppiflow_source_volumes()
    references_selected = ppiflow_staging.select_structure_files_from_artifacts(
        reference_artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=_patterns_from_config(config),
        max_files=_optional_config_int(config, "max_structures"),
    )
    models_selected = ppiflow_staging.select_structure_files_from_artifacts(
        model_artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
        max_files=_optional_config_int(config, "max_models"),
    )
    manifest = _candidate_manifest_frame_from_inputs(
        candidate_manifests or [],
        references_selected,
        step_name="DockQInput",
    )
    references = ppiflow_staging.candidate_structure_files_from_selected(
        references_selected,
        manifest_frame=manifest,
    )
    models = ppiflow_staging.candidate_structure_files_from_selected(
        models_selected,
        manifest_frame=manifest,
    )
    pairs = ppiflow_staging.prepare_dockq_pairs_by_candidate(
        references=references,
        models=models,
        mapping=config.get("mapping"),
    )
    if not pairs:
        raise ValueError("DockQ did not find any candidate pairs")
    dockq_args = config.get("dockq_args", "--short")
    if isinstance(dockq_args, str):
        dockq_args = shlex.split(dockq_args)
    return AppRunResult.model_validate(
        dockq_app.run_dockq_workflow.get_raw_f()(
            pairs=pairs,
            run_name=run_name,
            dockq_args=dockq_args,
        )
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
    candidate_manifests: list[WorkflowArtifact] | None = None,
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
    raw_dockq_threshold = config.get("dockq_threshold", 0.49)
    if not isinstance(raw_dockq_threshold, str | int | float):
        raise TypeError("dockq_threshold must be a string or number")
    ranked = ppiflow_tables.ranked_design_rows(
        structures=selected_structures,
        score_frames=score_frames,
        gentype=str(config.get("gentype") or "binder"),
        dockq_threshold=float(raw_dockq_threshold),
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
    for row in ranked:
        file_name, file_bytes = structure_by_key[str(row["design"])]
        (structures_dir / sanitize_filename(file_name)).write_bytes(file_bytes)
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
        warnings=warnings,
    )


def _stage_af3score_candidate_inputs(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> tuple[list[dict[str, object]], str, str]:
    """Stage candidate-keyed PDBs and retain their scientific identities."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts=artifacts,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=patterns,
        max_files=max_files,
    )
    candidates = ppiflow_staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=_candidate_manifest_frame_from_inputs(
            candidate_manifests or [],
            selected,
            step_name="AF3ScoreInput",
        ),
    )
    planned: list[tuple[ppiflow_staging.CandidateStructureFile, str, str]] = []
    input_names: set[str] = set()
    for candidate in candidates:
        pdb_name = f"{sanitize_filename(candidate.candidate_id)}.pdb"
        if pdb_name in input_names:
            raise ValueError(f"Duplicate AF3Score staged input name: {pdb_name}")
        input_names.add(pdb_name)
        planned.append((
            candidate,
            pdb_name,
            hashlib.sha256(candidate.data).hexdigest(),
        ))
    input_digests = {
        Path(pdb_name).stem: digest for _candidate, pdb_name, digest in planned
    }
    publication_key = hashlib.sha256(
        orjson.dumps(
            {
                "inputs": input_digests,
                "af3score": (
                    af3score_app.CONF.repo_commit_hash
                    or af3score_app.CONF.version
                    or "unknown"
                ),
            },
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()
    physical_run_name = f"ppiflow-af3score-{publication_key}"
    layout = AppRunLayout.from_run_root(
        Path(AF3SCORE_OUTPUT_MOUNTPOINT) / physical_run_name
    )
    if layout.inputs_dir.exists():
        shutil.rmtree(layout.inputs_dir)
    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, object]] = []
    for candidate, pdb_name, digest in planned:
        (layout.inputs_dir / pdb_name).write_bytes(candidate.data)
        staged.append({
            "candidate_id": candidate.candidate_id,
            "input_name": pdb_name,
            "scientific_payload": {
                "candidate_id": candidate.candidate_id,
                "content_sha256": digest,
                "source_path": candidate.source_path,
            },
        })
    AF3SCORE_OUTPUT_VOLUME.commit()
    return staged, physical_run_name, publication_key


def _rosetta_plan_artifact(plan: Mapping[str, object]) -> AppOutput:
    """Serialize one immutable PPIFlow Rosetta Task plan."""
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise TypeError("Rosetta plan Tasks must be a list")
    return AppOutput(
        name="rosetta_task_plan",
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=orjson.dumps(
                plan,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ),
            filename="rosetta_task_plan.json",
            media_type="application/json",
        ),
        metadata={
            "task_count": len(tasks),
            "run_name": str(plan["run_name"]),
            "run_id": str(plan["run_id"]),
        },
    )


def _load_rosetta_plan(path: Path) -> dict[str, object]:
    """Load and validate one materialized PPIFlow Rosetta Task plan."""
    value = orjson.loads(path.read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _ROSETTA_PLAN_SCHEMA_VERSION
    ):
        raise ValueError("PPIFlow Rosetta task plan schema is unsupported")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("PPIFlow Rosetta task plan has no Tasks")
    specs = tuple(RosettaTaskSpec.from_dict(task) for task in tasks)
    if len(specs) != value.get("num_jobs"):
        raise ValueError("PPIFlow Rosetta Task count does not match its plan")
    for field_name in (
        "worker_count",
        "claim_capacity",
        "max_parallel_per_worker",
    ):
        field_value = value.get(field_name)
        if not isinstance(field_value, int) or field_value < 1:
            raise ValueError(f"PPIFlow Rosetta {field_name} must be positive")
    return cast(dict[str, object], value)


def _read_rosetta_plan_artifacts(
    artifacts: Sequence[WorkflowArtifact],
) -> dict[str, object]:
    """Load the single Rosetta plan from a materialized workflow artifact."""
    if len(artifacts) != 1:
        raise ValueError(f"Expected one Rosetta task plan, found {len(artifacts)}")
    return _load_rosetta_plan(
        ppiflow_staging.artifact_mount_path(
            artifacts[0],
            PPI_FLOW_SOURCE_VOLUME_ROOTS,
        )
    )


def _rosetta_worker_policy(
    num_jobs: int,
    config: Mapping[str, object],
) -> tuple[int, int, int]:
    """Preserve Rosetta's pod cap while deriving pull-worker microbatches."""
    if num_jobs < 1:
        raise ValueError("Rosetta requires at least one Task")
    num_cpu_per_pod = min(30, num_jobs)
    max_num_pods = max(1, _config_int(config, "max_num_pods", 1))
    if config.get("max_child_calls") is not None:
        max_num_pods = min(
            max_num_pods,
            _config_int(config, "max_child_calls", 1),
        )
    worker_count = min(
        max_num_pods,
        (num_jobs + num_cpu_per_pod - 1) // num_cpu_per_pod,
    )
    claim_capacity = (num_jobs + worker_count - 1) // worker_count
    return worker_count, claim_capacity, min(30, claim_capacity)


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def prepare_ppiflow_rosetta_stage(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None = None,
    config: dict[str, object],
    step_name: str,
    run_name: str,
    run_id: str,
    node_id: str,
) -> AppRunResult:
    """Stage Rosetta inputs and publish a finite pull-worker Task plan."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts=artifacts,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=None,
        max_files=_optional_config_int(config, "max_structures"),
    )
    candidate_structures = ppiflow_staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=_candidate_manifest_frame_from_inputs(
            candidate_manifests or [],
            selected,
            step_name="RosettaInput",
        ),
    )
    if not candidate_structures:
        raise ValueError(f"{step_name} requires at least one Rosetta input")
    safe_run_name = sanitize_filename(run_name)
    safe_run_id = sanitize_filename(f"{run_id}-{node_id}")
    layout = AppRunLayout.from_run_root(
        Path(ROSETTA_OUTPUT_MOUNTPOINT) / f"{safe_run_name}-{safe_run_id}"
    )
    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    rosetta_script = config.get("rosetta_script")
    if rosetta_script is not None and not isinstance(rosetta_script, str):
        raise TypeError("rosetta_script must be text")
    script_content = (
        None
        if not rosetta_script
        else _resolve_rosetta_config_text(rosetta_script, "rosetta_script")
    )
    remote_script = None
    if script_content is not None:
        remote_script = "inputs/_script/workflow.xml"
        script_path = layout.run_root / remote_script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content, encoding="utf-8")
    flags_file = config.get("flags_file")
    if flags_file is not None and not isinstance(flags_file, str):
        raise TypeError("flags_file must be text")
    flags_content = (
        None
        if not flags_file
        else _resolve_rosetta_config_text(flags_file, "flags_file")
    )
    remote_flags = None
    if flags_content is not None:
        remote_flags = "inputs/_flags/workflow.flags"
        flags_path = layout.run_root / remote_flags
        flags_path.parent.mkdir(parents=True, exist_ok=True)
        flags_path.write_text(flags_content, encoding="utf-8")

    rosetta_rows = ppiflow_staging.rosetta_job_manifest_rows(
        candidate_structures,
        rosetta_binary=str(config.get("rosetta_binary", "relax")),
        rosetta_script=remote_script,
        flags_file=remote_flags,
    )
    task_specs = []
    for row, structure in zip(rosetta_rows, candidate_structures, strict=True):
        remote_pdb = str(row["pdb"])
        pdb_path = layout.run_root / remote_pdb
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_bytes(structure.data)
        task_specs.append(
            RosettaTaskSpec(
                task_key=str(row["candidate_id"]),
                index=_config_int(row, "index", 0),
                binary=str(row["binary"]),
                pdb=remote_pdb,
                rosetta_script=remote_script,
                flags_file=remote_flags,
                output_dir=str(row["expected_output_dir"]),
                worker_log=str(row["worker_log"]),
                expected_files=(str(row["expected_score_file"]),),
                input_sha256=hashlib.sha256(structure.data).hexdigest(),
                script_sha256=(
                    None
                    if script_content is None
                    else hashlib.sha256(script_content.encode()).hexdigest()
                ),
                flags_sha256=(
                    None
                    if flags_content is None
                    else hashlib.sha256(flags_content.encode()).hexdigest()
                ),
                candidate_id=str(row["candidate_id"]),
            )
        )
    job_manifest = ppiflow_staging.write_rosetta_job_manifest(
        rosetta_rows,
        layout.run_root / "rosetta_job_manifest.csv",
    )
    worker_count, claim_capacity, max_parallel = _rosetta_worker_policy(
        len(task_specs),
        config,
    )
    plan: dict[str, object] = {
        "schema_version": _ROSETTA_PLAN_SCHEMA_VERSION,
        "run_name": safe_run_name,
        "run_id": safe_run_id,
        "run_root": str(layout.run_root),
        "job_manifest": str(job_manifest),
        "num_jobs": len(task_specs),
        "worker_count": worker_count,
        "claim_capacity": claim_capacity,
        "max_parallel_per_worker": max_parallel,
        "tasks": [task.to_dict() for task in task_specs],
    }
    ROSETTA_OUTPUT_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[_rosetta_plan_artifact(plan)],
        metrics={
            "staged_candidates": len(task_specs),
            "worker_count": worker_count,
        },
    )


def _af3score_task_spec_value(task_spec: object, name: str) -> object:
    """Read one field from AF3Score's dataclass or a serialized test double."""
    if isinstance(task_spec, Mapping):
        return cast(Mapping[str, object], task_spec)[name]
    return getattr(task_spec, name)


def _af3score_chunk_payload(chunk: object) -> dict[str, str]:
    """Normalize one AF3Score chunk without exposing its dataclass to the ledger."""
    if isinstance(chunk, Mapping):
        payload = cast(Mapping[str, object], chunk)
        return {
            "batch_json_dir": str(payload["batch_json_dir"]),
            "batch_name": str(payload["batch_name"]),
            "batch_pdb_dir": str(payload["batch_pdb_dir"]),
        }
    chunk_spec = cast(af3score_app.ChunkSpec, chunk)
    return {
        "batch_json_dir": str(chunk_spec.batch_json_dir),
        "batch_name": str(chunk_spec.batch_name),
        "batch_pdb_dir": str(chunk_spec.batch_pdb_dir),
    }


def _af3score_plan_artifact(plan: Mapping[str, object]) -> AppOutput:
    """Serialize one operational AF3Score plan for downstream Task discovery."""
    candidates = plan["candidates"]
    if not isinstance(candidates, Sequence):
        raise TypeError("AF3Score plan candidates must be a sequence")
    return AppOutput(
        name="af3score_task_plan",
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=orjson.dumps(
                plan,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ),
            filename="af3score_task_plan.json",
            media_type="application/json",
        ),
        metadata={
            "candidate_count": len(candidates),
            "run_name": str(plan["run_name"]),
        },
    )


def _read_af3score_plan_artifacts(
    artifacts: Sequence[WorkflowArtifact],
) -> dict[str, object]:
    """Load the single prepared AF3Score plan from a workflow artifact."""
    if len(artifacts) != 1:
        raise ValueError(f"Expected one AF3Score task plan, found {len(artifacts)}")
    path = ppiflow_staging.artifact_mount_path(
        artifacts[0],
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("AF3Score task plan must be a JSON object")
    return value


@app.function(
    image=af3score_task_image,
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=AF3SCORE_TASK_VOLUME_MOUNTS,
)
def prepare_ppiflow_af3score_stage(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None = None,
    config: dict[str, object],
    step_name: str,
    run_name: str,
) -> AppRunResult:
    """Stage AF3Score candidates and publish its finite GPU Task plan."""
    staged, physical_run_name, publication_key = _stage_af3score_candidate_inputs(
        artifacts=artifacts,
        candidate_manifests=candidate_manifests,
        patterns=_patterns_from_config(config, default=("*.pdb",)),
        max_files=_optional_config_int(config, "max_structures"),
    )
    input_names = [str(record["input_name"]) for record in staged]
    if not input_names:
        raise ValueError(f"{step_name} requires at least one AF3Score input")
    input_digests = {
        Path(str(record["input_name"])).stem: str(
            cast(Mapping[str, object], record["scientific_payload"])["content_sha256"]
        )
        for record in staged
    }
    task_spec = af3score_app.af3score_prepare.get_raw_f()(
        run_name=physical_run_name,
        input_files=input_names,
        input_digests=input_digests,
        publication_key=publication_key,
        num_jobs=_config_int(
            config,
            "num_jobs",
            _config_int(
                config,
                "max_batches",
                _config_int(config, "max_child_calls", 10),
            ),
        ),
        prepare_workers=_config_int(config, "prepare_workers", 8),
    )
    chunks_by_input: dict[str, dict[str, str]] = {}
    raw_chunks = _af3score_task_spec_value(task_spec, "chunk_specs")
    if not isinstance(raw_chunks, Sequence):
        raise TypeError("AF3Score chunk_specs must be a sequence")
    for raw_chunk in raw_chunks:
        chunk = _af3score_chunk_payload(raw_chunk)
        for path in sorted(Path(chunk["batch_pdb_dir"]).glob("*.pdb")):
            if path.name in chunks_by_input:
                raise ValueError(f"AF3Score input {path.name!r} appears in two batches")
            chunks_by_input[path.name] = chunk
    pending_value = _af3score_task_spec_value(task_spec, "pending")
    if not isinstance(pending_value, int):
        raise TypeError("AF3Score pending Task count must be an integer")
    pending = pending_value
    if len(chunks_by_input) != pending:
        raise ValueError(
            "AF3Score prepared Task count does not match its batch directories: "
            f"{pending} pending, {len(chunks_by_input)} mapped"
        )
    candidates = [
        record | {"chunk": chunks_by_input.get(str(record["input_name"]))}
        for record in staged
    ]
    chunk_sizes: dict[str, int] = {}
    for candidate in candidates:
        chunk = candidate["chunk"]
        if isinstance(chunk, Mapping):
            chunk_payload = cast(Mapping[str, object], chunk)
            batch_name = str(chunk_payload["batch_name"])
            chunk_sizes[batch_name] = chunk_sizes.get(batch_name, 0) + 1
    for candidate in candidates:
        chunk = candidate["chunk"]
        if isinstance(chunk, dict):
            chunk_payload = cast(dict[str, object], chunk)
            chunk_payload["task_count"] = chunk_sizes[str(chunk_payload["batch_name"])]
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            _af3score_plan_artifact({
                "candidates": candidates,
                "input_files": input_names,
                "input_digests": input_digests,
                "publication_key": publication_key,
                "run_name": physical_run_name,
                "display_run_name": run_name,
            })
        ],
    )


@app.function(
    image=af3score_task_image,
    gpu=af3score_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=AF3SCORE_TASK_VOLUME_MOUNTS,
)
def run_ppiflow_af3score_batch(
    *,
    run_name: str,
    batch_name: str,
    batch_json_dir: str,
    batch_pdb_dir: str,
    task_keys: list[str],
    input_names: list[str],
    input_digests: dict[str, str],
    publication_key: str,
) -> dict[str, dict[str, object]]:
    """Run one AF3Score GPU batch and report each owned scientific Task."""
    if len(task_keys) != len(input_names) or not task_keys:
        raise ValueError(
            "AF3Score batch Task keys and inputs must be nonempty and align"
        )
    af3score_app.af3score_run.get_raw_f()(
        run_name=run_name,
        batch_name=batch_name,
        batch_json_dir=batch_json_dir,
        batch_pdb_dir=batch_pdb_dir,
        input_digests=input_digests,
        publication_key=publication_key,
    )
    layout = AppRunLayout.from_run_root(
        Path(AF3SCORE_OUTPUT_MOUNTPOINT) / sanitize_filename(run_name)
    )
    results: dict[str, dict[str, object]] = {}
    for task_key, input_name in zip(task_keys, input_names, strict=True):
        input_id = Path(input_name).stem
        complete = af3score_app._input_publication_ready(
            layout.outputs_dir,
            input_id,
            publication_key=publication_key,
            input_sha256=input_digests[input_id],
        )
        result = (
            AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name=f"af3score_{sanitize_filename(task_key)}",
                        kind=ArtifactKind.SCORES,
                        storage=volume_path_from_mount_path(
                            str(layout.outputs_dir / input_id),
                            AF3SCORE_OUTPUT_MOUNTPOINT,
                            AF3SCORE_OUTPUT_VOLUME_NAME,
                        ),
                        metadata={
                            "candidate_id": task_key,
                            "input_name": input_name,
                            "run_name": run_name,
                        },
                    )
                ],
            )
            if complete
            else AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[f"AF3Score output is incomplete for {task_key!r}"],
            )
        )
        results[task_key] = result.model_dump(mode="json")
    return results


@app.function(
    image=af3score_task_image,
    cpu=(0.125, 16.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    volumes=AF3SCORE_TASK_VOLUME_MOUNTS,
)
def postprocess_ppiflow_af3score_stage(
    *,
    plan_artifacts: list[WorkflowArtifact],
    step_name: str,
    run_id: str,
    node_id: str,
) -> AppRunResult:
    """Postprocess a kernel-completed AF3Score Task collection."""
    plan = _read_af3score_plan_artifacts(plan_artifacts)
    run_name = str(plan["run_name"])
    input_files = plan["input_files"]
    candidates = plan["candidates"]
    input_digests = plan.get("input_digests")
    publication_key = plan.get("publication_key")
    if (
        not isinstance(input_files, list)
        or not isinstance(candidates, list)
        or not isinstance(input_digests, dict)
        or not isinstance(publication_key, str)
    ):
        raise TypeError("AF3Score task plan contains invalid candidate data")
    metrics = af3score_app.af3score_postprocess.get_raw_f()(
        run_name=run_name,
        input_files=[str(value) for value in input_files],
        input_digests={str(key): str(value) for key, value in input_digests.items()},
        publication_key=publication_key,
    )
    metrics_csv = str(metrics["metrics_csv"])
    status = ppiflow_tables.score_table_status(
        requested_count=len(input_files),
        usable_rows=int(metrics.get("metrics_rows", 0)),
        failed_count=int(metrics.get("failed", 0)),
    )
    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        step_name=step_name,
        rows=[
            ppiflow_manifests.candidate_manifest_row(
                candidate_id=str(candidate_payload["candidate_id"]),
                stage_name=step_name,
                stage_role="score",
                operation_mode="af3score",
                candidate_status=status.value,
                source_path=str(candidate_payload["input_name"]),
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
            for candidate in candidates
            if isinstance(candidate, Mapping)
            for candidate_payload in (cast(Mapping[str, object], candidate),)
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
    image=ligandmpnn_task_image,
    gpu=ligandmpnn_app.CONF.gpu,
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=LIGANDMPNN_TASK_VOLUME_MOUNTS,
)
def run_ppiflow_ligandmpnn_candidate(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None,
    candidate_id: str,
    config: dict[str, object],
    step_name: str,
    run_name: str,
    script_mode: str,
    cli_args: dict[str, str | int | float | bool],
) -> AppRunResult:
    """Run LigandMPNN for one kernel-owned PPIFlow candidate."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=_patterns_from_config(config),
        max_files=_optional_config_int(config, "max_structures"),
    )
    selected_structures = [
        asdict(structure)
        for structure in ppiflow_staging.candidate_structure_files_from_selected(
            selected,
            manifest_frame=_candidate_manifest_frame_from_inputs(
                candidate_manifests or [],
                selected,
                step_name=step_name,
            ),
        )
    ]
    matches = [
        structure
        for structure in selected_structures
        if structure["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"LigandMPNN candidate {candidate_id!r} resolved to "
            f"{len(matches)} structures"
        )
    structure = matches[0]
    result = AppRunResult.model_validate(
        ligandmpnn_app.ligandmpnn_run.get_raw_f()(
            run_name=sanitize_filename(f"{run_name}-{candidate_id}"),
            script_mode=script_mode,
            struct_bytes=_bytes_payload(structure["data"], "structure data"),
            seeds=_parse_seed_values(config.get("seeds", [0])),
            cli_args=cli_args,
            bias_aa_per_residue_bytes=config.get("bias_aa_per_residue_bytes"),
            omit_aa_per_residue_bytes=config.get("omit_aa_per_residue_bytes"),
        )
    )
    outputs = _ligandmpnn_stage_outputs(
        result,
        candidate_id=candidate_id,
        step_name=step_name,
        selected_structure=str(structure["file_name"]),
    )
    candidate_files = _inline_output_file_records([
        output for output in outputs if output.kind == ArtifactKind.STRUCTURES
    ])
    return result.model_copy(
        update={
            "outputs": [
                output.model_copy(
                    update={
                        "metadata": dict(output.metadata)
                        | {"candidate_files": candidate_files}
                    }
                )
                if output.kind == ArtifactKind.STRUCTURES
                else output
                for output in outputs
            ]
        }
    )


@app.function(
    image=ppiflow_task_image,
    gpu=ppiflow_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_TASK_VOLUME_MOUNTS,
)
def run_ppiflow_partial_candidate(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None,
    candidate_id: str,
    config: dict[str, object],
    step_name: str,
    run_name: str,
) -> AppRunResult:
    """Run one kernel-owned PPIFlow partial-design candidate."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=_patterns_from_config(config),
        max_files=_optional_config_int(config, "max_structures"),
    )
    candidate_structures = ppiflow_staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=_candidate_manifest_frame_from_inputs(
            candidate_manifests or [],
            selected,
            step_name=step_name,
        ),
    )
    matches = [
        structure
        for structure in candidate_structures
        if structure.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"PPIFlow partial candidate {candidate_id!r} resolved to "
            f"{len(matches)} structures"
        )
    structure = matches[0]
    fixed_positions_by_candidate = _fixed_positions_by_candidate(
        artifacts,
        candidate_structures,
    )
    raw_args_template = deepcopy(config.get("args", config))
    if not isinstance(raw_args_template, dict):
        raise ValueError(f"PPIFlow step {step_name!r} args must be a mapping")
    field_name = "complex_pdb" if "complex_pdb" in raw_args_template else "input_pdb"
    staged_path = (
        Path(PPI_FLOW_OUTPUT_MOUNTPOINT)
        / sanitize_filename(run_name)
        / sanitize_filename(step_name)
        / sanitize_filename(candidate_id)
        / sanitize_filename(field_name)
        / sanitize_filename(structure.file_name)
    )
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_bytes(structure.data)
    PPI_FLOW_OUTPUT_VOLUME.commit()

    raw_args = deepcopy(raw_args_template)
    raw_args[field_name] = str(staged_path)
    if "fixed_positions" not in raw_args:
        fixed_positions = fixed_positions_by_candidate.get(candidate_id)
        if fixed_positions:
            raw_args["fixed_positions"] = fixed_positions
    app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
    result = AppRunResult.model_validate(
        ppiflow_app.ppiflow_run_workflow.get_raw_f()(
            args=app_args,
            run_name=sanitize_filename(f"{run_name}-{candidate_id}"),
        )
    )
    return _ppiflow_candidate_result(
        result,
        candidate_id=candidate_id,
        step_name=step_name,
        source_structure=structure.file_name,
    )


@app.function(
    image=alphafold3_task_image,
    gpu=alphafold3_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 131072),
    timeout=CONF.timeout,
    volumes=ALPHAFOLD3_TASK_VOLUME_MOUNTS,
)
def run_ppiflow_refold_candidate(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None,
    candidate_id: str,
    config: dict[str, object],
    step_name: str,
    run_name: str,
) -> AppRunResult:
    """Run AlphaFold3 refolding for one kernel-owned candidate Task."""
    # TODO: tune CPU/memory/timeout/GPU once ReFold candidate-stage telemetry exists.
    # TODO: sometimes PDB and score files mismatch in length; investigate
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts,
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=_patterns_from_config(config, default=("*.pdb",)),
        max_files=_optional_config_int(config, "max_structures"),
    )
    structures = ppiflow_staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=_candidate_manifest_frame_from_inputs(
            candidate_manifests or [],
            selected,
            step_name=step_name,
        ),
    )
    matches = [
        structure for structure in structures if structure.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"ReFold candidate {candidate_id!r} resolved to {len(matches)} structures"
        )
    structure = matches[0]
    candidate_run_name = sanitize_filename(f"{run_name}-{candidate_id}")
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=_run_one_refold_candidate(
            structure_name=structure.file_name,
            structure_bytes=structure.data,
            candidate_id=candidate_id,
            run_name=candidate_run_name,
            step_name=step_name,
            config=config,
        ),
        metrics={"candidate_id": candidate_id},
    )


def _rosetta_task_receipt(
    task: RosettaTaskSpec,
    task_fingerprint: str,
) -> AppOutput:
    """Return a small workflow-owned receipt for one validated Rosetta Task."""
    return AppOutput(
        name="rosetta_task_receipt",
        kind=ArtifactKind.REPORT,
        storage=InlineBytes(
            data=orjson.dumps(
                {
                    "task_key": task.task_key,
                    "task_fingerprint": task_fingerprint,
                    "candidate_id": task.candidate_id,
                    "expected_files": list(task.expected_files),
                },
                option=orjson.OPT_SORT_KEYS,
            ),
            filename=f"{sanitize_filename(task.task_key)}.json",
            media_type="application/json",
        ),
        metadata={"candidate_id": task.candidate_id or task.task_key},
    )


@app.function(
    image=rosetta_task_image,
    cpu=(0.125, 30.125),
    memory=(1024, 43008),
    timeout=CONF.timeout,
    volumes=ROSETTA_TASK_VOLUME_MOUNTS,
)
def run_ppiflow_rosetta_worker(
    coordinator,
    provider_call_id: str,
    run_name: str,
    run_id: str,
    claim_capacity: int,
    max_parallel: int,
) -> dict[str, int]:
    """Pull Rosetta Tasks from the workflow coordinator until none remain."""
    from biomodals.helper.shell import run_command

    layout = AppRunLayout.from_run_root(
        Path(ROSETTA_OUTPUT_MOUNTPOINT) / f"{run_name}-{run_id}"
    )

    def claim(request_id: str, capacity: int):
        return coordinator.claim_tasks.remote(
            provider_call_id,
            request_id,
            capacity,
        )

    def execute(assignment: WorkerAssignmentRecord) -> AppRunResult:
        payload = assignment.execution_payload
        if not isinstance(payload, Mapping):
            raise TypeError("Rosetta worker Task payload must be an object")
        task = RosettaTaskSpec.from_dict(payload.get("task"))
        try:
            execute_rosetta_task(
                run_root=layout.run_root,
                task=task,
                task_fingerprint=assignment.task_fingerprint,
                run_command=run_command,
            )
        except Exception as error:  # noqa: BLE001
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[str(error) or type(error).__name__],
                metrics={"candidate_id": task.candidate_id or task.task_key},
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[_rosetta_task_receipt(task, assignment.task_fingerprint)],
            metrics={"candidate_id": task.candidate_id or task.task_key},
        )

    def complete(
        assignment: WorkerAssignmentRecord,
        request_id: str,
        result: AppRunResult,
    ) -> None:
        coordinator.complete_task.remote(
            provider_call_id,
            assignment.task_key,
            request_id,
            result,
        )

    summary = drive_pull_worker(
        provider_call_id=UUID(provider_call_id),
        claim_capacity=claim_capacity,
        claim=claim,
        execute=execute,
        complete=complete,
        checkpoint_batch=ROSETTA_OUTPUT_VOLUME.commit,
        max_parallel=max_parallel,
    )
    return asdict(summary)


def _rosetta_task_outcomes_artifact(
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
) -> AppOutput:
    """Serialize terminal pull-Task outcomes for the remote finalizer."""
    return AppOutput(
        name="rosetta_task_outcomes",
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=orjson.dumps(
                {
                    "schema_version": _ROSETTA_PLAN_SCHEMA_VERSION,
                    "succeeded": sorted(results),
                    "errors": {
                        task_key: errors[task_key] for task_key in sorted(errors)
                    },
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ),
            filename="rosetta_task_outcomes.json",
            media_type="application/json",
        ),
    )


def _read_rosetta_task_outcomes(
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[set[str], dict[str, str]]:
    """Load one materialized Rosetta pull-Task outcome summary."""
    if len(artifacts) != 1:
        raise ValueError(
            f"Expected one Rosetta Task outcome artifact, found {len(artifacts)}"
        )
    path = ppiflow_staging.artifact_mount_path(
        artifacts[0],
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )
    value = orjson.loads(path.read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _ROSETTA_PLAN_SCHEMA_VERSION
        or not isinstance(value.get("succeeded"), list)
        or not isinstance(value.get("errors"), dict)
    ):
        raise ValueError("PPIFlow Rosetta Task outcomes are invalid")
    succeeded = {str(task_key) for task_key in value["succeeded"]}
    errors = {
        str(task_key): str(message) for task_key, message in value["errors"].items()
    }
    if succeeded & errors.keys():
        raise ValueError("Rosetta Task outcomes contain conflicting statuses")
    return succeeded, errors


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def finalize_ppiflow_rosetta_stage(
    *,
    plan_artifacts: list[WorkflowArtifact],
    outcome_artifacts: list[WorkflowArtifact],
    config: dict[str, object],
    step_name: str,
    run_id: str,
    node_id: str,
) -> AppRunResult:
    """Validate Rosetta publications and emit the existing stage artifacts."""
    _reload_ppiflow_source_volumes()
    plan = _read_rosetta_plan_artifacts(plan_artifacts)
    succeeded_tasks, task_errors = _read_rosetta_task_outcomes(outcome_artifacts)
    task_specs = tuple(
        RosettaTaskSpec.from_dict(task) for task in cast(list[object], plan["tasks"])
    )
    expected_task_keys = {task.task_key for task in task_specs}
    if succeeded_tasks | task_errors.keys() != expected_task_keys:
        raise ValueError("Rosetta Task outcomes do not match the staged plan")

    run_root = Path(str(plan["run_root"]))
    job_manifest = Path(str(plan["job_manifest"]))
    row_frame = pl.read_csv(job_manifest, infer_schema_length=0)
    rows = []
    successful_candidates = 0
    warnings = []
    for row in row_frame.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        expected_score = run_root / str(row["expected_score_file"])
        log_path = run_root / str(row["worker_log"])
        success = candidate_id in succeeded_tasks and expected_score.is_file()
        status = AppRunStatus.SUCCEEDED if success else AppRunStatus.FAILED
        error = task_errors.get(candidate_id)
        if error is None and not success:
            error = (
                "Rosetta Task reported success without its expected score file"
                if candidate_id in succeeded_tasks
                else "Rosetta Task did not publish a result"
            )
        if success:
            successful_candidates += 1
        elif error is not None:
            warnings.append(f"{candidate_id}: {error}")
        rows.append(
            ppiflow_manifests.candidate_manifest_row(
                candidate_id=candidate_id,
                stage_name=step_name,
                stage_role="rosetta",
                operation_mode=str(config.get("rosetta_binary", "relax")),
                candidate_status=status.value,
                source_path=str(row["pdb"]),
                derived_path=str(row["expected_output_dir"]),
                error=error,
                files=[
                    ppiflow_manifests.candidate_file_record(
                        role="score",
                        volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                        app_volume_path=str(
                            Path(str(plan["run_name"]) + "-" + str(plan["run_id"]))
                            / str(row["expected_score_file"])
                        ),
                        expected=True,
                    ),
                    ppiflow_manifests.candidate_file_record(
                        role="worker_log",
                        volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                        app_volume_path=str(
                            Path(str(plan["run_name"]) + "-" + str(plan["run_id"]))
                            / str(row["worker_log"])
                        ),
                        expected=log_path.is_file(),
                    ),
                ],
                summary={
                    "index": row["index"],
                    "num_pods": _config_int(plan, "worker_count", 0),
                },
            )
        )

    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        step_name=step_name,
        rows=rows,
    )
    return AppRunResult(
        status=(
            AppRunStatus.SUCCEEDED if successful_candidates else AppRunStatus.FAILED
        ),
        outputs=[
            volume_app_output(
                name="rosetta_outputs",
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(plan["run_root"]),
                mount_root=ROSETTA_OUTPUT_MOUNTPOINT,
                volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                metadata={
                    "step_name": step_name,
                    "run_name": str(plan["run_name"]),
                    "run_id": str(plan["run_id"]),
                    "num_jobs": _config_int(plan, "num_jobs", 0),
                    "num_pods": _config_int(plan, "worker_count", 0),
                    "structure_patterns": APP_RUN_OUTPUT_STRUCTURE_PATTERNS,
                },
            ),
            volume_app_output(
                name="rosetta_job_manifest",
                kind=ArtifactKind.TABLE,
                remote_path=str(plan["job_manifest"]),
                mount_root=ROSETTA_OUTPUT_MOUNTPOINT,
                volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={
                    "step_name": step_name,
                    "rows": _config_int(plan, "num_jobs", 0),
                },
            ),
            manifest_output,
        ],
        warnings=warnings,
        metrics={
            "successful_candidates": successful_candidates,
            "failed_candidates": len(rows) - successful_candidates,
        },
    )


_OPERATIONAL_CONFIG_KEYS = (
    "candidate_concurrency",
    "max_child_calls",
    "max_batches",
    "max_num_pods",
    "num_jobs",
    "prepare_workers",
)
_INPUT_DIGESTS_KEY = "_biomodals_input_sha256"


@dataclass
class _ConfiguredAppStepNode(AppBackedNode):
    """Base class for configured PPIFlow app-backed workflow nodes."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def _run_name(self, context: NodeRunContext) -> str:
        run_name = sanitize_filename(
            str(
                self.config.get("run_name")
                or f"{context.workload_run_key}-{self.step_name}"
            )
        )
        return run_name

    def _structure_inputs(
        self,
        context: NodeRunContext,
    ) -> list[WorkflowArtifact]:
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(
                f"PPIFlow workflow step {self.step_name!r} requires structure inputs"
            )
        return artifacts


@dataclass
class PPIFlowDesignNode(_ConfiguredAppStepNode):
    """Initial PPIFlow design step with candidate-manifest publication."""

    config: dict[str, Any] = field(default_factory=dict, metadata={"dag_hash": False})
    scientific_config: dict[str, Any] = field(default_factory=dict)

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare one direct PPIFlow design call for kernel submission."""
        raw_args = self.config.get("args", self.config)
        if not isinstance(raw_args, dict):
            raise ValueError(f"PPIFlow step {self.step_name!r} args must be a mapping")
        return RemoteNodeCall(
            function_name="run_ppiflow_design_stage",
            uses_gpu=True,
            kwargs={
                "args": ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args}),
                "run_name": self._run_name(context),
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
            runtime_image_key="ppiflow",
        )


@dataclass
class PPIFlowPartialNode(_ConfiguredAppStepNode, RemoteTaskWorkflowNode):
    """PPIFlow partial design with one kernel Task per candidate."""

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        """Discover stable candidate Tasks from the upstream manifest."""
        return _candidate_remote_tasks(
            context,
            max_candidates=_optional_config_int(self.config, "max_structures"),
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        """Prepare one partial-design candidate for kernel submission."""
        candidate_id = _candidate_task_id(context, task, step_name=self.step_name)
        return RemoteNodeCall(
            function_name="run_ppiflow_partial_candidate",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "candidate_id": candidate_id,
                "config": self.config,
                "step_name": self.step_name,
                "run_name": self._run_name(context),
            },
            runtime_image_key="ppiflow",
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish deterministic partial-design candidate outcomes."""
        return _finalize_candidate_tasks(
            context,
            step_name=self.step_name,
            stage_role="partial_design",
            operation_mode="ppiflow_partial",
            results=results,
            errors=errors,
        )


@dataclass
class LigandMPNNNode(_ConfiguredAppStepNode, RemoteTaskWorkflowNode):
    """LigandMPNN design with one kernel Task per input candidate."""

    def _model_type(self) -> str:
        """Return the configured sequence-design model."""
        return str(
            self.config.get(
                "model_type",
                "abmpnn" if self.step_name.startswith("AbMPNN") else "protein_mpnn",
            )
        )

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        """Discover stable candidate Tasks from the upstream manifest."""
        return _candidate_remote_tasks(
            context,
            max_candidates=_optional_config_int(self.config, "max_structures"),
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        """Prepare one LigandMPNN candidate for kernel submission."""
        candidate_id = _candidate_task_id(context, task, step_name=self.step_name)
        script_mode = str(self.config.get("script_mode", "run"))
        model_type = self._model_type()
        cli_kwargs = _ligandmpnn_cli_kwargs(
            self.config,
            script_mode=script_mode,
            model_type=model_type,
        )
        return RemoteNodeCall(
            function_name="run_ppiflow_ligandmpnn_candidate",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "candidate_id": candidate_id,
                "config": self.config,
                "run_name": self._run_name(context),
                "script_mode": script_mode,
                "cli_args": ligandmpnn_app.build_ligandmpnn_cli_args(**cli_kwargs),
                "step_name": self.step_name,
            },
            runtime_image_key="ligandmpnn",
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish deterministic sequence-design candidate outcomes."""
        return _finalize_candidate_tasks(
            context,
            step_name=self.step_name,
            stage_role="sequence_design",
            operation_mode=self._model_type(),
            results=results,
            errors=errors,
        )


@dataclass
class FlowPackerNode(_ConfiguredAppStepNode):
    """FlowPacker side-chain packing step."""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare the FlowPacker app call."""
        return RemoteNodeCall(
            function_name="run_ppiflow_flowpacker_stage",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "config": self.config,
                "run_name": self._run_name(context),
            },
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
class AF3ScorePrepareNode(_ConfiguredAppStepNode):
    """Prepare finite AF3Score candidate Tasks without nested Modal calls."""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare AF3Score inputs and its durable Task plan."""
        structures = context.inputs.get("structures") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return RemoteNodeCall(
            function_name="prepare_ppiflow_af3score_stage",
            uses_gpu=False,
            kwargs={
                "artifacts": structures,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "step_name": self.step_name,
                "run_name": self._run_name(context),
            },
            runtime_image_key="af3score-cpu",
        )


@dataclass
class AF3ScoreBatchNode(_ConfiguredAppStepNode, RemoteTaskWorkflowNode):
    """Schedule candidate Tasks through AF3Score's prepared GPU batches."""

    def _plan(self, context: NodeRunContext) -> dict[str, object]:
        return _read_af3score_plan_artifacts(context.inputs.get("af3score_plan") or [])

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        """Discover one Task per candidate that still needs GPU scoring."""
        plan = self._plan(context)
        candidates = plan.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("AF3Score task plan candidates must be a list")
        run_name = str(plan["run_name"])
        tasks = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError("AF3Score candidate plan entries must be objects")
            candidate_payload = cast(Mapping[str, object], candidate)
            chunk = candidate_payload.get("chunk")
            if chunk is None:
                continue
            if not isinstance(chunk, Mapping):
                raise TypeError("AF3Score candidate chunk must be an object")
            candidate_id = str(candidate_payload["candidate_id"])
            tasks.append(
                RemoteWorkflowTask(
                    task_key=candidate_id,
                    scientific_payload=candidate_payload["scientific_payload"],
                    execution_payload={
                        "candidate_id": candidate_id,
                        "chunk": dict(chunk),
                        "input_name": str(candidate_payload["input_name"]),
                        "run_name": run_name,
                    },
                )
            )
        return tuple(tasks)

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        """Prepare the single-Task tail of one AF3Score batch."""
        return self.prepare_remote_task_batch(context, (task,))

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[RemoteWorkflowTask, ...],
    ) -> RemoteNodeCall:
        """Prepare one direct AF3Score GPU call for compatible Tasks."""
        if not tasks:
            raise ValueError("AF3Score provider batch cannot be empty")
        plan = self._plan(context)
        input_digests = plan.get("input_digests")
        publication_key = plan.get("publication_key")
        if not isinstance(input_digests, Mapping) or not isinstance(
            publication_key, str
        ):
            raise TypeError("AF3Score task plan publication data is invalid")
        normalized_digests = {
            str(key): str(value) for key, value in input_digests.items()
        }
        payloads: list[Mapping[str, object]] = []
        chunks: list[Mapping[str, object]] = []
        for task in tasks:
            if not isinstance(task.execution_payload, Mapping):
                raise TypeError("AF3Score Task execution payload must be an object")
            payload = cast(Mapping[str, object], task.execution_payload)
            if str(payload["candidate_id"]) != task.task_key:
                raise ValueError("AF3Score Task identity does not match its payload")
            chunk = payload.get("chunk")
            if not isinstance(chunk, Mapping):
                raise TypeError("AF3Score Task chunk must be an object")
            payloads.append(payload)
            chunks.append(cast(Mapping[str, object], chunk))
        first = payloads[0]
        first_chunk = chunks[0]
        batch_name = str(first_chunk["batch_name"])
        run_name = str(first["run_name"])
        if any(
            str(payload["run_name"]) != run_name or chunk != first_chunk
            for payload, chunk in zip(payloads, chunks, strict=True)
        ):
            raise ValueError("AF3Score provider batch mixes incompatible chunks")
        task_count = first_chunk["task_count"]
        if not isinstance(task_count, int):
            raise TypeError("AF3Score batch Task count must be an integer")
        batch_input_ids = tuple(
            path.stem
            for path in sorted(Path(str(first_chunk["batch_json_dir"])).glob("*.json"))
            if path.is_file()
        )
        return RemoteNodeCall(
            function_name="run_ppiflow_af3score_batch",
            uses_gpu=True,
            kwargs={
                "run_name": run_name,
                "batch_name": batch_name,
                "batch_json_dir": str(first_chunk["batch_json_dir"]),
                "batch_pdb_dir": str(first_chunk["batch_pdb_dir"]),
                "task_keys": [task.task_key for task in tasks],
                "input_names": [str(payload["input_name"]) for payload in payloads],
                "input_digests": {
                    input_id: normalized_digests[input_id]
                    for input_id in batch_input_ids
                },
                "publication_key": publication_key,
            },
            runtime_image_key="af3score-gpu",
            compatibility_key=f"{run_name}:{batch_name}",
            max_tasks_per_call=task_count,
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        """Decode AF3Score's per-candidate outcomes from one GPU call."""
        del metadata
        if not isinstance(result, Mapping):
            raise TypeError("AF3Score batch result must be an object")
        result_by_task = cast(Mapping[str, object], result)
        return {
            task_key: AppRunResult.model_validate(result_by_task[task_key])
            for task_key in task_keys
        }

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Record the aggregate batch outcome; postprocessing publishes scores."""
        del context
        return AppRunResult(
            status=AppRunStatus.FAILED if errors else AppRunStatus.SUCCEEDED,
            warnings=[f"{key}: {errors[key]}" for key in sorted(errors)],
            metrics={
                "failed_candidates": len(errors),
                "scored_candidates": len(results),
            },
        )


@dataclass
class AF3ScoreNode(_ConfiguredAppStepNode):
    """Postprocess kernel-completed AF3Score candidate Tasks."""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare one CPU postprocessing call after all GPU batches finish."""
        plan_artifacts = context.inputs.get("af3score_plan") or []
        if not plan_artifacts:
            raise ValueError(f"{self.step_name} requires an AF3Score task plan")
        return RemoteNodeCall(
            function_name="postprocess_ppiflow_af3score_stage",
            uses_gpu=False,
            kwargs={
                "plan_artifacts": plan_artifacts,
                "step_name": self.step_name,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
            },
            runtime_image_key="af3score-cpu",
        )


@dataclass
class RosettaPrepareNode(_ConfiguredAppStepNode):
    """Stage PPIFlow candidates and publish a finite Rosetta Task plan."""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare the Rosetta input and job manifests without nested calls."""
        structures = context.inputs.get("structures") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return RemoteNodeCall(
            function_name="prepare_ppiflow_rosetta_stage",
            uses_gpu=False,
            kwargs={
                "artifacts": structures,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "step_name": self.step_name,
                "run_name": self._run_name(context),
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
            },
            runtime_image_key="rosetta-stage-cpu",
        )


@dataclass
class RosettaWorkerNode(_ConfiguredAppStepNode, RemotePullTaskWorkflowNode):
    """Execute staged Rosetta candidates through kernel-owned pull Tasks."""

    def _plan(self, context: NodeRunContext) -> dict[str, object]:
        artifacts = context.inputs.get("rosetta_plan") or []
        if len(artifacts) != 1:
            raise ValueError(f"Expected one Rosetta task plan, found {len(artifacts)}")
        return _load_rosetta_plan(context.resolve_workflow_artifact(artifacts[0]))

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        """Discover the complete staged Rosetta Task set in manifest order."""
        plan = self._plan(context)
        run_root = (
            Path(str(plan["run_root"]))
            .relative_to(ROSETTA_OUTPUT_MOUNTPOINT)
            .as_posix()
        )
        return tuple(
            RemoteWorkflowTask(
                task_key=task.task_key,
                scientific_payload=task.scientific_payload,
                execution_payload={
                    "task": task.to_dict(),
                    "run_root": run_root,
                },
            )
            for task in (
                RosettaTaskSpec.from_dict(value)
                for value in cast(list[object], plan["tasks"])
            )
        )

    def prepare_pull_worker(
        self,
        context: NodeRunContext,
    ) -> RemotePullWorkerCall:
        """Bind the derived worker pool to the workflow's Rosetta function."""
        plan = self._plan(context)
        return RemotePullWorkerCall(
            function_name="run_ppiflow_rosetta_worker",
            uses_gpu=False,
            claim_capacity=_config_int(plan, "claim_capacity", 0),
            kwargs={
                "run_name": str(plan["run_name"]),
                "run_id": str(plan["run_id"]),
                "claim_capacity": _config_int(plan, "claim_capacity", 0),
                "max_parallel": _config_int(
                    plan,
                    "max_parallel_per_worker",
                    0,
                ),
            },
            runtime_image_key="rosetta-cpu",
            compatibility_key=f"{plan['run_name']}:{plan['run_id']}",
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[WorkflowArtifact, ...],
    ) -> AvailabilityStatus:
        """Revalidate the fingerprint-bound Rosetta marker and required files."""
        del context, result, artifacts
        payload = task.execution_payload
        if not isinstance(payload, Mapping):
            raise TypeError("Rosetta Task execution payload must be an object")
        spec = RosettaTaskSpec.from_dict(payload.get("task"))
        run_root = payload.get("run_root")
        if not isinstance(run_root, str):
            raise TypeError("Rosetta Task run_root must be text")
        try:
            available = validate_task_publication_from_volume(
                ROSETTA_OUTPUT_VOLUME,
                run_root,
                spec,
                expected_fingerprint,
            )
        except Exception:  # noqa: BLE001
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish a deterministic outcome summary for remote finalization."""
        del context
        status = (
            AppRunStatus.PARTIAL
            if results and errors
            else AppRunStatus.SUCCEEDED
            if not errors
            else AppRunStatus.FAILED
        )
        return AppRunResult(
            status=status,
            outputs=[_rosetta_task_outcomes_artifact(results, errors)],
            warnings=[f"{task_key}: {errors[task_key]}" for task_key in sorted(errors)],
            metrics={
                "successful_candidates": len(results),
                "failed_candidates": len(errors),
            },
        )


@dataclass
class _RosettaNode(_ConfiguredAppStepNode):
    """Finalize one PPIFlow Rosetta stage from durable Task outcomes."""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Validate remote outputs and publish the established stage contract."""
        plan_artifacts = context.inputs.get("rosetta_plan") or []
        outcome_artifacts = context.inputs.get("rosetta_outcomes") or []
        if not plan_artifacts or not outcome_artifacts:
            raise ValueError(f"{self.step_name} requires Rosetta Task results")
        return RemoteNodeCall(
            function_name="finalize_ppiflow_rosetta_stage",
            uses_gpu=False,
            kwargs={
                "plan_artifacts": plan_artifacts,
                "outcome_artifacts": outcome_artifacts,
                "config": self.config,
                "step_name": self.step_name,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
            },
            runtime_image_key="rosetta-finalize-cpu",
        )


@dataclass
class RosettaFixNode(_RosettaNode):
    """Rosetta fixed-position analysis step."""


@dataclass
class RosettaRelaxNode(_RosettaNode):
    """Rosetta relaxation step."""


@dataclass
class ReFoldNode(_ConfiguredAppStepNode, RemoteTaskWorkflowNode):
    """AlphaFold3 refolding with one kernel Task per candidate."""

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        """Discover stable candidate Tasks from the upstream manifest."""
        return _candidate_remote_tasks(
            context,
            max_candidates=_optional_config_int(self.config, "max_structures"),
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        """Prepare one candidate wrapper for kernel submission."""
        candidate_id = _candidate_task_id(context, task, step_name=self.step_name)
        return RemoteNodeCall(
            function_name="run_ppiflow_refold_candidate",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "candidate_id": candidate_id,
                "config": self.config,
                "step_name": self.step_name,
                "run_name": self._run_name(context),
            },
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish deterministic candidate outcomes after all Tasks finish."""
        return _finalize_candidate_tasks(
            context,
            step_name=self.step_name,
            stage_role="refold",
            operation_mode="alphafold3",
            results=results,
            errors=errors,
        )


@dataclass
class DockQNode(_ConfiguredAppStepNode):
    """DockQ model/reference scoring step."""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare the DockQ app call."""
        model_artifacts = context.inputs.get("models") or []
        if not model_artifacts:
            raise ValueError(f"{self.step_name} requires model structure inputs")
        return RemoteNodeCall(
            function_name="run_ppiflow_dockq_stage",
            uses_gpu=False,
            kwargs={
                "reference_artifacts": self._structure_inputs(context),
                "model_artifacts": model_artifacts,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "run_name": self._run_name(context),
            },
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
class ExistingStructuresNode(RemoteWorkflowNode):
    """Reference existing structures for stage-2-only PPIFlow runs."""

    step_name: str
    storage: VolumePath
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare Stage2Input normalization for kernel submission."""
        return RemoteNodeCall(
            function_name="normalize_ppiflow_stage2_input",
            uses_gpu=False,
            kwargs={
                "storage": self.storage,
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class FilterStructuresNode(RemoteWorkflowNode):
    """Filter structures using score artifacts."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare score filtering for kernel submission."""
        structures = context.inputs.get("structures") or []
        scores = context.inputs.get("scores") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        if not scores:
            raise ValueError(f"{self.step_name} requires score inputs")
        return RemoteNodeCall(
            function_name="filter_ppiflow_artifacts",
            uses_gpu=False,
            kwargs={
                "structures": structures,
                "scores": scores,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class FixedPositionsNode(RemoteWorkflowNode):
    """Convert Rosetta residue energies into fixed-position constraints."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare fixed-position conversion for kernel submission."""
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return RemoteNodeCall(
            function_name="derive_ppiflow_fixed_positions",
            uses_gpu=False,
            kwargs={
                "artifacts": artifacts,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class RankNode(RemoteWorkflowNode):
    """Rank final designs."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare score-aware ranking for kernel submission."""
        structures = context.inputs.get("structures") or []
        score_artifacts = [
            artifact
            for input_name, artifact_list in context.inputs.items()
            if input_name != "structures"
            for artifact in artifact_list
        ]
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return RemoteNodeCall(
            function_name="rank_ppiflow_artifacts",
            uses_gpu=False,
            kwargs={
                "structures": structures,
                "score_artifacts": score_artifacts,
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class ReportNode(WorkflowNativeNode):
    """Write the final design report."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

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


def _inline_output_file_records(
    outputs: Sequence[AppOutput],
) -> list[dict[str, object]]:
    """Describe inline candidate outputs with one durable content digest."""
    return [
        ppiflow_manifests.candidate_file_record(
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


def _initial_ppiflow_candidate_rows(
    result: AppRunResult,
    *,
    step_name: str,
) -> list[dict[str, object]]:
    """Describe initial PPIFlow structures with stable content identities."""
    rows = []
    for output in result.outputs:
        if output.kind != ArtifactKind.STRUCTURES or not isinstance(
            output.storage, VolumePath
        ):
            continue
        artifact = WorkflowArtifact(
            artifact_id=sanitize_filename(output.name),
            producing_node_id=step_name,
            kind=ArtifactKind.STRUCTURES,
            storage=output.storage,
            metadata=output.metadata,
        )
        for structure in ppiflow_staging.selected_structure_file_records_from_artifact(
            artifact,
            PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS,
            PPI_FLOW_SOURCE_VOLUME_ROOTS,
        ):
            candidate_id = ppiflow_manifests.initial_candidate_id(
                stage_name=step_name,
                source_artifact_id=structure.artifact_id,
                source_path=structure.app_volume_path,
                basename=structure.file_name,
            )
            rows.append(
                ppiflow_manifests.candidate_manifest_row(
                    candidate_id=candidate_id,
                    stage_name=step_name,
                    stage_role="initial_design",
                    operation_mode="ppiflow",
                    candidate_status=AppRunStatus.SUCCEEDED.value,
                    source_artifact_id=structure.artifact_id,
                    source_path=structure.app_volume_path,
                    derived_path=structure.app_volume_path,
                    files=[
                        ppiflow_manifests.candidate_file_record(
                            role="structure",
                            volume_name=structure.volume_name,
                            app_volume_path=structure.app_volume_path,
                            path=structure.artifact_file_path,
                            media_type=structure.media_type,
                            size_bytes=structure.size_bytes,
                            content_sha256=structure.content_sha256,
                        )
                    ],
                )
            )
    if not rows:
        raise FileNotFoundError("PPIFlow design produced no structure candidates")
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def _write_candidate_manifest_output(
    *,
    run_id: str,
    node_id: str,
    step_name: str,
    rows: Sequence[Mapping[str, object]],
) -> AppOutput:
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
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


def _candidate_rows_for_task_discovery(
    context: NodeRunContext,
    *,
    max_candidates: int | None,
) -> list[dict[str, object]]:
    """Load active candidate rows from workflow-owned manifest artifacts."""
    artifacts = context.inputs.get("candidate_manifest") or []
    if not artifacts:
        raise ValueError(
            f"PPIFlow Node {context.node_id!r} requires a candidate manifest"
        )
    frames = [
        ppiflow_manifests.read_manifest(context.resolve_workflow_artifact(artifact))
        for artifact in artifacts
    ]
    frame = pl.concat(frames, how="diagonal") if len(frames) > 1 else frames[0]
    ppiflow_manifests.validate_manifest_frame(frame)
    active = frame.filter(pl.col("candidate_status") == AppRunStatus.SUCCEEDED.value)
    if max_candidates is not None:
        active = active.head(max_candidates)
    rows = active.to_dicts()
    for row in rows:
        files = row.get("files")
        has_structure_digest = isinstance(files, Sequence) and any(
            isinstance(file_record, Mapping)
            and file_record.get("role") in {"structure", "structures"}
            and bool(file_record.get("content_sha256"))
            for file_record in files
        )
        if not has_structure_digest:
            raise ValueError(
                "PPIFlow candidate manifest row "
                f"{row['candidate_id']!r} has no structure content digest"
            )
    return rows


def _candidate_remote_tasks(
    context: NodeRunContext,
    *,
    max_candidates: int | None,
) -> tuple[RemoteWorkflowTask, ...]:
    """Discover one stable kernel Task per active candidate row."""
    return tuple(
        RemoteWorkflowTask(
            task_key=str(row["candidate_id"]),
            scientific_payload=row,
            execution_payload={"candidate_id": str(row["candidate_id"])},
        )
        for row in _candidate_rows_for_task_discovery(
            context,
            max_candidates=max_candidates,
        )
    )


def _candidate_task_id(
    context: NodeRunContext,
    task: RemoteWorkflowTask,
    *,
    step_name: str,
) -> str:
    """Validate and return one persisted candidate Task identity."""
    if not isinstance(task.execution_payload, Mapping):
        raise TypeError(f"{step_name} Task execution payload must be a mapping")
    candidate_id = str(task.execution_payload["candidate_id"])
    if candidate_id != task.task_key or context.task_key != task.task_key:
        raise ValueError(f"{step_name} Task identity does not match its payload")
    return candidate_id


def _finalize_candidate_tasks(
    context: NodeRunContext,
    *,
    step_name: str,
    stage_role: str,
    operation_mode: str,
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
) -> AppRunResult:
    """Publish one deterministic manifest for terminal candidate Tasks."""
    candidate_ids = sorted((*results, *errors))
    rows = [
        ppiflow_manifests.candidate_manifest_row(
            candidate_id=candidate_id,
            stage_name=step_name,
            stage_role=stage_role,
            operation_mode=operation_mode,
            candidate_status=(
                AppRunStatus.SUCCEEDED.value
                if candidate_id in results
                else AppRunStatus.FAILED.value
            ),
            error=errors.get(candidate_id),
            files=(
                _task_result_structure_files(context, results[candidate_id])
                if candidate_id in results
                else ()
            ),
        )
        for candidate_id in candidate_ids
    ]
    status = (
        AppRunStatus.PARTIAL
        if results and errors
        else AppRunStatus.SUCCEEDED
        if not errors
        else AppRunStatus.FAILED
    )
    return AppRunResult(
        status=status,
        outputs=[_task_manifest_output(context, step_name, rows)],
        warnings=[
            f"{candidate_id}: {errors[candidate_id]}" for candidate_id in sorted(errors)
        ],
    )


def _task_manifest_output(
    context: NodeRunContext,
    step_name: str,
    rows: Sequence[Mapping[str, object]],
) -> AppOutput:
    """Write one task-aggregated manifest inside the workflow run Volume."""
    if context.volume_root is None or context.workflow_volume_name is None:
        raise RuntimeError("Workflow Volume context is unavailable")
    manifest_path = context.work_dir / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(rows, manifest_path)
    return ppiflow_manifests.manifest_artifact_output(
        manifest_path=manifest_path,
        mount_root=str(context.volume_root),
        volume_name=context.workflow_volume_name,
        stage_name=step_name,
        row_count=len(rows),
    )


def _task_result_structure_files(
    context: NodeRunContext,
    result: AppRunResult,
) -> list[dict[str, object]]:
    """Describe workflow-owned Task structures for downstream fingerprints."""
    if context.volume_root is None or context.workflow_volume_name is None:
        raise RuntimeError("Workflow Volume context is unavailable")
    records = []
    for output in result.outputs:
        candidate_files = output.metadata.get("candidate_files")
        if isinstance(candidate_files, Sequence):
            records.extend(
                dict(file_record)
                for file_record in candidate_files
                if isinstance(file_record, Mapping)
            )
            continue
        if (
            output.kind != ArtifactKind.STRUCTURES
            or not isinstance(output.storage, VolumePath)
            or output.storage.volume_name != context.workflow_volume_name
        ):
            continue
        root = output.storage.at_mountpoint(context.volume_root)
        paths = (
            [root]
            if root.is_file()
            else sorted(path for path in root.rglob("*") if path.is_file())
        )
        for path in paths:
            relative = path.relative_to(context.volume_root).as_posix()
            records.append(
                ppiflow_manifests.candidate_file_record(
                    role="structure",
                    workflow_path=relative,
                    volume_name=context.workflow_volume_name,
                    path=path.name
                    if root.is_file()
                    else path.relative_to(root).as_posix(),
                    media_type=output.storage.media_type,
                    size_bytes=path.stat().st_size,
                    content_sha256=_file_sha256(path),
                )
            )
    return records


def _ppiflow_candidate_result(
    result: AppRunResult,
    *,
    candidate_id: str,
    step_name: str,
    source_structure: str,
) -> AppRunResult:
    """Annotate one PPIFlow app result with candidate-keyed structure files."""
    outputs = []
    structure_count = 0
    for output in result.outputs:
        if (
            not isinstance(output.storage, VolumePath)
            or output.storage.volume_name != PPI_FLOW_OUTPUT_VOLUME_NAME
        ):
            outputs.append(output)
            continue
        root = output.storage.at_mountpoint(PPI_FLOW_OUTPUT_MOUNTPOINT)
        structure_paths = (
            [root]
            if root.is_file() and root.suffix.lower() in {".pdb", ".cif"}
            else sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".pdb", ".cif"}
            )
        )
        candidate_files = [
            ppiflow_manifests.candidate_file_record(
                role="structure",
                volume_name=PPI_FLOW_OUTPUT_VOLUME_NAME,
                app_volume_path=path.relative_to(PPI_FLOW_OUTPUT_MOUNTPOINT).as_posix(),
                path=(
                    path.name if root.is_file() else path.relative_to(root).as_posix()
                ),
                media_type=(
                    "chemical/x-pdb"
                    if path.suffix.lower() == ".pdb"
                    else "chemical/x-mmcif"
                ),
                size_bytes=path.stat().st_size,
                content_sha256=_file_sha256(path),
            )
            for path in structure_paths
        ]
        structure_count += len(candidate_files)
        outputs.append(
            output.model_copy(
                update={
                    "kind": ArtifactKind.STRUCTURES,
                    "metadata": dict(output.metadata)
                    | {
                        "candidate_id": candidate_id,
                        "candidate_files": candidate_files,
                        "source_structure": source_structure,
                        "step_name": step_name,
                        "structure_patterns": PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS,
                    },
                }
            )
        )
    if structure_count == 0:
        raise FileNotFoundError(
            f"PPIFlow candidate {candidate_id!r} produced no structure files"
        )
    return result.model_copy(update={"outputs": outputs})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if bool(config.get("search_msa", False)):
        raise ValueError(
            "PPIFlow ReFold does not yet support AlphaFold3 MSA search; "
            "provide an af3_config_json with populated fields or leave "
            "search_msa disabled"
        )
    enriched = resolve_msa_and_templates(
        conf,
        cast(Any, None),
        search_msa=False,
        search_protein_templates=False,
    )
    prepared = prepare_inference_run(
        enriched,
        recycle=_config_int(config, "recycle", 10),
        sample=_config_int(config, "sample", 5),
    )
    publication = RequestPublication.from_prepared(prepared)
    manifest = load_request_manifest(
        alphafold3_app.CONF.output_volume,
        publication,
    )
    if manifest is None:
        stage_inference_run(
            alphafold3_app.CONF.output_volume,
            prepared,
        )
        executor = InProcessInferenceExecutor(
            claim_function=alphafold3_app.claim_seed_prediction_work.get_raw_f(),
            inspect_function=alphafold3_app.inspect_seed_prediction_cache.get_raw_f(),
            worker_function=alphafold3_app.run_inference_pipeline.get_raw_f(),
            summary_function=alphafold3_app.finalize_inference_summary.get_raw_f(),
            request_function=alphafold3_app.finalize_inference_request.get_raw_f(),
        )
        result = coordinate_seed_predictions(
            prepared,
            executor,
            num_containers=1,
            active_wait_timeout_seconds=MAX_TIMEOUT + 900,
        )
        manifest = request_manifest_from_result(result)
    with TemporaryDirectory(prefix="biomodals-ppiflow-refold-") as temp_dir:
        archive_path = create_request_archive(
            alphafold3_app.CONF.output_volume,
            manifest,
            output_dir=temp_dir,
            display_name=run_name,
        )
        tarball_bytes = archive_path.read_bytes()
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
                    content_sha256=hashlib.sha256(data).hexdigest(),
                )
            ],
        )
        for name, data in selected_structures
    ]
    return pl.DataFrame(rows)


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


def _config_int(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{key} must be an integer")


def _optional_config_int(
    config: Mapping[str, object],
    key: str,
) -> int | None:
    if config.get(key) is None:
        return None
    return _config_int(config, key, 0)


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
) -> AF3Config:
    if config.get("af3_config_json") is not None:
        conf = AF3Config.model_validate_json(str(config["af3_config_json"]))
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

    return AF3Config(
        name=run_name,
        modelSeeds=[
            int(seed) for seed in _parse_seed_values(config.get("model_seeds", [1]))
        ],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
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
    max_child_calls: int | None = None,
) -> Workflow:
    """Build a PPIFlow workflow DAG from upstream-style YAML files."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _load_yaml_bytes(steps_yaml_bytes)
    if max_child_calls is not None:
        if max_child_calls < 1:
            raise ValueError("max_child_calls must be at least 1")
        steps_doc = _steps_doc_with_child_budget(steps_doc, max_child_calls)
    task = _task_section(task_doc)
    if max_child_calls is not None:
        task = dict(task)
        task["candidate_concurrency"] = min(
            int(task.get("candidate_concurrency", max_child_calls)),
            max_child_calls,
        )
    enabled = _enabled_section(task_doc)
    gentype = str(task.get("gentype") or task.get("design_mode") or "binder")
    candidate_concurrency = ppiflow_coordinators.candidate_concurrency_from_config(
        task,
        steps_doc,
    )
    workflow = Workflow("ppiflow-v2")
    report_table_inputs: dict[str, Any] = {}

    stage1_tail = None
    stage1_allows_partial = False
    if stage in {None, 1}:
        stage1_tail, stage1_allows_partial = _add_stage1_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            report_table_inputs=report_table_inputs,
            candidate_concurrency=candidate_concurrency,
        )

    if stage in {None, 2}:
        stage2_upstream = stage1_tail
        if stage == 2:
            stage2_upstream = workflow.add_node(
                _stage2_input_node(task, steps_doc),
                id="stage2-existing-input",
            )
        _add_stage2_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            upstream=stage2_upstream,
            upstream_allows_partial=stage1_allows_partial,
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
    report_table_inputs: dict[str, Any],
    candidate_concurrency: int,
):
    tail = None
    partial_tail = None
    if _step_enabled(enabled, "PPIFlowStep"):
        design_config = _step_cfg_with_candidate_concurrency(
            steps,
            "PPIFlowStep",
            candidate_concurrency,
        )
        runtime_config, scientific_config = _ppiflow_design_configs(design_config)
        tail = workflow.add_node(
            PPIFlowDesignNode(
                "PPIFlowStep",
                runtime_config,
                scientific_config,
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
                _step_cfg_with_candidate_concurrency(
                    steps,
                    step_name,
                    candidate_concurrency,
                ),
            ),
            id=node_id,
            inputs=_structure_inputs(tail),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
        )
        partial_tail = tail
        report_table_inputs["stage1_mpnn_seqs"] = tail.outputs(kind=ArtifactKind.TABLE)

    if _step_enabled(enabled, "FlowpackerStep_stage1"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage1",
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FlowpackerStep_stage1",
                    candidate_concurrency,
                ),
            ),
            id="stage1-flowpacker",
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage1"):
        score = _add_af3score_nodes(
            workflow=workflow,
            node_id="stage1-af3score",
            step_name="AF3scoreStep_stage1",
            config=_step_cfg_with_candidate_concurrency(
                steps,
                "AF3scoreStep_stage1",
                candidate_concurrency,
            ),
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )

    if _step_enabled(enabled, "FilterStep_stage1"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        tail = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage1",
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FilterStep_stage1",
                    candidate_concurrency,
                ),
            ),
            id="stage1-filter",
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None
        report_table_inputs["stage1_filter_tables"] = tail.outputs(
            kind=ArtifactKind.TABLE
        )
    return tail, partial_tail is not None


def _add_af3score_nodes(
    *,
    workflow: Workflow,
    node_id: str,
    step_name: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    accept_partial_from: list[Any] | None,
):
    """Add prepare, fixed-batch GPU, and postprocess Nodes for AF3Score."""
    prepare = workflow.add_node(
        AF3ScorePrepareNode(step_name, config),
        id=f"{node_id}-prepare",
        inputs=inputs,
        accept_partial_from=accept_partial_from,
    )
    batches = workflow.add_node(
        AF3ScoreBatchNode(step_name, config),
        id=f"{node_id}-batches",
        inputs={
            "af3score_plan": prepare.outputs(kind=ArtifactKind.TABLE),
        },
        allow_empty_result=True,
    )
    return workflow.add_node(
        AF3ScoreNode(step_name, config),
        id=node_id,
        inputs={
            "af3score_plan": prepare.outputs(kind=ArtifactKind.TABLE),
        },
        depends_on=[batches],
    )


def _add_rosetta_nodes(
    *,
    workflow: Workflow,
    node_id: str,
    step_name: str,
    finalizer_class: type[_RosettaNode],
    config: dict[str, Any],
    inputs: dict[str, Any],
    accept_partial_from: list[Any] | None,
):
    """Add prepare, pull-worker, and publication Nodes for Rosetta."""
    prepare = workflow.add_node(
        RosettaPrepareNode(step_name, config),
        id=f"{node_id}-prepare",
        inputs=inputs,
        accept_partial_from=accept_partial_from,
    )
    workers = workflow.add_node(
        RosettaWorkerNode(step_name, config),
        id=f"{node_id}-workers",
        inputs={
            "rosetta_plan": prepare.outputs(kind=ArtifactKind.TABLE),
        },
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    return workflow.add_node(
        finalizer_class(step_name, config),
        id=node_id,
        inputs={
            "rosetta_plan": prepare.outputs(kind=ArtifactKind.TABLE),
            "rosetta_outcomes": workers.outputs(kind=ArtifactKind.TABLE),
        },
        accept_partial_from=[workers],
    )


def _add_stage2_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    upstream,
    upstream_allows_partial: bool,
    report_table_inputs: dict[str, Any],
    candidate_concurrency: int,
) -> None:
    tail = upstream
    partial_tail = upstream if upstream_allows_partial else None
    if _step_enabled(enabled, "RosettaFixStep"):
        tail = _add_rosetta_nodes(
            workflow=workflow,
            node_id="stage2-rosetta-fix",
            step_name="RosettaFixStep",
            finalizer_class=RosettaFixNode,
            config=_step_cfg_with_candidate_concurrency(
                steps,
                "RosettaFixStep",
                candidate_concurrency,
            ),
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None

    if _step_enabled(enabled, "RosettaFixStep") and _step_enabled(
        enabled, "PartialStep"
    ):
        tail = workflow.add_node(
            FixedPositionsNode(
                "FixedPositions",
                {"gentype": gentype} | _step_cfg(steps, "FixedPositions"),
            ),
            id="stage2-fixed-positions",
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None

    if _step_enabled(enabled, "PartialStep"):
        tail = workflow.add_node(
            PPIFlowPartialNode(
                "PartialStep",
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "PartialStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-partial-ppiflow",
            inputs=_structure_inputs(tail),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = tail

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
                _step_cfg_with_candidate_concurrency(
                    steps,
                    step_name,
                    candidate_concurrency,
                ),
            ),
            id=node_id,
            inputs=_structure_inputs(tail),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = tail
        report_table_inputs["mpnn_seqs"] = tail.outputs(kind=ArtifactKind.TABLE)

    if _step_enabled(enabled, "FlowpackerStep_stage2"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage2",
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FlowpackerStep_stage2",
                    candidate_concurrency,
                ),
            ),
            id="stage2-flowpacker",
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage2"):
        score = _add_af3score_nodes(
            workflow=workflow,
            node_id="stage2-af3score",
            step_name="AF3scoreStep_stage2",
            config=_step_cfg_with_candidate_concurrency(
                steps,
                "AF3scoreStep_stage2",
                candidate_concurrency,
            ),
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )

    filtered = tail
    if _step_enabled(enabled, "FilterStep_stage2"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        filtered = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage2",
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "FilterStep_stage2",
                    candidate_concurrency,
                ),
            ),
            id="stage2-filter",
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None
        report_table_inputs["filter_tables"] = filtered.outputs(kind=ArtifactKind.TABLE)

    refold = None
    if _step_enabled(enabled, "ReFoldStep"):
        refold = workflow.add_node(
            ReFoldNode(
                "ReFoldStep",
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "ReFoldStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-alphafold3-refold",
            inputs=_structure_inputs(filtered),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
            accept_partial_from=_partial_sources(partial_tail),
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
                _step_cfg_with_candidate_concurrency(
                    steps,
                    "DockQStep",
                    candidate_concurrency,
                ),
            ),
            id="stage2-dockq",
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail, refold),
        )

    relaxed = None
    if _step_enabled(enabled, "RosettaRelaxStep"):
        inputs = _structure_inputs(filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        relaxed = _add_rosetta_nodes(
            workflow=workflow,
            node_id="stage2-rosetta-relax",
            step_name="RosettaRelaxStep",
            finalizer_class=RosettaRelaxNode,
            config=_step_cfg_with_candidate_concurrency(
                steps,
                "RosettaRelaxStep",
                candidate_concurrency,
            ),
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail),
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
                {"gentype": gentype} | _step_cfg(steps, "RankStep"),
            ),
            id="stage2-rank",
            inputs=inputs,
            accept_partial_from=_partial_sources(
                partial_tail if relaxed is None else None,
                refold,
            ),
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
            accept_partial_from=_partial_sources(
                partial_tail if rank is None else None,
                refold,
            ),
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


def _partial_sources(*sources: Any) -> list[Any] | None:
    """Return the present partial-result dependencies for one Node."""
    present = [source for source in sources if source is not None]
    return present or None


def _stage2_input_node(
    task: Mapping[str, Any],
    steps: Mapping[str, Any],
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


def _ppiflow_design_configs(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate runtime paths from content-based scientific input identity."""
    runtime_config = deepcopy(config)
    raw_digests = runtime_config.pop(_INPUT_DIGESTS_KEY, {})
    if not isinstance(raw_digests, Mapping):
        raise ValueError(f"{_INPUT_DIGESTS_KEY} must be a mapping")
    input_digests = {str(key): str(value) for key, value in raw_digests.items()}

    scientific_config = deepcopy(runtime_config)
    for key in _OPERATIONAL_CONFIG_KEYS:
        scientific_config.pop(key, None)
    scientific_args = scientific_config.get("args", scientific_config)
    if not isinstance(scientific_args, dict):
        raise ValueError("PPIFlow step args must be a mapping")
    for field_name in input_digests:
        scientific_args.pop(field_name, None)
    if input_digests:
        scientific_config["input_sha256"] = dict(sorted(input_digests.items()))
    return runtime_config, scientific_config


def _step_cfg_with_candidate_concurrency(
    steps: dict[str, Any],
    step_name: str,
    candidate_concurrency: int,
) -> dict[str, Any]:
    cfg = dict(_step_cfg(steps, step_name))
    cfg.setdefault("candidate_concurrency", candidate_concurrency)
    return cfg


def _steps_doc_with_child_budget(
    steps_doc: dict[str, Any], max_child_calls: int
) -> dict[str, Any]:
    capped: dict[str, Any] = {}
    for step_name, raw_cfg in steps_doc.items():
        if not isinstance(raw_cfg, dict):
            capped[step_name] = raw_cfg
            continue
        cfg = dict(raw_cfg)
        cfg["max_child_calls"] = max_child_calls
        for key in ("candidate_concurrency", "num_jobs", "max_batches", "max_num_pods"):
            if key in cfg and cfg[key] is not None:
                cfg[key] = min(int(cfg[key]), max_child_calls)
        capped[step_name] = cfg
    return capped


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
        input_digests: dict[str, str] = {}
        for field_name in _ppiflow_input_fields(app_args.args):
            current_value = getattr(app_args.args, field_name)
            current_path = Path(current_value)
            if current_path.is_absolute() and current_path.is_relative_to(volume_root):
                remote_storage = volume_path_from_mount_path(
                    str(current_path),
                    str(volume_root),
                    ppiflow_app.CONF.output_volume_name,
                )
                digest = hashlib.sha256()
                for chunk in ppiflow_app.CONF.output_volume.read_file(
                    remote_storage.path
                ):
                    digest.update(chunk)
                input_digests[field_name] = digest.hexdigest()
                continue

            local_path = current_path.expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(
                    f"PPIFlow {step_name} input {field_name!r} was not found "
                    f"locally or in the mounted output volume: {current_value}"
                )

            with local_path.open("rb") as input_file:
                content_sha256 = hashlib.file_digest(input_file, "sha256").hexdigest()
            input_digests[field_name] = content_sha256
            suffix = "".join(part.lower() for part in local_path.suffixes)
            remote_rel = (
                Path(run_id)
                / sanitize_filename(step_name)
                / sanitize_filename(field_name)
                / f"{content_sha256}{suffix}"
            )
            raw_args[field_name] = str(volume_root / remote_rel)
            uploads.append((local_path, remote_rel.as_posix()))
        if input_digests:
            cfg[_INPUT_DIGESTS_KEY] = input_digests

    if uploads:
        with ppiflow_app.CONF.output_volume.batch_upload(force=True) as batch:
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
    wait: bool = True,
    max_parallel: int = 16,
    max_child_calls: int | None = None,
    dry_run: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "development",
    deployment_name: str | None = None,
    deployment_version: int = 1,
    restart_from: str | None = None,
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
        wait: Wait locally for the remote workflow result. Disable to print the
            Modal function call id for asynchronous collection.
        max_parallel: Maximum ready workflow Nodes and active Provider Calls.
            Configured candidate concurrency may lower only the call limit.
        max_child_calls: Compatibility cap applied to stage fan-out settings
            and the Run-level active Provider Call limit.
        dry_run: Print the workflow DAG graph and skip orchestrator execution.
        use_deployed_coordinator: Submit through an exact named deployment.
        deployment_environment: Modal Environment containing the deployment.
        deployment_name: Modal app deployment name. Defaults to this workflow.
        deployment_version: Exact numeric Modal deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    if predecessor_execution_run_id is not None and not use_deployed_coordinator:
        raise ValueError("restart_from requires an exact deployed workflow coordinator")
    task_yaml_path = Path(task_yaml).expanduser().resolve()
    steps_yaml_path = Path(steps_yaml).expanduser().resolve()
    resolved_run_id = sanitize_filename(run_id or task_yaml_path.stem)
    task_yaml_bytes = task_yaml_path.read_bytes()
    steps_yaml_bytes = steps_yaml_path.read_bytes()
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    if dry_run:
        workflow = build_ppiflow_workflow(
            task_yaml_bytes=task_yaml_bytes,
            steps_yaml_bytes=steps_yaml_bytes,
            stage=stage,
            max_child_calls=max_child_calls,
        )
        print_workflow_dag(workflow.validate())
        return

    steps_doc = _stage_ppiflow_app_inputs(
        steps_doc=_load_yaml_bytes(steps_yaml_bytes),
        run_id=resolved_run_id,
        app_steps=_active_ppiflow_app_steps(task_doc, stage),
    )
    steps_doc = _inline_rosetta_config_files(steps_doc)
    provider_call_limit = min(
        max_parallel,
        ppiflow_coordinators.candidate_concurrency_from_config(
            _task_section(task_doc),
            steps_doc,
        ),
    )
    if max_child_calls is not None:
        provider_call_limit = min(provider_call_limit, max_child_calls)
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=task_yaml_bytes,
        steps_yaml_bytes=yaml.safe_dump(steps_doc).encode("utf-8"),
        stage=stage,
        max_child_calls=max_child_calls,
    )

    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        environment=(
            deployment_environment if use_deployed_coordinator else "development"
        ),
        deployment_name=(
            (deployment_name or CONF.name) if use_deployed_coordinator else CONF.name
        ),
        deployment_version=deployment_version if use_deployed_coordinator else 1,
    )
    coordinator = orchestrator.execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
    )
    orchestrator_kwargs = {
        "workflow": workflow,
        "workload_run_key": resolved_run_id,
        "max_parallel_nodes": max_parallel,
        "max_active_provider_calls": provider_call_limit,
        "max_active_gpu_provider_calls": provider_call_limit,
        "strict_external_artifact_checks": True,
        "external_artifact_checker_function_name": ("check_ppiflow_external_artifact"),
    }
    if not use_deployed_coordinator:
        orchestrator_kwargs["development_function_handles"] = {
            "run_ppiflow_design_stage": run_ppiflow_design_stage,
            "run_ppiflow_partial_candidate": run_ppiflow_partial_candidate,
            "run_ppiflow_ligandmpnn_candidate": run_ppiflow_ligandmpnn_candidate,
            "run_ppiflow_flowpacker_stage": run_ppiflow_flowpacker_stage,
            "prepare_ppiflow_af3score_stage": prepare_ppiflow_af3score_stage,
            "run_ppiflow_af3score_batch": run_ppiflow_af3score_batch,
            "postprocess_ppiflow_af3score_stage": (postprocess_ppiflow_af3score_stage),
            "prepare_ppiflow_rosetta_stage": prepare_ppiflow_rosetta_stage,
            "run_ppiflow_rosetta_worker": run_ppiflow_rosetta_worker,
            "finalize_ppiflow_rosetta_stage": finalize_ppiflow_rosetta_stage,
            "run_ppiflow_refold_candidate": run_ppiflow_refold_candidate,
            "run_ppiflow_dockq_stage": run_ppiflow_dockq_stage,
            "filter_ppiflow_artifacts": filter_ppiflow_artifacts,
            "derive_ppiflow_fixed_positions": derive_ppiflow_fixed_positions,
            "rank_ppiflow_artifacts": rank_ppiflow_artifacts,
            "normalize_ppiflow_stage2_input": normalize_ppiflow_stage2_input,
            "check_ppiflow_external_artifact": check_ppiflow_external_artifact,
        }
    print(
        f"Submitting PPIFlow workflow '{resolved_run_id}' with "
        f"{len(workflow.validate().nodes)} node(s)",
        flush=True,
    )
    if predecessor_execution_run_id is None:
        function_call = coordinator.run.spawn(**orchestrator_kwargs)
    else:
        function_call = coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor_execution_run_id),
            **orchestrator_kwargs,
        )
    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}",
        flush=True,
    )
    print(f"Execution Run ID: {execution_run_id}", flush=True)
    print(
        "Coordinator FunctionCall ID: "
        f"{getattr(function_call, 'object_id', function_call)}",
        flush=True,
    )
    if wait:
        result: AppRunResult | str = AppRunResult.model_validate(function_call.get())
    else:
        result = str(getattr(function_call, "object_id", function_call))
    if isinstance(result, AppRunResult):
        print(f"PPIFlow workflow run finished with status: {result.status}", flush=True)
    else:
        print(f"PPIFlow workflow run submitted. FunctionCall id: {result}", flush=True)
