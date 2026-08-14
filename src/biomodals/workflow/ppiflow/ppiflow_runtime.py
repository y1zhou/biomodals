"""PPIFlow-model task implementations for the PPIFlow workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from biomodals.app.design import ppiflow_app
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow.ppiflow import manifests, staging, tables
from biomodals.workflow.ppiflow.model_validation import (
    reload_ppiflow_model_volume,
)
from biomodals.workflow.ppiflow.runtime_context import (
    PPI_FLOW_OUTPUT_MOUNTPOINT,
    PPI_FLOW_OUTPUT_VOLUME_NAME,
    SOURCE_VOLUME_ROOTS,
    SOURCE_VOLUMES,
    reload_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    candidate_manifest_frame_from_inputs,
    file_sha256,
    optional_config_int,
    patterns_from_config,
    result_with_output_kind,
    write_candidate_manifest_output,
)

PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS = (
    "*.pdb",
    "*.cif",
    "**/*.pdb",
    "**/*.cif",
    "outputs/*.pdb",
    "outputs/**/*.pdb",
    "outputs/*.cif",
    "outputs/**/*.cif",
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
    reload_source_volumes()
    reload_ppiflow_model_volume()
    result = AppRunResult.model_validate(
        ppiflow_app.ppiflow_run_workflow.get_raw_f()(
            args=args,
            run_name=run_name,
        )
    )
    adapted = result_with_output_kind(
        result,
        ArtifactKind.STRUCTURES,
        {
            "step_name": step_name,
            "structure_patterns": PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS,
        },
    )
    rows = _initial_candidate_rows(adapted, step_name=step_name)
    return adapted.model_copy(
        update={
            "outputs": [
                *adapted.outputs,
                write_candidate_manifest_output(
                    run_id=run_id,
                    node_id=node_id,
                    step_name=step_name,
                    rows=rows,
                ),
            ]
        }
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
    reload_source_volumes()
    reload_ppiflow_model_volume()
    selected = staging.select_structure_files_from_artifacts(
        artifacts,
        SOURCE_VOLUME_ROOTS,
        patterns=patterns_from_config(config),
        max_files=optional_config_int(config, "max_structures"),
    )
    candidate_structures = staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=candidate_manifest_frame_from_inputs(
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
    fixed_positions = _fixed_positions_by_candidate(artifacts, candidate_structures)
    raw_args = deepcopy(config.get("args", config))
    if not isinstance(raw_args, dict):
        raise ValueError(f"PPIFlow step {step_name!r} args must be a mapping")
    field_name = "complex_pdb" if "complex_pdb" in raw_args else "input_pdb"
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
    SOURCE_VOLUMES[PPI_FLOW_OUTPUT_VOLUME_NAME].commit()

    app_args = validated_partial_args(
        raw_args,
        structure_path=str(staged_path),
        fixed_positions=fixed_positions.get(candidate_id),
    )
    result = AppRunResult.model_validate(
        ppiflow_app.ppiflow_run_workflow.get_raw_f()(
            args=app_args,
            run_name=sanitize_filename(f"{run_name}-{candidate_id}"),
        )
    )
    return _candidate_result(
        result,
        candidate_id=candidate_id,
        step_name=step_name,
        source_structure=structure.file_name,
    )


def validated_partial_args(
    raw_args: dict[str, Any],
    *,
    structure_path: str,
    fixed_positions: str | None,
) -> ppiflow_app.PPIFlowArgs:
    """Complete and validate PPIFlow partial-design arguments."""
    completed = raw_args.copy()
    structure_field = "complex_pdb" if "complex_pdb" in completed else "input_pdb"
    completed[structure_field] = structure_path
    if "fixed_positions" not in completed and fixed_positions is not None:
        completed["fixed_positions"] = fixed_positions
    return ppiflow_app.PPIFlowArgs.model_validate({"args": completed})


def _initial_candidate_rows(
    result: AppRunResult,
    *,
    step_name: str,
) -> list[dict[str, object]]:
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
        for structure in staging.selected_structure_file_records_from_artifact(
            artifact,
            PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS,
            SOURCE_VOLUME_ROOTS,
        ):
            candidate_id = manifests.initial_candidate_id(
                stage_name=step_name,
                source_artifact_id=structure.artifact_id,
                source_path=structure.app_volume_path,
                basename=structure.file_name,
            )
            rows.append(
                manifests.candidate_manifest_row(
                    candidate_id=candidate_id,
                    stage_name=step_name,
                    stage_role="initial_design",
                    operation_mode="ppiflow",
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
                            content_sha256=structure.content_sha256,
                        )
                    ],
                )
            )
    if not rows:
        raise FileNotFoundError("PPIFlow design produced no structure candidates")
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def _fixed_positions_by_candidate(
    artifacts: Sequence[WorkflowArtifact],
    selected_structures: Sequence[staging.CandidateStructureFile],
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
            tables.candidate_key(structure.file_name),
        }
        if structure.source_path:
            keys.add(tables.candidate_key(structure.source_path))
        for key in keys:
            if key in lookup:
                by_candidate[structure.candidate_id] = lookup[key]
                break
    return by_candidate


def _candidate_result(
    result: AppRunResult,
    *,
    candidate_id: str,
    step_name: str,
    source_structure: str,
) -> AppRunResult:
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
            manifests.candidate_file_record(
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
                content_sha256=file_sha256(path),
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
                        "files": [
                            ArtifactFile.model_validate(file_record).model_dump(
                                exclude_defaults=True,
                                exclude_none=True,
                            )
                            for file_record in candidate_files
                        ],
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
