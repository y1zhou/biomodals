"""LigandMPNN task implementation for the PPIFlow workflow."""

from dataclasses import asdict

from biomodals.app.design import ligandmpnn_app
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    ArtifactKind,
    ExecutionArtifact,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE
from biomodals.workflow.ppiflow import staging, tables
from biomodals.workflow.ppiflow.runtime_context import (
    SOURCE_VOLUME_ROOTS,
    reload_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    bytes_payload,
    candidate_manifest_frame_from_inputs,
    inline_csv_table_output,
    inline_output_file_records,
    optional_config_int,
    parse_seed_values,
    patterns_from_config,
    result_with_output_kind,
)


def run_ppiflow_ligandmpnn_candidate(
    *,
    artifacts: list[ExecutionArtifact],
    candidate_manifests: list[ExecutionArtifact] | None,
    candidate_id: str,
    config: dict[str, object],
    step_name: str,
    run_name: str,
    script_mode: str,
    cli_args: dict[str, str | int | float | bool],
) -> AppRunResult:
    """Run LigandMPNN for one kernel-owned PPIFlow candidate."""
    reload_source_volumes()
    selected = staging.select_structure_files_from_artifacts(
        artifacts,
        SOURCE_VOLUME_ROOTS,
        patterns=patterns_from_config(config),
        max_files=optional_config_int(config, "max_structures"),
    )
    structures = [
        asdict(structure)
        for structure in staging.candidate_structure_files_from_selected(
            selected,
            manifest_frame=candidate_manifest_frame_from_inputs(
                candidate_manifests or [],
                selected,
                step_name=step_name,
            ),
        )
    ]
    matches = [
        structure
        for structure in structures
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
            struct_bytes=bytes_payload(structure["data"], "structure data"),
            seeds=parse_seed_values(config.get("seeds", [0])),
            cli_args=cli_args,
            bias_aa_per_residue_bytes=config.get("bias_aa_per_residue_bytes"),
            omit_aa_per_residue_bytes=config.get("omit_aa_per_residue_bytes"),
        )
    )
    outputs = _stage_outputs(
        result,
        candidate_id=candidate_id,
        step_name=step_name,
        selected_structure=str(structure["file_name"]),
    )
    candidate_files = inline_output_file_records([
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


def _stage_outputs(
    result: AppRunResult,
    *,
    candidate_id: str,
    step_name: str,
    selected_structure: str,
) -> list[AppOutput]:
    adapted = result_with_output_kind(
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
            tables.mpnn_sequence_rows_from_fasta_files(
                staging.files_from_tar_zst_bytes(
                    output.storage.data,
                    suffixes=(".fa", ".faa", ".fasta"),
                ),
                stage_name=step_name,
                parent_candidate_id=candidate_id,
            )
        )
    sequence_output = inline_csv_table_output(
        name="mpnn_seqs",
        filename=f"{sanitize_filename(candidate_id)}_mpnn_seqs.csv",
        rows=sequence_rows,
        metadata={"candidate_id": candidate_id, "step_name": step_name},
    )
    return [*adapted.outputs, *([sequence_output] if sequence_output else [])]
