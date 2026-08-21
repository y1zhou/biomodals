"""AlphaFold3 ReFold task implementation for the PPIFlow workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold import alphafold3_app
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
    request_archive_member_for_role,
    request_manifest_from_result,
)
from biomodals.app.fold.alphafold3.search_pipeline import resolve_msa_and_templates
from biomodals.helper.app_run import volume_path_from_mount_path
from biomodals.helper.artifacts import publish_content_addressed_file
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    ExecutionArtifact,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE
from biomodals.workflow.ppiflow import manifests, staging, tables
from biomodals.workflow.ppiflow.runtime_context import (
    ALPHAFOLD3_OUTPUT_MOUNTPOINT,
    ALPHAFOLD3_OUTPUT_VOLUME_NAME,
    SOURCE_VOLUME_ROOTS,
    SOURCE_VOLUMES,
    reload_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    candidate_manifest_frame_from_inputs,
    config_int,
    inline_csv_table_output,
    optional_config_int,
    parse_seed_values,
    patterns_from_config,
)


def run_ppiflow_refold_candidate(
    *,
    artifacts: list[ExecutionArtifact],
    candidate_manifests: list[ExecutionArtifact] | None,
    candidate_id: str,
    config: dict[str, object],
    step_name: str,
    run_name: str,
) -> AppRunResult:
    """Run AlphaFold3 refolding for one kernel-owned candidate Task."""
    reload_source_volumes()
    selected = staging.select_structure_files_from_artifacts(
        artifacts,
        SOURCE_VOLUME_ROOTS,
        patterns=patterns_from_config(config, default=("*.pdb",)),
        max_files=optional_config_int(config, "max_structures"),
    )
    structures = staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=candidate_manifest_frame_from_inputs(
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
        outputs=_run_candidate(
            structure_name=structure.file_name,
            structure_bytes=structure.data,
            candidate_id=candidate_id,
            run_name=candidate_run_name,
            step_name=step_name,
            config=config,
        ),
        metrics={"candidate_id": candidate_id},
    )


def _run_candidate(
    *,
    structure_name: str,
    structure_bytes: bytes,
    candidate_id: str,
    run_name: str,
    step_name: str,
    config: Mapping[str, object],
) -> list[AppOutput]:
    conf = _af3_config(
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
        recycle=config_int(config, "recycle", 10),
        sample=config_int(config, "sample", 5),
    )
    publication = RequestPublication.from_prepared(prepared)
    output_volume = alphafold3_app.CONF.output_volume
    manifest = load_request_manifest(output_volume, publication)
    if manifest is None:
        stage_inference_run(output_volume, prepared)
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
            output_volume,
            manifest,
            output_dir=temp_dir,
            display_name=run_name,
        )
        publication_root = (
            Path(ALPHAFOLD3_OUTPUT_MOUNTPOINT)
            / "ppiflow"
            / "refold"
            / sanitize_filename(run_name)
        )
        published_archive, archive_size, archive_sha256 = (
            publish_content_addressed_file(archive_path, publication_root)
        )
        archive_file = ArtifactFile(
            path=published_archive.name,
            media_type=ZSTD_MEDIA_TYPE,
            size_bytes=archive_size,
            content_sha256=archive_sha256,
        )
        json_files = staging.files_from_tar_zst_path(
            published_archive,
            suffixes=(".json",),
        )
    SOURCE_VOLUMES[ALPHAFOLD3_OUTPUT_VOLUME_NAME].commit()
    best_model_member = request_archive_member_for_role(
        manifest,
        role="request_best_model",
        display_name=run_name,
    )
    best_summary_member = request_archive_member_for_role(
        manifest,
        role="request_best_summary_confidences",
        display_name=run_name,
    )
    best_summary_files = [item for item in json_files if item[0] == best_summary_member]
    if len(best_summary_files) != 1:
        raise ValueError(
            f"AlphaFold3 request archive requires member {best_summary_member!r}; "
            f"found {len(best_summary_files)}"
        )
    metric_rows = tables.refold_metric_rows_from_json_files(
        best_summary_files,
        candidate_id=candidate_id,
        stage_name=step_name,
    )
    metrics_output = inline_csv_table_output(
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
            storage=volume_path_from_mount_path(
                str(published_archive),
                ALPHAFOLD3_OUTPUT_MOUNTPOINT,
                ALPHAFOLD3_OUTPUT_VOLUME_NAME,
                ZSTD_MEDIA_TYPE,
            ),
            metadata={
                "step_name": step_name,
                "run_name": run_name,
                "candidate_id": candidate_id,
                "source_structure": structure_name,
                "archive_format": "tar.zst",
                "structure_patterns": (best_model_member,),
                "request_best_model_archive_member": best_model_member,
                "candidate_files": [
                    manifests.candidate_file_record(
                        role="structure",
                        volume_name=ALPHAFOLD3_OUTPUT_VOLUME_NAME,
                        app_volume_path=published_archive.relative_to(
                            ALPHAFOLD3_OUTPUT_MOUNTPOINT
                        ).as_posix(),
                        path=published_archive.name,
                        media_type=ZSTD_MEDIA_TYPE,
                        size_bytes=archive_size,
                        content_sha256=archive_sha256,
                    )
                ],
                "files": [
                    archive_file.model_dump(
                        exclude_defaults=True,
                        exclude_none=True,
                    )
                ],
            },
        )
    ]
    if metrics_output is not None:
        outputs.append(metrics_output)
    return outputs


def _af3_config(
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
        residue_key = (chain_id, line[22:27].strip())
        if residue_key in seen:
            continue
        seen.add(residue_key)
        chains.setdefault(chain_id, []).append(
            residue_map.get(line[17:20].strip().upper(), "X")
        )
    if not chains:
        raise ValueError(f"Could not derive AlphaFold3 sequence from {structure_name}")
    return AF3Config(
        name=run_name,
        modelSeeds=parse_seed_values(config.get("model_seeds", [1])),
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(id=chain_id, sequence="".join(sequence))
            )
            for chain_id, sequence in sorted(chains.items())
        ],
    )
