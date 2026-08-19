"""DockQ task implementation for the PPIFlow workflow."""

import shlex

from biomodals.app.score import dockq_app
from biomodals.schema import AppRunResult, ExecutionArtifact
from biomodals.workflow.ppiflow import staging
from biomodals.workflow.ppiflow.runtime_context import (
    SOURCE_VOLUME_ROOTS,
    reload_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    candidate_manifest_frame_from_inputs,
    optional_config_int,
    patterns_from_config,
)


def run_ppiflow_dockq_stage(
    *,
    reference_artifacts: list[ExecutionArtifact],
    model_artifacts: list[ExecutionArtifact],
    candidate_manifests: list[ExecutionArtifact] | None,
    config: dict[str, object],
    run_name: str,
) -> AppRunResult:
    """Select and pair candidate structures before invoking DockQ."""
    reload_source_volumes()
    selected_references = staging.select_structure_files_from_artifacts(
        reference_artifacts,
        SOURCE_VOLUME_ROOTS,
        patterns=patterns_from_config(config),
        max_files=optional_config_int(config, "max_structures"),
    )
    manifest = candidate_manifest_frame_from_inputs(
        candidate_manifests or [],
        selected_references,
        step_name="DockQInput",
    )
    references = staging.candidate_structure_files_from_selected(
        selected_references,
        manifest_frame=manifest,
    )
    models = []
    for artifact in model_artifacts:
        candidate_id = artifact.metadata.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(
                f"DockQ model artifact {artifact.artifact_id!r} has no "
                "canonical candidate_id"
            )
        best_model_member = artifact.metadata.get("request_best_model_archive_member")
        if best_model_member is not None and (
            not isinstance(best_model_member, str) or not best_model_member
        ):
            raise ValueError(
                f"DockQ model artifact {artifact.artifact_id!r} has an invalid "
                "request-ranked AlphaFold3 model member"
            )
        model_patterns = (
            (best_model_member,)
            if isinstance(best_model_member, str) and best_model_member
            else patterns_from_config(config)
        )
        selected = staging.select_structure_files_from_artifacts(
            [artifact],
            SOURCE_VOLUME_ROOTS,
            patterns=model_patterns,
        )
        if best_model_member is not None and len(selected) != 1:
            raise ValueError(
                f"DockQ model artifact {artifact.artifact_id!r} did not contain "
                "exactly one request-ranked AlphaFold3 model"
            )
        models.extend(
            staging.CandidateStructureFile(
                candidate_id=candidate_id,
                file_name=file_name,
                data=data,
                source_path=file_name,
            )
            for file_name, data in selected
        )
    models.sort(key=lambda item: item.file_name)
    if (max_models := optional_config_int(config, "max_models")) is not None:
        models = models[:max_models]
    pairs = staging.prepare_dockq_pairs_by_candidate(
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
