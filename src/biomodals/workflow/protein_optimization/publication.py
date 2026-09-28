"""One CSV scientific boundary plus a compact internal validation manifest."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal
from uuid import UUID

import polars as pl
from pydantic import BaseModel, ConfigDict

from biomodals.app.design.mutation_ridge.regression import (
    MAX_VALIDATION_BYTES,
    ValidationSummary,
)
from biomodals.execution.artifacts import execution_artifact_availability_errors
from biomodals.execution.nodes import NodeRunContext
from biomodals.helper.app_run import volume_app_output
from biomodals.helper.artifacts import (
    read_bounded_file_bytes,
    sha256_file,
    write_bytes_atomic,
)
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    ExecutionArtifact,
)
from biomodals.workflow.protein_optimization.design import OptimizationDesign
from biomodals.workflow.protein_optimization.validation import (
    exploration_folds,
    exploration_summary,
)

RESULT_SCHEMA = "protein_optimization/1"
MAX_MANIFEST_BYTES = MAX_VALIDATION_BYTES + 1024 * 1024


class OptimizationManifest(BaseModel):
    """Internal evidence needed to verify and explain the CSV, not a second download."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    execution_run_id: UUID
    design_digest: str
    scientific_versions: dict[str, str]
    mode: Literal["combination", "exploration"]
    direction: Literal["maximize", "minimize"]
    candidate_count: int
    chain_columns: dict[str, str]
    validation: ValidationSummary
    fitted_feature_widths: tuple[int, ...] = ()
    csv_sha256: str
    csv_bytes: int


def checked_artifact_path(artifact: ExecutionArtifact, roots: dict[str, Path]) -> Path:
    """Resolve only a known mounted Volume and verify every declared content digest."""
    root = roots.get(artifact.storage.volume_name)
    if root is None:
        raise ValueError("Scientific artifact belongs to an unexpected Volume")
    errors = execution_artifact_availability_errors(
        artifact, artifact_volume_name=artifact.storage.volume_name, volume_root=root
    )
    if errors:
        raise ValueError(
            "Scientific artifact failed integrity validation: " + "; ".join(errors)
        )
    return artifact.storage.at_mountpoint(root)


def publish_candidates(
    design: OptimizationDesign,
    context: NodeRunContext,
    *,
    roots: dict[str, Path],
    scored: ExecutionArtifact | None = None,
    features: ExecutionArtifact | None = None,
    validation: ExecutionArtifact | None = None,
) -> AppRunResult:
    """Join exact candidate IDs, preserve raw predictions and publish the whole CSV."""
    if context.volume_root is None or context.artifact_volume_name is None:
        raise ValueError("Publication requires the execution Volume")
    root = context.work_dir / "protein_optimization"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "candidates.csv"
    dataset = design.dataset()
    widths: tuple[int, ...] = ()
    count = design.candidate_count(dataset)
    if design.settings.mode == "combination":
        if scored is None or validation is None:
            raise ValueError("Ridge result and validation evidence are required")
        source = checked_artifact_path(scored, roots)
        summary = ValidationSummary.model_validate_json(
            read_bounded_file_bytes(
                checked_artifact_path(validation, roots),
                field_name="ridge validation",
                max_bytes=MAX_VALIDATION_BYTES,
            )
        )
        if scored.metadata["candidate_count"] != count:
            raise ValueError("Ridge result does not cover the admitted candidate space")
        shutil.copyfile(source, destination)
    elif count == 0:
        schema = {
            "id": pl.String,
            "mutations": pl.String,
            "predicted_label": pl.Float64,
            "n_mutations": pl.UInt32,
            "n_new_mutations": pl.UInt32,
            "warnings": pl.String,
            **{f"sequence_{chain}": pl.String for chain in dataset.parents},
        }
        pl.DataFrame(schema=schema).write_csv(destination)
        summary = ValidationSummary(
            regime="same_position_alternatives",
            training_variants=len(dataset.variants),
            evaluated_variants=0,
            folds=0,
            warnings=(
                "No novel candidates satisfy this design space; model extraction, fitting and validation were skipped.",
            ),
        )
    else:
        if features is None or scored is None:
            raise ValueError(
                "Exploration requires both frozen candidate features and native predictions"
            )
        directory = checked_artifact_path(features, roots)
        candidates = pl.read_parquet(directory / "candidates.parquet")
        predictions = pl.read_csv(
            checked_artifact_path(scored, roots),
            schema={"id": pl.String, "predicted_label": pl.Float64},
        )
        if (
            candidates.height != count
            or predictions.height != count
            or not predictions["predicted_label"].is_finite().fill_null(False).all()
        ):
            raise ValueError("Predictions do not cover every admitted novel candidate")
        result = candidates.join(
            predictions, on="id", how="left", validate="1:1", maintain_order="left"
        )
        if result["predicted_label"].null_count():
            raise ValueError(
                "Native predictions do not match exact candidate identities"
            )
        result.select(
            "id",
            "mutations",
            "predicted_label",
            "n_mutations",
            "n_new_mutations",
            "warnings",
            *[f"sequence_{chain}" for chain in dataset.parents],
        ).sort(
            ["predicted_label", "id"],
            descending=[design.settings.direction == "maximize", False],
        ).write_csv(destination)
        folds = exploration_folds(dataset.variants, seed=design.settings.seed)
        if folds:
            if validation is None:
                raise ValueError("Expected held-out validation predictions")
            held = pl.read_csv(checked_artifact_path(validation, roots))
            values = []
            for index, fold in enumerate(folds):
                rows = held.filter(pl.col("fold_index") == index)
                if rows["row_index"].to_list() != list(fold.test_indices):
                    raise ValueError(
                        "Validation predictions do not match the approved holdouts"
                    )
                values.append(rows["predicted_label"].to_numpy())
            if held.height != sum(len(fold.test_indices) for fold in folds):
                raise ValueError("Unexpected validation prediction rows")
        else:
            values = []
        summary = exploration_summary(
            dataset.variants, dataset.measurements["label"].to_numpy(), folds, values
        )
        widths = tuple(scored.metadata["fitted_feature_widths"])
    digest, size = sha256_file(destination), destination.stat().st_size
    manifest = OptimizationManifest(
        execution_run_id=context.execution_run_id,
        design_digest=design.digest(),
        scientific_versions=design.scientific_versions(),
        mode=design.settings.mode,
        direction=design.settings.direction,
        candidate_count=count,
        chain_columns={chain: f"sequence_{chain}" for chain in dataset.parents},
        validation=summary,
        fitted_feature_widths=widths,
        csv_sha256=digest,
        csv_bytes=size,
    )
    manifest_path = root / "manifest.json"
    write_bytes_atomic(manifest_path, manifest.model_dump_json().encode())
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="optimization_results",
                kind=ArtifactKind.DIRECTORY,
                remote_path=str(root),
                mount_root=str(context.volume_root),
                volume_name=context.artifact_volume_name,
                files=[
                    ArtifactFile(
                        path=destination.name, size_bytes=size, content_sha256=digest
                    ),
                    ArtifactFile(
                        path=manifest_path.name,
                        size_bytes=manifest_path.stat().st_size,
                        content_sha256=sha256_file(manifest_path),
                    ),
                ],
                metadata={"candidate_count": count, "design_digest": design.digest()},
            )
        ],
    )
