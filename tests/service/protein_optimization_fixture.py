"""Offline candidate publication: native ridge, deterministic fake exploration scores."""

from pathlib import Path
from uuid import UUID

import polars as pl

from biomodals.app.design.mutation_ridge.inputs import variant_sequences
from biomodals.app.design.mutation_ridge.regression import (
    ValidationSummary,
    write_combinations,
)
from biomodals.helper.artifacts import sha256_file
from biomodals.workflow.protein_optimization.execution import (
    OptimizationExecutionRequest,
)
from biomodals.workflow.protein_optimization.exploration import ExplorationSpace
from biomodals.workflow.protein_optimization.publication import OptimizationManifest


def result_publication(
    root: Path, job_id: UUID, request: OptimizationExecutionRequest
) -> tuple[Path, OptimizationManifest]:
    """Return the native schema without acquiring model weights or remote compute."""
    root.mkdir(parents=True, exist_ok=True)
    source = root / "candidates.csv"
    dataset = request.design.dataset()
    settings = request.design.settings
    if settings.mode == "combination":
        summary = write_combinations(
            dataset,
            source,
            max_mutations=settings.max_mutations,
            budget=settings.candidate_budget,
            higher_is_better=settings.direction == "maximize",
            seed=settings.seed,
        )
    else:
        variants = ExplorationSpace.build(dataset, settings).sample(
            settings.candidate_budget, seed=settings.seed
        )
        measured = set(dataset.vocabulary)
        schema = {
            "id": pl.String,
            "mutations": pl.String,
            "predicted_label": pl.Float64,
            "n_mutations": pl.Int64,
            "n_new_mutations": pl.Int64,
            "warnings": pl.String,
            **{f"sequence_{chain}": pl.String for chain in dataset.parents},
        }
        table = pl.DataFrame(
            [
                {
                    "id": f"candidate_{index + 1:09d}",
                    "mutations": ",".join(m.token for m in variant),
                    "predicted_label": float(index) / 10,
                    "n_mutations": len(variant),
                    "n_new_mutations": sum(m not in measured for m in variant),
                    "warnings": "Offline fixture prediction, not a scientific score",
                    **{
                        f"sequence_{chain}": sequence
                        for chain, sequence in variant_sequences(
                            dataset.parents, variant
                        ).items()
                    },
                }
                for index, variant in enumerate(variants)
            ],
            schema=schema,
        )
        table.sort(
            ["predicted_label", "id"],
            descending=[settings.direction == "maximize", False],
        ).write_csv(source)
        summary = ValidationSummary(
            regime="same_position_alternatives",
            training_variants=len(dataset.variants),
            evaluated_variants=0,
            folds=0,
            warnings=("Offline fixture has no native neural validation",),
        )
    manifest = OptimizationManifest(
        execution_run_id=job_id,
        design_digest=request.design.digest(),
        scientific_versions=request.scientific_versions,
        mode=settings.mode,
        direction=settings.direction,
        candidate_count=request.design.candidate_count(dataset),
        chain_columns={chain: f"sequence_{chain}" for chain in dataset.parents},
        validation=summary,
        csv_bytes=source.stat().st_size,
        csv_sha256=sha256_file(source),
    )
    return source, manifest
