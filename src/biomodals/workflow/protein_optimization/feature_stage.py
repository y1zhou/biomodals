"""Run-owned ESM feature tables consumed by the included generic TabPFN app."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import polars as pl
from pydantic import BaseModel, ConfigDict

from biomodals.app.design.mutation_ridge.inputs import variant_key, variant_sequences
from biomodals.app.misc.tabpfn.execution import TableFile, TabPFNRequest
from biomodals.app.misc.tabpfn.tables import TableFeature, TableSchema
from biomodals.helper.app_run import volume_app_output
from biomodals.helper.artifacts import (
    file_matches_sha256,
    read_bounded_file_bytes,
    sha256_file,
    write_bytes_atomic,
)
from biomodals.schema import AppRunResult, AppRunStatus, ArtifactFile, ArtifactKind
from biomodals.workflow.protein_optimization.design import (
    PCA_COMPONENTS,
    OptimizationDesign,
)
from biomodals.workflow.protein_optimization.exploration import ExplorationSpace
from biomodals.workflow.protein_optimization.features import (
    native_encoder,
    variant_features,
)
from biomodals.workflow.protein_optimization.validation import exploration_folds


class FeaturePublication(BaseModel):
    """Compact complete marker; interruption before it is safe to recompute."""

    model_config = ConfigDict(extra="forbid")
    design_digest: str
    scientific_versions: dict[str, str]
    result: AppRunResult


def feature_tables(
    design_json: str,
    *,
    output_key: str,
    volume_root: Path,
    volume_name: str,
    model_root: Path,
    scientific_versions: dict[str, str],
) -> AppRunResult:
    """Write content-bound feature files; labels never enter the frozen encoder."""
    design = OptimizationDesign.model_validate_json(design_json)
    if (
        design.settings.mode != "exploration"
        or design.scientific_versions() != scientific_versions
    ):
        raise ValueError("Target deployment changed Exploration scientific versions")
    if re.fullmatch(r"[0-9a-f]{64}", output_key) is None:
        raise ValueError("Invalid run-owned output key")
    root = volume_root / "inputs" / output_key[:32]
    marker = root / ".result.json"
    if marker.exists():
        saved = FeaturePublication.model_validate_json(
            read_bounded_file_bytes(
                marker, field_name="feature publication", max_bytes=32 * 1024 * 1024
            )
        )
        if (
            saved.design_digest != design.digest()
            or saved.scientific_versions != scientific_versions
        ):
            raise ValueError(
                "Feature publication belongs to a different scientific request"
            )
        records = saved.result.outputs[0].metadata["files"]
        if {record["path"] for record in records} != {
            "training.parquet",
            "inference.parquet",
            "candidates.parquet",
        } or any(
            not file_matches_sha256(
                root / record["path"], record["size_bytes"], record["content_sha256"]
            )
            for record in records
        ):
            raise ValueError("Published feature tables failed integrity validation")
        return saved.result
    dataset = design.dataset()
    candidates = ExplorationSpace.build(dataset, design.settings).sample(
        design.settings.candidate_budget, seed=design.settings.seed
    )
    if not candidates:
        raise ValueError("Empty Exploration spaces must skip scientific providers")
    matrix, chain_order = variant_features(
        dataset.parents, (*dataset.variants, *candidates), native_encoder(model_root)
    )
    names = [f"feature_{i:05d}" for i in range(matrix.shape[1])]
    schema = TableSchema(
        target="label",
        identifier="id",
        features=tuple(TableFeature(name=name) for name in names),
    )
    n_train = len(dataset.variants)
    candidate_ids = [f"candidate_{i:09d}" for i in range(1, len(candidates) + 1)]
    root.mkdir(parents=True, exist_ok=True)
    training = pl.DataFrame(matrix[:n_train], schema=names).with_columns(
        pl.Series("id", [f"measurement_{i:09d}" for i in range(n_train)]),
        dataset.measurements["label"],
    )
    inference = pl.DataFrame(matrix[n_train:], schema=names).with_columns(
        pl.Series("id", candidate_ids)
    )
    training.write_parquet(root / "training.parquet", compression="zstd")
    inference.write_parquet(root / "inference.parquet", compression="zstd")
    known = set(dataset.vocabulary)
    known_sites = {m.site for m in known}
    known_chains = {m.chain for m in known}
    rows = []
    for candidate_id, variant in zip(candidate_ids, candidates, strict=True):
        warnings = []
        if any(m.site not in known_sites for m in variant):
            warnings.append("Previously unmeasured position")
        if any(m.chain not in known_chains for m in variant):
            warnings.append("Previously unmeasured chain")
        rows.append({
            "id": candidate_id,
            "mutations": variant_key(variant),
            "n_mutations": len(variant),
            "n_new_mutations": sum(m not in known for m in variant),
            "warnings": "; ".join(warnings) or None,
            **{
                f"sequence_{chain}": seq
                for chain, seq in variant_sequences(dataset.parents, variant).items()
            },
        })
    pl.DataFrame(
        rows,
        schema_overrides={
            "warnings": pl.String,
            "n_mutations": pl.UInt32,
            "n_new_mutations": pl.UInt32,
        },
    ).write_parquet(root / "candidates.parquet", compression="zstd")
    files = [
        ArtifactFile(
            path=name,
            size_bytes=(root / name).stat().st_size,
            content_sha256=sha256_file(root / name),
        )
        for name in ("training.parquet", "inference.parquet", "candidates.parquet")
    ]
    request = TabPFNRequest(
        training=TableFile(
            path=str((root / files[0].path).relative_to(volume_root)),
            size_bytes=cast(int, files[0].size_bytes),
            sha256=cast(str, files[0].content_sha256),
        ),
        inference=TableFile(
            path=str((root / files[1].path).relative_to(volume_root)),
            size_bytes=cast(int, files[1].size_bytes),
            sha256=cast(str, files[1].content_sha256),
        ),
        table_schema=schema,
        seed=design.settings.seed,
        pca_components=PCA_COMPONENTS,
        validation_folds=tuple(
            fold.test_indices
            for fold in exploration_folds(dataset.variants, seed=design.settings.seed)
        ),
    )
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="features",
                kind=ArtifactKind.DIRECTORY,
                remote_path=str(root),
                mount_root=str(volume_root),
                volume_name=volume_name,
                files=files,
                metadata={
                    "tabpfn_request": request.model_dump(mode="json"),
                    "chain_order": list(chain_order),
                    "candidate_count": len(candidates),
                },
            )
        ],
    )
    write_bytes_atomic(
        marker,
        FeaturePublication(
            design_digest=design.digest(),
            scientific_versions=scientific_versions,
            result=result,
        )
        .model_dump_json()
        .encode(),
    )
    return result
