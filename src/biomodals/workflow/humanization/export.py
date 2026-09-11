"""Self-contained, manifested humanization result directories."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import orjson
import polars as pl

from biomodals.app.design.pabnativ2.models import HUMANIZATION_PROTOCOL
from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import AppOutput, AppRunResult, ArtifactKind, VolumePath
from biomodals.workflow.humanization.artifacts import archive_members
from biomodals.workflow.humanization.contracts import (
    AntibodyPair,
    HumanizationCandidate,
)
from biomodals.workflow.humanization.ranking import RANKING_POLICY
from biomodals.workflow.humanization.settings import HumanizationSettings

IMGT_MUTATION_SCHEMA = {
    "parent_id": pl.String,
    "candidate_id": pl.String,
    "chain": pl.String,
    "position": pl.Int64,
    "insertion_code": pl.String,
    "parent_residue": pl.String,
    "candidate_residue": pl.String,
    "region": pl.String,
    "change_type": pl.String,
}

GENERATION_SCHEMA = {
    "parent_id": pl.String,
    "method": pl.String,
    "source_id": pl.String,
    "seed": pl.Int64,
    "root_seed": pl.Int64,
    "iteration": pl.Int64,
    "attempt_index": pl.Int64,
    "outcome": pl.String,
    "reason": pl.String,
    "vh": pl.String,
    "vl": pl.String,
}


def generation_table(
    context: NodeRunContext,
    selection: pl.DataFrame,
    parents: Sequence[AntibodyPair],
    settings: HumanizationSettings,
    errors: Mapping[str, str],
) -> pl.DataFrame:
    """Join generation outcomes to the exact union without repeating sequences."""
    rows = [
        row
        for name, artifacts in context.inputs.items()
        if name.startswith("generation_native_")
        for artifact in artifacts
        if artifact.source_app_output_name == "generation_outcomes"
        for row in orjson.loads(context.resolve_artifact(artifact).read_bytes())
    ]
    for task_key, reason in errors.items():
        method, index, *replicate = task_key.split("-")
        parameters = settings.method_arguments(method)
        seed = parameters.get("seed")
        if method == "pabnativ2":
            seed = settings.pabnativ2_seeds[int(replicate[-1]) if replicate else 0]
        rows.append({
            "parent_id": parents[int(index)].id,
            "method": method,
            "root_seed": seed,
            "outcome": "failed",
            "reason": reason,
        })
    union = selection.select("parent_id", "candidate_id", "vh", "vl", "is_parent")
    return (
        pl
        .DataFrame(rows, schema=GENERATION_SCHEMA)
        .join(union, on=["parent_id", "vh", "vl"], how="left", validate="m:1")
        .with_columns(
            pl
            .when((pl.col("outcome") == "generated") & pl.col("is_parent"))
            .then(pl.lit("no_op"))
            .otherwise(pl.col("outcome"))
            .alias("outcome")
        )
        .drop("vh", "vl", "is_parent")
        .sort("parent_id", "method", "seed", "iteration", "attempt_index", "source_id")
    )


def export_results(
    context: NodeRunContext,
    candidates: Sequence[HumanizationCandidate],
    table: pl.DataFrame,
    mutations: pl.DataFrame,
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
    settings: HumanizationSettings,
    scientific_versions: Mapping[str, str],
    parents: Sequence[AntibodyPair],
) -> AppOutput:
    """Publish a terminal directory covering every retained scientific artifact."""
    if context.volume_root is None or context.artifact_volume_name is None:
        raise RuntimeError("Result export requires the execution volume")
    root = context.work_dir / "humanization"
    root.resolve().relative_to(context.volume_root.resolve())
    root.mkdir(parents=True, exist_ok=True)
    scores = root / "scores"
    scores.mkdir(exist_ok=True)
    table.write_csv(root / "selection.csv")
    parent_by_candidate = {
        candidate.candidate_id: candidate.parent_id for candidate in candidates
    }
    generation_errors = orjson.loads(context.read_input_bytes("generation_errors"))
    generation_table(
        context, table, parents, settings, generation_errors
    ).write_parquet(root / "generation.parquet", compression="zstd")

    # Scratch shards stay off the execution volume and out of the final bundle.
    with TemporaryDirectory(prefix="humanization_scores_") as temporary:
        detail_shards: dict[str, list[pl.LazyFrame]] = {}
        for task_key, result in results.items():
            for output in result.outputs:
                if output.name in {
                    "evaluation",
                    "annotation",
                    "imgt_mutations",
                    "evaluated_union",
                    "generation_complete",
                }:
                    continue
                if (
                    not isinstance(output.storage, VolumePath)
                    or output.storage.volume_name != context.artifact_volume_name
                ):
                    raise ValueError(
                        "Native results must be materialized in the execution volume"
                    )
                source = (context.volume_root / output.storage.path).resolve()
                source.relative_to(context.volume_root.resolve())
                if (
                    not task_key.startswith(("sapiens-", "humatch-", "pabnativ2-"))
                    or output.storage.media_type != "application/zstd"
                ):
                    continue
                method, candidate_id = task_key.split("-", maxsplit=1)
                for filename, content in archive_members(source.read_bytes()).items():
                    if filename.endswith(".parquet"):
                        detail_name = f"{method}_{Path(filename).stem}"
                        shard = (
                            Path(temporary) / f"{detail_name}-{candidate_id}.parquet"
                        )
                        shard.write_bytes(content)
                        detail_shards.setdefault(detail_name, []).append(
                            pl.scan_parquet(shard).with_columns(
                                pl.lit(parent_by_candidate[candidate_id]).alias(
                                    "parent_id"
                                ),
                                pl.lit(candidate_id).alias("candidate_id"),
                            )
                        )
        for name, shards in detail_shards.items():
            pl.concat(shards).sink_parquet(
                scores / f"{name}.parquet", compression="zstd"
            )
    mutations.write_parquet(root / "imgt_mutations.parquet", compression="zstd")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "content_sha256": digest,
        })
    manifest = {
        "schema_version": 3,
        "execution_run_id": str(context.execution_run_id),
        "status": "partial" if errors else "succeeded",
        "parameters": settings.model_dump(),
        "generation_seeds": {"pabnativ2": settings.pabnativ2_seeds},
        "scientific_versions": dict(scientific_versions),
        "protocols": {"pabnativ2": HUMANIZATION_PROTOCOL},
        "candidate_count": len(candidates),
        "generation_failure_count": len(generation_errors),
        "selection_semantics": "one row per parent/exact pair; per-parent Pareto tiers and diverse panel order; unranked values are null; no composite fitness score",
        "ranking_policy": RANKING_POLICY,
        "cdr_definition": "imgt",
        "files": files,
    }
    content = orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
    (root / "manifest.json").write_bytes(content)
    files.append({
        "path": "manifest.json",
        "size_bytes": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    })
    return AppOutput(
        name="humanization_results",
        kind=ArtifactKind.DIRECTORY,
        storage=VolumePath(
            volume_name=context.artifact_volume_name,
            path=str(root.relative_to(context.volume_root)),
        ),
        metadata={
            "files": files,
            "candidate_count": len(candidates),
            "status": manifest["status"],
        },
    )
