"""Self-contained, manifested humanization result directories."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import orjson
import polars as pl

from biomodals.app.design.pabnativ2.models import HUMANIZATION_PROTOCOL
from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import AppOutput, AppRunResult, ArtifactKind, VolumePath
from biomodals.workflow.humanization.artifacts import archive_members
from biomodals.workflow.humanization.contracts import HumanizationCandidate
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


def export_results(
    context: NodeRunContext,
    candidates: Sequence[HumanizationCandidate],
    table: pl.DataFrame,
    mutations: pl.DataFrame,
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
    settings: HumanizationSettings,
    scientific_versions: Mapping[str, str],
) -> AppOutput:
    """Publish a terminal directory covering every retained scientific artifact."""
    if context.volume_root is None or context.artifact_volume_name is None:
        raise RuntimeError("Result export requires the execution volume")
    root = context.work_dir / "humanization"
    root.resolve().relative_to(context.volume_root.resolve())
    root.mkdir(parents=True, exist_ok=True)
    native = root / "native"
    native.mkdir(exist_ok=True)
    scores = root / "scores"
    scores.mkdir(exist_ok=True)
    table.write_csv(root / "selection.csv")
    parent_by_candidate = {
        candidate.candidate_id: candidate.parent_id for candidate in candidates
    }
    artifact_sources = []
    scoring_publications = []

    for artifact in (
        artifact
        for name, artifacts in context.inputs.items()
        if name.startswith("generation_native_")
        for artifact in artifacts
    ):
        if artifact.source_app_output_name in {
            "parents",
            "generated",
            "generation_errors",
        }:
            continue
        source = context.resolve_artifact(artifact)
        name = hashlib.sha256(artifact.artifact_id.encode()).hexdigest()
        destination = native / f"generation-{name}{source.suffix}"
        shutil.copyfile(source, destination)
        artifact_sources.append({
            "file": str(destination.relative_to(root)),
            "artifact_id": artifact.artifact_id,
            "output_name": artifact.source_app_output_name,
            "metadata": artifact.metadata,
        })

    # Scratch shards stay off the execution volume and out of the final bundle.
    with TemporaryDirectory(prefix="humanization_scores_") as temporary:
        detail_shards: dict[str, list[pl.LazyFrame]] = {}
        for task_key, result in results.items():
            for index, output in enumerate(result.outputs):
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
                    destination = native / f"{task_key}-{index}{source.suffix}"
                    shutil.copyfile(source, destination)
                    artifact_sources.append({
                        "file": str(destination.relative_to(root)),
                        "task_key": task_key,
                        "output_name": output.name,
                        "metadata": output.metadata,
                    })
                    continue
                method, candidate_id = task_key.split("-", maxsplit=1)
                publication = {
                    "task_key": task_key,
                    "output_name": output.name,
                    "metadata": output.metadata,
                }
                for filename, content in archive_members(source.read_bytes()).items():
                    if filename == "manifest.json":
                        publication["manifest"] = orjson.loads(content)
                    elif filename in {"input.csv", "summary.csv"}:
                        # Exact pairs and every native summary field are in selection.csv.
                        continue
                    elif filename.endswith(".parquet"):
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
                    else:
                        destination = native / f"{task_key}-{filename}"
                        destination.write_bytes(content)
                        artifact_sources.append({
                            "file": str(destination.relative_to(root)),
                            "task_key": task_key,
                            "output_name": output.name,
                            "archive_member": filename,
                        })
                scoring_publications.append(publication)
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
        "schema_version": 2,
        "execution_run_id": str(context.execution_run_id),
        "status": "partial" if errors else "succeeded",
        "parameters": settings.model_dump(),
        "scientific_versions": dict(scientific_versions),
        "protocols": {"pabnativ2": HUMANIZATION_PROTOCOL},
        "candidate_count": len(candidates),
        "candidate_provenance": [
            candidate.model_dump(include={"parent_id", "candidate_id", "origins"})
            for candidate in candidates
        ],
        "errors": dict(errors),
        "generation_errors": orjson.loads(
            context.read_input_bytes("generation_errors")
        ),
        "native_publications": artifact_sources,
        "scoring_publications": scoring_publications,
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
