"""Self-contained, manifested humanization result directories."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path

import orjson
import polars as pl

from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import AppOutput, AppRunResult, ArtifactKind, VolumePath
from biomodals.workflow.humanization.artifacts import archive_members
from biomodals.workflow.humanization.contracts import HumanizationCandidate
from biomodals.workflow.humanization.settings import HumanizationSettings


def export_results(
    context: NodeRunContext,
    candidates: Sequence[HumanizationCandidate],
    table: pl.DataFrame,
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
    table.write_csv(root / "selection.csv")
    table.write_parquet(root / "selection.parquet", compression="zstd")
    (root / "candidates.fasta").write_text(
        "".join(
            f">{candidate.candidate_id}_{chain.upper()}\n{getattr(candidate, chain)}\n"
            for candidate in candidates
            for chain in ("vh", "vl")
        )
    )
    (root / "candidate_provenance.json").write_bytes(
        orjson.dumps([candidate.model_dump() for candidate in candidates])
    )
    parent_by_candidate = {
        candidate.candidate_id: candidate.parent_id for candidate in candidates
    }
    detail_shards: dict[str, list[Path]] = {}
    mutations = []
    artifact_sources = []

    for artifact in context.inputs.get("generation_native", []):
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

    for task_key, result in results.items():
        for index, output in enumerate(result.outputs):
            if (
                not isinstance(output.storage, VolumePath)
                or output.storage.volume_name != context.artifact_volume_name
            ):
                raise ValueError(
                    "Native results must be materialized in the execution volume"
                )
            source = (context.volume_root / output.storage.path).resolve()
            source.relative_to(context.volume_root.resolve())
            destination = native / f"{task_key}-{index}{source.suffix}"
            shutil.copyfile(source, destination)
            artifact_sources.append({
                "file": str(destination.relative_to(root)),
                "task_key": task_key,
                "output_name": output.name,
                "metadata": output.metadata,
            })
            if output.name == "imgt_mutations":
                mutations.extend(orjson.loads(source.read_bytes()))
            if (
                not task_key.startswith(("sapiens-", "humatch-", "pabnativ2-"))
                or output.storage.media_type != "application/zstd"
            ):
                continue
            method, candidate_id = task_key.split("-", maxsplit=1)
            for filename, content in archive_members(source.read_bytes()).items():
                if not filename.endswith(".parquet"):
                    continue
                frame = pl.read_parquet(BytesIO(content)).with_columns(
                    pl.lit(parent_by_candidate[candidate_id]).alias("parent_id"),
                    pl.lit(candidate_id).alias("candidate_id"),
                )
                detail_name = f"{method}_{Path(filename).stem}"
                shard = native / f"{detail_name}-{candidate_id}.parquet"
                frame.write_parquet(shard, compression="zstd")
                detail_shards.setdefault(detail_name, []).append(shard)
    for name, shards in detail_shards.items():
        pl.scan_parquet(shards).sink_parquet(
            root / f"{name}.parquet", compression="zstd"
        )
    mutation_schema = {
        "parent_id": pl.String,
        "candidate_id": pl.String,
        "chain": pl.String,
        "numbering_scheme": pl.String,
        "cdr_definition": pl.String,
        "position": pl.Int64,
        "insertion_code": pl.String,
        "parent_residue": pl.String,
        "candidate_residue": pl.String,
        "region": pl.String,
        "change_type": pl.String,
    }
    pl.DataFrame(mutations, schema=mutation_schema).write_parquet(
        root / "imgt_mutations.parquet", compression="zstd"
    )
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
        "schema_version": 1,
        "execution_run_id": str(context.execution_run_id),
        "status": "partial" if errors else "succeeded",
        "parameters": settings.model_dump(),
        "scientific_versions": dict(scientific_versions),
        "candidate_count": len(candidates),
        "errors": dict(errors),
        "native_publications": artifact_sources,
        "selection_semantics": "one row per parent/exact pair; no composite ranking; missing scores are null",
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
