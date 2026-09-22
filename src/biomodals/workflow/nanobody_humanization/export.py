"""Compact content-bound single-domain results; native trees stay operational."""

from collections.abc import Mapping, Sequence

import orjson
import polars as pl

from biomodals.execution.nodes import NodeRunContext
from biomodals.helper.artifacts import file_size_sha256
from biomodals.schema import AppOutput, ArtifactKind, VolumePath
from biomodals.workflow.nanobody_humanization.preparation import (
    PreparedVH,
    preparation_digest,
)
from biomodals.workflow.nanobody_humanization.ranking import RANKING_POLICY
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings

SELECTION_COLUMNS = (
    "parent_id",
    "candidate_id",
    "vh",
    "vh_pI",
    "vh_v_gene",
    "vh_j_gene",
    "is_parent",
    "panel_order",
    "quality_tier",
    "generating_methods",
    "vh_mutations",
    "abnativ2_vh_nativeness",
    "abnativ2_vhh_nativeness",
    "abnativ2_vh_nativeness_delta",
    "abnativ2_vhh_nativeness_delta",
    "evaluation_complete",
    "abnativ2_vh_nativeness_error",
    "abnativ2_vhh_nativeness_error",
    "annotation_error",
)


def export_results(
    context: NodeRunContext,
    selection: pl.DataFrame,
    generation: pl.DataFrame,
    mutations: pl.DataFrame,
    germlines: pl.DataFrame,
    details: Mapping[str, pl.DataFrame],
    parents: Sequence[PreparedVH],
    settings: NanobodySettings,
    scientific_versions: Mapping[str, str],
    *,
    incomplete: bool,
) -> AppOutput:
    """Write one ordered CSV and consolidated evidence with a sorted inventory."""
    if context.volume_root is None or context.artifact_volume_name is None:
        raise RuntimeError("Result publication requires the execution volume")
    root = context.work_dir / "nanobody_humanization"
    relative = root.resolve().relative_to(context.volume_root.resolve())
    root.mkdir(parents=True, exist_ok=True)
    selection.select(SELECTION_COLUMNS).write_csv(root / "selection.csv")
    # Exact sequence identity lives in selection; rejected samples have no union row.
    generation.with_columns(
        pl.when(pl.col("candidate_id").is_null()).then(pl.col("vh")).alias("vh")
    ).sort("parent_id", "method", "attempt_index").write_parquet(
        root / "generation.parquet", compression="zstd"
    )
    mutations.write_parquet(root / "imgt_mutations.parquet", compression="zstd")
    germlines.write_parquet(root / "germlines.parquet", compression="zstd")
    if details:
        (root / "scores").mkdir(exist_ok=True)
        for name, frame in sorted(details.items()):
            frame.write_parquet(root / "scores" / f"{name}.parquet", compression="zstd")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            size, digest = file_size_sha256(path)
            files.append({
                "path": str(path.relative_to(root)),
                "size_bytes": size,
                "content_sha256": digest,
            })
    manifest = {
        "schema_version": 1,
        "execution_run_id": str(context.execution_run_id),
        "status": "partial" if incomplete else "succeeded",
        "parameters": settings.model_dump(),
        "scientific_versions": dict(scientific_versions),
        "preparation_digest": preparation_digest(parents),
        "candidate_count": selection.height,
        "ranking_policy": RANKING_POLICY,
        "files": files,
    }
    path = root / "manifest.json"
    path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS))
    size, digest = file_size_sha256(path)
    files.append({"path": path.name, "size_bytes": size, "content_sha256": digest})
    return AppOutput(
        name="nanobody_results",
        kind=ArtifactKind.DIRECTORY,
        storage=VolumePath(
            volume_name=context.artifact_volume_name, path=str(relative)
        ),
        metadata={
            "files": sorted(files, key=lambda row: row["path"]),
            "candidate_count": selection.height,
            "status": manifest["status"],
        },
    )
