"""Public-domain inputs, native local annotation and deterministic fake scores."""

from pathlib import Path
from uuid import UUID

import polars as pl

from biomodals.execution.nodes import NodeRunContext
from biomodals.workflow.nanobody_humanization.annotation import annotation_tables
from biomodals.workflow.nanobody_humanization.execution import NanobodyExecutionRequest
from biomodals.workflow.nanobody_humanization.export import export_results
from biomodals.workflow.nanobody_humanization.ranking import rank_panel
from biomodals.workflow.nanobody_humanization.tables import (
    GENERATION_SCHEMA,
    KEY,
    add_scores,
    candidate_union,
    mutation_table,
)

VHH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"


def result_directory(
    root: Path, job_id: UUID, request: NanobodyExecutionRequest
) -> Path:
    """Exercise real union/ranking/publication logic without invoking any model."""
    generation = []
    for parent in request.parents:
        index = next(
            i for i in range(len(parent.sequence)) if i not in parent.protected_indices
        )
        replacements = [aa for aa in "AEG" if aa != parent.sequence[index]][:2]
        for method, aa in zip(("abnativ2_vhh", "hudiff_nb"), replacements, strict=True):
            generation.append({
                "parent_id": parent.id,
                "method": method,
                "attempt_index": 1,
                "seed": request.settings.root_seed if method == "hudiff_nb" else None,
                "vh": parent.sequence[:index] + aa + parent.sequence[index + 1 :],
                "error": None,
            })
    candidates, origins = candidate_union(
        request.parents, pl.DataFrame(generation, schema=GENERATION_SCHEMA)
    )
    mutations = mutation_table(candidates, request.parents)
    scores = {
        model: candidates.select(
            "candidate_id",
            pl
            .when(pl.col("is_parent"))
            .then(0.8)
            .otherwise(0.85 if model == "VH2" else 0.9)
            .alias("score"),
            pl.lit(None, dtype=pl.String).alias("error"),
        )
        for model in ("VH2", "VHH2")
    }
    annotations, germlines = annotation_tables(candidates)
    selection = rank_panel(add_scores(candidates, mutations, scores), mutations).join(
        annotations, on=KEY, how="left", validate="1:1", maintain_order="left"
    )
    context = NodeRunContext(
        execution_run_id=job_id,
        workload_run_key=request.run_name,
        node_id="evaluate",
        task_key="evaluate",
        work_dir=root / "workflow-runs" / str(job_id) / "nodes/evaluate/result",
        cache_dir=root / "cache",
        inputs={},
        volume_root=root,
        artifact_volume_name="fixture",
    )
    output = export_results(
        context,
        selection,
        origins,
        mutations,
        germlines,
        {},
        request.parents,
        request.settings,
        request.scientific_versions,
        incomplete=False,
    )
    return root / output.storage.path
