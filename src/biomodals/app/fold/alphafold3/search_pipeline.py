"""Request-level AlphaFold 3 MSA and protein-template coordination.

The coordinator owns phase ordering, cache reconciliation, validation, and
failure policy. Its executor interface hides Modal calls so the same production
behavior can be exercised synchronously in local tests.
"""

from __future__ import annotations

from typing import Protocol

from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.inference_inputs import (
    validate_submitted_af3_input,
    validate_upstream_af3_input,
)
from biomodals.app.fold.alphafold3.input_enrichment import (
    apply_msa_resolution,
    apply_template_results,
    chain_msa_states,
    fill_missing_msa_for_inference,
    missing_raw_searches,
    plan_template_searches,
    reduce_msa_assembly_results,
    reduce_template_cache_results,
    validate_template_result,
)
from biomodals.app.fold.alphafold3.msa_search import (
    SEARCH_MAX_PARALLEL_SHARDS,
    SEARCH_N_CPU,
    MsaAssemblyTask,
    RawSearchTask,
    plan_msa_resolution,
    sequence_hash,
)
from biomodals.app.fold.alphafold3.template_search import TemplateTask

type SearchOutcome = dict[str, object] | Exception


class SearchExecutor(Protocol):
    """Remote-call interface required by the request-level coordinator."""

    def inspect_raw(
        self,
        tasks: tuple[RawSearchTask, ...],
    ) -> tuple[dict[str, object], ...]:
        """Return cache status for every raw database search."""
        ...

    def run_raw(
        self,
        tasks: tuple[RawSearchTask, ...],
        *,
        max_parallel: int,
    ) -> tuple[SearchOutcome, ...]:
        """Run missing raw searches with bounded request-level fanout."""
        ...

    def run_assemblies(
        self,
        tasks: tuple[MsaAssemblyTask, ...],
        *,
        max_parallel: int,
    ) -> tuple[SearchOutcome, ...]:
        """Assemble requested MSA fields from durable raw results."""
        ...

    def inspect_templates(
        self,
        tasks: tuple[TemplateTask, ...],
    ) -> tuple[dict[str, object], ...]:
        """Return canonical protein-template cache status."""
        ...

    def run_templates(
        self,
        tasks: tuple[TemplateTask, ...],
        *,
        max_parallel: int,
    ) -> tuple[SearchOutcome, ...]:
        """Run missing canonical and request-local template searches."""
        ...


def validate_search_worker_budget(max_parallel_search_workers: int) -> int:
    """Validate the request-wide remote-worker budget."""
    if (
        isinstance(max_parallel_search_workers, bool)
        or not isinstance(max_parallel_search_workers, int)
        or not 1 <= max_parallel_search_workers <= 32
    ):
        raise ValueError("max_parallel_search_workers must be between 1 and 32")
    return max_parallel_search_workers


def resolve_msa_and_templates(
    config: AF3Config,
    executor: SearchExecutor,
    *,
    search_msa: bool = True,
    search_protein_templates: bool = True,
    max_parallel_search_workers: int = 4,
) -> AF3Config:
    """Populate missing MSA/template fields through one validated deep seam."""
    worker_budget = validate_search_worker_budget(max_parallel_search_workers)
    conf = validate_submitted_af3_input(config)
    if not search_msa:
        return validate_upstream_af3_input(fill_missing_msa_for_inference(conf))

    states = chain_msa_states(conf)
    plan = plan_msa_resolution(states)
    cache_statuses = (
        executor.inspect_raw(plan.raw_searches) if plan.raw_searches else ()
    )
    missing_raw = missing_raw_searches(plan.raw_searches, cache_statuses)

    print(
        "🧬 Sharded MSA search plan: "
        f"{len(cache_statuses) - len(missing_raw)} cached, "
        f"{len(missing_raw)} missing, worker cap {worker_budget}; each database "
        f"worker runs at most {SEARCH_MAX_PARALLEL_SHARDS} shard searches "
        f"with {SEARCH_N_CPU} HMMER CPUs each (request-wide theoretical cap "
        f"{worker_budget * SEARCH_MAX_PARALLEL_SHARDS} shard searches / "
        f"{worker_budget * SEARCH_MAX_PARALLEL_SHARDS * SEARCH_N_CPU} HMMER "
        "CPU slots)."
    )
    search_outcomes = (
        executor.run_raw(missing_raw, max_parallel=worker_budget) if missing_raw else ()
    )
    search_failures = [
        {
            "database_id": task.database_id,
            "polymer": task.polymer,
            "sequence_sha256": task.sequence_hash,
            "error_type": type(outcome).__name__,
            "message": str(outcome),
        }
        for task, outcome in zip(missing_raw, search_outcomes, strict=True)
        if isinstance(outcome, Exception)
    ]
    if search_failures:
        raise RuntimeError(
            "Incomplete Raw Database MSA tasks; rerun to reuse successful "
            f"siblings: {search_failures}"
        )

    assembly_outcomes = (
        executor.run_assemblies(plan.assemblies, max_parallel=worker_budget)
        if plan.assemblies
        else ()
    )
    assembly_failures = [
        {
            "polymer": task.polymer,
            "sequence_sha256": sequence_hash(task.sequence),
            "error_type": type(outcome).__name__,
            "message": str(outcome),
        }
        for task, outcome in zip(
            plan.assemblies,
            assembly_outcomes,
            strict=True,
        )
        if isinstance(outcome, Exception)
    ]
    if assembly_failures:
        raise RuntimeError(
            "Incomplete MSA assembly tasks; raw database results remain "
            f"reusable: {assembly_failures}"
        )

    assembly_resolution = reduce_msa_assembly_results(
        plan.assemblies,
        tuple(outcome for outcome in assembly_outcomes if isinstance(outcome, dict)),
    )
    apply_msa_resolution(
        conf,
        states,
        assembly_resolution,
        search_protein_templates=search_protein_templates,
    )

    if not search_protein_templates:
        return validate_upstream_af3_input(conf)

    template_plan = plan_template_searches(
        conf,
        states,
        assembly_resolution.canonical_sequences,
    )
    canonical_tasks = template_plan.canonical_tasks
    template_statuses = (
        executor.inspect_templates(canonical_tasks) if canonical_tasks else ()
    )
    cache_resolution = reduce_template_cache_results(
        canonical_tasks,
        template_statuses,
    )
    templates_by_identity = cache_resolution.templates_by_identity
    missing_canonical = cache_resolution.missing_tasks
    request_local_tasks = template_plan.request_local_tasks
    worker_tasks = missing_canonical + request_local_tasks
    print(
        "🧬 Protein template search plan: "
        f"{len(canonical_tasks) - len(missing_canonical)} cached, "
        f"{len(missing_canonical)} missing canonical, "
        f"{len(request_local_tasks)} request-local, "
        f"worker cap {worker_budget}."
    )
    template_outcomes = (
        executor.run_templates(worker_tasks, max_parallel=worker_budget)
        if worker_tasks
        else ()
    )
    template_failures: list[dict[str, object]] = []
    for task, outcome in zip(worker_tasks, template_outcomes, strict=True):
        if isinstance(outcome, Exception):
            template_failures.append({
                "sequence_sha256": sequence_hash(task.sequence),
                "unpaired_msa_sha256": task.unpaired_msa_sha256,
                "publish_canonical": task.publish_canonical,
                "error_type": type(outcome).__name__,
                "message": str(outcome),
            })
            continue
        try:
            if not isinstance(outcome, dict):
                raise RuntimeError(f"Invalid protein template result: {outcome!r}")
            templates_by_identity[task.template_identity] = validate_template_result(
                task,
                outcome,
                allowed_statuses=(
                    frozenset({"published", "reused"})
                    if task.publish_canonical
                    else frozenset({"request-local"})
                ),
            )
        except Exception as error:
            template_failures.append({
                "sequence_sha256": sequence_hash(task.sequence),
                "unpaired_msa_sha256": task.unpaired_msa_sha256,
                "publish_canonical": task.publish_canonical,
                "error_type": type(error).__name__,
                "message": str(error),
            })
    if template_failures:
        raise RuntimeError(
            "Incomplete protein template tasks; completed canonical results "
            f"remain reusable: {template_failures}"
        )

    apply_template_results(conf, template_plan, templates_by_identity)
    return validate_upstream_af3_input(conf)
