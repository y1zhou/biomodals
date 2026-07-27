"""Modal adapters for AlphaFold 3 production orchestration.

The production app injects decorated Modal functions here. This module owns
one-time profile fan-out, payload marshaling, bounded blocking calls,
spawn/poll behavior, and staging; scientific coordinators remain independent
of Modal.
"""

# Protocol implementations inherit their method contracts.
# ruff: noqa: D102

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO

import modal

from biomodals.app.fold.alphafold3.inference_inputs import (
    PreparedInferenceRun,
    serialize_af3_input,
)
from biomodals.app.fold.alphafold3.inference_pipeline import (
    InferenceBatchOutcome,
    InferenceExecutor,
)
from biomodals.app.fold.alphafold3.msa_search import (
    MsaAssemblyTask,
    RawSearchTask,
)
from biomodals.app.fold.alphafold3.profile_builder import (
    plan_missing_profile_builds,
)
from biomodals.app.fold.alphafold3.profiles import profile_build_slot_budget
from biomodals.app.fold.alphafold3.search_pipeline import SearchExecutor
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    SeedClaimPlan,
    partition_claimed_seeds,
    seed_claim_plan_from_dict,
)
from biomodals.app.fold.alphafold3.template_search import TemplateTask
from biomodals.helper.task_budget import bounded_map


def execute_profile_setup(
    inspect_function: modal.Function,
    build_function: modal.Function,
    finalize_function: modal.Function,
    *,
    seqkit_threads: int,
    source_policy: str,
) -> dict[str, object]:
    """Build all missing profiles concurrently, then finalize their registry."""
    print("🧬 Inspecting fixed sharded database profiles...")
    inventory = inspect_function.remote()
    inputs = plan_missing_profile_builds(
        inventory,
        seqkit_threads,
        source_policy,
    )
    missing = [database_id for database_id, _, _ in inputs]
    budget = profile_build_slot_budget(len(inputs), seqkit_threads)
    print(
        "🧬 Effective profile-build fanout: "
        f"{budget['builder_containers']} containers, "
        f"{budget['maximum_effective_worker_slots']} configured worker slots."
    )

    results: list[dict[str, object] | BaseException] = []
    if missing:
        print(
            "🧬 Submitting missing database profiles concurrently: "
            f"{', '.join(missing)}"
        )
        results = list(
            build_function.starmap(
                inputs,
                return_exceptions=True,
            )
        )
        failures = [
            {
                "database_id": database_id,
                "error_type": type(result).__name__,
                "message": str(result),
            }
            for database_id, result in zip(missing, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(
                "One or more sharded database builders failed after all "
                f"submitted builders completed: {failures}"
            )
    else:
        print("🧬 All fixed profiles are already valid; no builders submitted.")

    print("🧬 Running final inventory and workspace cleanup...")
    return {
        "status": "complete",
        "initial_inventory": inventory,
        "builder_results": results,
        "final_inventory": finalize_function.remote(),
    }


def _bounded_remote_outcomes[TaskT](
    tasks: tuple[TaskT, ...],
    invoke: Callable[[TaskT], dict[str, object]],
    *,
    max_parallel: int,
) -> tuple[dict[str, object] | Exception, ...]:
    """Run blocking remote calls while preserving task order and failures."""

    def capture(task: TaskT) -> dict[str, object] | Exception:
        try:
            return invoke(task)
        except Exception as exc:
            return exc

    return tuple(bounded_map(tasks, capture, max_parallel=max_parallel))


@dataclass(frozen=True, slots=True)
class ModalSearchExecutor(SearchExecutor):
    """Bind the search coordinator to decorated Modal functions."""

    inspect_raw_function: modal.Function
    raw_search_function: modal.Function
    msa_assembly_function: modal.Function
    inspect_templates_function: modal.Function
    template_search_function: modal.Function

    def inspect_raw(
        self,
        tasks: tuple[RawSearchTask, ...],
    ) -> tuple[dict[str, object], ...]:
        inputs = [(task.database_id, task.sequence) for task in tasks]
        return tuple(self.inspect_raw_function.remote(inputs))

    def run_raw(
        self,
        tasks: tuple[RawSearchTask, ...],
        *,
        max_parallel: int,
    ) -> tuple[dict[str, object] | Exception, ...]:
        return _bounded_remote_outcomes(
            tasks,
            lambda task: self.raw_search_function.remote(
                task.database_id,
                task.sequence,
            ),
            max_parallel=max_parallel,
        )

    def run_assemblies(
        self,
        tasks: tuple[MsaAssemblyTask, ...],
        *,
        max_parallel: int,
    ) -> tuple[dict[str, object] | Exception, ...]:
        return _bounded_remote_outcomes(
            tasks,
            lambda task: self.msa_assembly_function.remote(
                task.polymer,
                task.sequence,
                task.include_unpaired,
                task.include_paired,
            ),
            max_parallel=max_parallel,
        )

    def inspect_templates(
        self,
        tasks: tuple[TemplateTask, ...],
    ) -> tuple[dict[str, object], ...]:
        inputs = [
            (
                task.sequence,
                task.unpaired_msa_sha256,
                task.max_template_date,
            )
            for task in tasks
        ]
        return tuple(self.inspect_templates_function.remote(inputs))

    def run_templates(
        self,
        tasks: tuple[TemplateTask, ...],
        *,
        max_parallel: int,
    ) -> tuple[dict[str, object] | Exception, ...]:
        return _bounded_remote_outcomes(
            tasks,
            lambda task: self.template_search_function.remote(
                task.sequence,
                task.unpaired_msa,
                task.publish_canonical,
                task.max_template_date,
            ),
            max_parallel=max_parallel,
        )


def stage_inference_run(
    output_volume: modal.Volume,
    prepared: PreparedInferenceRun,
) -> None:
    """Upload normalized inputs and custom templates to the output Volume."""
    with output_volume.batch_upload(force=True) as batch:
        for upload in prepared.uploads:
            batch.put_file(
                BytesIO(upload.content),
                f"/{upload.relative_path.as_posix()}",
            )


def _worker_failure(
    batch: tuple[ClaimedSeed, ...],
    error_type: str,
    message: str,
) -> dict[str, object]:
    return {
        "seeds": [item.seed for item in batch],
        "error_type": error_type,
        "message": message,
    }


def _worker_seeds(
    value: object,
    allowed: set[int],
) -> set[int] | None:
    if not isinstance(value, list):
        return None
    seeds: set[int] = set()
    for seed in value:
        if isinstance(seed, bool) or not isinstance(seed, int):
            return None
        seeds.add(seed)
    return seeds if seeds.issubset(allowed) else None


def _run_claimed_seed_batches(
    worker_function: modal.Function,
    prepared: PreparedInferenceRun,
    claimed_seeds: tuple[ClaimedSeed, ...],
    *,
    recycle: int,
    sample_count: int,
    max_workers: int,
    poll_timeout_seconds: int,
) -> InferenceBatchOutcome:
    batches = partition_claimed_seeds(claimed_seeds, max_workers)
    json_bytes = serialize_af3_input(prepared.worker_config)
    calls = [
        (
            worker_function.spawn(
                json_bytes,
                prepared.run_id,
                recycle,
                sample_count,
                [item.to_dict() for item in batch],
            ),
            batch,
        )
        for batch in batches
    ]

    published: set[int] = set()
    reused: set[int] = set()
    failures: list[dict[str, object]] = []
    while calls:
        pending: list[tuple[modal.FunctionCall, tuple[ClaimedSeed, ...]]] = []
        for function_call, batch in calls:
            try:
                result = function_call.get(timeout=poll_timeout_seconds)
            except TimeoutError:
                pending.append((function_call, batch))
                continue
            except Exception as exc:
                failures.append(_worker_failure(batch, type(exc).__name__, str(exc)))
                continue

            if not isinstance(result, dict) or result.get("run_id") != prepared.run_id:
                failures.append(
                    _worker_failure(batch, "InvalidWorkerResult", repr(result))
                )
                continue
            allowed = {item.seed for item in batch}
            batch_published = _worker_seeds(result.get("published_seeds"), allowed)
            batch_reused = _worker_seeds(result.get("reused_seeds"), allowed)
            if batch_published is None or batch_reused is None:
                failures.append(
                    _worker_failure(batch, "InvalidWorkerResult", repr(result))
                )
                continue
            published.update(batch_published)
            reused.update(batch_reused)
        calls = pending

    return InferenceBatchOutcome(
        published_seeds=frozenset(published),
        reused_seeds=frozenset(reused),
        failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class ModalInferenceExecutor(InferenceExecutor):
    """Bind the inference coordinator to decorated Modal functions."""

    claim_function: modal.Function
    inspect_function: modal.Function
    worker_function: modal.Function
    summary_function: modal.Function
    request_function: modal.Function

    def claim_seeds(
        self,
        run_id: str,
        seeds: tuple[int, ...],
        *,
        sample_count: int,
    ) -> SeedClaimPlan:
        raw_plan = self.claim_function.remote(run_id, list(seeds), sample_count)
        return seed_claim_plan_from_dict(raw_plan)

    def inspect_seeds(
        self,
        run_id: str,
        seeds: tuple[int, ...],
        *,
        sample_count: int,
    ) -> tuple[dict[str, object], ...]:
        return tuple(self.inspect_function.remote(run_id, list(seeds), sample_count))

    def run_claimed(
        self,
        prepared: PreparedInferenceRun,
        claimed_seeds: tuple[ClaimedSeed, ...],
        *,
        recycle: int,
        sample_count: int,
        max_workers: int,
        poll_timeout_seconds: int,
    ) -> InferenceBatchOutcome:
        return _run_claimed_seed_batches(
            self.worker_function,
            prepared,
            claimed_seeds,
            recycle=recycle,
            sample_count=sample_count,
            max_workers=max_workers,
            poll_timeout_seconds=poll_timeout_seconds,
        )

    def finalize_summary(
        self,
        prepared: PreparedInferenceRun,
        *,
        sample_count: int,
    ) -> dict[str, object]:
        return self.summary_function.remote(
            serialize_af3_input(prepared.worker_config),
            prepared.run_id,
            sample_count,
        )

    def finalize_request(
        self,
        prepared: PreparedInferenceRun,
        *,
        sample_count: int,
        reused_seeds: tuple[int, ...],
        published_seeds: tuple[int, ...],
    ) -> dict[str, object]:
        return self.request_function.remote(
            prepared.run_id,
            prepared.request_id,
            list(prepared.submitted_seeds),
            list(prepared.normalized_seeds),
            sample_count,
            prepared.display_name,
            list(reused_seeds),
            list(published_seeds),
        )
