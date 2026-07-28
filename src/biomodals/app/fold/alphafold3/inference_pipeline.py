"""Request-level AlphaFold 3 seed reconciliation and publication.

This coordinator owns the durable seed state machine while its executor
interface hides Modal spawning, polling, and serialization. Completion markers
remain the sole evidence used to reuse or publish predictions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from biomodals.app.fold.alphafold3.inference_inputs import PreparedInferenceRun
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    SeedClaimPlan,
)


@dataclass(frozen=True, slots=True)
class InferenceBatchOutcome:
    """Observed result of all GPU workers launched for one owned seed set."""

    published_seeds: frozenset[int]
    reused_seeds: frozenset[int]
    failures: tuple[dict[str, object], ...]


class InferenceExecutor(Protocol):
    """Modal-call interface required by the request-level coordinator."""

    def claim_seeds(
        self,
        run_id: str,
        seeds: tuple[int, ...],
        *,
        sample_count: int,
    ) -> SeedClaimPlan:
        """Reuse or claim every currently pending seed."""
        ...

    def inspect_seeds(
        self,
        run_id: str,
        seeds: tuple[int, ...],
        *,
        sample_count: int,
    ) -> tuple[dict[str, object], ...]:
        """Inspect completion markers for an ordered seed set."""
        ...

    def run_claimed(
        self,
        prepared: PreparedInferenceRun,
        claimed_seeds: tuple[ClaimedSeed, ...],
        *,
        max_workers: int,
        poll_timeout_seconds: int,
    ) -> InferenceBatchOutcome:
        """Run all seed generations owned by this request."""
        ...

    def finalize_summary(
        self,
        prepared: PreparedInferenceRun,
    ) -> dict[str, object]:
        """Rebuild the accumulated non-regressing run summary."""
        ...

    def finalize_request(
        self,
        prepared: PreparedInferenceRun,
    ) -> dict[str, object]:
        """Publish the immutable request view over completed seeds."""
        ...


def _completed_seed_set(
    executor: InferenceExecutor,
    run_id: str,
    seeds: tuple[int, ...],
    sample_count: int,
) -> set[int]:
    statuses = executor.inspect_seeds(
        run_id,
        seeds,
        sample_count=sample_count,
    )
    if len(statuses) != len(seeds):
        raise RuntimeError("Seed marker inspection returned the wrong result count")
    completed: set[int] = set()
    for seed, status in zip(seeds, statuses, strict=True):
        if status.get("run_id") != run_id or status.get("seed") != seed:
            raise RuntimeError(f"Invalid seed marker inspection result: {status!r}")
        if status.get("status") == "reused":
            completed.add(seed)
        elif status.get("status") != "missing":
            raise RuntimeError(f"Invalid seed marker inspection result: {status!r}")
    return completed


def coordinate_seed_predictions(
    prepared: PreparedInferenceRun,
    executor: InferenceExecutor,
    *,
    num_containers: int,
    active_wait_timeout_seconds: int | float,
    worker_poll_timeout_seconds: int = 30,
    active_poll_seconds: int | float = 30,
) -> dict[str, object]:
    """Reconcile, run once, summarize, and publish one inference request."""
    if (
        isinstance(num_containers, bool)
        or not isinstance(num_containers, int)
        or num_containers < 1
    ):
        raise ValueError("num_containers must be a positive integer")
    if (
        isinstance(active_wait_timeout_seconds, bool)
        or not isinstance(active_wait_timeout_seconds, int | float)
        or active_wait_timeout_seconds <= 0
    ):
        raise ValueError("active_wait_timeout_seconds must be positive")
    if (
        isinstance(worker_poll_timeout_seconds, bool)
        or not isinstance(worker_poll_timeout_seconds, int)
        or worker_poll_timeout_seconds < 1
    ):
        raise ValueError("worker_poll_timeout_seconds must be a positive integer")
    if (
        isinstance(active_poll_seconds, bool)
        or not isinstance(active_poll_seconds, int | float)
        or active_poll_seconds <= 0
    ):
        raise ValueError("active_poll_seconds must be positive")

    requested = prepared.normalized_seeds
    sample_count = prepared.sample_count
    pending = set(requested)
    reused: set[int] = set()
    published: set[int] = set()
    failures: list[dict[str, object]] = []
    attempted: set[int] = set()
    deadline = time.monotonic() + active_wait_timeout_seconds

    while pending:
        plan = executor.claim_seeds(
            prepared.run_id,
            tuple(sorted(pending)),
            sample_count=sample_count,
        )
        reused.update(plan.reused_seeds)
        pending.difference_update(plan.reused_seeds)
        if plan.owned:
            owned_seeds = {item.seed for item in plan.owned}
            if attempted.intersection(owned_seeds):
                raise RuntimeError("Refusing to retry a surfaced seed failure")
            attempted.update(owned_seeds)
            batch = executor.run_claimed(
                prepared,
                plan.owned,
                max_workers=num_containers,
                poll_timeout_seconds=worker_poll_timeout_seconds,
            )
            completed_owned = _completed_seed_set(
                executor,
                prepared.run_id,
                tuple(sorted(owned_seeds)),
                sample_count,
            )
            published.update(batch.published_seeds)
            published.update(completed_owned - batch.reused_seeds)
            reused.update(batch.reused_seeds)
            pending.difference_update(owned_seeds)
            failures.extend(batch.failures)
            for seed in sorted(owned_seeds - completed_owned):
                if not any(
                    isinstance(
                        failure_seeds := failure.get("seeds"),
                        list,
                    )
                    and seed in failure_seeds
                    for failure in batch.failures
                ):
                    failures.append({
                        "seeds": [seed],
                        "error_type": "IncompleteSeedPrediction",
                        "message": "Worker returned without a valid seed marker",
                    })

        active_seeds = {item.seed for item in plan.active}
        if pending and pending != active_seeds:
            raise RuntimeError("Seed claim plan did not account for every pending seed")
        if pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append({
                    "seeds": sorted(pending),
                    "error_type": "ActiveSeedTimeout",
                    "message": "Timed out waiting for concurrent seed owners",
                })
                break
            time.sleep(min(active_poll_seconds, remaining))

    completed = _completed_seed_set(
        executor,
        prepared.run_id,
        requested,
        sample_count,
    )
    reused.update(completed.difference(reused, published))
    incomplete = set(requested) - completed
    summary: dict[str, object] | None = None
    if completed:
        summary = executor.finalize_summary(prepared)
    result: dict[str, object] = {
        "run_id": prepared.run_id,
        "request_id": prepared.request_id,
        "requested_seeds": list(requested),
        "reused_seeds": sorted(reused),
        "published_seeds": sorted(published),
        "completed_seeds": sorted(completed),
        "incomplete_seeds": sorted(incomplete),
        "failures": failures,
        "summary": summary,
    }
    if incomplete:
        raise RuntimeError(
            "Incomplete AlphaFold3 seed predictions; completed siblings remain "
            f"reusable and no failed seed was retried: {result}"
        )
    result["request"] = executor.finalize_request(prepared)
    return result
