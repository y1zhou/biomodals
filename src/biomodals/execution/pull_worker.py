"""Provider-neutral mechanics for one SQLite-backed pull worker."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeAlias, TypeVar
from uuid import UUID

from biomodals.execution.model import (
    PullTaskClaim,
    WorkerAssignmentRecord,
)

_Result = TypeVar("_Result")
PullWorkerCompletion: TypeAlias = (  # noqa: UP040 - Python 3.11 task images
    tuple[WorkerAssignmentRecord, str, _Result]
)


@dataclass(frozen=True)
class PullWorkerSummary:
    """Small durable-envelope value returned by a completed worker call."""

    claimed_tasks: int
    claim_requests: int


def size_pull_worker_pool(
    task_count: int,
    *,
    max_worker_calls: int,
    max_parallel_per_worker: int,
) -> tuple[int, int]:
    """Return the admitted worker count and bounded per-worker claim size."""
    if task_count < 1:
        raise ValueError("task_count must be positive")
    if max_worker_calls < 1:
        raise ValueError("max_worker_calls must be positive")
    if max_parallel_per_worker < 1:
        raise ValueError("max_parallel_per_worker must be positive")
    worker_count = min(
        max_worker_calls,
        (task_count + max_parallel_per_worker - 1) // max_parallel_per_worker,
    )
    claim_capacity = min(
        max_parallel_per_worker,
        (task_count + worker_count - 1) // worker_count,
    )
    return worker_count, claim_capacity


def drive_pull_worker(  # noqa: UP047 - Python 3.11 task images
    *,
    provider_call_id: UUID,
    claim_capacity: int,
    claim: Callable[[str, int], PullTaskClaim],
    execute: Callable[[WorkerAssignmentRecord], _Result],
    complete_and_claim: Callable[
        [tuple[PullWorkerCompletion[_Result], ...], str, int],
        PullTaskClaim,
    ],
    checkpoint_batch: Callable[[], None] | None = None,
    max_parallel: int | None = None,
) -> PullWorkerSummary:
    """Claim, execute, and report deterministic Task microbatches until empty.

    ``checkpoint_batch`` runs after every nonempty batch finishes execution and
    before any of that batch's completions are reported. ``complete_and_claim``
    durably applies those completions together with the next claim.
    """
    if claim_capacity < 1:
        raise ValueError("claim_capacity must be positive")
    parallelism = claim_capacity if max_parallel is None else max_parallel
    if parallelism < 1:
        raise ValueError("max_parallel must be positive")

    claimed_tasks = 0
    claim_ordinal = 0
    claimed = claim(f"{provider_call_id}:claim:{claim_ordinal}", claim_capacity)
    while True:
        assignments = claimed.assignments
        if not assignments:
            return PullWorkerSummary(
                claimed_tasks=claimed_tasks,
                claim_requests=claim_ordinal + 1,
            )

        with ThreadPoolExecutor(
            max_workers=min(parallelism, len(assignments))
        ) as executor:
            results = tuple(executor.map(execute, assignments))
        if checkpoint_batch is not None:
            checkpoint_batch()
        claimed_tasks += len(assignments)
        claim_ordinal += 1
        claimed = complete_and_claim(
            tuple(
                (
                    assignment,
                    f"{provider_call_id}:complete:{assignment.task_fingerprint}",
                    result,
                )
                for assignment, result in zip(assignments, results, strict=True)
            ),
            f"{provider_call_id}:claim:{claim_ordinal}",
            claim_capacity,
        )
