"""Provider-neutral mechanics for one SQLite-backed pull worker."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from uuid import UUID

from biomodals.execution.model import (
    PullTaskClaim,
    WorkerAssignmentRecord,
)


@dataclass(frozen=True)
class PullWorkerSummary:
    """Small durable-envelope value returned by a completed worker call."""

    claimed_tasks: int
    claim_requests: int


def drive_pull_worker[Result](
    *,
    provider_call_id: UUID,
    claim_capacity: int,
    claim: Callable[[str, int], PullTaskClaim],
    execute: Callable[[WorkerAssignmentRecord], Result],
    complete: Callable[[WorkerAssignmentRecord, str, Result], None],
    checkpoint_batch: Callable[[], None] | None = None,
    max_parallel: int | None = None,
) -> PullWorkerSummary:
    """Claim, execute, and report deterministic Task microbatches until empty.

    ``checkpoint_batch`` runs after every nonempty batch finishes execution and
    before any of that batch's completions are reported.
    """
    if claim_capacity < 1:
        raise ValueError("claim_capacity must be positive")
    parallelism = claim_capacity if max_parallel is None else max_parallel
    if parallelism < 1:
        raise ValueError("max_parallel must be positive")

    claimed_tasks = 0
    claim_ordinal = 0
    while True:
        claim_request_id = f"{provider_call_id}:claim:{claim_ordinal}"
        claimed = claim(claim_request_id, claim_capacity)
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
        for assignment, result in zip(assignments, results, strict=True):
            completion_request_id = (
                f"{provider_call_id}:complete:{assignment.task_fingerprint}"
            )
            complete(assignment, completion_request_id, result)
        claimed_tasks += len(assignments)
        claim_ordinal += 1
