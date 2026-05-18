"""Helpers for workflow node worker-pool bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from biomodals.helper.shell import sanitize_filename


@dataclass(frozen=True)
class WorkerTask:
    """One unit of work submitted to a node-local worker pool."""

    task_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerTaskResult:
    """Completion record returned by one worker task."""

    task_id: str
    succeeded: bool
    error: str | None = None


@dataclass(frozen=True)
class WorkerPoolSummary:
    """Aggregated completion state for one worker pool."""

    task_count: int
    succeeded_count: int
    failed_count: int
    failed_task_ids: list[str]

    @property
    def succeeded(self) -> bool:
        """Return whether every submitted task succeeded."""
        return self.failed_count == 0


class QueueLike(Protocol):
    """Minimal queue protocol used by pure worker-pool helpers."""

    def put(self, item: WorkerTask) -> object:
        """Put one worker task on a queue."""


def build_worker_pool_name(workflow_name: str, run_id: str, node_id: str) -> str:
    """Build a deterministic Modal object name for one node worker pool."""
    parts = [workflow_name, run_id, node_id, "workers"]
    return "-".join(sanitize_filename(part) for part in parts)


def build_worker_task_id(node_id: str, task_index: int) -> str:
    """Build a deterministic task id scoped to one workflow node."""
    if task_index < 0:
        raise ValueError("task_index must be non-negative")
    return sanitize_filename(f"{node_id}-task-{task_index:06d}")


def bounded_worker_count(max_workers: int, task_count: int) -> int:
    """Return the worker count needed for a fixed-size worker pool."""
    if task_count < 1:
        return 0
    return max(1, min(max_workers, task_count))


def enqueue_worker_tasks(queue: QueueLike, tasks: Iterable[WorkerTask]) -> int:
    """Enqueue worker tasks through a queue-like object and return the count."""
    queued_count = 0
    for task in tasks:
        queue.put(task)
        queued_count += 1
    return queued_count


def summarize_worker_results(
    results: Iterable[WorkerTaskResult],
) -> WorkerPoolSummary:
    """Aggregate worker task results into one pool summary."""
    task_count = 0
    succeeded_count = 0
    failed_task_ids: list[str] = []
    for result in results:
        task_count += 1
        if result.succeeded:
            succeeded_count += 1
        else:
            failed_task_ids.append(result.task_id)
    failed_count = len(failed_task_ids)
    return WorkerPoolSummary(
        task_count=task_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        failed_task_ids=failed_task_ids,
    )
