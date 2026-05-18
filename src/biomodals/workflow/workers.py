"""Helpers for workflow node worker-pool bookkeeping."""

from __future__ import annotations

from biomodals.helper.shell import sanitize_filename


def build_worker_pool_name(workflow_name: str, run_id: str, node_id: str) -> str:
    """Build a deterministic Modal object name for one node worker pool."""
    parts = [workflow_name, run_id, node_id, "workers"]
    return "-".join(sanitize_filename(part) for part in parts)


def bounded_worker_count(max_workers: int, task_count: int) -> int:
    """Return the worker count needed for a fixed-size worker pool."""
    if task_count < 1:
        return 0
    return max(1, min(max_workers, task_count))
