"""Small helpers for bounded local fan-out."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def bounded_map(  # noqa: UP047
    items: Iterable[T],
    worker: Callable[[T], R],
    *,
    max_parallel: int | None = None,
) -> list[R]:
    """Run blocking work with at most ``max_parallel`` active calls."""
    item_list = list(items)
    if not item_list:
        return []
    if max_parallel is None:
        max_parallel = len(item_list)
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")

    max_workers = min(max_parallel, len(item_list))
    if max_workers == 1:
        return [worker(item) for item in item_list]

    results: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker, item): index for index, item in enumerate(item_list)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[index] for index in range(len(item_list))]


def batches_for_total_concurrency(  # noqa: UP047
    items: Iterable[T],
    *,
    max_batches: int,
    max_workers_per_batch: int,
    total_concurrency: int,
) -> tuple[list[list[T]], int]:
    """Split work into batches while capping total active workers.

    Returns ordered batches and the local worker count each batch should use.
    If there are many items, local workers per batch are reduced so
    ``len(batches) * workers_per_batch`` does not exceed ``total_concurrency``.
    """
    item_list = list(items)
    if not item_list:
        return [], 1
    if max_batches < 1:
        raise ValueError("max_batches must be at least 1")
    if max_workers_per_batch < 1:
        raise ValueError("max_workers_per_batch must be at least 1")
    if total_concurrency < 1:
        raise ValueError("total_concurrency must be at least 1")

    worker_sized_batch_count = (
        len(item_list) + max_workers_per_batch - 1
    ) // max_workers_per_batch
    batch_count = min(
        len(item_list),
        max_batches,
        total_concurrency,
        max(1, worker_sized_batch_count),
    )
    workers_per_batch = max(
        1,
        min(max_workers_per_batch, total_concurrency // batch_count),
    )
    batch_size = (len(item_list) + batch_count - 1) // batch_count
    batches = [
        item_list[index : index + batch_size]
        for index in range(0, len(item_list), batch_size)
    ]
    return batches, workers_per_batch
