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
