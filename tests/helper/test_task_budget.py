"""Tests for bounded task helpers."""

# ruff: noqa: D103

from threading import Lock
from time import sleep

import pytest

from biomodals.helper.task_budget import batches_for_total_concurrency, bounded_map


def test_bounded_map_preserves_order_and_limits_active_workers() -> None:
    lock = Lock()
    active = 0
    max_active = 0

    def worker(value: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        sleep(0.01)
        with lock:
            active -= 1
        return value * 2

    assert bounded_map(range(5), worker, max_parallel=2) == [0, 2, 4, 6, 8]
    assert max_active <= 2


def test_bounded_map_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        bounded_map([1], lambda item: item, max_parallel=0)


def test_batches_for_total_concurrency_caps_total_workers() -> None:
    batches, workers = batches_for_total_concurrency(
        range(1000),
        max_batches=32,
        max_workers_per_batch=32,
        total_concurrency=32,
    )

    assert len(batches) == 32
    assert workers == 1
    assert [item for batch in batches for item in batch] == list(range(1000))
    assert len(batches) * workers <= 32


def test_batches_for_total_concurrency_uses_local_workers_for_small_jobs() -> None:
    batches, workers = batches_for_total_concurrency(
        range(2),
        max_batches=2,
        max_workers_per_batch=32,
        total_concurrency=2,
    )

    assert batches == [[0, 1]]
    assert workers == 2
    assert len(batches) * workers <= 2
