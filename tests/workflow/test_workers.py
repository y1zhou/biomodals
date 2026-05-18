"""Tests for workflow worker-pool helpers."""

# ruff: noqa: D103

from biomodals.workflow.workers import bounded_worker_count, build_worker_pool_name


def test_build_worker_pool_name_is_sanitized_and_run_scoped() -> None:
    assert (
        build_worker_pool_name("PPI Flow", "run/1", "score:a")
        == "PPI Flow-run_1-score:a-workers"
    )


def test_bounded_worker_count_zero_for_no_tasks() -> None:
    assert bounded_worker_count(max_workers=4, task_count=0) == 0


def test_bounded_worker_count_caps_by_max_workers() -> None:
    assert bounded_worker_count(max_workers=4, task_count=10) == 4


def test_bounded_worker_count_uses_task_count_when_smaller() -> None:
    assert bounded_worker_count(max_workers=4, task_count=2) == 2


def test_bounded_worker_count_never_returns_zero_for_positive_tasks() -> None:
    assert bounded_worker_count(max_workers=0, task_count=2) == 1
