"""Tests for workflow worker-pool helpers."""

# ruff: noqa: D103

from biomodals.workflow.core.workers import (
    WorkerTask,
    WorkerTaskResult,
    bounded_worker_count,
    build_worker_pool_name,
    build_worker_task_id,
    enqueue_worker_tasks,
    summarize_worker_results,
)


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


def test_build_worker_task_id_is_deterministic_and_sanitized() -> None:
    assert build_worker_task_id("score:a", 12) == "score:a-task-000012"


def test_enqueue_worker_tasks_uses_queue_put() -> None:
    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    queue = FakeQueue()
    tasks = [
        WorkerTask(task_id="task-1", payload={"input": "a.pdb"}),
        WorkerTask(task_id="task-2", payload={"input": "b.pdb"}),
    ]

    queued_count = enqueue_worker_tasks(queue, tasks)

    assert queued_count == 2
    assert queue.items == tasks


def test_summarize_worker_results_aggregates_completion_status() -> None:
    summary = summarize_worker_results([
        WorkerTaskResult(task_id="task-1", succeeded=True),
        WorkerTaskResult(task_id="task-2", succeeded=False, error="failed"),
    ])

    assert summary.task_count == 2
    assert summary.succeeded_count == 1
    assert summary.failed_count == 1
    assert summary.failed_task_ids == ["task-2"]
    assert not summary.succeeded
