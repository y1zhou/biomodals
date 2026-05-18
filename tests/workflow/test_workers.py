"""Tests for workflow worker-pool helpers."""

# ruff: noqa: D103

from biomodals.workflow.core.workers import (
    WorkerTask,
    WorkerTaskResult,
    bounded_worker_count,
    build_worker_pool_name,
    build_worker_task_id,
    create_worker_queue,
    enqueue_worker_tasks,
    gather_worker_pool_results,
    spawn_worker_pool,
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


def test_create_worker_queue_uses_injected_queue_factory() -> None:
    calls = []

    class FakeQueueFactory:
        @staticmethod
        def from_name(name, *, create_if_missing):
            calls.append((name, create_if_missing))
            return "queue"

    queue = create_worker_queue(
        "demo-run-node-workers",
        queue_factory=FakeQueueFactory,
    )

    assert queue == "queue"
    assert calls == [("demo-run-node-workers", True)]


def test_spawn_worker_pool_spawns_fixed_count_workers() -> None:
    calls = []

    class FakeWorkerFunction:
        def spawn(self, queue, **kwargs):
            calls.append((queue, kwargs))
            return f"call-{len(calls)}"

    spawned = spawn_worker_pool(
        FakeWorkerFunction(),
        queue="queue",
        worker_count=3,
        node_id="score",
    )

    assert spawned == ["call-1", "call-2", "call-3"]
    assert calls == [
        ("queue", {"node_id": "score"}),
        ("queue", {"node_id": "score"}),
        ("queue", {"node_id": "score"}),
    ]


def test_gather_worker_pool_results_uses_injected_gather() -> None:
    calls = ["call-1", "call-2"]

    def fake_gather(*function_calls):
        assert function_calls == tuple(calls)
        return [
            WorkerTaskResult(task_id="task-1", succeeded=True),
            WorkerTaskResult(task_id="task-2", succeeded=True),
        ]

    results = gather_worker_pool_results(calls, gather=fake_gather)

    assert [result.task_id for result in results] == ["task-1", "task-2"]


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
