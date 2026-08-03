"""Node Task aggregation tests."""

# ruff: noqa: D103, S106

import sqlite3

from biomodals.execution import (
    AvailabilityStatus,
    NodeAggregationPolicy,
    NodeStatus,
    TaskStatus,
)
from biomodals.execution.scheduler import aggregate_task_status_counts

from .provider_call_helpers import (
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
)


def test_task_aggregation_policies_are_strict_and_non_vacuous() -> None:
    assert (
        aggregate_task_status_counts(
            NodeAggregationPolicy.COLLECT_ALL,
            {TaskStatus.SUCCEEDED: 2},
        )
        == NodeStatus.SUCCEEDED
    )
    assert (
        aggregate_task_status_counts(
            NodeAggregationPolicy.COLLECT_ALL,
            {TaskStatus.SUCCEEDED: 1, TaskStatus.FAILED: 1},
        )
        == NodeStatus.FAILED
    )
    assert (
        aggregate_task_status_counts(
            NodeAggregationPolicy.ALLOW_PARTIAL,
            {TaskStatus.SUCCEEDED: 1, TaskStatus.FAILED: 1},
        )
        == NodeStatus.PARTIAL
    )
    assert (
        aggregate_task_status_counts(
            NodeAggregationPolicy.ALLOW_PARTIAL,
            {TaskStatus.FAILED: 1, TaskStatus.SKIPPED: 1},
        )
        == NodeStatus.FAILED
    )
    assert (
        aggregate_task_status_counts(
            NodeAggregationPolicy.FAIL_FAST,
            {TaskStatus.FAILED: 1, TaskStatus.RUNNING: 1},
        )
        is None
    )
    assert aggregate_task_status_counts(NodeAggregationPolicy.COLLECT_ALL, {}) is None


def test_node_reconciliation_reads_status_counts_not_task_payloads() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(connection=connection, task_count=100)
    connection.execute(
        """
        UPDATE execution_tasks
        SET status = ?, completed_at = ?, updated_at = ?
        WHERE execution_run_id = ? AND node_key = ?
        """,
        (TaskStatus.SUCCEEDED.value, 110, 110, str(RUN_ID), "inference"),
    )
    statements: list[str] = []
    connection.set_trace_callback(statements.append)

    node = repository.reconcile_node_tasks(RUN_ID, "inference", now=111)

    connection.set_trace_callback(None)
    task_selects = [
        statement for statement in statements if "FROM execution_tasks" in statement
    ]
    assert node.status == NodeStatus.SUCCEEDED
    assert task_selects
    assert all("scientific_payload_json" not in statement for statement in task_selects)


def test_fail_fast_skips_only_unowned_siblings_and_drains_owned_work() -> None:
    repository = create_repository(
        task_count=3,
        aggregation_policy=NodeAggregationPolicy.FAIL_FAST,
    )
    persist_fixed_policy(
        repository,
        ("seed-1",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-1",),
        submission_token="owned",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None
    repository.fail_task(
        RUN_ID,
        "inference",
        "seed-0",
        message="invalid input",
        now=111,
    )

    draining = repository.reconcile_node_tasks(
        RUN_ID,
        "inference",
        now=112,
    )
    tasks = {
        task.task_key: task.status
        for task in repository.list_tasks(RUN_ID, "inference")
    }
    assert draining.status == NodeStatus.RUNNING
    assert tasks == {
        "seed-0": TaskStatus.FAILED,
        "seed-1": TaskStatus.RUNNING,
        "seed-2": TaskStatus.SKIPPED,
    }

    repository.attach_provider_call(
        claim.call.provider_call_id,
        provider_call_handle_id="fc-owned",
        now=119,
    )
    repository.record_provider_call_result(
        claim.call.provider_call_id,
        result_envelope={"tasks": {"seed-1": {"path": "/outputs/seed-1"}}},
        now=120,
    )
    repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-1",
        AvailabilityStatus.AVAILABLE,
        now=121,
    )
    failed = repository.reconcile_node_tasks(
        RUN_ID,
        "inference",
        now=122,
    )
    assert failed.status == NodeStatus.FAILED


def test_allow_partial_persists_partial_only_after_all_tasks_are_terminal() -> None:
    repository = create_repository(
        task_count=2,
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-0",
        AvailabilityStatus.AVAILABLE,
        now=110,
    )
    repository.fail_task(
        RUN_ID,
        "inference",
        "seed-1",
        message="seed failed",
        now=111,
    )

    node = repository.reconcile_node_tasks(
        RUN_ID,
        "inference",
        now=112,
    )

    assert node.status == NodeStatus.PARTIAL
    assert node.completed_at == 112
