"""Task discovery and identity tests."""

# ruff: noqa: D103

import sqlite3
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    NodeStatus,
    SqliteExecutionRepository,
    TaskPlan,
    TaskStatus,
)

RUN_ID = UUID("d4e4744e-aacf-4478-92d6-a58681805162")


def _repository(*, allow_empty_result: bool = False) -> SqliteExecutionRepository:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    repository.create_run(
        execution_run_id=RUN_ID,
        plan=ExecutionPlan(
            workload_name="alphafold3",
            nodes=(
                NodePlan(
                    node_key="inference",
                    allow_empty_result=allow_empty_result,
                ),
            ),
        ),
        deployment=DeploymentIdentity("production", "biomodals-af3", 23),
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
        now=100,
    )
    repository.start_node(RUN_ID, "inference", now=101)
    return repository


def test_task_fingerprint_excludes_operational_execution_payload() -> None:
    task = TaskPlan(
        task_key="seed-1",
        scientific_payload={"seed": 1},
        execution_payload={"gpu": "H100", "batch_size": 8},
    )
    differently_scheduled = TaskPlan(
        task_key="seed-1",
        scientific_payload={"seed": 1},
        execution_payload={"gpu": "A100", "batch_size": 1},
    )

    expected = "f78a24e9761a7a49668a455e94a61c00432135a101f19f973dd6256e384f9310"
    assert (
        task.fingerprint(
            workload_plan_fingerprint="plan-digest",
            node_key="inference",
        )
        == expected
    )
    assert (
        differently_scheduled.fingerprint(
            workload_plan_fingerprint="plan-digest",
            node_key="inference",
        )
        == expected
    )


def test_task_fingerprint_rejects_non_finite_numbers() -> None:
    task = TaskPlan(
        task_key="seed-1",
        scientific_payload={"confidence": float("nan")},
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        task.fingerprint(
            workload_plan_fingerprint="plan-digest",
            node_key="inference",
        )


def test_discovery_persists_complete_ordered_task_set_and_fingerprints() -> None:
    repository = _repository()
    task_plans = (
        TaskPlan(
            task_key="seed-5",
            scientific_payload={"seed": 5},
            execution_payload={"input_path": "/staging/5.json"},
        ),
        TaskPlan(
            task_key="seed-7",
            scientific_payload={"seed": 7},
            execution_payload={"input_path": "/staging/7.json"},
        ),
    )

    tasks = repository.discover_tasks(
        RUN_ID,
        "inference",
        task_plans,
        now=102,
    )

    assert [
        (
            task.task_key,
            task.ordinal,
            task.status,
            task.scientific_payload,
            task.execution_payload,
        )
        for task in tasks
    ] == [
        (
            "seed-5",
            0,
            TaskStatus.PENDING,
            {"seed": 5},
            {"input_path": "/staging/5.json"},
        ),
        (
            "seed-7",
            1,
            TaskStatus.PENDING,
            {"seed": 7},
            {"input_path": "/staging/7.json"},
        ),
    ]
    assert [task.fingerprint for task in tasks] == [
        plan.fingerprint(
            workload_plan_fingerprint=repository.get_run(
                RUN_ID
            ).plan.workload_plan_fingerprint,
            node_key="inference",
        )
        for plan in task_plans
    ]
    assert repository.get_node(RUN_ID, "inference").discovery_complete
    assert repository.list_tasks(RUN_ID, "inference") == tasks


def test_discovery_rejects_duplicate_keys_without_partial_rows() -> None:
    repository = _repository()

    with pytest.raises(ValueError, match="duplicate Task key 'seed-5'"):
        repository.discover_tasks(
            RUN_ID,
            "inference",
            (
                TaskPlan(task_key="seed-5", scientific_payload={"seed": 5}),
                TaskPlan(task_key="seed-5", scientific_payload={"seed": 7}),
            ),
            now=102,
        )

    assert repository.list_tasks(RUN_ID, "inference") == ()
    assert not repository.get_node(RUN_ID, "inference").discovery_complete


def test_discovery_is_a_single_checkpoint() -> None:
    repository = _repository()
    repository.discover_tasks(
        RUN_ID,
        "inference",
        (TaskPlan(task_key="seed-5", scientific_payload={"seed": 5}),),
        now=102,
    )

    with pytest.raises(ValueError, match="Task discovery is already complete"):
        repository.discover_tasks(
            RUN_ID,
            "inference",
            (TaskPlan(task_key="seed-7", scientific_payload={"seed": 7}),),
            now=103,
        )


def test_empty_discovery_fails_by_default_without_a_synthetic_task() -> None:
    repository = _repository()

    assert repository.discover_tasks(RUN_ID, "inference", (), now=102) == ()

    node = repository.get_node(RUN_ID, "inference")
    assert node.discovery_complete
    assert node.status == NodeStatus.FAILED
    assert node.error_message == "Node discovered no Tasks"


def test_allowed_empty_discovery_waits_for_explicit_result_validation() -> None:
    repository = _repository(allow_empty_result=True)

    assert repository.discover_tasks(RUN_ID, "inference", (), now=102) == ()

    node = repository.get_node(RUN_ID, "inference")
    assert node.discovery_complete
    assert node.status == NodeStatus.RUNNING
    assert node.error_message is None


def test_caller_can_fail_discovery_without_inventing_a_task() -> None:
    repository = _repository()

    failed = repository.fail_node(
        RUN_ID,
        "inference",
        message="Could not enumerate candidate inputs",
        now=102,
    )

    assert failed.status == NodeStatus.FAILED
    assert failed.discovery_complete
    assert failed.error_message == "Could not enumerate candidate inputs"
    assert repository.list_tasks(RUN_ID, "inference") == ()


def test_caller_cannot_fail_a_node_while_tasks_remain_active() -> None:
    repository = _repository()
    repository.discover_tasks(
        RUN_ID,
        "inference",
        (TaskPlan(task_key="seed-5", scientific_payload={"seed": 5}),),
        now=102,
    )

    with pytest.raises(ValueError, match="Tasks remain active"):
        repository.fail_node(
            RUN_ID,
            "inference",
            message="Aggregate publication failed",
            now=103,
        )
