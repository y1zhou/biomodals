"""Execution Task publication-observation tests."""

# ruff: noqa: D103

import sqlite3
from uuid import UUID

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    ResultProvenance,
    RunStatus,
    RunStatusReason,
    SqliteExecutionRepository,
    TaskPlan,
    TaskStatus,
)

RUN_ID = UUID("d4e4744e-aacf-4478-92d6-a58681805162")


def _repository() -> SqliteExecutionRepository:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    repository.create_run(
        execution_run_id=RUN_ID,
        plan=ExecutionPlan(
            workload_name="demo",
            nodes=(NodePlan(node_key="inference"),),
        ),
        deployment=DeploymentIdentity("production", "biomodals-demo", 3),
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        now=100,
    )
    repository.start_node(RUN_ID, "inference", now=101)
    repository.discover_tasks(
        RUN_ID,
        "inference",
        (TaskPlan(task_key="seed-1", scientific_payload={"seed": 1}),),
        now=102,
    )
    return repository


def test_available_unowned_task_publication_is_cache_success() -> None:
    repository = _repository()

    task = repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-1",
        AvailabilityStatus.AVAILABLE,
        now=110,
    )

    assert task.status == TaskStatus.SUCCEEDED
    assert task.result_observation == AvailabilityStatus.AVAILABLE
    assert task.result_observed_at == 110
    assert task.result_provenance == ResultProvenance.CACHE
    assert task.started_at is None
    assert task.completed_at == 110


def test_missing_unowned_task_remains_pending() -> None:
    repository = _repository()

    task = repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-1",
        AvailabilityStatus.MISSING,
        now=110,
    )

    assert task.status == TaskStatus.PENDING
    assert task.result_observation == AvailabilityStatus.MISSING
    assert task.result_provenance is None


def test_unknown_task_publication_suspends_without_changing_ownership() -> None:
    repository = _repository()

    task = repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-1",
        AvailabilityStatus.UNKNOWN,
        now=110,
    )

    run = repository.get_run(RUN_ID)
    assert task.status == TaskStatus.PENDING
    assert run.status == RunStatus.SUSPENDED
    assert run.status_reason == RunStatusReason.RESULT_VALIDATION_UNKNOWN
