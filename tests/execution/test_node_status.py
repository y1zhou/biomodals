"""Execution Node publication-observation tests."""

# ruff: noqa: D103

import sqlite3
from uuid import UUID

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodeDependency,
    NodePlan,
    NodeStatus,
    ResultProvenance,
    RunStatus,
    RunStatusReason,
    SqliteExecutionRepository,
    TaskPlan,
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
            nodes=(NodePlan(node_key="result"),),
        ),
        deployment=DeploymentIdentity("production", "biomodals-demo", 3),
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        now=100,
    )
    return repository


def test_available_initial_node_publication_is_cache_success() -> None:
    repository = _repository()

    node = repository.record_node_result_observation(
        RUN_ID,
        "result",
        AvailabilityStatus.AVAILABLE,
        now=110,
    )

    assert node.status == NodeStatus.SUCCEEDED
    assert node.result_observation == AvailabilityStatus.AVAILABLE
    assert node.result_observed_at == 110
    assert node.result_provenance == ResultProvenance.CACHE
    assert node.started_at is None
    assert node.completed_at == 110
    assert repository.get_run(RUN_ID).status == RunStatus.PENDING


def test_missing_node_publication_authorizes_no_state_transition() -> None:
    repository = _repository()

    node = repository.record_node_result_observation(
        RUN_ID,
        "result",
        AvailabilityStatus.MISSING,
        now=110,
    )

    assert node.status == NodeStatus.PENDING
    assert node.result_observation == AvailabilityStatus.MISSING
    assert node.result_provenance is None


def test_unknown_node_publication_suspends_until_explicit_resume() -> None:
    repository = _repository()

    node = repository.record_node_result_observation(
        RUN_ID,
        "result",
        AvailabilityStatus.UNKNOWN,
        now=110,
    )

    run = repository.get_run(RUN_ID)
    assert node.status == NodeStatus.PENDING
    assert run.status == RunStatus.SUSPENDED
    assert run.status_reason == RunStatusReason.RESULT_VALIDATION_UNKNOWN

    repository.resume_run(RUN_ID, now=120)
    conclusive = repository.record_node_result_observation(
        RUN_ID,
        "result",
        AvailabilityStatus.MISSING,
        now=121,
    )
    assert conclusive.status == NodeStatus.PENDING
    assert repository.get_run(RUN_ID).status == RunStatus.RUNNING


def test_available_running_node_publication_records_current_run_provenance() -> None:
    repository = _repository()
    repository.start_node(RUN_ID, "result", now=105)

    node = repository.record_node_result_observation(
        RUN_ID,
        "result",
        AvailabilityStatus.AVAILABLE,
        now=110,
    )

    assert node.status == NodeStatus.SUCCEEDED
    assert node.result_provenance == ResultProvenance.CURRENT_RUN
    assert node.started_at == 105
    assert node.completed_at == 110


def test_dependency_failure_durably_skips_every_unreachable_successor() -> None:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    repository.create_run(
        execution_run_id=RUN_ID,
        plan=ExecutionPlan(
            workload_name="demo",
            nodes=(
                NodePlan(node_key="prepare"),
                NodePlan(
                    node_key="simulate",
                    dependencies=(NodeDependency("prepare"),),
                ),
                NodePlan(
                    node_key="result",
                    dependencies=(NodeDependency("simulate"),),
                ),
            ),
        ),
        deployment=DeploymentIdentity("production", "biomodals-demo", 3),
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        now=100,
    )
    repository.start_node(RUN_ID, "prepare", now=101)
    repository.discover_tasks(
        RUN_ID,
        "prepare",
        (TaskPlan("task", {"input": "x"}),),
        now=102,
    )
    repository.record_task_result_observation(
        RUN_ID,
        "prepare",
        "task",
        AvailabilityStatus.MISSING,
        now=103,
    )
    repository.fail_task(
        RUN_ID,
        "prepare",
        "task",
        message="preparation failed",
        now=104,
    )
    repository.reconcile_node_tasks(RUN_ID, "prepare", now=105)

    skipped = repository.skip_unreachable_nodes(RUN_ID, now=106)

    assert [node.node_key for node in skipped] == ["simulate", "result"]
    assert all(node.status == NodeStatus.SKIPPED for node in skipped)
