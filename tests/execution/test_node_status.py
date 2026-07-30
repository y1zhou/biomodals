"""Execution Node publication-observation tests."""

# ruff: noqa: D103

import sqlite3
from uuid import UUID

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    NodeStatus,
    ResultProvenance,
    RunStatus,
    RunStatusReason,
    SqliteExecutionRepository,
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
