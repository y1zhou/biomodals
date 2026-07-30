"""Execution Run outcome tests."""

# ruff: noqa: D103

import sqlite3
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionPlan,
    NodeDependency,
    NodePlan,
    NodeStatus,
    RunStatus,
    RunStatusReason,
    SqliteExecutionRepository,
    terminal_run_outcome,
)


def _branched_plan() -> ExecutionPlan:
    return ExecutionPlan(
        workload_name="branched",
        nodes=(
            NodePlan(node_key="source"),
            NodePlan(
                node_key="result-a",
                dependencies=(NodeDependency(node_key="source"),),
            ),
            NodePlan(
                node_key="result-b",
                dependencies=(NodeDependency(node_key="source"),),
            ),
        ),
    )


def test_run_outcome_is_strictly_driven_by_terminal_nodes() -> None:
    plan = _branched_plan()

    assert (
        terminal_run_outcome(
            plan,
            {
                "source": NodeStatus.FAILED,
                "result-a": NodeStatus.SUCCEEDED,
                "result-b": NodeStatus.PARTIAL,
            },
        )
        == RunStatus.PARTIAL
    )
    assert (
        terminal_run_outcome(
            plan,
            {
                "source": NodeStatus.SUCCEEDED,
                "result-a": NodeStatus.SUCCEEDED,
                "result-b": NodeStatus.FAILED,
            },
        )
        == RunStatus.FAILED
    )
    assert (
        terminal_run_outcome(
            plan,
            {
                "source": NodeStatus.SUCCEEDED,
                "result-a": NodeStatus.CANCELLED,
                "result-b": NodeStatus.FAILED,
            },
        )
        == RunStatus.CANCELLED
    )


def _repository() -> SqliteExecutionRepository:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    return repository


def _create_run(repository: SqliteExecutionRepository) -> UUID:
    execution_run_id = UUID("d4e4744e-aacf-4478-92d6-a58681805162")
    repository.create_run(
        execution_run_id=execution_run_id,
        plan=_branched_plan(),
        deployment=DeploymentIdentity(
            environment="production",
            deployment_name="biomodals-short-md",
            deployment_version=17,
        ),
        max_active_provider_calls=8,
        max_active_gpu_provider_calls=2,
        now=100,
    )
    return execution_run_id


def test_run_reason_vocabulary_is_closed() -> None:
    assert tuple(RunStatusReason) == (
        RunStatusReason.COORDINATOR_ERROR,
        RunStatusReason.RESULT_VALIDATION_UNKNOWN,
        RunStatusReason.SUBMISSION_OUTCOME_UNKNOWN,
        RunStatusReason.PROVIDER_OUTCOME_UNKNOWN,
        RunStatusReason.CANCELLATION_OUTCOME_UNKNOWN,
        RunStatusReason.REQUIRED_WORK_FAILED,
        RunStatusReason.DEPLOYMENT_UNAVAILABLE,
    )


def test_repository_enforces_run_transitions_and_reason_compatibility() -> None:
    repository = _repository()
    execution_run_id = _create_run(repository)

    running = repository.transition_run(
        execution_run_id,
        RunStatus.RUNNING,
        now=110,
    )
    assert running.status == RunStatus.RUNNING
    assert running.started_at == 110

    with pytest.raises(ValueError, match="requires a status reason"):
        repository.transition_run(
            execution_run_id,
            RunStatus.SUSPENDED,
            now=120,
        )
    with pytest.raises(ValueError, match="not valid for suspended"):
        repository.transition_run(
            execution_run_id,
            RunStatus.SUSPENDED,
            reason=RunStatusReason.PROVIDER_OUTCOME_UNKNOWN,
            now=120,
        )

    suspended = repository.transition_run(
        execution_run_id,
        RunStatus.SUSPENDED,
        reason=RunStatusReason.COORDINATOR_ERROR,
        message="coordinator stopped",
        now=120,
    )
    assert suspended.status_reason == RunStatusReason.COORDINATOR_ERROR
    assert suspended.status_message == "coordinator stopped"

    with pytest.raises(ValueError, match="explicit resume"):
        repository.transition_run(
            execution_run_id,
            RunStatus.RUNNING,
            now=130,
        )

    resumed = repository.resume_run(execution_run_id, now=130)
    assert resumed.status == RunStatus.RUNNING
    assert resumed.status_reason is None
    assert resumed.status_message is None

    failed = repository.transition_run(
        execution_run_id,
        RunStatus.FAILED,
        reason=RunStatusReason.REQUIRED_WORK_FAILED,
        message="terminal output unavailable",
        now=140,
    )
    assert failed.completed_at == 140

    with pytest.raises(ValueError, match="terminal Run"):
        repository.transition_run(
            execution_run_id,
            RunStatus.RUNNING,
            now=150,
        )
