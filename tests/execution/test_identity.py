"""Execution Run identity and persisted-plan tests."""

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
    SqliteExecutionRepository,
    UnsupportedExecutionSchemaVersionError,
)


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        workload_name="alphafold3",
        workload_run_key="friendly-name",
        scientific_payload={"input_sha256": "abc"},
        scientific_versions={"model": "3"},
        nodes=(
            NodePlan(node_key="search"),
            NodePlan(
                node_key="inference",
                dependencies=(NodeDependency(node_key="search"),),
            ),
        ),
    )


def test_create_run_persists_opaque_identity_plan_and_nodes() -> None:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    execution_run_id = UUID("d4e4744e-aacf-4478-92d6-a58681805162")
    deployment = DeploymentIdentity(
        environment="production",
        deployment_name="biomodals-alphafold3",
        deployment_version=23,
    )

    created = repository.create_run(
        execution_run_id=execution_run_id,
        plan=_plan(),
        deployment=deployment,
        max_active_provider_calls=10,
        max_active_gpu_provider_calls=4,
        now=100,
    )

    assert created.execution_run_id == execution_run_id
    assert created.predecessor_execution_run_id is None
    assert created.plan == _plan()
    assert created.deployment == deployment
    assert created.status == RunStatus.PENDING
    assert created.max_active_provider_calls == 10
    assert created.max_active_gpu_provider_calls == 4
    assert created.created_at == 100
    assert created.updated_at == 100
    assert created.started_at is None
    assert created.completed_at is None
    assert repository.get_run(execution_run_id) == created

    nodes = repository.list_nodes(execution_run_id)
    assert [(node.node_key, node.ordinal, node.status) for node in nodes] == [
        ("search", 0, NodeStatus.PENDING),
        ("inference", 1, NodeStatus.PENDING),
    ]
    assert nodes[1].dependencies == (NodeDependency(node_key="search"),)


def test_run_identity_is_not_inferred_from_workload_key() -> None:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    deployment = DeploymentIdentity("production", "biomodals-af3", 23)

    first = repository.create_run(
        execution_run_id=UUID("d4e4744e-aacf-4478-92d6-a58681805162"),
        plan=_plan(),
        deployment=deployment,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        now=100,
    )
    second = repository.create_run(
        execution_run_id=UUID("c3eca2bb-2ab2-4b14-bd63-bd2077bb8bc4"),
        plan=_plan(),
        deployment=deployment,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        now=101,
    )

    assert first.execution_run_id != second.execution_run_id
    assert first.plan.workload_run_key == second.plan.workload_run_key


@pytest.mark.parametrize(
    ("total", "gpu", "message"),
    [
        (0, 0, "max_active_provider_calls must be positive"),
        (1, -1, "max_active_gpu_provider_calls cannot be negative"),
        (1, 2, "cannot exceed max_active_provider_calls"),
    ],
)
def test_create_run_rejects_invalid_call_limits(
    total: int,
    gpu: int,
    message: str,
) -> None:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()

    with pytest.raises(ValueError, match=message):
        repository.create_run(
            execution_run_id=UUID("d4e4744e-aacf-4478-92d6-a58681805162"),
            plan=_plan(),
            deployment=DeploymentIdentity("production", "biomodals-af3", 23),
            max_active_provider_calls=total,
            max_active_gpu_provider_calls=gpu,
            now=100,
        )


def test_repository_rejects_unknown_schema_version() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE execution_schema (singleton INTEGER PRIMARY KEY, version INTEGER)"
    )
    connection.execute("INSERT INTO execution_schema VALUES (1, 999)")

    with pytest.raises(
        UnsupportedExecutionSchemaVersionError,
        match="Unsupported execution schema version 999",
    ):
        SqliteExecutionRepository(connection).initialize_schema()
