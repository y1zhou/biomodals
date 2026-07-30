"""Successor Execution Run compatibility tests."""

# ruff: noqa: D103

import sqlite3
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    RunStatus,
    SqliteExecutionRepository,
)

PREDECESSOR_ID = UUID("d4e4744e-aacf-4478-92d6-a58681805162")
SUCCESSOR_ID = UUID("c3eca2bb-2ab2-4b14-bd63-bd2077bb8bc4")


def _plan(*, input_digest: str = "abc") -> ExecutionPlan:
    return ExecutionPlan(
        workload_name="alphafold3",
        workload_run_key="request-1",
        scientific_payload={"input_sha256": input_digest},
        scientific_versions={"model": "3"},
        nodes=(NodePlan(node_key="result"),),
    )


def _repository(*, terminal: bool = True) -> SqliteExecutionRepository:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    repository.create_run(
        execution_run_id=PREDECESSOR_ID,
        plan=_plan(),
        deployment=DeploymentIdentity("prod", "af3-coordinator", 7),
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
        now=100,
    )
    if terminal:
        repository.start_node(PREDECESSOR_ID, "result", now=105)
        repository.discover_tasks(PREDECESSOR_ID, "result", (), now=106)
        repository.finalize_run_from_results(PREDECESSOR_ID, now=110)
    return repository


def test_successor_gets_new_identity_and_may_change_operational_policy() -> None:
    repository = _repository()
    deployment = DeploymentIdentity("prod", "af3-coordinator", 8)

    successor = repository.create_successor_run(
        predecessor_execution_run_id=PREDECESSOR_ID,
        execution_run_id=SUCCESSOR_ID,
        deployment=deployment,
        max_active_provider_calls=8,
        max_active_gpu_provider_calls=4,
        now=120,
    )

    assert successor.execution_run_id == SUCCESSOR_ID
    assert successor.predecessor_execution_run_id == PREDECESSOR_ID
    assert successor.plan == _plan()
    assert successor.deployment == deployment
    assert successor.max_active_provider_calls == 8
    assert successor.max_active_gpu_provider_calls == 4
    assert successor.status == RunStatus.PENDING
    assert repository.get_run(PREDECESSOR_ID).status == RunStatus.FAILED


def test_launch_time_restart_requires_same_scientific_fingerprint() -> None:
    repository = _repository()

    with pytest.raises(ValueError, match="Workload Plan Fingerprint"):
        repository.create_successor_run(
            predecessor_execution_run_id=PREDECESSOR_ID,
            execution_run_id=SUCCESSOR_ID,
            plan=_plan(input_digest="changed"),
            deployment=DeploymentIdentity("prod", "af3-coordinator", 8),
            max_active_provider_calls=4,
            max_active_gpu_provider_calls=2,
            now=120,
        )

    assert len(repository.list_nodes(PREDECESSOR_ID)) == 1
    with pytest.raises(LookupError):
        repository.get_run(SUCCESSOR_ID)


def test_nonterminal_predecessor_fails_closed() -> None:
    repository = _repository(terminal=False)

    with pytest.raises(ValueError, match="predecessor Run is not terminal"):
        repository.create_successor_run(
            predecessor_execution_run_id=PREDECESSOR_ID,
            execution_run_id=SUCCESSOR_ID,
            deployment=DeploymentIdentity("prod", "af3-coordinator", 8),
            max_active_provider_calls=4,
            max_active_gpu_provider_calls=2,
            now=120,
        )


def test_successor_lineage_may_cross_physical_repositories() -> None:
    predecessor_repository = _repository()
    predecessor = predecessor_repository.validate_successor_source(PREDECESSOR_ID)
    successor_repository = SqliteExecutionRepository(sqlite3.connect(":memory:"))
    successor_repository.initialize_schema()

    successor = successor_repository.create_run(
        execution_run_id=SUCCESSOR_ID,
        predecessor_execution_run_id=predecessor.execution_run_id,
        plan=predecessor.plan,
        deployment=DeploymentIdentity("prod", "af3-coordinator", 8),
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
        now=120,
    )

    assert successor.predecessor_execution_run_id == PREDECESSOR_ID
