"""Tests for the workflow-owned physical run store."""

# ruff: noqa: D103

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
)
from biomodals.workflow.core.run_store import (
    UnsupportedWorkflowRunStoreError,
    WorkflowRunStore,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def test_run_store_uses_execution_identity_and_shared_schema(tmp_path: Path) -> None:
    store = WorkflowRunStore(tmp_path, RUN_ID)

    with store.transaction():
        store.execution.create_run(
            execution_run_id=RUN_ID,
            plan=ExecutionPlan(
                workload_name="workflow:demo",
                workload_run_key="user-chosen-name",
                nodes=(NodePlan(node_key="design"),),
            ),
            deployment=DeploymentIdentity("main", "DemoWorkflow", 4),
            max_active_provider_calls=8,
            max_active_gpu_provider_calls=2,
            now=100,
        )

    assert store.state_root == (
        tmp_path
        / ".biomodals"
        / "execution"
        / "runs"
        / "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    )
    assert "user-chosen-name" not in str(store.ledger_path)
    assert store.output_root == (
        tmp_path / "workflow-runs" / "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    )
    assert store.output_root != store.state_root
    assert store.execution.get_run(RUN_ID).plan.workload_run_key == "user-chosen-name"
    tables = {
        str(row[0])
        for row in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "execution_runs" in tables
    assert "workflow_artifacts" in tables
    assert "runs" not in tables
    assert "attempts" not in tables


def test_transaction_rolls_back_both_repository_views(tmp_path: Path) -> None:
    store = WorkflowRunStore(tmp_path, RUN_ID)

    with pytest.raises(RuntimeError, match="stop"):
        with store.transaction():
            store.execution.create_run(
                execution_run_id=RUN_ID,
                plan=ExecutionPlan(
                    workload_name="workflow:demo",
                    nodes=(NodePlan(node_key="design"),),
                ),
                deployment=DeploymentIdentity("main", "DemoWorkflow", 4),
                max_active_provider_calls=8,
                max_active_gpu_provider_calls=2,
                now=100,
            )
            raise RuntimeError("stop")

    assert (
        store.connection.execute("SELECT COUNT(*) FROM execution_runs").fetchone()[0]
        == 0
    )


def test_closing_for_volume_sync_reopens_repository_views(tmp_path: Path) -> None:
    store = WorkflowRunStore(tmp_path, RUN_ID)
    first_connection = store.connection

    with store.transaction():
        store.execution.create_run(
            execution_run_id=RUN_ID,
            plan=ExecutionPlan(
                workload_name="workflow:demo",
                nodes=(NodePlan(node_key="design"),),
            ),
            deployment=DeploymentIdentity("main", "DemoWorkflow", 4),
            max_active_provider_calls=8,
            max_active_gpu_provider_calls=2,
            now=100,
        )
    with store.closed_for_volume_sync():
        with pytest.raises(sqlite3.ProgrammingError):
            first_connection.execute("SELECT 1")

    assert store.connection is not first_connection
    assert store.execution.get_run(RUN_ID).status.value == "pending"


def test_existing_legacy_or_unrecognized_ledger_is_rejected(tmp_path: Path) -> None:
    store = WorkflowRunStore(tmp_path, RUN_ID)
    store.state_root.mkdir(parents=True)
    with sqlite3.connect(store.ledger_path) as connection:
        connection.execute("CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY)")

    with pytest.raises(UnsupportedWorkflowRunStoreError, match="fresh"):
        _ = store.connection
