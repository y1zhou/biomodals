"""Tests for durable Direct CLI App Run storage."""

from pathlib import Path
from uuid import UUID

import pytest

from biomodals.execution import DeploymentIdentity, ExecutionPlan, NodePlan
from biomodals.helper.app_execution import AppExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def test_app_execution_store_uses_the_reserved_run_namespace(
    tmp_path: Path,
) -> None:
    """App execution state is isolated from scientific output paths."""
    store = AppExecutionRunStore(tmp_path, RUN_ID)

    assert store.state_root == (
        tmp_path / ".biomodals" / "execution" / "runs" / str(RUN_ID)
    )
    assert store.ledger_path == store.state_root / "ledger.sqlite3"
    assert not store.ledger_path.exists()


def test_app_execution_store_persists_the_shared_repository(tmp_path: Path) -> None:
    """Closing and reopening retains one kernel Execution Run."""
    store = AppExecutionRunStore(tmp_path, RUN_ID)
    plan = ExecutionPlan("example", (NodePlan("run"),))
    with store.transaction():
        store.execution.create_run(
            execution_run_id=RUN_ID,
            plan=plan,
            deployment=DeploymentIdentity("main", "Example", 3),
            max_active_provider_calls=2,
            max_active_gpu_provider_calls=1,
            now=10,
        )
    store.close()

    reopened = AppExecutionRunStore(tmp_path, RUN_ID)
    assert reopened.execution.get_run(RUN_ID).plan == plan
    reopened.close()


def test_app_execution_store_closes_sqlite_during_volume_sync(
    tmp_path: Path,
) -> None:
    """A mounted SQLite file is never open while its Volume is synchronized."""
    store = AppExecutionRunStore(tmp_path, RUN_ID)
    original = store.connection

    with store.closed_for_volume_sync():
        with pytest.raises(RuntimeError, match="closed for Volume synchronization"):
            _ = store.connection

    assert store.connection is not original
    store.close()
