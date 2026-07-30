"""Read-only execution snapshot tests."""

# ruff: noqa: D103

from biomodals.execution import ActiveProviderCallCounts, RunStatus

from .provider_call_helpers import RUN_ID, create_repository


def test_snapshot_projects_one_consistent_execution_view() -> None:
    repository = create_repository(task_count=2)

    snapshot = repository.snapshot(RUN_ID)

    assert snapshot.run.status == RunStatus.RUNNING
    assert [node.node_key for node in snapshot.nodes] == ["inference"]
    assert [task.task_key for task in snapshot.tasks] == ["seed-0", "seed-1"]
    assert snapshot.provider_calls == ()
    assert snapshot.active_provider_calls == ActiveProviderCallCounts(total=0, gpu=0)
