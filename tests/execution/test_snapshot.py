"""Read-only execution snapshot and overview tests."""

# ruff: noqa: D103, S105, S106

import sqlite3

from biomodals.execution import (
    ActiveProviderCallCounts,
    NodeTaskStatusCounts,
    ProviderCallStatus,
    RunStatus,
)

from .provider_call_helpers import (
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
)


def test_snapshot_projects_one_consistent_execution_view() -> None:
    repository = create_repository(task_count=2)

    snapshot = repository.snapshot(RUN_ID)

    assert snapshot.run.status == RunStatus.RUNNING
    assert [node.node_key for node in snapshot.nodes] == ["inference"]
    assert [task.task_key for task in snapshot.tasks] == ["seed-0", "seed-1"]
    assert snapshot.provider_calls == ()
    assert snapshot.active_provider_calls == ActiveProviderCallCounts(total=0, gpu=0)


def test_overview_preserves_lifecycle_state_without_reading_tasks() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(connection=connection, task_count=2)
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    active = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-1",),
        submission_token="call-0",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    completed = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="call-1",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=111,
    )
    assert completed is not None and active is not None
    repository.attach_provider_call(
        completed.call.provider_call_id,
        provider_call_handle_id="fc-completed",
        now=112,
    )
    repository.record_provider_call_result(
        completed.call.provider_call_id,
        result_envelope={"path": "/outputs/seed-0"},
        now=113,
    )
    repository.attach_provider_call(
        active.call.provider_call_id,
        provider_call_handle_id="fc-active",
        now=114,
    )
    statements: list[str] = []
    connection.set_trace_callback(statements.append)

    overview = repository.overview(RUN_ID)

    connection.set_trace_callback(None)
    assert overview.run == repository.get_run(RUN_ID)
    assert overview.nodes == repository.list_nodes(RUN_ID)
    assert [
        call.submission_token for call in overview.representative_provider_calls
    ] == ["call-0"]
    assert overview.representative_provider_calls[0].status == (
        ProviderCallStatus.ATTACHED
    )
    assert overview.representative_provider_calls[0].provider_call_handle_id == (
        "fc-active"
    )
    assert overview.representative_provider_calls[0].result_envelope is None
    assert overview.active_provider_calls == ActiveProviderCallCounts(total=1, gpu=1)
    assert overview.node_task_status_counts == (
        NodeTaskStatusCounts(node_key="inference", running=2),
    )
    assert all("scientific_payload_json" not in statement for statement in statements)

    repository.cancel_provider_call(
        active.call.provider_call_id,
        message="test cleanup",
        now=115,
    )

    [representative] = repository.overview(RUN_ID).representative_provider_calls
    assert representative.submission_token == "call-1"
    assert representative.status == ProviderCallStatus.SUCCEEDED
    assert representative.result_envelope == {"path": "/outputs/seed-0"}


def test_provider_call_page_is_bounded_and_node_scoped() -> None:
    repository = create_repository(task_count=2)
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    first = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="call-0",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    second = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-1",),
        submission_token="call-1",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=111,
    )
    assert first is not None and second is not None

    page = repository.provider_call_page(RUN_ID, node_key="inference", limit=1)

    assert [call.provider_call_id for call in page.calls] == [
        first.call.provider_call_id
    ]
    assert page.calls[0].function_name == GPU_BINDING.function_name
    assert page.next_cursor == first.call.provider_call_id
    final_page = repository.provider_call_page(
        RUN_ID,
        node_key="inference",
        cursor=page.next_cursor,
        limit=1,
    )
    assert [call.provider_call_id for call in final_page.calls] == [
        second.call.provider_call_id
    ]
    assert final_page.next_cursor is None

    newest = repository.provider_call_page(
        RUN_ID,
        node_key="inference",
        limit=1,
        newest_first=True,
    )
    assert newest.calls[0].provider_call_id == second.call.provider_call_id
    older = repository.provider_call_page(
        RUN_ID,
        node_key="inference",
        cursor=newest.next_cursor,
        limit=1,
        newest_first=True,
    )
    assert older.calls[0].provider_call_id == first.call.provider_call_id
