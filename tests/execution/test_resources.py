"""Run-local total and GPU Provider Call limit tests."""

# ruff: noqa: D103, S106

import sqlite3

from biomodals.execution import ActiveProviderCallCounts

from .provider_call_helpers import (
    CPU_BINDING,
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
)


def test_active_call_count_uses_the_status_index() -> None:
    connection = sqlite3.connect(":memory:")
    create_repository(connection=connection)

    query_plan = connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT COUNT(*), COALESCE(SUM(uses_gpu), 0)
        FROM execution_provider_calls
        WHERE execution_run_id = ?
            AND status IN (?, ?, ?, ?, ?)
        """,
        (
            str(RUN_ID),
            "submitting",
            "attached",
            "running",
            "outcome_unknown",
            "state_unknown",
        ),
    ).fetchall()

    assert any(
        "execution_provider_calls_status_created_idx" in str(row[3])
        for row in query_plan
    )


def test_preclaim_atomically_enforces_total_and_gpu_subset_limits() -> None:
    repository = create_repository(
        task_count=4,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
    )
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    persist_fixed_policy(
        repository,
        ("seed-2", "seed-3"),
        binding=CPU_BINDING,
        compatibility_key="cpu",
    )

    gpu = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="gpu-0",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    blocked_gpu = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-1",),
        submission_token="gpu-1",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=111,
    )
    cpu = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-2",),
        submission_token="cpu-0",
        binding=CPU_BINDING,
        compatibility_key="cpu",
        now=112,
    )
    blocked_total = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-3",),
        submission_token="cpu-1",
        binding=CPU_BINDING,
        compatibility_key="cpu",
        now=113,
    )

    assert gpu is not None
    assert blocked_gpu is None
    assert cpu is not None
    assert blocked_total is None
    assert repository.active_provider_call_counts(RUN_ID) == ActiveProviderCallCounts(
        total=2, gpu=1
    )


def test_unknown_calls_retain_slots_and_durable_success_releases_them() -> None:
    repository = create_repository(
        task_count=3,
        max_active_provider_calls=1,
        max_active_gpu_provider_calls=1,
    )
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="gpu-0",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None

    repository.mark_submission_outcome_unknown(
        claim.call.provider_call_id,
        message="spawn response lost",
        now=111,
    )
    assert (
        repository.preclaim_fixed_batch(
            RUN_ID,
            "inference",
            ("seed-1",),
            submission_token="gpu-1",
            binding=GPU_BINDING,
            compatibility_key="gpu",
            now=112,
        )
        is None
    )

    repository.record_provider_call_result(
        claim.call.provider_call_id,
        result_envelope={"tasks": {"seed-0": {"path": "/outputs/seed-0"}}},
        now=120,
    )
    assert repository.active_provider_call_counts(RUN_ID) == ActiveProviderCallCounts(
        total=0, gpu=0
    )
    assert (
        repository.preclaim_fixed_batch(
            RUN_ID,
            "inference",
            ("seed-1",),
            submission_token="gpu-1",
            binding=GPU_BINDING,
            compatibility_key="gpu",
            now=121,
        )
        is not None
    )


def test_node_call_counts_separate_active_from_selected_history() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(
        connection=connection,
        task_count=2,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=2,
    )
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    completed = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="completed",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    active = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-1",),
        submission_token="active",
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
    statements: list[str] = []
    connection.set_trace_callback(statements.append)

    assert repository.active_provider_call_counts_by_node(RUN_ID) == {"inference": 1}
    assert repository.provider_call_counts_by_node(RUN_ID, ("inference",)) == {
        "inference": (2, 1)
    }
    statement_count = len(statements)
    assert repository.provider_call_counts_by_node(RUN_ID, ()) == {}

    connection.set_trace_callback(None)
    assert len(statements) == statement_count


def test_selected_node_call_counts_use_the_node_status_index() -> None:
    connection = sqlite3.connect(":memory:")
    create_repository(connection=connection)
    query_plan = connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT node_key, COUNT(*)
        FROM execution_provider_calls
        WHERE execution_run_id = ? AND node_key IN (?, ?)
        GROUP BY node_key
        """,
        (str(RUN_ID), "inference", "other"),
    ).fetchall()

    assert any(
        "execution_provider_calls_node_status_idx" in str(row[3]) for row in query_plan
    )
