"""Provider Call lifecycle tests."""

# ruff: noqa: D103, S106

import sqlite3

import pytest

from biomodals.execution import ProviderCallStatus, RunStatus, RunStatusReason

from .provider_call_helpers import (
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
)


def _repository_with_policy():
    repository = create_repository()
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    return repository


def test_call_attachment_running_and_success_require_durable_identity_and_result() -> (
    None
):
    repository = _repository_with_policy()
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None

    attached = repository.attach_provider_call(
        claim.call.provider_call_id,
        provider_call_handle_id="fc-123",
        now=111,
    )
    running = repository.mark_provider_call_running(
        claim.call.provider_call_id,
        now=112,
    )
    succeeded = repository.record_provider_call_result(
        claim.call.provider_call_id,
        result_envelope={"tasks": {"seed-0": {"path": "/outputs/seed-0"}}},
        now=120,
    )

    assert attached.status == ProviderCallStatus.ATTACHED
    assert attached.provider_call_handle_id == "fc-123"
    assert running.status == ProviderCallStatus.RUNNING
    assert succeeded.status == ProviderCallStatus.SUCCEEDED
    assert succeeded.result_envelope == {
        "tasks": {"seed-0": {"path": "/outputs/seed-0"}}
    }
    assert succeeded.completed_at == 120


def test_attached_unknown_state_projects_to_run_and_can_be_reconciled() -> None:
    repository = _repository_with_policy()
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None
    repository.attach_provider_call(
        claim.call.provider_call_id,
        provider_call_handle_id="fc-123",
        now=111,
    )

    unknown = repository.mark_provider_call_state_unknown(
        claim.call.provider_call_id,
        message="Modal state lookup was inconclusive",
        now=112,
    )

    run = repository.get_run(RUN_ID)
    assert unknown.status == ProviderCallStatus.STATE_UNKNOWN
    assert run.status == RunStatus.STATE_UNKNOWN
    assert run.status_reason == RunStatusReason.PROVIDER_OUTCOME_UNKNOWN

    running = repository.mark_provider_call_running(
        claim.call.provider_call_id,
        now=120,
    )
    assert running.status == ProviderCallStatus.RUNNING
    assert repository.get_run(RUN_ID).status == RunStatus.RUNNING


def test_call_lifecycle_rejects_missing_attachment_identity_and_illegal_rewrites() -> (
    None
):
    repository = _repository_with_policy()
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None

    with pytest.raises(ValueError, match="handle ID cannot be empty"):
        repository.attach_provider_call(
            claim.call.provider_call_id,
            provider_call_handle_id="",
            now=111,
        )
    with pytest.raises(
        ValueError, match="cannot mark submitting Provider Call running"
    ):
        repository.mark_provider_call_running(
            claim.call.provider_call_id,
            now=112,
        )
    with pytest.raises(ValueError, match="cannot cancel submitting Provider Call"):
        repository.cancel_provider_call(
            claim.call.provider_call_id,
            message="cancel requested before attachment",
            now=113,
        )


def test_listing_provider_calls_uses_bounded_ownership_queries() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(
        connection=connection,
        task_count=3,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=3,
    )
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1", "seed-2"),
        binding=GPU_BINDING,
        compatibility_key="gpu",
    )
    for index in range(3):
        claim = repository.preclaim_fixed_batch(
            RUN_ID,
            "inference",
            (f"seed-{index}",),
            submission_token=f"batch-{index}",
            binding=GPU_BINDING,
            compatibility_key="gpu",
            now=110 + index,
        )
        assert claim is not None

    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    calls = repository.list_provider_calls(RUN_ID)
    connection.set_trace_callback(None)

    task_selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
        and "FROM execution_tasks" in statement
    ]
    assert [call.task_keys for call in calls] == [
        ("seed-0",),
        ("seed-1",),
        ("seed-2",),
    ]
    assert len(task_selects) == 2
