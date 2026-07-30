"""Provider Call lifecycle tests."""

# ruff: noqa: D103, S106

import pytest

from biomodals.execution import ProviderCallStatus, RunStatus, RunStatusReason

from .provider_call_helpers import GPU_BINDING, RUN_ID, create_repository


def test_call_attachment_running_and_success_require_durable_identity_and_result() -> (
    None
):
    repository = create_repository()
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
    repository = create_repository()
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
    repository = create_repository()
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
