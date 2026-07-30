"""Provider Result Envelope durability tests."""

# ruff: noqa: D103, S106

import pytest

from biomodals.execution import AvailabilityStatus, ProviderCallStatus, TaskStatus

from .provider_call_helpers import GPU_BINDING, RUN_ID, create_repository


def test_call_success_and_task_scientific_completion_are_separate() -> None:
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

    call = repository.record_provider_call_result(
        claim.call.provider_call_id,
        result_envelope={"tasks": {"seed-0": {"path": "/outputs/seed-0"}}},
        now=120,
    )

    assert call.status == ProviderCallStatus.SUCCEEDED
    assert (
        repository.get_task(
            RUN_ID,
            "inference",
            "seed-0",
        ).status
        == TaskStatus.RUNNING
    )


def test_non_json_result_does_not_change_call_or_release_slot() -> None:
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

    with pytest.raises(TypeError):
        repository.record_provider_call_result(
            claim.call.provider_call_id,
            result_envelope={"bad": object()},
            now=120,
        )

    assert (
        repository.get_provider_call(claim.call.provider_call_id).status
        == ProviderCallStatus.ATTACHED
    )


def test_conclusive_call_failure_fails_only_unfinished_owned_tasks() -> None:
    repository = create_repository(task_count=2)
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0", "seed-1"),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="gpu",
        now=110,
    )
    assert claim is not None
    repository.record_task_result_observation(
        RUN_ID,
        "inference",
        "seed-0",
        # An early worker completion report already validated this publication.
        observation=AvailabilityStatus.AVAILABLE,
        now=111,
    )

    repository.fail_provider_call(
        claim.call.provider_call_id,
        message="provider input failed",
        now=120,
    )

    tasks = {
        task.task_key: task.status
        for task in repository.list_tasks(RUN_ID, "inference")
    }
    assert tasks == {
        "seed-0": TaskStatus.SUCCEEDED,
        "seed-1": TaskStatus.FAILED,
    }
