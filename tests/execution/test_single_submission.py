"""Single-submission Provider Call ownership tests."""

# ruff: noqa: D103, S106

import pytest

from biomodals.execution import ProviderCallStatus, TaskStatus

from .provider_call_helpers import GPU_BINDING, RUN_ID, create_repository


def test_only_new_durable_preclaim_authorizes_spawn() -> None:
    repository = create_repository()

    first = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0", "seed-1"),
        submission_token="batch-0",
        binding=GPU_BINDING,
        compatibility_key="model-weights-v3",
        now=110,
    )
    duplicate = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0", "seed-1"),
        submission_token="batch-0",
        binding=GPU_BINDING,
        compatibility_key="model-weights-v3",
        now=111,
    )

    assert first is not None
    assert first.spawn_authorized
    assert first.call.status == ProviderCallStatus.SUBMITTING
    assert duplicate is not None
    assert not duplicate.spawn_authorized
    assert duplicate.call.provider_call_id == first.call.provider_call_id
    assert len(repository.list_provider_calls(RUN_ID)) == 1
    assert {
        task.task_key: (task.status, task.provider_call_id)
        for task in repository.list_tasks(RUN_ID, "inference")
    }["seed-0"] == (TaskStatus.RUNNING, first.call.provider_call_id)


def test_task_cannot_acquire_a_second_remote_owner() -> None:
    repository = create_repository()
    repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="first",
        binding=GPU_BINDING,
        compatibility_key="model-weights-v3",
        now=110,
    )

    with pytest.raises(ValueError, match="is not ready for Provider Call ownership"):
        repository.preclaim_fixed_batch(
            RUN_ID,
            "inference",
            ("seed-0",),
            submission_token="second",
            binding=GPU_BINDING,
            compatibility_key="model-weights-v3",
            now=111,
        )

    assert len(repository.list_provider_calls(RUN_ID)) == 1


def test_reusing_submission_token_for_other_work_is_a_conflict() -> None:
    repository = create_repository()
    repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="model-weights-v3",
        now=110,
    )

    with pytest.raises(ValueError, match="submission token was reused"):
        repository.preclaim_fixed_batch(
            RUN_ID,
            "inference",
            ("seed-1",),
            submission_token="batch",
            binding=GPU_BINDING,
            compatibility_key="model-weights-v3",
            now=111,
        )


def test_abandoned_submitting_call_becomes_unknown_and_never_reauthorizes() -> None:
    repository = create_repository()
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="model-weights-v3",
        now=110,
    )
    assert claim is not None

    recovered = repository.mark_submission_outcome_unknown(
        claim.call.provider_call_id,
        message="coordinator stopped after preclaim",
        now=120,
    )
    duplicate = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0",),
        submission_token="batch",
        binding=GPU_BINDING,
        compatibility_key="model-weights-v3",
        now=121,
    )

    assert recovered.status == ProviderCallStatus.OUTCOME_UNKNOWN
    assert duplicate is not None
    assert not duplicate.spawn_authorized
    assert duplicate.call.status == ProviderCallStatus.OUTCOME_UNKNOWN
