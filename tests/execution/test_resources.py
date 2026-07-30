"""Run-local total and GPU Provider Call limit tests."""

# ruff: noqa: D103, S106

from biomodals.execution import ActiveProviderCallCounts

from .provider_call_helpers import (
    CPU_BINDING,
    GPU_BINDING,
    RUN_ID,
    create_repository,
)


def test_preclaim_atomically_enforces_total_and_gpu_subset_limits() -> None:
    repository = create_repository(
        task_count=4,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
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
