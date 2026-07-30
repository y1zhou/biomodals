"""Pure fixed-batch construction tests."""

# ruff: noqa: D103

import pytest

from biomodals.execution import ProviderBinding
from biomodals.execution.scheduler import (
    TaskDispatchDescriptor,
    form_fixed_batches,
)

GPU = ProviderBinding(
    "production",
    "biomodals-af3",
    23,
    "run_inference",
    True,
    "af3-gpu",
)
CPU = ProviderBinding(
    "production",
    "biomodals-af3",
    23,
    "run_search",
    False,
    "af3-search",
)


def _task(
    task_key: str,
    ordinal: int,
    *,
    binding: ProviderBinding = GPU,
    compatibility_key: str = "same-model",
    max_tasks_per_call: int = 2,
) -> TaskDispatchDescriptor:
    return TaskDispatchDescriptor(
        node_key="inference",
        node_ordinal=2,
        task_key=task_key,
        task_ordinal=ordinal,
        binding=binding,
        compatibility_key=compatibility_key,
        max_tasks_per_call=max_tasks_per_call,
        depth=2,
        unblocking_span=1,
    )


def test_fixed_batches_preserve_compatibility_and_encounter_order() -> None:
    batches = form_fixed_batches((
        _task("seed-0", 0),
        _task("seed-1", 1, compatibility_key="other-model"),
        _task("seed-2", 2),
        _task("seed-3", 3),
        _task("search-0", 4, binding=CPU),
    ))

    assert [
        (
            batch.binding.function_name,
            batch.compatibility_key,
            batch.task_keys,
            batch.task_ordinal,
        )
        for batch in batches
    ] == [
        ("run_inference", "same-model", ("seed-0", "seed-2"), 0),
        ("run_inference", "same-model", ("seed-3",), 3),
        ("run_inference", "other-model", ("seed-1",), 1),
        ("run_search", "same-model", ("search-0",), 4),
    ]


def test_fixed_batches_never_span_nodes() -> None:
    second_node = TaskDispatchDescriptor(
        node_key="ranking",
        node_ordinal=3,
        task_key="seed-1",
        task_ordinal=0,
        binding=GPU,
        compatibility_key="same-model",
        max_tasks_per_call=2,
        depth=3,
        unblocking_span=0,
    )

    batches = form_fixed_batches((_task("seed-0", 0), second_node))

    assert [batch.task_keys for batch in batches] == [("seed-0",), ("seed-1",)]


def test_fixed_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_tasks_per_call must be positive"):
        form_fixed_batches((_task("seed-0", 0, max_tasks_per_call=0),))
