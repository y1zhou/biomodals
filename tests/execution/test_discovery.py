"""Task discovery and identity tests."""

# ruff: noqa: D103

import pytest

from biomodals.execution import TaskPlan


def test_task_fingerprint_excludes_operational_execution_payload() -> None:
    task = TaskPlan(
        task_key="seed-1",
        scientific_payload={"seed": 1},
        execution_payload={"gpu": "H100", "batch_size": 8},
    )
    differently_scheduled = TaskPlan(
        task_key="seed-1",
        scientific_payload={"seed": 1},
        execution_payload={"gpu": "A100", "batch_size": 1},
    )

    expected = "f78a24e9761a7a49668a455e94a61c00432135a101f19f973dd6256e384f9310"
    assert (
        task.fingerprint(
            workload_plan_fingerprint="plan-digest",
            node_key="inference",
        )
        == expected
    )
    assert (
        differently_scheduled.fingerprint(
            workload_plan_fingerprint="plan-digest",
            node_key="inference",
        )
        == expected
    )


def test_task_fingerprint_rejects_non_finite_numbers() -> None:
    task = TaskPlan(
        task_key="seed-1",
        scientific_payload={"confidence": float("nan")},
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        task.fingerprint(
            workload_plan_fingerprint="plan-digest",
            node_key="inference",
        )
