"""Execution status vocabulary tests."""

# ruff: noqa: D103

from biomodals.execution import (
    AvailabilityStatus,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    TaskStatus,
)


def test_run_status_vocabulary_and_terminality() -> None:
    assert tuple(RunStatus) == (
        RunStatus.PENDING,
        RunStatus.RUNNING,
        RunStatus.CANCEL_REQUESTED,
        RunStatus.SUSPENDED,
        RunStatus.STATE_UNKNOWN,
        RunStatus.SUCCEEDED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    )
    assert {status for status in RunStatus if status.is_terminal} == {
        RunStatus.SUCCEEDED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }


def test_node_status_vocabulary_and_terminality() -> None:
    assert tuple(NodeStatus) == (
        NodeStatus.PENDING,
        NodeStatus.RUNNING,
        NodeStatus.SUCCEEDED,
        NodeStatus.PARTIAL,
        NodeStatus.FAILED,
        NodeStatus.CANCELLED,
        NodeStatus.SKIPPED,
    )
    assert {status for status in NodeStatus if status.is_terminal} == {
        NodeStatus.SUCCEEDED,
        NodeStatus.PARTIAL,
        NodeStatus.FAILED,
        NodeStatus.CANCELLED,
        NodeStatus.SKIPPED,
    }


def test_task_status_vocabulary_and_terminality() -> None:
    assert tuple(TaskStatus) == (
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    )
    assert {status for status in TaskStatus if status.is_terminal} == {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }


def test_provider_call_status_vocabulary_and_terminality() -> None:
    assert tuple(ProviderCallStatus) == (
        ProviderCallStatus.SUBMITTING,
        ProviderCallStatus.ATTACHED,
        ProviderCallStatus.RUNNING,
        ProviderCallStatus.OUTCOME_UNKNOWN,
        ProviderCallStatus.STATE_UNKNOWN,
        ProviderCallStatus.SUCCEEDED,
        ProviderCallStatus.FAILED,
        ProviderCallStatus.CANCELLED,
    )
    assert {status for status in ProviderCallStatus if status.is_terminal} == {
        ProviderCallStatus.SUCCEEDED,
        ProviderCallStatus.FAILED,
        ProviderCallStatus.CANCELLED,
    }


def test_availability_status_has_exact_tri_state_vocabulary() -> None:
    assert tuple(AvailabilityStatus) == (
        AvailabilityStatus.AVAILABLE,
        AvailabilityStatus.MISSING,
        AvailabilityStatus.UNKNOWN,
    )
