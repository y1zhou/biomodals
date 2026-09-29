"""Execution status vocabulary tests."""

# ruff: noqa: D103

from biomodals.execution import (
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    TaskStatus,
)


def test_run_status_vocabulary_and_terminality() -> None:
    assert {status for status in RunStatus if status.is_terminal} == {
        RunStatus.SUCCEEDED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }


def test_node_status_vocabulary_and_terminality() -> None:
    assert {status for status in NodeStatus if status.is_terminal} == {
        NodeStatus.SUCCEEDED,
        NodeStatus.PARTIAL,
        NodeStatus.FAILED,
        NodeStatus.CANCELLED,
        NodeStatus.SKIPPED,
    }


def test_task_status_vocabulary_and_terminality() -> None:
    assert {status for status in TaskStatus if status.is_terminal} == {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }


def test_provider_call_status_vocabulary_and_terminality() -> None:
    assert {status for status in ProviderCallStatus if status.is_terminal} == {
        ProviderCallStatus.SUCCEEDED,
        ProviderCallStatus.FAILED,
        ProviderCallStatus.CANCELLED,
    }
