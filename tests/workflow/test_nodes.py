"""Tests for reusable workflow node helpers."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from uuid import UUID

import pytest

from biomodals.workflow.core.nodes import AppBackedNode, NodeRunContext


def test_app_backed_node_requires_caller_owned_remote_preparation(
    tmp_path: Path,
) -> None:
    node = AppBackedNode()
    context = NodeRunContext(
        execution_run_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        workload_run_key="demo",
        node_id="remote",
        task_key="node",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={},
    )

    with pytest.raises(NotImplementedError):
        node.prepare_remote(context)


def test_app_backed_node_owns_no_modal_lookup_or_submission_api() -> None:
    assert not hasattr(AppBackedNode, "app_name")
    assert not hasattr(AppBackedNode, "function_name")
    assert not hasattr(AppBackedNode, "load_app_function")
    assert not hasattr(AppBackedNode, "invoke_app_function")
    assert not hasattr(AppBackedNode, "submit_remote")
