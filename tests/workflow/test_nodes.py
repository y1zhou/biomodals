"""Tests for reusable workflow node helpers."""

# ruff: noqa: D101,D102,D103,D107

from biomodals.schema import NodeExecutionPolicy, NodePlacement
from biomodals.workflow.core.nodes import AppBackedNode


def test_app_backed_node_is_marker_base_for_remote_app_work() -> None:
    node = AppBackedNode()

    assert node.execution_policy == NodeExecutionPolicy.RERUN
    assert node.placement == NodePlacement.REMOTE


def test_app_backed_node_no_longer_owns_modal_lookup_api() -> None:
    assert not hasattr(AppBackedNode, "app_name")
    assert not hasattr(AppBackedNode, "function_name")
    assert not hasattr(AppBackedNode, "load_app_function")
    assert not hasattr(AppBackedNode, "invoke_app_function")
