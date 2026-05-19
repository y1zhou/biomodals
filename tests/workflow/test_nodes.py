"""Tests for reusable workflow node helpers."""

# ruff: noqa: D101,D102,D103,D107

from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow.core.nodes import AppBackedNode, NodeRunContext


class FakeRemoteFunction:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


class FakeAppNode(AppBackedNode):
    app_name = "FlowPacker"
    function_name = "run_flowpacker_workflow"

    def __init__(self, remote_function):
        self.remote_function = remote_function

    def load_app_function(self):
        return self.remote_function

    def build_app_function_kwargs(self, context):
        return {"run_name": context.run_id, "node_id": context.node_id}


def test_app_backed_node_invokes_loaded_modal_function(tmp_path) -> None:
    remote_function = FakeRemoteFunction()
    node = FakeAppNode(remote_function)

    result = node.run(
        NodeRunContext(
            run_id="run-1",
            node_id="pack",
            attempt_id="attempt-1",
            cache_dir=tmp_path / "cache",
            inputs={},
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert remote_function.calls == [{"run_name": "run-1", "node_id": "pack"}]
