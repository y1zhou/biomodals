"""Tests for the mocked workflow orchestrator boundary."""

# ruff: noqa: D101,D102,D103,D107

import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow import Workflow
from biomodals.workflow.core import orchestrator
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNativeNode
from biomodals.workflow.core.runtime import WorkflowRuntime


class SucceedNode(WorkflowNativeNode):
    def run(self, context):
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


class RaisingNode(WorkflowNativeNode):
    def run(self, context):
        raise RuntimeError("remote failed")


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def _raw_orchestrator() -> tuple[Any, Any]:
    raw_cls = cast(Any, orchestrator.WorkflowOrchestrator)._get_user_cls()
    return raw_cls, raw_cls()


def test_orchestrator_method_uses_runtime_from_definition(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeRuntime:
        @classmethod
        def from_definition(
            cls,
            *,
            workflow_name: str,
            workflow_definition: dict[str, object],
            volume_root: Path,
            workflow_volume_name: str,
            workflow_volume=None,
            remote_node_runner=None,
            remote_node_function_name=None,
            function_call_resolver=None,
        ):
            calls["workflow_name"] = workflow_name
            calls["workflow_definition"] = workflow_definition
            calls["volume_root"] = volume_root
            calls["workflow_volume_name"] = workflow_volume_name
            calls["workflow_volume"] = workflow_volume
            calls["remote_node_runner"] = remote_node_runner
            calls["remote_node_function_name"] = remote_node_function_name
            calls["function_call_resolver"] = function_call_resolver
            return cls()

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            calls["run_id"] = run_id
            calls["force"] = force
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    raw_cls, instance = _raw_orchestrator()
    result = raw_cls.run_workflow_orchestrator._get_raw_f()(
        instance,
        workflow_name="demo",
        run_id="run-1",
        workflow_definition={"nodes": []},
        force=True,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls == {
        "workflow_name": "demo",
        "workflow_definition": {"nodes": []},
        "volume_root": Path(orchestrator.CONF.output_volume_mountpoint),
        "workflow_volume_name": "WorkflowOrchestrator-outputs",
        "workflow_volume": orchestrator.OUT_VOLUME,
        "remote_node_runner": calls["remote_node_runner"],
        "remote_node_function_name": orchestrator.REMOTE_NODE_FUNCTION_NAME,
        "function_call_resolver": calls["function_call_resolver"],
        "run_id": "run-1",
        "force": True,
    }
    assert calls["remote_node_runner"] is not None
    assert callable(calls["function_call_resolver"])


def test_orchestrator_method_passes_remote_node_runner_and_resolver(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeRuntime:
        @classmethod
        def from_definition(
            cls,
            *,
            workflow_name: str,
            workflow_definition: dict[str, object],
            volume_root: Path,
            workflow_volume_name: str,
            workflow_volume=None,
            remote_node_runner=None,
            remote_node_function_name=None,
            function_call_resolver=None,
        ):
            calls["remote_node_runner"] = remote_node_runner
            calls["remote_node_function_name"] = remote_node_function_name
            calls["function_call_resolver"] = function_call_resolver
            return cls()

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    raw_cls, instance = _raw_orchestrator()
    result = raw_cls.run_workflow_orchestrator._get_raw_f()(
        instance,
        workflow_name="demo",
        run_id="run-1",
        workflow_definition={"nodes": []},
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["remote_node_runner"] is not None
    assert calls["remote_node_function_name"] == orchestrator.REMOTE_NODE_FUNCTION_NAME
    assert callable(calls["function_call_resolver"])


def test_orchestrator_enter_closes_stale_runtime_before_reload(monkeypatch) -> None:
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    raw_cls, instance = _raw_orchestrator()
    runtime = FakeRuntime()
    instance._runtime = runtime

    raw_cls.enter._get_raw_f()(instance)

    assert runtime.close_count == 1
    assert instance._runtime is None
    assert volume.reload_count == 1


def test_spawn_remote_workflow_node_rejects_non_function_call(
    tmp_path: Path,
) -> None:
    class BadRemoteMethod:
        def spawn(self, *args):
            return object()

    context = NodeRunContext(
        run_id="run-1",
        node_id="remote",
        attempt_id="attempt-1",
        cache_dir=tmp_path / "cache",
        inputs={},
    )

    with pytest.raises(TypeError, match="FunctionCall"):
        orchestrator._spawn_remote_workflow_node(
            cast(Any, BadRemoteMethod()),
            SucceedNode(),
            context,
        )


def test_orchestrator_remote_node_method_commits_after_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)
    context = NodeRunContext(
        run_id="run-1",
        node_id="remote",
        attempt_id="attempt-1",
        cache_dir=tmp_path / "cache",
        inputs={},
    )

    with pytest.raises(RuntimeError, match="remote failed"):
        raw_cls, instance = _raw_orchestrator()
        raw_cls.run_remote_workflow_node._get_raw_f()(
            instance,
            RaisingNode(),
            context,
        )

    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_submit_workflow_run_waits_for_remote_result() -> None:
    calls: dict[str, object] = {}

    class FakeOrchestratorFunction:
        def remote(self, **kwargs):
            calls.update(kwargs)
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    result = orchestrator.submit_workflow_run(
        orchestrator_function=cast(Any, FakeOrchestratorFunction()),
        workflow_name="demo",
        run_id="run-1",
        workflow_definition={"nodes": []},
        force=True,
    )

    assert result == AppRunResult(status=AppRunStatus.SUCCEEDED)
    assert calls == {
        "workflow_name": "demo",
        "run_id": "run-1",
        "workflow_definition": {"nodes": []},
        "force": True,
    }


def test_submit_workflow_run_can_spawn_without_waiting() -> None:
    calls: dict[str, object] = {}

    class FakeCall:
        object_id = "fc-123"

    class FakeOrchestratorFunction:
        def spawn(self, **kwargs):
            calls.update(kwargs)
            return FakeCall()

    function_call_id = orchestrator.submit_workflow_run(
        orchestrator_function=cast(Any, FakeOrchestratorFunction()),
        workflow_name="demo",
        run_id="run-1",
        workflow_definition={"nodes": []},
        wait=False,
    )

    assert function_call_id == "fc-123"
    assert calls["workflow_name"] == "demo"
    assert calls["run_id"] == "run-1"


def test_load_workflow_definition_rejects_serialized_dict_factory(
    monkeypatch,
) -> None:
    module = types.ModuleType("fake_workflow_factory")
    module.build = lambda: {"nodes": []}
    monkeypatch.setitem(sys.modules, "fake_workflow_factory", module)

    with pytest.raises(TypeError, match="must return a Workflow"):
        orchestrator.load_workflow_definition("fake_workflow_factory:build")


def test_load_workflow_definition_accepts_importable_module_factory(
    monkeypatch,
) -> None:
    workflow = Workflow("demo")
    module = types.ModuleType("importable_workflow_factory")
    module.build = lambda: workflow
    monkeypatch.setitem(sys.modules, "importable_workflow_factory", module)

    assert (
        orchestrator.load_workflow_definition("importable_workflow_factory:build")
        is workflow
    )


def test_runtime_from_definition_accepts_python_workflow(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(SucceedNode(), id="ok")

    runtime = WorkflowRuntime.from_definition(
        workflow_name="demo",
        workflow_definition=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    assert runtime.workflow is workflow
    assert runtime.run(run_id="run-1").status == AppRunStatus.SUCCEEDED


def test_orchestrator_modal_app_uses_python_313_runtime() -> None:
    assert orchestrator.CONF.python_version == "3.13"
    assert orchestrator.WorkflowOrchestrator is not None
