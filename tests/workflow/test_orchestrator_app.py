"""Tests for the mocked workflow orchestrator boundary."""

# ruff: noqa: D101,D102,D103,D107

import sys
import types
from pathlib import Path

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


def test_orchestrator_helper_uses_runtime_from_definition(monkeypatch) -> None:
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
        ):
            calls["workflow_name"] = workflow_name
            calls["workflow_definition"] = workflow_definition
            calls["volume_root"] = volume_root
            calls["workflow_volume_name"] = workflow_volume_name
            calls["workflow_volume"] = workflow_volume
            calls["remote_node_runner"] = remote_node_runner
            return cls()

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            calls["run_id"] = run_id
            calls["force"] = force
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    result = orchestrator.run_workflow_definition(
        workflow_name="demo",
        run_id="run-1",
        workflow_definition={"nodes": []},
        volume_root=Path("/workflow-outputs"),
        workflow_volume_name="WorkflowOrchestrator-outputs",
        force=True,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls == {
        "workflow_name": "demo",
        "workflow_definition": {"nodes": []},
        "volume_root": Path("/workflow-outputs"),
        "workflow_volume_name": "WorkflowOrchestrator-outputs",
        "workflow_volume": None,
        "remote_node_runner": None,
        "run_id": "run-1",
        "force": True,
    }


def test_orchestrator_helper_passes_remote_node_runner(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def remote_runner(node, context):
        raise AssertionError("not executed by boundary test")

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
        ):
            calls["remote_node_runner"] = remote_node_runner
            return cls()

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    result = orchestrator.run_workflow_definition(
        workflow_name="demo",
        run_id="run-1",
        workflow_definition={"nodes": []},
        volume_root=Path("/workflow-outputs"),
        workflow_volume_name="WorkflowOrchestrator-outputs",
        remote_node_runner=remote_runner,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["remote_node_runner"] is remote_runner


def test_run_remote_node_with_volume_commits_after_exception(tmp_path: Path) -> None:
    volume = FakeVolume()
    context = NodeRunContext(
        run_id="run-1",
        node_id="remote",
        attempt_id="attempt-1",
        cache_dir=tmp_path / "cache",
        inputs={},
    )

    with pytest.raises(RuntimeError, match="remote failed"):
        orchestrator.run_remote_node_with_volume(
            node=RaisingNode(),
            context=context,
            workflow_volume=volume,
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
        orchestrator_function=FakeOrchestratorFunction(),
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
        orchestrator_function=FakeOrchestratorFunction(),
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
    setattr(module, "build", lambda: {"nodes": []})
    monkeypatch.setitem(sys.modules, "fake_workflow_factory", module)

    with pytest.raises(TypeError, match="must return a Workflow"):
        orchestrator.load_workflow_definition("fake_workflow_factory:build")


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
