"""Tests for the mocked workflow orchestrator boundary."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from typing import Any, cast

import pytest

from biomodals.helper.constant import WORKFLOW_ORCHESTRATOR_VOLUME_NAME
from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow import Workflow
from biomodals.workflow.core import orchestrator
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNativeNode


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


def test_orchestrator_run_constructs_runtime(monkeypatch) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(
            self,
            *,
            workflow: Workflow,
            volume_root: Path,
            workflow_volume_name: str | None = None,
            workflow_volume=None,
            remote_node_runner=None,
            remote_node_function_name=None,
            function_call_resolver=None,
            max_ready_workers: int = 32,
        ):
            calls["workflow"] = workflow
            calls["volume_root"] = volume_root
            calls["workflow_volume_name"] = workflow_volume_name
            calls["workflow_volume"] = workflow_volume
            calls["remote_node_runner"] = remote_node_runner
            calls["remote_node_function_name"] = remote_node_function_name
            calls["function_call_resolver"] = function_call_resolver
            calls["max_ready_workers"] = max_ready_workers

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            calls["run_id"] = run_id
            calls["force"] = force
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    raw_cls, instance = _raw_orchestrator()
    workflow = Workflow("demo")
    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=workflow,
        run_id="run-1",
        force=True,
        max_ready_workers=7,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls == {
        "workflow": workflow,
        "volume_root": Path(orchestrator.CONF.output_volume_mountpoint),
        "workflow_volume_name": WORKFLOW_ORCHESTRATOR_VOLUME_NAME,
        "workflow_volume": volume,
        "remote_node_runner": calls["remote_node_runner"],
        "remote_node_function_name": orchestrator.REMOTE_NODE_FUNCTION_NAME,
        "function_call_resolver": calls["function_call_resolver"],
        "max_ready_workers": 7,
        "run_id": "run-1",
        "force": True,
        "closed": True,
    }
    assert calls["remote_node_runner"] is not None
    assert callable(calls["function_call_resolver"])
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_orchestrator_run_passes_remote_node_runner_and_resolver(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(
            self,
            *,
            workflow: Workflow,
            volume_root: Path,
            workflow_volume_name: str | None = None,
            workflow_volume=None,
            remote_node_runner=None,
            remote_node_function_name=None,
            function_call_resolver=None,
            max_ready_workers: int = 32,
        ) -> None:
            self.remote_node_runner = remote_node_runner
            calls["remote_node_runner"] = remote_node_runner
            calls["remote_node_function_name"] = remote_node_function_name
            calls["function_call_resolver"] = function_call_resolver
            calls["max_ready_workers"] = max_ready_workers

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            assert self.remote_node_runner is not None
            context = NodeRunContext(
                run_id=run_id,
                node_id="remote",
                attempt_id="attempt-1",
                cache_dir=tmp_path / "cache",
                inputs={},
            )
            calls["remote_call"] = self.remote_node_runner(SucceedNode(), context)
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    class FakeFunctionCall:
        object_id = "fc-remote"

        def get(self):
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    raw_cls, instance = _raw_orchestrator()
    instance.run_node = cast(
        Any,
        type(
            "FakeRemoteMethod",
            (),
            {
                "__name__": "run_node",
                "spawn": lambda self, node, context: (
                    calls.update({
                        "remote_method": self,
                        "remote_node": node,
                        "remote_context": context,
                    })
                    or FakeFunctionCall()
                ),
            },
        )(),
    )
    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("demo"),
        run_id="run-1",
        max_ready_workers=9,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["remote_node_runner"] is not None
    assert calls["remote_node_function_name"] == orchestrator.REMOTE_NODE_FUNCTION_NAME
    assert callable(calls["function_call_resolver"])
    assert calls["max_ready_workers"] == 9
    assert getattr(calls["remote_method"], "__name__", None) == "run_node"
    assert isinstance(calls["remote_node"], SucceedNode)
    remote_context = cast(NodeRunContext, calls["remote_context"])
    remote_call = cast(Any, calls["remote_call"])
    assert remote_context.run_id == "run-1"
    assert remote_call.object_id == "fc-remote"
    assert calls["closed"] is True
    assert volume.reload_count == 1
    assert volume.commit_count == 1


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


def test_orchestrator_exit_cancels_active_remote_calls_once(monkeypatch) -> None:
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(self) -> None:
            self.cancel_count = 0
            self.close_count = 0

        def cancel_active_remote_calls(self, *, terminate_containers: bool) -> None:
            assert terminate_containers is True
            self.cancel_count += 1

        def close(self) -> None:
            self.close_count += 1

    raw_cls, instance = _raw_orchestrator()
    runtime = FakeRuntime()
    instance._runtime = runtime

    raw_cls.exit._get_raw_f()(instance)
    raw_cls.exit._get_raw_f()(instance)

    assert runtime.cancel_count == 1
    assert runtime.close_count == 1
    assert instance._runtime is None
    assert volume.commit_count == 2


def test_orchestrator_exit_still_closes_and_commits_when_cancel_fails(
    monkeypatch,
) -> None:
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(self) -> None:
            self.close_count = 0

        def cancel_active_remote_calls(self, *, terminate_containers: bool) -> None:
            raise RuntimeError("cancel failed")

        def close(self) -> None:
            self.close_count += 1

    raw_cls, instance = _raw_orchestrator()
    runtime = FakeRuntime()
    instance._runtime = runtime

    raw_cls.exit._get_raw_f()(instance)

    assert runtime.close_count == 1
    assert instance._runtime is None
    assert volume.commit_count == 1


def test_orchestrator_remote_node_runner_rejects_non_function_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", FakeVolume())

    class FakeRuntime:
        def __init__(self, *, remote_node_runner=None, **kwargs) -> None:
            self.remote_node_runner = remote_node_runner

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            assert self.remote_node_runner is not None
            context = NodeRunContext(
                run_id=run_id,
                node_id="remote",
                attempt_id="attempt-1",
                cache_dir=tmp_path / "cache",
                inputs={},
            )
            calls["remote_call"] = self.remote_node_runner(SucceedNode(), context)
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    raw_cls, instance = _raw_orchestrator()
    instance.run_node = cast(
        Any,
        type(
            "BadRemoteMethod",
            (),
            {"spawn": lambda self, *args: object()},
        )(),
    )
    with pytest.raises(TypeError, match="FunctionCall"):
        raw_cls.run._get_raw_f()(
            instance,
            workflow=Workflow("demo"),
            run_id="run-1",
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
        raw_cls.run_node._get_raw_f()(
            instance,
            RaisingNode(),
            context,
        )

    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_orchestrator_rejects_serialized_workflow_dict() -> None:
    raw_cls, instance = _raw_orchestrator()

    with pytest.raises(TypeError, match="Workflow object"):
        raw_cls.run._get_raw_f()(
            instance,
            workflow={"nodes": []},
            run_id="run-1",
        )


def test_orchestrator_modal_app_uses_python_313_runtime() -> None:
    assert orchestrator.CONF.python_version == "3.13"
    assert orchestrator.WorkflowOrchestrator is not None
    assert orchestrator.OUT_VOLUME_NAME == WORKFLOW_ORCHESTRATOR_VOLUME_NAME


def test_orchestrator_app_exposes_only_class_remote_surface() -> None:
    functions = orchestrator.app._local_state.functions

    assert "WorkflowOrchestrator.*" in functions
    assert "run_workflow_orchestrator" not in functions
    assert "run_remote_workflow_node" not in functions
    assert "submit_workflow_orchestrator_task" not in functions
