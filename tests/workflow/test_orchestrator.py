"""Tests for the mocked workflow orchestrator boundary."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from typing import Any, cast

import pytest

from biomodals.helper.constant import WORKFLOW_ORCHESTRATOR_VOLUME_NAME
from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow import Workflow
from biomodals.workflow.core import orchestrator


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
            function_call_resolver=None,
            max_ready_workers: int = 32,
            strict_external_artifact_checks: bool = False,
            external_artifact_checker=None,
        ):
            calls["workflow"] = workflow
            calls["volume_root"] = volume_root
            calls["workflow_volume_name"] = workflow_volume_name
            calls["workflow_volume"] = workflow_volume
            calls["function_call_resolver"] = function_call_resolver
            calls["max_ready_workers"] = max_ready_workers
            calls["strict_external_artifact_checks"] = strict_external_artifact_checks
            calls["external_artifact_checker"] = external_artifact_checker

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
        "function_call_resolver": calls["function_call_resolver"],
        "max_ready_workers": 7,
        "strict_external_artifact_checks": False,
        "external_artifact_checker": None,
        "run_id": "run-1",
        "force": True,
        "closed": True,
    }
    assert callable(calls["function_call_resolver"])
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_orchestrator_run_passes_function_call_resolver(monkeypatch) -> None:
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
            function_call_resolver=None,
            max_ready_workers: int = 32,
            strict_external_artifact_checks: bool = False,
            external_artifact_checker=None,
        ) -> None:
            calls["function_call_resolver"] = function_call_resolver
            calls["max_ready_workers"] = max_ready_workers
            calls["strict_external_artifact_checks"] = strict_external_artifact_checks
            calls["external_artifact_checker"] = external_artifact_checker

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            calls["run_id"] = run_id
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    raw_cls, instance = _raw_orchestrator()
    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("demo"),
        run_id="run-1",
        max_ready_workers=9,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert callable(calls["function_call_resolver"])
    assert calls["max_ready_workers"] == 9
    assert calls["strict_external_artifact_checks"] is False
    assert calls["external_artifact_checker"] is None
    assert calls["run_id"] == "run-1"
    assert calls["closed"] is True
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_orchestrator_run_passes_external_artifact_checker(monkeypatch) -> None:
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
            function_call_resolver=None,
            max_ready_workers: int = 32,
            strict_external_artifact_checks: bool = False,
            external_artifact_checker=None,
        ) -> None:
            calls["strict_external_artifact_checks"] = strict_external_artifact_checks
            calls["external_artifact_checker"] = external_artifact_checker

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)

    def external_checker(_artifact) -> list[str]:
        return []

    raw_cls, instance = _raw_orchestrator()
    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("demo"),
        run_id="run-1",
        strict_external_artifact_checks=True,
        external_artifact_checker=external_checker,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["strict_external_artifact_checks"] is True
    assert calls["external_artifact_checker"] is external_checker
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


def test_orchestrator_exit_preserves_active_remote_calls(monkeypatch) -> None:
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

    assert runtime.cancel_count == 0
    assert runtime.close_count == 1
    assert instance._runtime is None
    assert volume.commit_count == 2


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
