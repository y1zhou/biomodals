"""Tests for the workflow coordinator's Modal boundary."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from biomodals.execution import DeploymentIdentity, ProviderBinding
from biomodals.helper.constant import WORKFLOW_ORCHESTRATOR_VOLUME_NAME
from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow import Workflow
from biomodals.workflow.core import orchestrator

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


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


def test_orchestrator_binds_exact_execution_identity_and_call_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def run(self, *, workload_run_key: str) -> AppRunResult:
            calls["workload_run_key"] = workload_run_key
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_orchestrator()
    workflow = Workflow("demo")

    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=workflow,
        execution_run_id=str(RUN_ID),
        workload_run_key="friendly-name",
        deployment_environment="main",
        deployment_name="DemoWorkflow",
        deployment_version=7,
        max_active_provider_calls=9,
        max_active_gpu_provider_calls=3,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["init"] == {
        "workflow": workflow,
        "execution_run_id": RUN_ID,
        "deployment": DeploymentIdentity("main", "DemoWorkflow", 7),
        "volume_root": Path(orchestrator.CONF.output_volume_mountpoint),
        "workflow_volume_name": WORKFLOW_ORCHESTRATOR_VOLUME_NAME,
        "workflow_volume": volume,
        "modal_driver": None,
        "max_active_provider_calls": 9,
        "max_active_gpu_provider_calls": 3,
        "strict_external_artifact_checks": False,
        "external_artifact_checker": None,
    }
    assert calls["workload_run_key"] == "friendly-name"
    assert calls["closed"] is True
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_orchestrator_uses_explicit_handles_only_for_development_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeHandle:
        def __init__(self) -> None:
            self.hydrated = False

        def hydrate(self) -> None:
            self.hydrated = True

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)

        def run(self, *, workload_run_key: str) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            pass

    handle = FakeHandle()
    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_orchestrator()
    raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("demo"),
        execution_run_id=str(RUN_ID),
        workload_run_key="demo",
        deployment_environment="development",
        deployment_name="DemoWorkflow",
        deployment_version=1,
        development_function_handles={"compute": handle},
    )

    driver = calls["modal_driver"]
    resolved = driver.resolve(
        ProviderBinding("development", "DemoWorkflow", 1, "compute", False)
    )
    assert resolved is handle
    assert handle.hydrated is True


def test_orchestrator_passes_external_artifact_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)

        def run(self, *, workload_run_key: str) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            pass

    def external_checker(_artifact: object) -> list[str]:
        return []

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_orchestrator()
    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("demo"),
        execution_run_id=str(RUN_ID),
        workload_run_key="demo",
        deployment_environment="main",
        deployment_name="DemoWorkflow",
        deployment_version=7,
        strict_external_artifact_checks=True,
        external_artifact_checker=external_checker,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["strict_external_artifact_checks"] is True
    assert calls["external_artifact_checker"] is external_checker


def test_orchestrator_enter_and_exit_close_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = FakeVolume()
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)

    class FakeRuntime:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    raw_cls, instance = _raw_orchestrator()
    stale_runtime = FakeRuntime()
    instance._runtime = stale_runtime

    raw_cls.enter._get_raw_f()(instance)

    assert stale_runtime.close_count == 1
    assert instance._runtime is None
    assert volume.reload_count == 1

    active_runtime = FakeRuntime()
    instance._runtime = active_runtime
    raw_cls.exit._get_raw_f()(instance)
    raw_cls.exit._get_raw_f()(instance)

    assert active_runtime.close_count == 1
    assert instance._runtime is None
    assert volume.commit_count == 2


@pytest.mark.parametrize(
    ("workflow", "execution_run_id", "deployment_version", "message"),
    [
        ({"nodes": []}, str(RUN_ID), 7, "Workflow object"),
        (Workflow("demo"), "not-a-uuid", 7, "badly formed"),
        (Workflow("demo"), str(RUN_ID), 0, "positive"),
    ],
)
def test_orchestrator_rejects_invalid_identity(
    workflow: object,
    execution_run_id: str,
    deployment_version: int,
    message: str,
) -> None:
    raw_cls, instance = _raw_orchestrator()

    with pytest.raises((TypeError, ValueError), match=message):
        raw_cls.run._get_raw_f()(
            instance,
            workflow=workflow,
            execution_run_id=execution_run_id,
            workload_run_key="demo",
            deployment_environment="main",
            deployment_name="DemoWorkflow",
            deployment_version=deployment_version,
        )


def test_orchestrator_modal_app_exposes_only_class_remote_surface() -> None:
    functions = orchestrator.app._local_state.functions

    assert orchestrator.CONF.python_version == "3.13"
    assert orchestrator.OUT_VOLUME_NAME == WORKFLOW_ORCHESTRATOR_VOLUME_NAME
    assert "WorkflowOrchestrator.*" in functions
    assert "run_workflow_orchestrator" not in functions
    assert "run_remote_workflow_node" not in functions
