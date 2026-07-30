"""Tests for the workflow coordinator's Modal boundary."""

# ruff: noqa: D101,D102,D103,D107

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionRunNotFoundError,
    ProviderBinding,
    RunStatus,
    RunStatusReason,
)
from biomodals.helper.constant import WORKFLOW_ORCHESTRATOR_VOLUME_NAME
from biomodals.schema import AppOutput, AppRunResult, AppRunStatus, ArtifactKind
from biomodals.schema.storage import InlineBytes
from biomodals.workflow import Workflow, WorkflowNativeNode
from biomodals.workflow.core import orchestrator
from biomodals.workflow.core.nodes import NodeRunContext
from biomodals.workflow.core.run_store import WorkflowRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "DemoWorkflow", 7)


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


class FakeHandle:
    def __init__(self) -> None:
        self.hydrated = False

    def hydrate(self) -> None:
        self.hydrated = True

    def remote(self, value: object) -> object:
        return value


@dataclass
class TextNode(WorkflowNativeNode):
    text: str

    def run(self, _context: NodeRunContext) -> AppRunResult:
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="result",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=self.text.encode(),
                        filename="result.txt",
                        media_type="text/plain",
                    ),
                )
            ],
        )


def _raw_coordinator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    volume: FakeVolume,
    *,
    execution_run_id: str = str(RUN_ID),
    deployment_environment: str = DEPLOYMENT.environment,
    deployment_name: str = DEPLOYMENT.deployment_name,
    deployment_version: int = DEPLOYMENT.deployment_version,
) -> tuple[Any, Any]:
    monkeypatch.setattr(orchestrator, "OUT_VOLUME", volume)
    monkeypatch.setattr(
        orchestrator.CONF,
        "output_volume_mountpoint",
        str(tmp_path),
    )
    raw_cls = cast(Any, orchestrator.ExecutionCoordinator)._get_user_cls()
    instance = raw_cls()
    instance.execution_run_id = execution_run_id
    instance.deployment_environment = deployment_environment
    instance.deployment_name = deployment_name
    instance.deployment_version = deployment_version
    raw_cls.enter._get_raw_f()(instance)
    return raw_cls, instance


def test_coordinator_binds_parameterized_identity_and_persists_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs
            self.store = WorkflowRunStore(tmp_path, RUN_ID)

        def run(self, *, workload_run_key: str, synchronize: object) -> AppRunResult:
            calls["workload_run_key"] = workload_run_key
            calls["synchronize"] = synchronize
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_coordinator(monkeypatch, tmp_path, volume)
    workflow = Workflow("demo")

    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=workflow,
        workload_run_key="friendly-name",
        max_active_provider_calls=9,
        max_active_gpu_provider_calls=3,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    init = cast(dict[str, object], calls["init"])
    assert init["workflow"] is workflow
    assert init["execution_run_id"] == RUN_ID
    assert init["deployment"] == DEPLOYMENT
    assert init["volume_root"] == tmp_path
    assert init["workflow_volume_name"] == WORKFLOW_ORCHESTRATOR_VOLUME_NAME
    assert init["workflow_volume"] is volume
    assert init["max_active_provider_calls"] == 9
    assert init["max_active_gpu_provider_calls"] == 3
    assert calls["workload_run_key"] == "friendly-name"
    assert calls["closed"] is True
    assert volume.reload_count == 1
    assert volume.commit_count == 2

    store = WorkflowRunStore(tmp_path, RUN_ID)
    plan = pickle.loads(store.read_workflow_plan())  # noqa: S301
    assert isinstance(plan, orchestrator.WorkflowCoordinatorPlan)
    assert plan.workflow is not workflow
    assert (
        plan.identity
        == orchestrator.WorkflowCoordinatorPlan(
            workflow,
            "friendly-name",
            9,
            3,
        ).identity
    )


def test_coordinator_rejects_a_changed_plan_for_the_same_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()

    class FakeRuntime:
        def __init__(self, **_kwargs: object) -> None:
            self.store = WorkflowRunStore(tmp_path, RUN_ID)

        def run(self, **_kwargs: object) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            pass

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_coordinator(monkeypatch, tmp_path, volume)
    raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("first"),
        workload_run_key="demo",
    )

    with pytest.raises(ValueError, match="does not match"):
        raw_cls.run._get_raw_f()(
            instance,
            workflow=Workflow("changed"),
            workload_run_key="demo",
        )


def test_coordinator_uses_explicit_handles_only_for_development_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)
            self.store = WorkflowRunStore(tmp_path, RUN_ID)

        def run(self, **_kwargs: object) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            pass

    handle = FakeHandle()
    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        deployment_environment="development",
        deployment_version=1,
    )
    raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("demo"),
        workload_run_key="demo",
        development_function_handles={"compute": handle},
    )

    driver = calls["modal_driver"]
    resolved = driver.resolve(
        ProviderBinding("development", "DemoWorkflow", 1, "compute", False)
    )
    assert resolved is handle
    assert handle.hydrated is True


def test_coordinator_resolves_persisted_external_checker_by_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()
    checker = FakeHandle()

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)
            self.store = WorkflowRunStore(tmp_path, RUN_ID)

        def run(self, **_kwargs: object) -> AppRunResult:
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            pass

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        deployment_environment="development",
        deployment_version=1,
    )
    raw_cls.run._get_raw_f()(
        instance,
        workflow=Workflow("demo"),
        workload_run_key="demo",
        strict_external_artifact_checks=True,
        external_artifact_checker_function_name="check_external",
        development_function_handles={"check_external": checker},
    )

    resolved_checker = calls["external_artifact_checker"]
    assert resolved_checker.__self__ is checker
    assert checker.hydrated is True
    assert calls["strict_external_artifact_checks"] is True


def test_status_and_terminal_cancel_are_read_only_kernel_views(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, instance = _raw_coordinator(monkeypatch, tmp_path, volume)
    workflow = Workflow("demo")
    workflow.add_node(TextNode("complete"), id="write")

    result = raw_cls.run._get_raw_f()(
        instance,
        workflow=workflow,
        workload_run_key="demo",
        development_function_handles={},
    )
    status = raw_cls.status._get_raw_f()(instance)
    cancelled = raw_cls.cancel._get_raw_f()(instance)

    assert result.status == AppRunStatus.SUCCEEDED
    assert status.run.status == RunStatus.SUCCEEDED
    assert cancelled.run.status == RunStatus.SUCCEEDED
    assert status.run.deployment == DEPLOYMENT


def test_status_does_not_create_state_for_an_unknown_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_cls, instance = _raw_coordinator(monkeypatch, tmp_path, FakeVolume())
    store = WorkflowRunStore(tmp_path, RUN_ID)

    with pytest.raises(ExecutionRunNotFoundError):
        raw_cls.status._get_raw_f()(instance)

    assert not store.ledger_path.exists()


def test_resume_reloads_the_persisted_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    volume = FakeVolume()
    plan = orchestrator.WorkflowCoordinatorPlan(
        workflow=Workflow("demo"),
        workload_run_key="friendly-name",
    )
    store = WorkflowRunStore(tmp_path, RUN_ID)
    store.write_workflow_plan(pickle.dumps(plan))
    with store.transaction():
        store.execution.create_run(
            execution_run_id=RUN_ID,
            plan=orchestrator.execution_plan(
                plan.workflow.validate(),
                workload_run_key=plan.workload_run_key,
            ),
            deployment=DEPLOYMENT,
            max_active_provider_calls=32,
            max_active_gpu_provider_calls=32,
            now=100,
        )
        store.execution.transition_run(
            RUN_ID,
            RunStatus.SUSPENDED,
            reason=RunStatusReason.COORDINATOR_ERROR,
            message="test suspension",
            now=101,
        )
    store.close()

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs
            self.store = WorkflowRunStore(tmp_path, RUN_ID)

        def resume(
            self,
            *,
            workload_run_key: str,
            synchronize: object,
        ) -> AppRunResult:
            calls["workload_run_key"] = workload_run_key
            calls["synchronize"] = synchronize
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(orchestrator, "WorkflowRuntime", FakeRuntime)
    raw_cls, instance = _raw_coordinator(monkeypatch, tmp_path, volume)

    result = raw_cls.resume._get_raw_f()(instance)

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls["workload_run_key"] == "friendly-name"
    assert cast(dict[str, object], calls["init"])["deployment"] == DEPLOYMENT
    assert calls["closed"] is True


def test_enter_and_exit_close_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, instance = _raw_coordinator(monkeypatch, tmp_path, volume)

    class FakeRuntime:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    active_runtime = FakeRuntime()
    instance._runtime = active_runtime
    raw_cls.exit._get_raw_f()(instance)
    raw_cls.exit._get_raw_f()(instance)

    assert active_runtime.close_count == 1
    assert instance._runtime is None
    assert volume.commit_count == 2


@pytest.mark.parametrize(
    ("execution_run_id", "deployment_version", "message"),
    [
        ("not-a-uuid", 7, "badly formed"),
        (str(RUN_ID), 0, "positive"),
    ],
)
def test_coordinator_rejects_invalid_parameterized_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    execution_run_id: str,
    deployment_version: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _raw_coordinator(
            monkeypatch,
            tmp_path,
            FakeVolume(),
            execution_run_id=execution_run_id,
            deployment_version=deployment_version,
        )


def test_coordinator_plan_rejects_invalid_workflow_or_limits() -> None:
    with pytest.raises(TypeError, match="Workflow object"):
        orchestrator.WorkflowCoordinatorPlan(
            cast(Any, {"nodes": []}),
            "demo",
        )
    with pytest.raises(ValueError, match="positive"):
        orchestrator.WorkflowCoordinatorPlan(Workflow("demo"), "demo", 0)
    with pytest.raises(ValueError, match="between zero"):
        orchestrator.WorkflowCoordinatorPlan(Workflow("demo"), "demo", 2, 3)
    with pytest.raises(ValueError, match="checker function"):
        orchestrator.WorkflowCoordinatorPlan(
            Workflow("demo"),
            "demo",
            strict_external_artifact_checks=True,
        )


def test_orchestrator_modal_app_exposes_standard_coordinator_surface() -> None:
    functions = orchestrator.app._local_state.functions

    assert orchestrator.CONF.python_version == "3.13"
    assert orchestrator.OUT_VOLUME_NAME == WORKFLOW_ORCHESTRATOR_VOLUME_NAME
    assert "ExecutionCoordinator.*" in functions
    assert "WorkflowOrchestrator.*" not in functions
    assert "run_workflow_orchestrator" not in functions
