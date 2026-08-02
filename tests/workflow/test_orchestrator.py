"""Tests for the workflow coordinator's Modal boundary."""

# ruff: noqa: D101,D102,D103,D107

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionRunNotFoundError,
    NodeAggregationPolicy,
    NodeStatus,
    ProviderBinding,
    ResultProvenance,
    RunStatus,
    RunStatusReason,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.helper.constant import WORKFLOW_ORCHESTRATOR_VOLUME_NAME
from biomodals.schema import AppOutput, AppRunResult, AppRunStatus, ArtifactKind
from biomodals.schema.storage import InlineBytes
from biomodals.workflow import Workflow, WorkflowNativeNode
from biomodals.workflow.core import orchestrator
from biomodals.workflow.core.nodes import (
    NodeRunContext,
    RemoteNodeCall,
    RemoteTaskWorkflowNode,
    RemoteWorkflowTask,
)
from biomodals.workflow.core.run_store import WorkflowRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SUCCESSOR_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
DEPLOYMENT = DeploymentIdentity("main", "DemoWorkflow", 7)
SUCCESSOR_DEPLOYMENT = DeploymentIdentity("main", "DemoWorkflow", 8)


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
    workers: int = field(default=1, metadata={"dag_hash": False})

    def run(self, context: NodeRunContext) -> AppRunResult:
        del context
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


@dataclass
class FanoutNode(RemoteTaskWorkflowNode):
    texts: tuple[str, ...]

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        del context
        return tuple(
            RemoteWorkflowTask(
                task_key=f"candidate-{ordinal}",
                scientific_payload={"text": text},
                execution_payload={"text": text},
            )
            for ordinal, text in enumerate(self.texts)
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        del context
        return RemoteNodeCall(
            function_name="run_candidate",
            uses_gpu=False,
            kwargs={
                "task_key": task.task_key,
                "text": task.execution_payload["text"],
            },
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results,
        errors,
    ) -> AppRunResult:
        del context
        status = (
            AppRunStatus.PARTIAL
            if results and errors
            else AppRunStatus.SUCCEEDED
            if not errors
            else AppRunStatus.FAILED
        )
        return AppRunResult(
            status=status,
            outputs=[
                AppOutput(
                    name="summary",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=",".join(results).encode(),
                        filename="summary.txt",
                        media_type="text/plain",
                    ),
                )
            ],
        )


class FanoutDriver:
    def __init__(self, *, failing_tasks: set[str] | None = None) -> None:
        self.failing_tasks = failing_tasks or set()
        self.spawned: list[str] = []
        self.results: dict[str, AppRunResult] = {}

    def resolve(self, binding: ProviderBinding) -> str:
        return binding.function_name

    def spawn(self, function, *, args, kwargs) -> str:
        del function, args
        task_key = str(kwargs["task_key"])
        self.spawned.append(task_key)
        call_id = f"fc-{task_key}"
        self.results[call_id] = TextNode(str(kwargs["text"])).run(
            cast(NodeRunContext, None)
        )
        return call_id

    def observe(self, provider_call_handle_id: str) -> ModalCallObservation:
        task_key = provider_call_handle_id.removeprefix("fc-")
        if task_key in self.failing_tasks:
            return ModalCallObservation(
                ModalCallObservationKind.FAILED,
                message=f"{task_key} failed",
            )
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=self.results[provider_call_handle_id],
        )

    def cancel(self, provider_call_handle_id: str) -> None:
        del provider_call_handle_id


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
        max_parallel_nodes=4,
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
    assert init["max_parallel_nodes"] == 4
    assert init["max_active_provider_calls"] == 9
    assert init["max_active_gpu_provider_calls"] == 3
    assert calls["workload_run_key"] == "friendly-name"
    assert calls["closed"] is True
    assert volume.reload_count == 1
    assert volume.commit_count == 1

    store = WorkflowRunStore(tmp_path, RUN_ID)
    plan = pickle.loads(store.read_workflow_plan())  # noqa: S301
    assert isinstance(plan, orchestrator.WorkflowCoordinatorPlan)
    assert plan.workflow is not workflow
    assert (
        plan.identity
        == orchestrator.WorkflowCoordinatorPlan(
            workflow=workflow,
            workload_run_key="friendly-name",
            max_parallel_nodes=4,
            max_active_provider_calls=9,
            max_active_gpu_provider_calls=3,
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

    driver = cast(Any, calls["modal_driver"])
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

    resolved_checker = cast(Any, calls["external_artifact_checker"])
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


def test_restart_creates_an_idempotent_successor_from_cached_publications(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    workflow = Workflow("demo")
    workflow.add_node(TextNode("complete"), id="write")
    raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=workflow,
        workload_run_key="demo",
        development_function_handles={},
    )
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )

    first = raw_cls.restart._get_raw_f()(
        successor_coordinator,
        predecessor_execution_run_id=str(RUN_ID),
        predecessor_deployment_environment=DEPLOYMENT.environment,
        predecessor_deployment_name=DEPLOYMENT.deployment_name,
        predecessor_deployment_version=DEPLOYMENT.deployment_version,
    )
    second = raw_cls.restart._get_raw_f()(
        successor_coordinator,
        predecessor_execution_run_id=str(RUN_ID),
        predecessor_deployment_environment=DEPLOYMENT.environment,
        predecessor_deployment_name=DEPLOYMENT.deployment_name,
        predecessor_deployment_version=DEPLOYMENT.deployment_version,
    )

    assert first.status == AppRunStatus.SUCCEEDED
    assert second.status == AppRunStatus.SUCCEEDED
    store = WorkflowRunStore(tmp_path, SUCCESSOR_ID)
    successor = store.execution.get_run(SUCCESSOR_ID)
    node = store.execution.get_node(SUCCESSOR_ID, "write")
    publication = store.artifacts.load_node_output_artifacts("write")
    assert successor.predecessor_execution_run_id == RUN_ID
    assert successor.deployment == SUCCESSOR_DEPLOYMENT
    assert successor.status == RunStatus.SUCCEEDED
    assert node.status == NodeStatus.SUCCEEDED
    assert node.result_provenance == ResultProvenance.CACHE
    assert str(RUN_ID) in publication[0].storage.path
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM workflow_node_results"
        ).fetchone()[0]
        == 1
    )
    store.close()


def test_launch_restart_prepares_candidate_before_driving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    workflow = Workflow("demo")
    workflow.add_node(TextNode("complete", workers=8), id="write")
    raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=workflow,
        workload_run_key="demo",
        max_parallel_nodes=7,
        max_active_provider_calls=8,
        max_active_gpu_provider_calls=4,
        development_function_handles={},
    )
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )
    candidate_workflow = Workflow("demo")
    candidate_workflow.add_node(TextNode("complete", workers=2), id="write")

    raw_cls.prepare_restart_from._get_raw_f()(
        successor_coordinator,
        predecessor_execution_run_id=str(RUN_ID),
        workflow=candidate_workflow,
        workload_run_key="demo",
        max_parallel_nodes=1,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=2,
    )

    store = WorkflowRunStore(tmp_path, SUCCESSOR_ID)
    successor = store.execution.get_run(SUCCESSOR_ID)
    assert successor.predecessor_execution_run_id == RUN_ID
    assert successor.deployment == SUCCESSOR_DEPLOYMENT
    assert successor.max_active_provider_calls == 3
    assert successor.max_active_gpu_provider_calls == 2
    successor_plan = pickle.loads(store.read_workflow_plan())  # noqa: S301
    assert successor_plan.max_parallel_nodes == 1
    successor_node = successor_plan.workflow.validate().nodes["write"].node
    assert successor_node.workers == 2
    assert getattr(successor_coordinator, "_runtime", None) is None
    store.close()

    result = raw_cls.drive_prepared._get_raw_f()(successor_coordinator)

    assert result.status == AppRunStatus.SUCCEEDED


def test_generic_restart_prepares_successor_before_driving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    workflow = Workflow("demo")
    workflow.add_node(TextNode("complete"), id="write")
    raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=workflow,
        workload_run_key="demo",
        development_function_handles={},
    )
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )

    raw_cls.prepare_restart._get_raw_f()(
        successor_coordinator,
        predecessor_execution_run_id=str(RUN_ID),
        predecessor_deployment_environment=DEPLOYMENT.environment,
        predecessor_deployment_name=DEPLOYMENT.deployment_name,
        predecessor_deployment_version=DEPLOYMENT.deployment_version,
    )

    store = WorkflowRunStore(tmp_path, SUCCESSOR_ID)
    assert store.execution.get_run(SUCCESSOR_ID).predecessor_execution_run_id == RUN_ID
    assert getattr(successor_coordinator, "_runtime", None) is None
    store.close()

    result = raw_cls.drive_prepared._get_raw_f()(successor_coordinator)

    assert result.status == AppRunStatus.SUCCEEDED


def test_launch_restart_rejects_changed_scientific_plan_before_creating_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    predecessor_workflow = Workflow("demo")
    predecessor_workflow.add_node(TextNode("original"), id="write")
    raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=predecessor_workflow,
        workload_run_key="demo",
        development_function_handles={},
    )
    candidate_workflow = Workflow("demo")
    candidate_workflow.add_node(TextNode("changed"), id="write")
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )

    with pytest.raises(ValueError, match="Workload Plan Fingerprint"):
        raw_cls.prepare_restart_from._get_raw_f()(
            successor_coordinator,
            predecessor_execution_run_id=str(RUN_ID),
            workflow=candidate_workflow,
            workload_run_key="demo",
        )

    assert not WorkflowRunStore(tmp_path, SUCCESSOR_ID).ledger_path.exists()


def test_launch_restart_rejects_changed_workload_run_key_before_creating_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    workflow = Workflow("demo")
    workflow.add_node(TextNode("unchanged"), id="write")
    raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=workflow,
        workload_run_key="original",
        development_function_handles={},
    )
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )

    with pytest.raises(ValueError, match="Workload Run Key"):
        raw_cls.prepare_restart_from._get_raw_f()(
            successor_coordinator,
            predecessor_execution_run_id=str(RUN_ID),
            workflow=workflow,
            workload_run_key="changed",
        )

    successor_store = WorkflowRunStore(tmp_path, SUCCESSOR_ID)
    assert not successor_store.ledger_path.exists()
    assert not successor_store.workflow_plan_path.exists()


def test_restart_reuses_successful_task_publications_from_partial_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    predecessor_driver = FanoutDriver(failing_tasks={"candidate-1"})
    monkeypatch.setattr(
        orchestrator,
        "ModalCallDriver",
        lambda **_kwargs: predecessor_driver,
    )
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    workflow = Workflow("fanout")
    workflow.add_node(
        FanoutNode(("alpha", "beta")),
        id="fanout",
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )

    predecessor_result = raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=workflow,
        workload_run_key="fanout",
    )

    assert predecessor_result.status == AppRunStatus.PARTIAL
    assert predecessor_driver.spawned == ["candidate-0", "candidate-1"]

    successor_driver = FanoutDriver()
    monkeypatch.setattr(
        orchestrator,
        "ModalCallDriver",
        lambda **_kwargs: successor_driver,
    )
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )

    successor_result = raw_cls.restart._get_raw_f()(
        successor_coordinator,
        predecessor_execution_run_id=str(RUN_ID),
        predecessor_deployment_environment=DEPLOYMENT.environment,
        predecessor_deployment_name=DEPLOYMENT.deployment_name,
        predecessor_deployment_version=DEPLOYMENT.deployment_version,
    )

    assert successor_result.status == AppRunStatus.SUCCEEDED
    assert successor_driver.spawned == ["candidate-1"]
    store = WorkflowRunStore(tmp_path, SUCCESSOR_ID)
    cached = store.execution.get_task(SUCCESSOR_ID, "fanout", "candidate-0")
    repaired = store.execution.get_task(SUCCESSOR_ID, "fanout", "candidate-1")
    assert cached.result_provenance == ResultProvenance.CACHE
    assert repaired.result_provenance == ResultProvenance.CURRENT_RUN
    assert store.execution.get_node(SUCCESSOR_ID, "fanout").status == (
        NodeStatus.SUCCEEDED
    )
    store.close()


def test_restart_recomputes_a_missing_predecessor_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    workflow = Workflow("demo")
    workflow.add_node(TextNode("replacement"), id="write")
    raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=workflow,
        workload_run_key="demo",
        development_function_handles={},
    )
    predecessor_store = WorkflowRunStore(tmp_path, RUN_ID)
    publication = predecessor_store.artifacts.load_node_output_artifacts("write")[0]
    predecessor_store.close()
    (tmp_path / publication.storage.path).unlink()
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )

    result = raw_cls.restart._get_raw_f()(
        successor_coordinator,
        predecessor_execution_run_id=str(RUN_ID),
        predecessor_deployment_environment=DEPLOYMENT.environment,
        predecessor_deployment_name=DEPLOYMENT.deployment_name,
        predecessor_deployment_version=DEPLOYMENT.deployment_version,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    successor_store = WorkflowRunStore(tmp_path, SUCCESSOR_ID)
    node = successor_store.execution.get_node(SUCCESSOR_ID, "write")
    task = successor_store.execution.get_task(SUCCESSOR_ID, "write", "node")
    publication = successor_store.artifacts.load_node_output_artifacts("write")[0]
    assert node.status == NodeStatus.SUCCEEDED
    assert task.result_provenance == ResultProvenance.CURRENT_RUN
    assert str(SUCCESSOR_ID) in publication.storage.path
    successor_store.close()


def test_restart_rejects_a_mismatched_predecessor_deployment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = FakeVolume()
    raw_cls, predecessor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
    )
    raw_cls.run._get_raw_f()(
        predecessor_coordinator,
        workflow=Workflow("demo"),
        workload_run_key="demo",
        development_function_handles={},
    )
    raw_cls, successor_coordinator = _raw_coordinator(
        monkeypatch,
        tmp_path,
        volume,
        execution_run_id=str(SUCCESSOR_ID),
        deployment_version=SUCCESSOR_DEPLOYMENT.deployment_version,
    )

    with pytest.raises(ValueError, match="Predecessor Deployment Identity"):
        raw_cls.restart._get_raw_f()(
            successor_coordinator,
            predecessor_execution_run_id=str(RUN_ID),
            predecessor_deployment_environment=DEPLOYMENT.environment,
            predecessor_deployment_name=DEPLOYMENT.deployment_name,
            predecessor_deployment_version=DEPLOYMENT.deployment_version + 1,
        )


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
    assert volume.commit_count == 0


def test_workflow_cancel_does_not_start_a_second_driver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Concurrent cancellation leaves exactly one workflow drive owner."""
    raw_cls, instance = _raw_coordinator(monkeypatch, tmp_path, FakeVolume())
    plan = orchestrator.WorkflowCoordinatorPlan(Workflow("demo"), "demo")

    class FakeRuntime:
        def __init__(self) -> None:
            self.started = Event()
            self.cancel_requested = Event()
            self.status = RunStatus.RUNNING
            self.active_drivers = 0
            self.max_active_drivers = 0
            self.closed_while_driving = False
            self._lock = Lock()

        def run(self, **_kwargs: object) -> AppRunResult:
            with self._lock:
                self.active_drivers += 1
                self.max_active_drivers = max(
                    self.max_active_drivers,
                    self.active_drivers,
                )
            self.started.set()
            assert self.cancel_requested.wait(timeout=1)
            sleep(0.05)
            self.status = RunStatus.CANCELLED
            with self._lock:
                self.active_drivers -= 1
            return AppRunResult(status=AppRunStatus.FAILED)

        def cancel(self) -> None:
            self.status = RunStatus.CANCEL_REQUESTED
            self.cancel_requested.set()

        def close(self) -> None:
            with self._lock:
                self.closed_while_driving |= self.active_drivers > 0

    runtime = FakeRuntime()

    def open_runtime(*_args: object, **_kwargs: object) -> FakeRuntime:
        instance._runtime = runtime
        return runtime

    def snapshot():
        return SimpleNamespace(
            run=SimpleNamespace(
                execution_run_id=RUN_ID,
                deployment=DEPLOYMENT,
                status=runtime.status,
            )
        )

    instance._persist_or_verify_plan = lambda candidate: candidate
    instance._load_plan = lambda: plan
    instance._open_runtime = open_runtime
    instance._require_ledger = lambda: None
    instance._verified_snapshot = snapshot
    errors: list[BaseException] = []

    def call(operation, *args: object, **kwargs: object) -> None:
        try:
            operation(instance, *args, **kwargs)
        except BaseException as error:  # pragma: no cover - assertion aid
            errors.append(error)

    run_thread = Thread(
        target=call,
        args=(raw_cls.run._get_raw_f(),),
        kwargs={"workflow": plan.workflow, "workload_run_key": "demo"},
    )
    run_thread.start()
    assert runtime.started.wait(timeout=1)
    cancel_thread = Thread(target=call, args=(raw_cls.cancel._get_raw_f(),))
    cancel_thread.start()
    run_thread.join(timeout=2)
    cancel_thread.join(timeout=2)

    assert not run_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert errors == []
    assert runtime.max_active_drivers == 1
    assert not runtime.closed_while_driving


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
    with pytest.raises(ValueError, match="max_parallel_nodes"):
        orchestrator.WorkflowCoordinatorPlan(
            Workflow("demo"),
            "demo",
            max_parallel_nodes=0,
        )
    with pytest.raises(ValueError, match="between zero"):
        orchestrator.WorkflowCoordinatorPlan(Workflow("demo"), "demo", 2, 3)
    with pytest.raises(ValueError, match="checker function"):
        orchestrator.WorkflowCoordinatorPlan(
            Workflow("demo"),
            "demo",
            strict_external_artifact_checks=True,
        )


def test_coordinator_handle_resolves_the_exact_deployed_class_version() -> None:
    calls: dict[str, object] = {}

    class FakeCoordinator:
        def __init__(self, **kwargs: object) -> None:
            calls["parameters"] = kwargs

    def resolve(*args: object, **kwargs: object) -> type[FakeCoordinator]:
        calls["lookup_args"] = args
        calls["lookup_kwargs"] = kwargs
        return FakeCoordinator

    handle = orchestrator.execution_coordinator_handle(
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        use_deployed_coordinator=True,
        class_resolver=resolve,
    )

    assert isinstance(handle, FakeCoordinator)
    assert calls["lookup_args"] == ("DemoWorkflow", "ExecutionCoordinator")
    assert calls["lookup_kwargs"] == {
        "environment_name": "main",
        "version": 7,
    }
    assert calls["parameters"] == {
        "execution_run_id": str(RUN_ID),
        "deployment_environment": "main",
        "deployment_name": "DemoWorkflow",
        "deployment_version": 7,
    }


def test_orchestrator_modal_app_exposes_standard_coordinator_surface() -> None:
    functions = orchestrator.app._local_state.functions

    assert orchestrator.CONF.python_version == "3.13"
    assert orchestrator.OUT_VOLUME_NAME == WORKFLOW_ORCHESTRATOR_VOLUME_NAME
    assert "ExecutionCoordinator.*" in functions
    assert "WorkflowOrchestrator.*" not in functions
    assert "run_workflow_orchestrator" not in functions
