"""Kernel-backed workflow runtime integration tests."""

# ruff: noqa: D101, D102, D103, D107

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from uuid import UUID

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    NodeAggregationPolicy,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    TaskStatus,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    InlineBytes,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.nodes import (
    NodeRunContext,
    RemoteNodeCall,
    RemotePullTaskWorkflowNode,
    RemotePullWorkerCall,
    RemoteTaskWorkflowNode,
    RemoteWorkflowNode,
    RemoteWorkflowTask,
    WorkflowNativeNode,
)
from biomodals.workflow.core.runtime import WorkflowRuntime

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "DemoWorkflow", 7)


@dataclass
class TextNode(WorkflowNativeNode):
    text: str
    seen: list[NodeRunContext] = field(
        default_factory=list,
        repr=False,
        metadata={"dag_hash": False},
    )

    def run(self, context: NodeRunContext) -> AppRunResult:
        self.seen.append(context)
        return _text_result(self.text)


@dataclass
class RemoteTextNode(RemoteWorkflowNode):
    text: str
    function_name: str
    uses_gpu: bool = False

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        return RemoteNodeCall(
            function_name=self.function_name,
            uses_gpu=self.uses_gpu,
            kwargs={"text": self.text},
            metadata={"node": context.node_id},
            runtime_image_key=("gpu" if self.uses_gpu else "cpu"),
        )


@dataclass
class RemoteFanoutNode(RemoteTaskWorkflowNode):
    texts: tuple[str, ...]
    prepare_calls: list[str] = field(
        default_factory=list,
        repr=False,
        metadata={"dag_hash": False},
    )
    finalized_results: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(
        default_factory=list,
        repr=False,
        metadata={"dag_hash": False},
    )

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
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
        self.prepare_calls.append(task.task_key)
        payload = dict(task.execution_payload)
        return RemoteNodeCall(
            function_name="run_candidate",
            uses_gpu=False,
            kwargs={
                "task_key": task.task_key,
                "text": payload["text"],
            },
            metadata={"task_key": context.task_key},
            runtime_image_key="fanout",
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        self.finalized_results.append((tuple(results), tuple(errors)))
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


@dataclass
class BatchedRemoteFanoutNode(RemoteFanoutNode):
    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
    ) -> RemoteNodeCall:
        payload = dict(task.execution_payload)
        return RemoteNodeCall(
            function_name="run_candidate_batch",
            uses_gpu=True,
            kwargs={"task_keys": [task.task_key], "texts": [payload["text"]]},
            metadata={"task_keys": [context.task_key]},
            runtime_image_key="fanout",
            compatibility_key="same-batch",
            max_tasks_per_call=2,
        )

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[RemoteWorkflowTask, ...],
    ) -> RemoteNodeCall:
        return RemoteNodeCall(
            function_name="run_candidate_batch",
            uses_gpu=True,
            kwargs={
                "task_keys": [task.task_key for task in tasks],
                "texts": [dict(task.execution_payload)["text"] for task in tasks],
            },
            metadata={"task_keys": [task.task_key for task in tasks]},
            runtime_image_key="fanout",
            compatibility_key="same-batch",
            max_tasks_per_call=2,
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: object,
        metadata: Mapping[str, object],
    ) -> Mapping[str, AppRunResult]:
        assert metadata["task_keys"] == list(task_keys)
        if not isinstance(result, Mapping):
            raise TypeError("batch result must be a mapping")
        result_by_task = cast(Mapping[str, object], result)
        return {
            task_key: AppRunResult.model_validate(result_by_task[task_key])
            for task_key in task_keys
        }


@dataclass
class PullFanoutNode(RemotePullTaskWorkflowNode):
    texts: tuple[str, ...]
    publication_observation: AvailabilityStatus | None = field(
        default=None,
        repr=False,
        metadata={"dag_hash": False},
    )
    publication_probes: list[str] = field(
        default_factory=list,
        repr=False,
        metadata={"dag_hash": False},
    )
    finalized_results: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(
        default_factory=list,
        repr=False,
        metadata={"dag_hash": False},
    )

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[RemoteWorkflowTask, ...]:
        return tuple(
            RemoteWorkflowTask(
                task_key=f"candidate-{ordinal}",
                scientific_payload={"text": text},
                execution_payload={"text": text},
            )
            for ordinal, text in enumerate(self.texts)
        )

    def prepare_pull_worker(
        self,
        context: NodeRunContext,
    ) -> RemotePullWorkerCall:
        return RemotePullWorkerCall(
            function_name="run_pull_worker",
            uses_gpu=False,
            claim_capacity=2,
            kwargs={"node_id": context.node_id},
            runtime_image_key="pull-cpu",
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[WorkflowArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, expected_fingerprint, result, artifacts
        self.publication_probes.append(task.task_key)
        return self.publication_observation

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        self.finalized_results.append((tuple(results), tuple(errors)))
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED if not errors else AppRunStatus.PARTIAL,
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


class FakeModalDriver:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.results: dict[str, object] = {}
        self.cancelled: list[str] = []

    def resolve(self, binding):
        self.events.append(
            "resolve:"
            f"{binding.environment}/{binding.app_name}/"
            f"{binding.app_version}/{binding.function_name}"
        )
        return binding.function_name

    def spawn(self, function, *, args, kwargs):
        call_id = f"fc-{function}"
        self.events.append(f"spawn:{function}")
        self.results[call_id] = _text_result(str(kwargs["text"]))
        return call_id

    def observe(self, provider_call_handle_id):
        self.events.append(f"observe:{provider_call_handle_id}")
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=self.results[provider_call_handle_id],
        )

    def cancel(self, provider_call_handle_id):
        self.cancelled.append(provider_call_handle_id)


class CancellingModalDriver(FakeModalDriver):
    def observe(self, provider_call_handle_id):
        self.events.append(f"observe:{provider_call_handle_id}")
        if provider_call_handle_id in self.cancelled:
            return ModalCallObservation(ModalCallObservationKind.CANCELLED)
        return ModalCallObservation(ModalCallObservationKind.RUNNING)


class StateUnknownUntilCancelledModalDriver(CancellingModalDriver):
    def observe(self, provider_call_handle_id):
        self.events.append(f"observe:{provider_call_handle_id}")
        if provider_call_handle_id in self.cancelled:
            return ModalCallObservation(ModalCallObservationKind.CANCELLED)
        return ModalCallObservation(ModalCallObservationKind.STATE_UNKNOWN)


class FanoutModalDriver(FakeModalDriver):
    def __init__(self, *, failing_tasks: set[str] | None = None) -> None:
        super().__init__()
        self.failing_tasks = failing_tasks or set()

    def spawn(self, function, *, args, kwargs):
        task_key = str(kwargs["task_key"])
        call_id = f"fc-{task_key}"
        self.events.append(f"spawn:{task_key}")
        self.results[call_id] = _text_result(str(kwargs["text"]))
        return call_id

    def observe(self, provider_call_handle_id):
        task_key = provider_call_handle_id.removeprefix("fc-")
        self.events.append(f"observe:{task_key}")
        if task_key in self.failing_tasks:
            return ModalCallObservation(
                ModalCallObservationKind.FAILED,
                message=f"{task_key} failed",
            )
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=self.results[provider_call_handle_id],
        )


class PullModalDriver(FakeModalDriver):
    def __init__(self) -> None:
        super().__init__()
        self.spawn_kwargs: list[dict[str, object]] = []

    def spawn(self, function, *, args, kwargs):
        call_id = f"fc-pull-{len(self.spawn_kwargs)}"
        self.events.append(f"spawn:{function}")
        self.spawn_kwargs.append(dict(kwargs))
        self.results[call_id] = {"claimed_tasks": 0}
        return call_id


class BatchedFanoutModalDriver(FakeModalDriver):
    def spawn(self, function, *, args, kwargs):
        del function, args
        task_keys = tuple(kwargs["task_keys"])
        texts = tuple(kwargs["texts"])
        call_id = "fc-" + "-".join(task_keys)
        self.events.append("spawn:" + ",".join(task_keys))
        self.results[call_id] = {
            task_key: _text_result(str(text)).model_dump(mode="json")
            for task_key, text in zip(task_keys, texts, strict=True)
        }
        return call_id


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


class CoordinatorInterrupted(BaseException):
    """Model a hard coordinator interruption outside application handling."""


@dataclass
class InterruptOnceNode(WorkflowNativeNode):
    interrupted: bool = field(default=False, metadata={"dag_hash": False})
    calls: int = field(default=0, metadata={"dag_hash": False})

    def run(self, context: NodeRunContext) -> AppRunResult:
        self.calls += 1
        if not self.interrupted:
            self.interrupted = True
            raise CoordinatorInterrupted
        return _text_result("recovered")


@dataclass
class FailingNode(WorkflowNativeNode):
    calls: int = field(default=0, metadata={"dag_hash": False})

    def run(self, context: NodeRunContext) -> AppRunResult:
        self.calls += 1
        raise ValueError("scientific input is invalid")


def _text_result(text: str) -> AppRunResult:
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="text",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=text.encode(),
                    filename="result.txt",
                    media_type="text/plain",
                ),
            )
        ],
    )


def _runtime(
    tmp_path: Path,
    workflow: Workflow,
    *,
    driver: FakeModalDriver | None = None,
    volume: FakeVolume | None = None,
    max_parallel_nodes: int = 32,
    max_calls: int = 8,
    max_gpu_calls: int = 4,
    pull_worker_coordinator: object | None = None,
) -> WorkflowRuntime:
    return WorkflowRuntime(
        workflow=workflow,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
        modal_driver=driver,
        max_parallel_nodes=max_parallel_nodes,
        max_active_provider_calls=max_calls,
        max_active_gpu_provider_calls=max_gpu_calls,
        pull_worker_coordinator=pull_worker_coordinator,
        now=iter(range(100, 1000)).__next__,
        poll_interval_seconds=0,
    )


def test_initialization_reuses_the_host_volume_view(
    tmp_path: Path,
) -> None:
    workflow = Workflow("remote")
    workflow.add_node(
        RemoteTextNode("hello", "remote_text"),
        id="remote",
    )
    volume = FakeVolume()
    runtime = _runtime(tmp_path, workflow, volume=volume)

    runtime._initialize("friendly-name")

    assert volume.reloads == 0
    assert volume.commits == 0
    runtime.attach(workload_run_key="friendly-name")
    runtime.attach(workload_run_key="friendly-name")
    assert volume.reloads == 0
    runtime.refresh_publications(workload_run_key="friendly-name")
    assert volume.reloads == 1
    runtime.close()


def test_running_provider_poll_does_not_synchronize_the_workflow_volume(
    tmp_path: Path,
) -> None:
    workflow = Workflow("remote")
    workflow.add_node(
        RemoteTextNode("hello", "remote_text"),
        id="remote",
    )
    volume = FakeVolume()
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=CancellingModalDriver(),
        volume=volume,
    )
    runtime._initialize("friendly-name")
    runtime.advance_once()
    commits = volume.commits
    reloads = volume.reloads

    runtime.advance_once()

    assert volume.commits == commits
    assert volume.reloads == reloads
    runtime.close()


def test_provider_result_payload_is_file_backed_outside_the_ledger(
    tmp_path: Path,
) -> None:
    workflow = Workflow("remote-envelope")
    workflow.add_node(RemoteTextNode("large-return", "remote_text"), id="remote")
    runtime = _runtime(tmp_path, workflow, driver=FakeModalDriver())

    result = runtime.run(workload_run_key="remote-envelope")

    assert result.status == AppRunStatus.SUCCEEDED
    [call] = runtime.store.execution.list_provider_calls(RUN_ID)
    envelope = cast(dict[str, dict[str, object]], call.result_envelope)
    reference = envelope["result_file"]
    result_path = runtime.store.output_root / str(reference["path"])
    content = result_path.read_bytes()
    assert len(content) == reference["size_bytes"]
    assert b"bGFyZ2UtcmV0dXJu" in content
    assert b"bGFyZ2UtcmV0dXJu" not in runtime.store.ledger_path.read_bytes()
    runtime.close()


def test_provider_publication_reload_follows_success_observation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow = Workflow("remote")
    workflow.add_node(
        RemoteTextNode("hello", "remote_text"),
        id="remote",
    )
    driver = FakeModalDriver()
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=driver,
        volume=FakeVolume(),
    )
    runtime._initialize("friendly-name")
    runtime.advance_once()
    events: list[str] = []
    original_observe = driver.observe
    original_reload = runtime._volume_sync.reload
    original_publish = runtime._publish_provider_result

    def observe(provider_call_handle_id):
        events.append("observe")
        return original_observe(provider_call_handle_id)

    def reload() -> None:
        events.append("reload")
        original_reload()

    def publish(call) -> None:
        events.append("publish")
        original_publish(call)

    monkeypatch.setattr(driver, "observe", observe)
    monkeypatch.setattr(runtime._volume_sync, "reload", reload)
    monkeypatch.setattr(runtime, "_publish_provider_result", publish)

    runtime.advance_once()

    assert events == ["observe", "reload", "publish"]
    runtime.close()


def test_local_dag_uses_kernel_state_and_attempt_free_paths(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    first_node = TextNode("first")
    first = workflow.add_node(first_node, id="first")
    second_node = TextNode("second")
    workflow.add_node(
        second_node,
        id="second",
        inputs={"upstream": first.outputs(kind=ArtifactKind.REPORT)},
    )
    runtime = _runtime(tmp_path, workflow)

    result = runtime.run(workload_run_key="friendly-name")

    assert result.status == AppRunStatus.SUCCEEDED
    snapshot = runtime.store.execution.snapshot(RUN_ID)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert [node.status for node in snapshot.nodes] == [
        NodeStatus.SUCCEEDED,
        NodeStatus.SUCCEEDED,
    ]
    assert [task.status.value for task in snapshot.tasks] == [
        "succeeded",
        "succeeded",
    ]
    tables = {
        str(row[0])
        for row in runtime.store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "attempts" not in tables
    assert "remote_calls" not in tables
    assert first_node.seen[0].workload_run_key == "friendly-name"
    assert first_node.seen[0].execution_run_id == RUN_ID
    assert "attempt" not in str(first_node.seen[0].work_dir)
    assert [artifact.artifact_id for artifact in second_node.seen[0].inputs["upstream"]]
    assert (
        runtime.store.output_root.joinpath(
            "nodes",
            "second",
            "result",
            "second-text",
            "result.txt",
        ).read_text()
        == "second"
    )


def test_independent_remote_nodes_spawn_before_results_are_polled(
    tmp_path: Path,
) -> None:
    workflow = Workflow("parallel")
    workflow.add_node(
        RemoteTextNode("gpu", "run_gpu", uses_gpu=True),
        id="gpu",
    )
    workflow.add_node(RemoteTextNode("cpu", "run_cpu"), id="cpu")
    driver = FakeModalDriver()
    volume = FakeVolume()
    runtime = _runtime(tmp_path, workflow, driver=driver, volume=volume)

    result = runtime.run(workload_run_key="parallel")

    assert result.status == AppRunStatus.SUCCEEDED
    assert driver.events[:4] == [
        "resolve:main/DemoWorkflow/7/run_gpu",
        "resolve:main/DemoWorkflow/7/run_cpu",
        "spawn:run_gpu",
        "spawn:run_cpu",
    ]
    assert driver.events[4:] == [
        "observe:fc-run_gpu",
        "observe:fc-run_cpu",
    ]
    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    assert [call.status for call in calls] == [
        ProviderCallStatus.SUCCEEDED,
        ProviderCallStatus.SUCCEEDED,
    ]
    assert volume.commits > 0
    assert volume.reloads > 0


def test_node_parallelism_is_independent_from_provider_call_limits(
    tmp_path: Path,
) -> None:
    workflow = Workflow("node-limit")
    workflow.add_node(RemoteTextNode("first", "run_first"), id="first")
    workflow.add_node(RemoteTextNode("second", "run_second"), id="second")
    driver = CancellingModalDriver()
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=driver,
        max_parallel_nodes=1,
        max_calls=2,
        max_gpu_calls=0,
    )
    runtime._initialize("node-limit")

    runtime.advance_once()

    snapshot = runtime.store.execution.snapshot(RUN_ID)
    assert [node.status for node in snapshot.nodes] == [
        NodeStatus.RUNNING,
        NodeStatus.PENDING,
    ]
    assert len(snapshot.provider_calls) == 1


def test_cancel_requested_workflow_reconciles_provider_cancellation(
    tmp_path: Path,
) -> None:
    workflow = Workflow("cancel")
    workflow.add_node(RemoteTextNode("remote", "run_remote"), id="remote")
    driver = CancellingModalDriver()
    runtime = _runtime(tmp_path, workflow, driver=driver)
    runtime._initialize("cancel")
    runtime.advance_once()

    runtime.cancel()
    assert runtime.store.execution.get_run(RUN_ID).status == (
        RunStatus.CANCEL_REQUESTED
    )

    runtime.advance_once()

    snapshot = runtime.store.execution.snapshot(RUN_ID)
    assert snapshot.run.status == RunStatus.CANCELLED
    assert {call.status for call in snapshot.provider_calls} == {
        ProviderCallStatus.CANCELLED
    }


def test_unknown_workflow_prunes_call_after_terminal_publication_appears(
    tmp_path: Path,
) -> None:
    workflow = Workflow("unknown-pruning")
    workflow.add_node(RemoteTextNode("remote", "run_remote"), id="remote")
    driver = StateUnknownUntilCancelledModalDriver()
    runtime = _runtime(tmp_path, workflow, driver=driver)
    runtime._initialize("unknown-pruning")
    runtime.advance_once()
    call = runtime.store.execution.list_provider_calls(RUN_ID)[0]
    with runtime.store.transaction():
        runtime.store.execution.mark_provider_call_state_unknown(
            call.provider_call_id,
            message="Modal state lookup was inconclusive",
            now=11,
        )
    cached_path = runtime.store.output_root / "cached" / "result.txt"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text("cached")
    storage = VolumePath(
        volume_name="Workflow-outputs",
        path=cached_path.relative_to(tmp_path).as_posix(),
        media_type="text/plain",
    )
    with runtime.store.transaction():
        runtime.store.artifacts.record_node_publication(
            "remote",
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="text",
                        kind=ArtifactKind.REPORT,
                        storage=storage,
                    )
                ],
            ),
            artifacts=(
                WorkflowArtifact(
                    artifact_id="remote-text",
                    producing_node_id="remote",
                    kind=ArtifactKind.REPORT,
                    storage=storage,
                    files=[ArtifactFile(path="result.txt", size_bytes=6)],
                ),
            ),
            now=12,
        )
    runtime._checkpoint()

    result = runtime.resume(workload_run_key="unknown-pruning")

    assert result.status == AppRunStatus.SUCCEEDED
    assert driver.cancelled == ["fc-run_remote"]
    call = runtime.store.execution.list_provider_calls(RUN_ID)[0]
    assert call.status == ProviderCallStatus.CANCELLED


def test_remote_task_node_discovers_and_publishes_independent_tasks(
    tmp_path: Path,
) -> None:
    workflow = Workflow("fanout")
    node = RemoteFanoutNode(("alpha", "beta"))
    fanout = workflow.add_node(node, id="fanout")
    downstream = TextNode("joined")
    workflow.add_node(
        downstream,
        id="downstream",
        inputs={"candidates": fanout.outputs(kind=ArtifactKind.REPORT)},
    )
    driver = FanoutModalDriver()
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=driver,
        max_parallel_nodes=1,
        max_calls=2,
        max_gpu_calls=0,
    )

    result = runtime.run(workload_run_key="fanout")

    assert result.status == AppRunStatus.SUCCEEDED
    assert driver.events[:3] == [
        "resolve:main/DemoWorkflow/7/run_candidate",
        "spawn:candidate-0",
        "spawn:candidate-1",
    ]
    tasks = runtime.store.execution.list_tasks(RUN_ID, "fanout")
    assert [task.task_key for task in tasks] == ["candidate-0", "candidate-1"]
    assert [task.status for task in tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]
    assert len(runtime.store.execution.list_provider_calls(RUN_ID)) == 2
    assert node.finalized_results == [
        (("candidate-0", "candidate-1"), ()),
    ]
    task_artifact_ids = [
        runtime.store.artifacts.load_task_output_artifacts(
            "fanout",
            task.task_key,
        )[0].artifact_id
        for task in tasks
    ]
    assert len(set(task_artifact_ids)) == 2
    assert {
        artifact.artifact_id for artifact in downstream.seen[0].inputs["candidates"]
    }.issuperset(task_artifact_ids)


def test_remote_task_admission_reuses_persisted_dispatch_policy(
    tmp_path: Path,
) -> None:
    workflow = Workflow("bounded-fanout")
    node = RemoteFanoutNode(tuple(f"value-{index}" for index in range(20)))
    workflow.add_node(node, id="fanout")
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=FanoutModalDriver(),
        max_calls=2,
        max_gpu_calls=0,
    )
    runtime._initialize("bounded-fanout")

    runtime.advance_once()

    assert len(node.prepare_calls) == 22
    node.prepare_calls.clear()

    runtime.advance_once()

    assert node.prepare_calls == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    runtime.close()


def test_remote_task_node_batches_compatible_tasks_into_one_call(
    tmp_path: Path,
) -> None:
    workflow = Workflow("batched-fanout")
    node = BatchedRemoteFanoutNode(("alpha", "beta"))
    workflow.add_node(node, id="fanout")
    driver = BatchedFanoutModalDriver()
    runtime = _runtime(tmp_path, workflow, driver=driver)

    result = runtime.run(workload_run_key="batched-fanout")

    assert result.status == AppRunStatus.SUCCEEDED
    assert [event for event in driver.events if event.startswith("spawn:")] == [
        "spawn:candidate-0,candidate-1"
    ]
    [provider_call] = runtime.store.execution.list_provider_calls(RUN_ID)
    assert provider_call.task_keys == ("candidate-0", "candidate-1")
    assert [
        task.status for task in runtime.store.execution.list_tasks(RUN_ID, "fanout")
    ] == [TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED]
    assert node.finalized_results == [
        (("candidate-0", "candidate-1"), ()),
    ]


def test_pull_task_node_uses_durable_claims_and_worker_publications(
    tmp_path: Path,
) -> None:
    workflow = Workflow("pull-fanout")
    node = PullFanoutNode(("alpha", "beta", "gamma"))
    workflow.add_node(
        node,
        id="fanout",
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    driver = PullModalDriver()
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=driver,
        max_calls=2,
        max_gpu_calls=0,
        pull_worker_coordinator="run-pool",
        volume=(volume := FakeVolume()),
    )
    definition = workflow.validate()
    runtime._definition = definition
    runtime._workload_run_key = "pull-fanout"
    runtime._ensure_run(definition, "pull-fanout")

    runtime.advance_once()

    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    assert len(calls) == 2
    assert all(kwargs["coordinator"] == "run-pool" for kwargs in driver.spawn_kwargs)
    for call_index, call in enumerate(calls):
        claim = runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{call_index}",
            capacity=2,
        )
        commits = volume.commits
        runtime.complete_pull_tasks(
            call.provider_call_id,
            tuple(
                (
                    assignment.task_key,
                    f"complete-{assignment.task_key}",
                    _text_result(str(dict(assignment.execution_payload)["text"])),
                )
                for assignment in claim.assignments
            ),
        )
        assert volume.commits == commits + 1

    runtime.advance_once()

    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.SUCCEEDED
    assert [
        task.status for task in runtime.store.execution.list_tasks(RUN_ID, "fanout")
    ] == [TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED]
    assert node.finalized_results == [
        (("candidate-0", "candidate-1", "candidate-2"), ()),
    ]


def test_pull_task_node_uses_workload_publication_probe(tmp_path: Path) -> None:
    workflow = Workflow("pull-publication-probe")
    node = PullFanoutNode(
        ("alpha",),
        publication_observation=AvailabilityStatus.MISSING,
    )
    workflow.add_node(node, id="fanout")
    driver = PullModalDriver()
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=driver,
        max_calls=1,
        max_gpu_calls=0,
        pull_worker_coordinator="run-pool",
    )
    definition = workflow.validate()
    runtime._definition = definition
    runtime._workload_run_key = "pull-publication-probe"
    runtime._ensure_run(definition, "pull-publication-probe")
    runtime.advance_once()
    [call] = runtime.store.execution.list_provider_calls(RUN_ID)
    [assignment] = runtime.claim_pull_tasks(
        call.provider_call_id,
        request_id="claim",
        capacity=1,
    ).assignments

    task = runtime.complete_pull_task(
        call.provider_call_id,
        assignment.task_key,
        request_id="complete",
        result=_text_result("alpha"),
    )

    assert task.status == TaskStatus.FAILED
    assert node.publication_probes == ["candidate-0"]


def test_remote_task_node_can_publish_partial_outcomes(tmp_path: Path) -> None:
    workflow = Workflow("partial-fanout")
    node = RemoteFanoutNode(("alpha", "beta"))
    workflow.add_node(
        node,
        id="fanout",
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    driver = FanoutModalDriver(failing_tasks={"candidate-1"})
    runtime = _runtime(tmp_path, workflow, driver=driver)

    result = runtime.run(workload_run_key="partial-fanout")

    assert result.status == AppRunStatus.PARTIAL
    assert runtime.store.execution.get_node(RUN_ID, "fanout").status == (
        NodeStatus.PARTIAL
    )
    assert [
        task.status for task in runtime.store.execution.list_tasks(RUN_ID, "fanout")
    ] == [TaskStatus.SUCCEEDED, TaskStatus.FAILED]
    assert node.finalized_results == [
        (("candidate-0",), ("candidate-1",)),
    ]
    assert runtime.store.artifacts.load_task_result("fanout", "candidate-0") is not None
    assert runtime.store.artifacts.load_task_result("fanout", "candidate-1") is None


def test_remote_task_node_fail_fast_stops_unowned_siblings(tmp_path: Path) -> None:
    workflow = Workflow("fail-fast-fanout")
    node = RemoteFanoutNode(("alpha", "beta", "gamma"))
    workflow.add_node(
        node,
        id="fanout",
        aggregation_policy=NodeAggregationPolicy.FAIL_FAST,
    )
    driver = FanoutModalDriver(failing_tasks={"candidate-0"})
    runtime = _runtime(
        tmp_path,
        workflow,
        driver=driver,
        max_calls=1,
        max_gpu_calls=1,
    )

    result = runtime.run(workload_run_key="fail-fast-fanout")

    assert result.status == AppRunStatus.FAILED
    assert [event for event in driver.events if event.startswith("spawn:")] == [
        "spawn:candidate-0"
    ]
    assert [
        task.status for task in runtime.store.execution.list_tasks(RUN_ID, "fanout")
    ] == [TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.SKIPPED]
    assert node.finalized_results == [
        ((), ("candidate-0", "candidate-1", "candidate-2")),
    ]


def test_remote_task_node_requires_explicit_empty_publication(
    tmp_path: Path,
) -> None:
    workflow = Workflow("empty-fanout")
    node = RemoteFanoutNode(())
    workflow.add_node(node, id="fanout", allow_empty_result=True)
    driver = FanoutModalDriver()
    runtime = _runtime(tmp_path, workflow, driver=driver)

    result = runtime.run(workload_run_key="empty-fanout")

    assert result.status == AppRunStatus.SUCCEEDED
    assert runtime.store.execution.list_tasks(RUN_ID, "fanout") == ()
    assert node.finalized_results == [((), ())]
    assert not any(event.startswith("spawn:") for event in driver.events)
    assert runtime.store.artifacts.load_node_result("fanout") is not None


def test_durable_provider_envelope_is_published_without_another_spawn(
    tmp_path: Path,
) -> None:
    workflow = Workflow("recover")
    workflow.add_node(RemoteTextNode("answer", "run_remote"), id="remote")
    driver = FakeModalDriver()
    runtime = _runtime(tmp_path, workflow, driver=driver)
    definition = workflow.validate()
    runtime._definition = definition
    runtime._workload_run_key = "recover"
    runtime._ensure_run(definition, "recover")
    runtime.advance_once()
    runtime._provider.repository = runtime.store.execution
    ((_, completed),) = runtime._provider.reconcile_provider_calls(
        RUN_ID,
        required_node_keys={"remote"},
        encode_result=runtime._prepare_result_envelope,
        finalize_result=runtime._finalize_result_envelope,
        now=200,
    )
    assert completed.status == ProviderCallStatus.SUCCEEDED
    spawn_count = driver.events.count("spawn:run_remote")

    result = runtime.run(workload_run_key="recover")

    assert result.status == AppRunStatus.SUCCEEDED
    assert driver.events.count("spawn:run_remote") == spawn_count == 1
    assert runtime.store.artifacts.load_node_result("remote") is not None


def test_interrupted_local_task_recovers_without_an_attempt_or_new_task(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class RecordingVolume(FakeVolume):
        def commit(self) -> None:
            super().commit()
            events.append("checkpoint")

    class RecordingInterruptNode(InterruptOnceNode):
        def run(self, context: NodeRunContext) -> AppRunResult:
            events.append("run")
            return super().run(context)

    workflow = Workflow("local-recovery")
    node = RecordingInterruptNode()
    workflow.add_node(node, id="local")
    runtime = _runtime(tmp_path, workflow, volume=RecordingVolume())

    try:
        runtime.run(workload_run_key="local-recovery")
    except CoordinatorInterrupted:
        pass
    else:  # pragma: no cover - test must exercise the hard interruption
        raise AssertionError("Coordinator interruption was not raised")
    interrupted_task = runtime.store.execution.get_task(RUN_ID, "local", "node")
    assert interrupted_task.status.value == "running"
    assert interrupted_task.local_owned is True
    assert events == ["checkpoint", "run"]

    result = runtime.run(workload_run_key="local-recovery")

    assert result.status == AppRunStatus.SUCCEEDED
    recovered_task = runtime.store.execution.get_task(RUN_ID, "local", "node")
    assert recovered_task.status.value == "succeeded"
    assert recovered_task.started_at == interrupted_task.started_at
    assert node.calls == 2


def test_caught_local_failure_is_terminal_and_never_reentered(tmp_path: Path) -> None:
    workflow = Workflow("failure")
    node = FailingNode()
    workflow.add_node(node, id="local")
    runtime = _runtime(tmp_path, workflow)

    first = runtime.run(workload_run_key="failure")
    second = runtime.run(workload_run_key="failure")

    assert first.status == AppRunStatus.FAILED
    assert second.status == AppRunStatus.FAILED
    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.FAILED
    assert node.calls == 1


def test_cached_terminal_publication_prunes_its_ancestor_closure(
    tmp_path: Path,
) -> None:
    workflow = Workflow("cached")
    ancestor_node = TextNode("should-not-run")
    ancestor = workflow.add_node(ancestor_node, id="ancestor")
    terminal_node = TextNode("should-not-run")
    workflow.add_node(
        terminal_node,
        id="terminal",
        inputs={"upstream": ancestor.outputs()},
    )
    runtime = _runtime(tmp_path, workflow)
    definition = workflow.validate()
    runtime._definition = definition
    runtime._workload_run_key = "cached"
    runtime._ensure_run(definition, "cached")
    cached_path = runtime.store.output_root / "cached" / "result.txt"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text("cached")
    storage = VolumePath(
        volume_name="Workflow-outputs",
        path=cached_path.relative_to(tmp_path).as_posix(),
        media_type="text/plain",
    )
    cached_result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="text",
                kind=ArtifactKind.REPORT,
                storage=storage,
            )
        ],
    )
    cached_artifact = WorkflowArtifact(
        artifact_id="terminal-text",
        producing_node_id="terminal",
        kind=ArtifactKind.REPORT,
        storage=storage,
        files=[ArtifactFile(path="result.txt", size_bytes=6)],
    )
    uncheckable_ancestor_storage = VolumePath(
        volume_name="External-outputs",
        path="ancestor/result.txt",
        media_type="text/plain",
    )
    with runtime.store.transaction():
        runtime.store.artifacts.record_node_publication(
            "ancestor",
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="text",
                        kind=ArtifactKind.REPORT,
                        storage=uncheckable_ancestor_storage,
                    )
                ],
            ),
            artifacts=(
                WorkflowArtifact(
                    artifact_id="ancestor-text",
                    producing_node_id="ancestor",
                    kind=ArtifactKind.REPORT,
                    storage=uncheckable_ancestor_storage,
                ),
            ),
            now=98,
        )
        runtime.store.artifacts.record_node_publication(
            "terminal",
            result=cached_result,
            artifacts=(cached_artifact,),
            now=99,
        )
    runtime._checkpoint()

    result = runtime.run(workload_run_key="cached")

    assert result.status == AppRunStatus.SUCCEEDED
    nodes = runtime.store.execution.list_nodes(RUN_ID)
    assert [node.status for node in nodes] == [
        NodeStatus.SKIPPED,
        NodeStatus.SUCCEEDED,
    ]
    assert ancestor_node.seen == []
    assert terminal_node.seen == []


def test_missing_copied_publication_is_discarded_and_recomputed(
    tmp_path: Path,
) -> None:
    workflow = Workflow("repair-cache")
    node = TextNode("recomputed")
    workflow.add_node(node, id="terminal")
    runtime = _runtime(tmp_path, workflow)
    definition = workflow.validate()
    runtime._definition = definition
    runtime._workload_run_key = "repair-cache"
    runtime._ensure_run(definition, "repair-cache")
    storage = VolumePath(
        volume_name="Workflow-outputs",
        path="missing/result.txt",
        media_type="text/plain",
    )
    with runtime.store.transaction():
        runtime.store.artifacts.record_node_publication(
            "terminal",
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="text",
                        kind=ArtifactKind.REPORT,
                        storage=storage,
                    )
                ],
            ),
            artifacts=(
                WorkflowArtifact(
                    artifact_id="terminal-text",
                    producing_node_id="terminal",
                    kind=ArtifactKind.REPORT,
                    storage=storage,
                    files=[ArtifactFile(path="result.txt")],
                ),
            ),
            now=99,
        )
    runtime._checkpoint()

    result = runtime.run(workload_run_key="repair-cache")

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(node.seen) == 1
    publication = runtime.store.artifacts.load_node_result("terminal")
    assert publication is not None
    assert isinstance(publication.outputs[0].storage, VolumePath)
    assert publication.outputs[0].storage.path != "missing/result.txt"


def test_unchecked_external_publication_suspends_instead_of_authorizing_work(
    tmp_path: Path,
) -> None:
    workflow = Workflow("unknown-cache")
    node = TextNode("should-not-run")
    workflow.add_node(node, id="terminal")
    runtime = _runtime(tmp_path, workflow)
    definition = workflow.validate()
    runtime._definition = definition
    runtime._workload_run_key = "unknown-cache"
    runtime._ensure_run(definition, "unknown-cache")
    storage = VolumePath(
        volume_name="External-outputs",
        path="terminal/result.txt",
    )
    with runtime.store.transaction():
        runtime.store.artifacts.record_node_publication(
            "terminal",
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="text",
                        kind=ArtifactKind.REPORT,
                        storage=storage,
                    )
                ],
            ),
            artifacts=(
                WorkflowArtifact(
                    artifact_id="terminal-text",
                    producing_node_id="terminal",
                    kind=ArtifactKind.REPORT,
                    storage=storage,
                ),
            ),
            now=99,
        )
    runtime._checkpoint()

    result = runtime.run(workload_run_key="unknown-cache")

    run = runtime.store.execution.get_run(RUN_ID)
    assert result.status == AppRunStatus.PARTIAL
    assert run.status == RunStatus.SUSPENDED
    assert run.status_reason is not None
    assert run.status_reason.value == "result_validation_unknown"
    assert node.seen == []
