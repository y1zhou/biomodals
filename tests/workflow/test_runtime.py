"""Kernel-backed workflow runtime integration tests."""

# ruff: noqa: D101, D102, D103, D107

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from biomodals.execution import (
    DeploymentIdentity,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
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
    RemoteWorkflowNode,
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
    max_calls: int = 8,
    max_gpu_calls: int = 4,
) -> WorkflowRuntime:
    return WorkflowRuntime(
        workflow=workflow,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
        modal_driver=driver,
        max_active_provider_calls=max_calls,
        max_active_gpu_provider_calls=max_gpu_calls,
        now=iter(range(100, 1000)).__next__,
        poll_interval_seconds=0,
    )


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
        "spawn:run_gpu",
        "resolve:main/DemoWorkflow/7/run_cpu",
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
    call = runtime.store.execution.list_provider_calls(RUN_ID)[0]
    runtime._provider.repository = runtime.store.execution
    completed = runtime._provider.reconcile_provider_call(
        call.provider_call_id,
        encode_result=lambda value: {
            "result": value.model_dump(mode="json"),
        },
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
    workflow = Workflow("local-recovery")
    node = InterruptOnceNode()
    workflow.add_node(node, id="local")
    runtime = _runtime(tmp_path, workflow)

    try:
        runtime.run(workload_run_key="local-recovery")
    except CoordinatorInterrupted:
        pass
    else:  # pragma: no cover - test must exercise the hard interruption
        raise AssertionError("Coordinator interruption was not raised")
    interrupted_task = runtime.store.execution.get_task(RUN_ID, "local", "node")
    assert interrupted_task.status.value == "running"
    assert interrupted_task.local_owned is True

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
    with runtime.store.transaction():
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
