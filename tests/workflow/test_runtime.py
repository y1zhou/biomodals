"""Tests for the local workflow runtime scheduler."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from threading import Barrier, BrokenBarrierError

from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    NodeExecutionPolicy,
    NodePlacement,
    NodeStatus,
    RunStatus,
    VolumePath,
    WorkflowArtifact,
    WorkflowRun,
)
from biomodals.workflow import Workflow
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import WorkflowNativeNode
from biomodals.workflow.core.runtime import WorkflowRuntime


class FakeNode(WorkflowNativeNode):
    def __init__(
        self,
        *,
        result: AppRunResult | None = None,
        calls: list[str] | None = None,
        policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN,
    ):
        self.result = result or AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="output",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(data=b"ok", filename="output.txt"),
                )
            ],
        )
        self.calls = calls
        self.execution_policy = policy
        self.seen_cache_dir: Path | None = None
        self.seen_inputs = None

    def run(self, context):
        self.seen_cache_dir = context.cache_dir
        self.seen_inputs = context.inputs
        if self.calls is not None:
            self.calls.append(context.node_id)
        return self.result


class ExplodingNode(WorkflowNativeNode):
    def run(self, context):
        raise AssertionError(f"{context.node_id} should not run")


class RuntimeErrorNode(WorkflowNativeNode):
    def run(self, context):
        raise RuntimeError(f"{context.node_id} exploded")


class CommitObservedNode(WorkflowNativeNode):
    def __init__(self, volume: "FakeVolume"):
        self.volume = volume
        self.commit_count_at_run = -1

    def run(self, context):
        self.commit_count_at_run = self.volume.commit_count
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


class RemoteOnlyNode(WorkflowNativeNode):
    placement = NodePlacement.REMOTE

    def run(self, context):
        raise AssertionError("remote placement should use remote_node_runner")


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def test_completed_nodes_are_skipped(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(ExplodingNode(), id="done")
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.record_artifacts([
        WorkflowArtifact(
            artifact_id="artifact-1",
            producing_node_id="done",
            kind=ArtifactKind.REPORT,
            storage=VolumePath(volume_name="Workflow-outputs", path="done"),
        )
    ])
    ledger.mark_node_succeeded("done", ["artifact-1"])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert runtime.executed_waves == []


def test_independent_ready_nodes_run_in_same_scheduler_wave(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    upstream = workflow.add_node(FakeNode(), id="design")
    calls: list[str] = []
    workflow.add_node(
        FakeNode(calls=calls),
        id="score-a",
        inputs={"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)},
    )
    workflow.add_node(
        FakeNode(calls=calls),
        id="score-b",
        inputs={"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)},
    )
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.record_artifacts([
        WorkflowArtifact(
            artifact_id="design-artifact",
            producing_node_id="design",
            kind=ArtifactKind.STRUCTURES,
            storage=VolumePath(volume_name="Workflow-outputs", path="design"),
        )
    ])
    ledger.mark_node_succeeded("design", ["design-artifact"])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    runtime.run(run_id="run-1")

    assert set(calls) == {"score-a", "score-b"}
    assert runtime.executed_waves == [["score-a", "score-b"]]


def test_independent_ready_nodes_execute_concurrently(tmp_path: Path) -> None:
    barrier = Barrier(2, timeout=0.5)
    workflow = Workflow("demo")

    class BarrierNode(WorkflowNativeNode):
        def run(self, context):
            try:
                barrier.wait()
            except BrokenBarrierError:
                return AppRunResult(status=AppRunStatus.FAILED)
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    workflow.add_node(BarrierNode(), id="one")
    workflow.add_node(BarrierNode(), id="two")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert runtime.executed_waves == [["one", "two"]]


def test_failed_node_prevents_downstream_nodes_from_running(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    failed = workflow.add_node(
        FakeNode(result=AppRunResult(status=AppRunStatus.FAILED)),
        id="fail",
    )
    workflow.add_node(ExplodingNode(), id="downstream", depends_on=[failed])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.FAILED
    assert runtime.executed_waves == [["fail"]]


def test_single_node_exception_marks_node_and_run_failed(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(RuntimeErrorNode(), id="fail")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.FAILED
    assert result.warnings == ["fail exploded"]
    assert runtime.ledger.load_run("demo", "run-1").status == RunStatus.FAILED
    status = runtime.ledger._load_node_status_or_default("fail")
    assert status.status == NodeStatus.FAILED
    assert status.error == "fail exploded"


def test_rerun_policy_runs_incomplete_nodes(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    calls: list[str] = []
    workflow.add_node(FakeNode(calls=calls), id="incomplete")
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.mark_node_running("incomplete", "attempt-old")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls == ["incomplete"]


def test_resume_policy_receives_durable_cache_path(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    node = FakeNode(policy=NodeExecutionPolicy.RESUME)
    workflow.add_node(node, id="long")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    runtime.run(run_id="run-1")

    assert node.seen_cache_dir == tmp_path / "demo/run-1/nodes/long/cache"
    assert node.seen_cache_dir is not None
    assert node.seen_cache_dir.exists()


def test_runtime_commits_node_start_before_node_execution(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = FakeVolume()
    node = CommitObservedNode(volume)
    workflow.add_node(node, id="long")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert node.commit_count_at_run >= 3


def test_remote_placement_uses_injected_remote_runner(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(RemoteOnlyNode(), id="remote")
    calls = []

    def remote_runner(node, context):
        calls.append((node, context.node_id))
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        remote_node_runner=remote_runner,
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls == [(workflow.validate().nodes["remote"].node, "remote")]


def test_runtime_passes_selected_upstream_artifacts_to_node_context(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    upstream = workflow.add_node(ExplodingNode(), id="design")
    downstream = FakeNode()
    workflow.add_node(
        downstream,
        id="score",
        inputs={"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)},
    )
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.record_artifacts([
        WorkflowArtifact(
            artifact_id="design-structures",
            producing_node_id="design",
            kind=ArtifactKind.STRUCTURES,
            storage=VolumePath(
                volume_name="Workflow-outputs",
                path="demo/run-1/nodes/design/outputs",
            ),
        )
    ])
    ledger.mark_node_succeeded("design", ["design-structures"])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert downstream.seen_inputs == {
        "structures": [
            WorkflowArtifact(
                artifact_id="design-structures",
                producing_node_id="design",
                kind=ArtifactKind.STRUCTURES,
                storage=VolumePath(
                    volume_name="Workflow-outputs",
                    path="demo/run-1/nodes/design/outputs",
                ),
            )
        ]
    }
    status = runtime.ledger._load_node_status_or_default("score")
    assert status.input_artifact_ids == ["design-structures"]


def test_runtime_records_succeeded_run_status(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(FakeNode(), id="one")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert runtime.ledger.load_run("demo", "run-1").status == RunStatus.SUCCEEDED


def test_runtime_records_failed_run_status(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(
        FakeNode(result=AppRunResult(status=AppRunStatus.FAILED)),
        id="fail",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.FAILED
    assert runtime.ledger.load_run("demo", "run-1").status == RunStatus.FAILED


def test_runtime_reloads_and_commits_workflow_volume(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(FakeNode(), id="one")
    volume = FakeVolume()

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert volume.reload_count >= 1
    assert volume.commit_count >= 1
