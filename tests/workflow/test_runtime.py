"""Tests for the local workflow runtime scheduler."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event, Thread

import orjson
import pytest
from pydantic import BaseModel, Field

from biomodals.helper.styling import strip_ansi
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
from biomodals.workflow.core._runtime import hashing
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import RemoteNodeSubmission, WorkflowNativeNode
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
    def __init__(self, volume: FakeVolume):
        self.volume = volume
        self.commit_count_at_run = -1

    def run(self, context):
        self.commit_count_at_run = self.volume.commit_count
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


class RemoteOnlyNode(WorkflowNativeNode):
    placement = NodePlacement.REMOTE

    def run(self, context):
        raise AssertionError("remote placement should use submit_remote")


class DirectSubmitNode(WorkflowNativeNode):
    placement = NodePlacement.REMOTE

    def __init__(self, *, call: FakeRemoteCall):
        self.call = call
        self.submitted_contexts: list[str] = []
        self.processed_metadata: list[dict[str, object]] = []

    def submit_remote(self, context):
        self.submitted_contexts.append(context.node_id)
        return RemoteNodeSubmission(
            function_call=self.call,
            function_name="direct_app_function",
            metadata={"selected": "A1 A3"},
        )

    def process_remote_result(self, result, metadata):
        self.processed_metadata.append(dict(metadata))
        for output in result.outputs:
            output.metadata.setdefault("selected", str(metadata["selected"]))
        return result

    def run(self, context):
        raise AssertionError("direct remote placement should submit a FunctionCall")


class FakeRemoteCall:
    def __init__(
        self,
        *,
        object_id: str,
        result: AppRunResult,
        on_get=None,
        effects: list[object] | None = None,
    ):
        self.object_id = object_id
        self.result = result
        self.on_get = on_get
        self.effects = effects or []
        self.get_timeouts: list[float | int | None] = []
        self.cancel_calls: list[bool] = []

    def get(self, timeout=None):
        self.get_timeouts.append(timeout)
        if self.on_get is not None:
            self.on_get(timeout)
        if self.effects:
            effect = self.effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect
        return self.result

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancel_calls.append(terminate_containers)


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0
        self.on_commit = None
        self.on_reload = None

    def commit(self) -> None:
        if self.on_commit is not None:
            self.on_commit()
        self.commit_count += 1

    def reload(self) -> None:
        if self.on_reload is not None:
            self.on_reload()
        self.reload_count += 1


class SnapshotVolume(FakeVolume):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.committed_paths: set[str] = set()

    def commit(self) -> None:
        super().commit()
        self.committed_paths = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }


class HashSettings(BaseModel):
    visible: str
    hidden: str = Field(repr=False)


@dataclass
class ConfiguredNode(WorkflowNativeNode):
    settings: HashSettings
    output_path: Path

    def run(self, context):
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


@dataclass
class BytesConfiguredNode(WorkflowNativeNode):
    payload: bytes

    def run(self, context):
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


@dataclass
class RuntimeHandleNode(WorkflowNativeNode):
    handle: object = field(metadata={"dag_hash": False})

    def run(self, context):
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


def record_artifacts_with_manifests(
    ledger: WorkflowLedger,
    artifacts: list[WorkflowArtifact],
) -> None:
    ledger.record_artifacts(artifacts)
    artifact_dir = ledger.run_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        artifact_dir.joinpath(f"{artifact.artifact_id}.json").write_text(
            artifact.model_dump_json(indent=2),
            encoding="utf-8",
        )


def test_volume_sync_closes_open_ledger_connection(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = FakeVolume()
    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )
    runtime.ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    assert runtime.ledger._connection is not None

    def assert_ledger_closed() -> None:
        assert runtime.ledger._connection is None

    volume.on_reload = assert_ledger_closed
    volume.on_commit = assert_ledger_closed

    runtime._volume_sync.reload()
    runtime.ledger.load_run("demo", "run-1")
    assert runtime.ledger._connection is not None

    runtime._volume_sync.commit()

    assert volume.reload_count == 1
    assert volume.commit_count == 1
    assert runtime.ledger._connection is None


def test_completed_nodes_are_skipped(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(ExplodingNode(), id="done")
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    output_path = tmp_path / "done"
    output_path.write_text("done\n", encoding="utf-8")
    record_artifacts_with_manifests(
        ledger,
        [
            WorkflowArtifact(
                artifact_id="artifact-1",
                producing_node_id="done",
                kind=ArtifactKind.REPORT,
                storage=VolumePath(volume_name="Workflow-outputs", path="done"),
            )
        ],
    )
    ledger.mark_node_succeeded("done", ["artifact-1"])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert runtime.executed_waves == []


def test_completed_node_with_missing_workflow_artifact_is_rerun(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    calls: list[str] = []
    workflow.add_node(FakeNode(calls=calls), id="done")
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.record_artifacts([
        WorkflowArtifact(
            artifact_id="artifact-1",
            producing_node_id="done",
            kind=ArtifactKind.REPORT,
            storage=VolumePath(
                volume_name="Workflow-outputs",
                path="demo/run-1/nodes/done/attempts/attempt-1/output.txt",
            ),
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
    assert calls == ["done"]
    assert runtime.executed_waves == [["done"]]


def test_completed_node_with_missing_artifact_manifest_is_rerun(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    calls: list[str] = []
    workflow.add_node(FakeNode(calls=calls), id="one")
    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    first = runtime.run(run_id="run-1")
    manifest_path = tmp_path / "demo" / "run-1" / "artifacts" / "one-output.json"
    manifest_path.unlink()
    second = runtime.run(run_id="run-1")

    assert first.status == AppRunStatus.SUCCEEDED
    assert second.status == AppRunStatus.SUCCEEDED
    assert calls == ["one", "one"]
    assert manifest_path.exists()


def test_remote_completed_node_with_missing_materialized_archive_is_rerun(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    node = DirectSubmitNode(
        call=FakeRemoteCall(
            object_id="fc-remote",
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="archive",
                        kind=ArtifactKind.ARCHIVE,
                        storage=InlineBytes(
                            data=b"archive",
                            filename="designs.tar.zst",
                            media_type="application/zstd",
                        ),
                    )
                ],
            ),
        )
    )
    workflow.add_node(node, id="remote")
    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    first = runtime.run(run_id="run-1")
    materialized_path = (
        tmp_path
        / "demo"
        / "run-1"
        / "nodes"
        / "remote"
        / "attempts"
        / "attempt-1"
        / "remote-archive"
        / "designs.tar.zst"
    )
    materialized_path.unlink()
    second = runtime.run(run_id="run-1")

    assert first.status == AppRunStatus.SUCCEEDED
    assert second.status == AppRunStatus.SUCCEEDED
    assert node.submitted_contexts == ["remote", "remote"]
    assert materialized_path.exists()


def test_force_replaces_existing_run_ledger_and_reruns_completed_nodes(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    calls: list[str] = []
    workflow.add_node(FakeNode(calls=calls), id="one")
    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    first = runtime.run(run_id="run-1")
    stale_file = tmp_path / "demo" / "run-1" / "nodes" / "one" / "cache" / "stale"
    stale_file.write_text("old", encoding="utf-8")
    second = runtime.run(run_id="run-1", force=True)

    assert first.status == AppRunStatus.SUCCEEDED
    assert second.status == AppRunStatus.SUCCEEDED
    assert calls == ["one", "one"]
    assert not stale_file.exists()


def test_force_reset_commits_deleted_run_before_recreate(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = SnapshotVolume(tmp_path)
    workflow.add_node(FakeNode(), id="one")
    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )

    assert runtime.run(run_id="run-1").status == AppRunStatus.SUCCEEDED
    stale_path = tmp_path / "demo" / "run-1" / "nodes" / "stale.txt"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale\n", encoding="utf-8")
    runtime._volume_sync.commit()
    assert "demo/run-1/nodes/stale.txt" in volume.committed_paths

    assert runtime.run(run_id="run-1", force=True).status == AppRunStatus.SUCCEEDED
    assert "demo/run-1/nodes/stale.txt" not in volume.committed_paths


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
    output_dir = tmp_path / "design"
    output_dir.mkdir()
    record_artifacts_with_manifests(
        ledger,
        [
            WorkflowArtifact(
                artifact_id="design-artifact",
                producing_node_id="design",
                kind=ArtifactKind.STRUCTURES,
                storage=VolumePath(volume_name="Workflow-outputs", path="design"),
            )
        ],
    )
    ledger.mark_node_succeeded("design", ["design-artifact"])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    runtime.run(run_id="run-1")

    assert set(calls) == {"score-a", "score-b"}
    assert runtime.executed_waves == [["score-a", "score-b"]]


def test_independent_remote_nodes_execute_concurrently(tmp_path: Path) -> None:
    barrier = Barrier(2, timeout=0.5)
    workflow = Workflow("demo")

    class BarrierRemoteCall(FakeRemoteCall):
        def get(self, timeout=None):
            try:
                barrier.wait()
            except BrokenBarrierError:
                return AppRunResult(status=AppRunStatus.FAILED)
            return super().get(timeout=timeout)

    workflow.add_node(
        DirectSubmitNode(
            call=BarrierRemoteCall(
                object_id="fc-one",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
            )
        ),
        id="one",
    )
    workflow.add_node(
        DirectSubmitNode(
            call=BarrierRemoteCall(
                object_id="fc-two",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
            )
        ),
        id="two",
    )

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


def test_partial_node_marks_run_failed_and_blocks_downstream(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    partial = workflow.add_node(
        FakeNode(result=AppRunResult(status=AppRunStatus.PARTIAL)),
        id="partial",
    )
    workflow.add_node(ExplodingNode(), id="downstream", depends_on=[partial])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.PARTIAL
    assert runtime.ledger.load_run("demo", "run-1").status == RunStatus.FAILED
    status = runtime.ledger._load_node_status_or_default("partial")
    assert status.status == NodeStatus.FAILED
    assert status.error == "Node returned partial status"
    assert runtime.executed_waves == [["partial"]]


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
    assert status.error is not None
    assert "Traceback" in status.error
    assert "RuntimeError: fail exploded" in status.error


def test_runtime_logs_dag_and_node_state_transitions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow = Workflow("demo")
    first = workflow.add_node(FakeNode(), id="prepare")
    workflow.add_node(FakeNode(), id="produce", depends_on=[first])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    stdout = capsys.readouterr().out
    assert "\x1b[" not in stdout
    plain_stdout = strip_ansi(stdout)
    assert "[workflow] Starting workflow 'demo' run 'run-1'" in plain_stdout
    assert (
        "[workflow] DAG graph: node_id [placement; class] <- dependency" in plain_stdout
    )
    assert "[workflow]   prepare [orchestrator; FakeNode] <- -" in plain_stdout
    assert "[workflow]   produce [orchestrator; FakeNode] <- prepare" in plain_stdout
    assert "tests.workflow.test_runtime.FakeNode" not in plain_stdout
    assert "<- prepare" in plain_stdout
    assert "[workflow] Node started: prepare attempt=attempt-1" in plain_stdout
    assert "[workflow] Node succeeded: prepare attempt=attempt-1" in plain_stdout
    assert "[workflow] Node started: produce attempt=attempt-1" in plain_stdout
    assert "[workflow] Node succeeded: produce attempt=attempt-1" in plain_stdout


def test_runtime_logs_failed_node_transition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    stdout = capsys.readouterr().out
    plain_stdout = strip_ansi(stdout)
    assert "[workflow] Node started: fail attempt=attempt-1" in plain_stdout
    assert "[workflow] Node failed: fail attempt=attempt-1" in plain_stdout


def test_runtime_colorizes_node_state_transitions_when_forced(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("BIOMODALS_WORKFLOW_COLOR", "1")
    workflow = Workflow("demo")
    first = workflow.add_node(FakeNode(), id="prepare")
    workflow.add_node(FakeNode(), id="produce", depends_on=[first])

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    stdout = capsys.readouterr().out
    plain_stdout = strip_ansi(stdout)
    assert "\x1b[" in stdout
    assert "[workflow] Node started: prepare" in plain_stdout
    assert "[workflow] Node succeeded: prepare" in plain_stdout
    assert "[workflow] Starting workflow 'demo'" in plain_stdout
    assert "[workflow]   prepare [orchestrator; FakeNode] <- -" in plain_stdout
    assert "[workflow]   produce [orchestrator; FakeNode] <- prepare" in plain_stdout


def test_runtime_color_can_be_disabled_with_biomodals_no_color(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIOMODALS_NO_COLOR", "1")
    workflow = Workflow("demo")
    workflow.add_node(FakeNode(), id="prepare")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    stdout = capsys.readouterr().out
    assert "\x1b[" not in stdout
    plain_stdout = strip_ansi(stdout)
    assert "[workflow] Node started: prepare attempt=attempt-1" in plain_stdout
    assert "[workflow] Node succeeded: prepare attempt=attempt-1" in plain_stdout


def test_runtime_dag_hash_uses_stable_json_for_dataclass_node_config() -> None:
    first_workflow = Workflow("demo")
    first_workflow.add_node(
        ConfiguredNode(
            settings=HashSettings(visible="same", hidden="one"),
            output_path=Path("outputs/report.txt"),
        ),
        id="configured",
    )
    second_workflow = Workflow("demo")
    second_workflow.add_node(
        ConfiguredNode(
            settings=HashSettings(visible="same", hidden="two"),
            output_path=Path("outputs/report.txt"),
        ),
        id="configured",
    )
    repeated_workflow = Workflow("demo")
    repeated_workflow.add_node(
        ConfiguredNode(
            settings=HashSettings(visible="same", hidden="one"),
            output_path=Path("outputs/report.txt"),
        ),
        id="configured",
    )

    first_hash = hashing.dag_hash(first_workflow.validate())
    second_hash = hashing.dag_hash(second_workflow.validate())
    repeated_hash = hashing.dag_hash(repeated_workflow.validate())

    assert first_hash != second_hash
    assert first_hash == repeated_hash


def test_runtime_dag_hash_supports_bytes_in_dataclass_node_config() -> None:
    first_workflow = Workflow("demo")
    first_workflow.add_node(BytesConfiguredNode(payload=b"ATOM 1\n"), id="configured")
    second_workflow = Workflow("demo")
    second_workflow.add_node(BytesConfiguredNode(payload=b"ATOM 2\n"), id="configured")
    repeated_workflow = Workflow("demo")
    repeated_workflow.add_node(
        BytesConfiguredNode(payload=b"ATOM 1\n"), id="configured"
    )

    first_hash = hashing.dag_hash(first_workflow.validate())
    second_hash = hashing.dag_hash(second_workflow.validate())
    repeated_hash = hashing.dag_hash(repeated_workflow.validate())

    assert first_hash != second_hash
    assert first_hash == repeated_hash


def test_runtime_dag_hash_skips_dataclass_fields_marked_excluded() -> None:
    first_workflow = Workflow("demo")
    first_workflow.add_node(RuntimeHandleNode(handle=object()), id="node")
    second_workflow = Workflow("demo")
    second_workflow.add_node(RuntimeHandleNode(handle=object()), id="node")

    assert hashing.dag_hash(first_workflow.validate()) == (
        hashing.dag_hash(second_workflow.validate())
    )


def test_runtime_dag_hash_ignores_internal_node_placement() -> None:
    orchestrator_node = ConfiguredNode(
        settings=HashSettings(visible="same", hidden="same"),
        output_path=Path("outputs/report.txt"),
    )
    remote_node = ConfiguredNode(
        settings=HashSettings(visible="same", hidden="same"),
        output_path=Path("outputs/report.txt"),
    )
    remote_node.placement = NodePlacement.REMOTE

    first_workflow = Workflow("demo")
    first_workflow.add_node(orchestrator_node, id="configured")
    second_workflow = Workflow("demo")
    second_workflow.add_node(remote_node, id="configured")

    assert hashing.dag_hash(first_workflow.validate()) == (
        hashing.dag_hash(second_workflow.validate())
    )


def test_running_node_without_recoverable_call_is_not_duplicated(
    tmp_path: Path,
) -> None:
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

    assert result.status == AppRunStatus.PARTIAL
    assert calls == []
    assert runtime.ledger.load_run("demo", "run-1").status == RunStatus.RUNNING


def test_rerun_policy_discards_failed_attempt_state(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    calls: list[str] = []
    workflow.add_node(FakeNode(calls=calls), id="incomplete")
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.mark_node_running("incomplete", "attempt-old")
    ledger.record_attempt_started("incomplete", "attempt-old")
    ledger.mark_node_failed("incomplete", "old failure")
    old_cache = tmp_path / "demo" / "run-1" / "nodes" / "incomplete" / "cache" / "old"
    old_cache.parent.mkdir(parents=True)
    old_cache.write_text("old", encoding="utf-8")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls == ["incomplete"]
    assert not old_cache.exists()
    status = runtime.ledger._load_node_status_or_default("incomplete")
    assert status.attempts == ["attempt-1"]


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


def test_downstream_node_sees_committed_inline_artifact(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = SnapshotVolume(tmp_path)
    upstream = workflow.add_node(
        FakeNode(
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="report",
                        kind=ArtifactKind.REPORT,
                        storage=InlineBytes(data=b"ok\n", filename="report.txt"),
                    )
                ],
            )
        ),
        id="upstream",
    )

    class CommittedReaderNode(WorkflowNativeNode):
        def run(self, context):
            artifacts = context.inputs["report"]
            storage_path = artifacts[0].storage.path
            assert storage_path in volume.committed_paths
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    workflow.add_node(
        CommittedReaderNode(),
        id="downstream",
        inputs={"report": upstream.outputs(kind=ArtifactKind.REPORT)},
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED


def test_failed_node_logs_are_committed_before_run_returns(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    volume = SnapshotVolume(tmp_path)
    workflow.add_node(
        FakeNode(
            result=AppRunResult(
                status=AppRunStatus.FAILED,
                logs=[
                    AppOutput(
                        name="stderr",
                        kind=ArtifactKind.LOGS,
                        storage=InlineBytes(
                            data=b"missing input\n",
                            filename="stderr.log",
                        ),
                    )
                ],
                warnings=["remote file not found"],
            )
        ),
        id="failed",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.FAILED
    assert (
        "demo/run-1/nodes/failed/attempts/attempt-1/logs/failed-logs-stderr/stderr.log"
    ) in volume.committed_paths
    assert "demo/run-1/artifacts/failed-logs-stderr.json" in volume.committed_paths


def test_remote_placement_requires_direct_submission(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(RemoteOnlyNode(), id="remote")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.FAILED
    assert result.warnings
    assert "must implement submit_remote" in result.warnings[0]


def test_remote_placement_prefers_direct_node_submission(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    node = DirectSubmitNode(
        call=FakeRemoteCall(
            object_id="fc-direct",
            result=AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="output",
                        kind=ArtifactKind.REPORT,
                        storage=InlineBytes(data=b"ok", filename="output.txt"),
                    )
                ],
            ),
        )
    )
    workflow.add_node(node, id="remote")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert node.submitted_contexts == ["remote"]
    assert node.call.get_timeouts == [0]
    assert node.processed_metadata == [{"selected": "A1 A3"}]
    with sqlite3.connect(tmp_path / "demo" / "run-1" / "ledger.sqlite3") as conn:
        remote_call = conn.execute(
            """
            SELECT function_name, metadata_json
            FROM remote_calls
            WHERE call_id = 'fc-direct'
            """
        ).fetchone()
        artifact_metadata = conn.execute(
            """
            SELECT metadata_json
            FROM artifacts
            WHERE artifact_id = 'remote-output'
            """
        ).fetchone()[0]
    assert remote_call[0] == "direct_app_function"
    assert orjson.loads(remote_call[1]) == {
        "result_status": "succeeded",
        "selected": "A1 A3",
    }
    assert orjson.loads(artifact_metadata) == {"selected": "A1 A3"}


def test_remote_placement_records_function_call_before_waiting(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    observed_statuses: list[str] = []

    def observe_remote_call(timeout):
        with sqlite3.connect(tmp_path / "demo" / "run-1" / "ledger.sqlite3") as conn:
            row = conn.execute(
                "SELECT status FROM remote_calls WHERE call_id = 'fc-new'"
            ).fetchone()
        observed_statuses.append(row[0])

    workflow.add_node(
        DirectSubmitNode(
            call=FakeRemoteCall(
                object_id="fc-new",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
                on_get=observe_remote_call,
            )
        ),
        id="remote",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert observed_statuses == ["submitted"]
    with sqlite3.connect(tmp_path / "demo" / "run-1" / "ledger.sqlite3") as conn:
        final_status = conn.execute(
            "SELECT status FROM remote_calls WHERE call_id = 'fc-new'"
        ).fetchone()[0]
    assert final_status == "succeeded"


def test_remote_placement_records_configured_function_name(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    workflow.add_node(
        DirectSubmitNode(
            call=FakeRemoteCall(
                object_id="fc-named",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
            )
        ),
        id="remote",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    with sqlite3.connect(tmp_path / "demo" / "run-1" / "ledger.sqlite3") as conn:
        function_name = conn.execute(
            "SELECT function_name FROM remote_calls WHERE call_id = 'fc-named'"
        ).fetchone()[0]
    assert function_name == "direct_app_function"


def test_remote_call_failure_after_timeout_is_recorded(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(
        DirectSubmitNode(
            call=FakeRemoteCall(
                object_id="fc-fail",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
                effects=[TimeoutError(), RuntimeError("remote exploded")],
            )
        ),
        id="remote",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.FAILED
    with sqlite3.connect(tmp_path / "demo" / "run-1" / "ledger.sqlite3") as conn:
        row = conn.execute(
            "SELECT status, error FROM remote_calls WHERE call_id = 'fc-fail'"
        ).fetchone()
    assert row == ("failed", "remote exploded")


def test_runtime_cleanup_cancels_in_flight_remote_function_call(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    waiting_for_result = Event()
    release_result = Event()

    class BlockingRemoteCall(FakeRemoteCall):
        def get(self, timeout=None):
            self.get_timeouts.append(timeout)
            if timeout == 0:
                raise TimeoutError()
            waiting_for_result.set()
            release_result.wait(timeout=2)
            raise RuntimeError("cancelled after cleanup")

        def cancel(self, terminate_containers: bool = False) -> None:
            super().cancel(terminate_containers=terminate_containers)
            release_result.set()

    remote_call = BlockingRemoteCall(
        object_id="fc-blocking",
        result=AppRunResult(status=AppRunStatus.SUCCEEDED),
    )
    workflow.add_node(DirectSubmitNode(call=remote_call), id="remote")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    results: list[AppRunResult] = []
    thread = Thread(
        target=lambda: results.append(runtime.run(run_id="run-1")),
        daemon=True,
    )
    thread.start()

    assert waiting_for_result.wait(timeout=2)
    try:
        runtime.cancel_active_remote_calls(terminate_containers=True)
    finally:
        release_result.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert remote_call.cancel_calls == [True]
    assert results[0].status == AppRunStatus.FAILED


def test_remote_success_records_materialized_app_result(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    workflow.add_node(
        DirectSubmitNode(
            call=FakeRemoteCall(
                object_id="fc-success",
                result=AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        AppOutput(
                            name="archive",
                            kind=ArtifactKind.ARCHIVE,
                            storage=InlineBytes(
                                data=b"ok",
                                filename="archive.tar.zst",
                                media_type="application/zstd",
                            ),
                        )
                    ],
                ),
            )
        ),
        id="remote",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    with sqlite3.connect(tmp_path / "demo" / "run-1" / "ledger.sqlite3") as conn:
        row = conn.execute(
            """
            SELECT app_result_json
            FROM attempts
            WHERE node_id = 'remote' AND attempt_id = 'attempt-1'
            """
        ).fetchone()
    data = orjson.loads(row[0])
    storage = data["outputs"][0]["storage"]
    assert storage == {
        "kind": "volume_path",
        "volume_name": "Workflow-outputs",
        "path": "demo/run-1/nodes/remote/attempts/attempt-1/remote-archive/archive.tar.zst",
        "media_type": "application/zstd",
    }
    assert "data" not in storage
    assert (tmp_path / storage["path"]).read_bytes() == b"ok"


def test_remote_success_reloads_volume_before_materializing_outputs(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    volume = FakeVolume()
    workflow.add_node(
        DirectSubmitNode(
            call=FakeRemoteCall(
                object_id="fc-reload",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
            )
        ),
        id="remote",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert volume.reload_count >= 2


def test_remote_reload_waits_for_orchestrator_node_volume_access(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    local_running = Event()
    local_can_finish = Event()
    local_finished = Event()
    reload_happened_while_local_running = False

    class GuardedVolume(FakeVolume):
        def reload(self) -> None:
            nonlocal reload_happened_while_local_running
            if local_running.is_set() and not local_finished.is_set():
                reload_happened_while_local_running = True
                local_can_finish.set()
            super().reload()

    class SlowLocalNode(WorkflowNativeNode):
        def run(self, context):
            local_running.set()
            local_can_finish.wait(timeout=1)
            local_finished.set()
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    class FinishingRemoteCall(FakeRemoteCall):
        def get(self, timeout=None):
            local_running.wait(timeout=1)
            return super().get(timeout=timeout)

    volume = GuardedVolume()
    workflow.add_node(SlowLocalNode(), id="local")
    workflow.add_node(
        DirectSubmitNode(
            call=FinishingRemoteCall(
                object_id="fc-remote",
                result=AppRunResult(status=AppRunStatus.SUCCEEDED),
            )
        ),
        id="remote",
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        workflow_volume=volume,
        max_ready_workers=2,
    )

    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert reload_happened_while_local_running is False


def test_remote_recovery_reattaches_existing_call_before_rerun(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    node = DirectSubmitNode(
        call=FakeRemoteCall(
            object_id="fc-unused",
            result=AppRunResult(status=AppRunStatus.SUCCEEDED),
        )
    )
    workflow.add_node(node, id="remote")
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.mark_node_running(
        "remote",
        "attempt-old",
        placement=NodePlacement.REMOTE,
    )
    ledger.record_attempt_started("remote", "attempt-old")
    ledger.record_remote_call(
        call_id="fc-old",
        node_id="remote",
        attempt_id="attempt-old",
        function_name="direct_app_function",
        call_kind="node",
    )
    resolved: list[str] = []
    existing_call = FakeRemoteCall(
        object_id="fc-old",
        result=AppRunResult(status=AppRunStatus.SUCCEEDED),
    )

    def resolve_call(call_id: str):
        resolved.append(call_id)
        return existing_call

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        function_call_resolver=resolve_call,
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert resolved == ["fc-old"]
    assert node.submitted_contexts == []
    assert existing_call.get_timeouts == [0]
    status = runtime.ledger._load_node_status_or_default("remote")
    assert status.status == NodeStatus.SUCCEEDED


def test_remote_recovery_processes_direct_submission_metadata(
    tmp_path: Path,
) -> None:
    workflow = Workflow("demo")
    node = DirectSubmitNode(
        call=FakeRemoteCall(
            object_id="fc-unused",
            result=AppRunResult(status=AppRunStatus.SUCCEEDED),
        )
    )
    workflow.add_node(node, id="remote")
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    ledger.mark_node_running(
        "remote",
        "attempt-old",
        placement=NodePlacement.REMOTE,
    )
    ledger.record_attempt_started("remote", "attempt-old")
    recovered_output = (
        tmp_path / "demo" / "run-1" / "nodes" / "remote" / "attempts" / "attempt-old"
    )
    recovered_output.joinpath("remote-output").mkdir(parents=True)
    ledger.record_app_result(
        "remote",
        "attempt-old",
        AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="output",
                    kind=ArtifactKind.REPORT,
                    storage=VolumePath(
                        volume_name="Workflow-outputs",
                        path="demo/run-1/nodes/remote/attempts/attempt-old/remote-output",
                    ),
                    metadata={"selected": "B5"},
                )
            ],
        ),
    )
    ledger.record_remote_call(
        call_id="fc-old-direct",
        node_id="remote",
        attempt_id="attempt-old",
        function_name="direct_app_function",
        call_kind="node",
        status="succeeded",
        metadata={"selected": "B5"},
    )

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert node.submitted_contexts == []
    assert node.processed_metadata == []
    with sqlite3.connect(tmp_path / "demo" / "run-1" / "ledger.sqlite3") as conn:
        artifact_metadata = conn.execute(
            """
            SELECT metadata_json
            FROM artifacts
            WHERE artifact_id = 'remote-output'
            """
        ).fetchone()[0]
    assert orjson.loads(artifact_metadata) == {"selected": "B5"}


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
    output_dir = tmp_path / "demo" / "run-1" / "nodes" / "design" / "outputs"
    output_dir.mkdir(parents=True)
    record_artifacts_with_manifests(
        ledger,
        [
            WorkflowArtifact(
                artifact_id="design-structures",
                producing_node_id="design",
                kind=ArtifactKind.STRUCTURES,
                storage=VolumePath(
                    volume_name="Workflow-outputs",
                    path="demo/run-1/nodes/design/outputs",
                ),
            )
        ],
    )
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


def test_runtime_records_dag_hash_and_timestamps(tmp_path: Path) -> None:
    workflow = Workflow("demo")
    workflow.add_node(FakeNode(), id="one")

    runtime = WorkflowRuntime(
        workflow=workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    result = runtime.run(run_id="run-1")
    run = runtime.ledger.load_run("demo", "run-1")

    assert result.status == AppRunStatus.SUCCEEDED
    assert run.dag_hash
    assert run.created_at <= run.updated_at


def test_runtime_rejects_resume_when_dag_hash_changed(tmp_path: Path) -> None:
    first_workflow = Workflow("demo")
    first_workflow.add_node(FakeNode(), id="one")
    first_runtime = WorkflowRuntime(
        workflow=first_workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )
    first_runtime.run(run_id="run-1")

    second_workflow = Workflow("demo")
    second_workflow.add_node(FakeNode(), id="renamed")
    second_runtime = WorkflowRuntime(
        workflow=second_workflow,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
    )

    with pytest.raises(ValueError, match="DAG hash"):
        second_runtime.run(run_id="run-1")


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
