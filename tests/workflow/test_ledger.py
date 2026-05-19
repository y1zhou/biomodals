"""Tests for the filesystem-backed workflow ledger."""

# ruff: noqa: D103

from pathlib import Path
from typing import Any, cast

from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    NodeStatus,
    RunStatus,
    VolumePath,
    WorkflowArtifact,
    WorkflowRun,
)
from biomodals.workflow.core.ledger import WorkflowLedger


def test_create_run_writes_and_loads_run_json(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    run = WorkflowRun(
        workflow_name="ppiflow",
        run_id="run-1",
        dag_hash="abc123",
    )

    created = ledger.create_run(run)
    loaded = ledger.load_run("ppiflow", "run-1")

    assert created == run
    assert loaded == run
    assert tmp_path.joinpath("ppiflow", "run-1", "run.json").exists()


def test_mark_run_status_updates_run_json(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))

    updated = ledger.mark_run_status(RunStatus.RUNNING)
    loaded = ledger.load_run("ppiflow", "run-1")

    assert updated.status == RunStatus.RUNNING
    assert loaded.status == RunStatus.RUNNING


def test_node_status_attempt_app_result_and_artifacts_are_recorded(
    tmp_path: Path,
) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))

    status = ledger.mark_node_running("design", "attempt-1")
    attempt = ledger.record_attempt_started("design", "attempt-1")
    result_path = ledger.record_app_result(
        "design",
        "attempt-1",
        AppRunResult(status=AppRunStatus.SUCCEEDED),
    )
    artifact = WorkflowArtifact(
        artifact_id="artifact-1",
        producing_node_id="design",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(
            volume_name="Workflow-outputs",
            path="ppiflow/run-1/artifacts/artifact-1",
        ),
    )
    artifact_paths = ledger.record_artifacts([artifact])
    succeeded = ledger.mark_node_succeeded("design", ["artifact-1"])

    assert status.status == NodeStatus.RUNNING
    assert attempt.attempt_id == "attempt-1"
    assert (
        result_path
        == tmp_path / "ppiflow/run-1/nodes/design/attempts/attempt-1/app_result.json"
    )
    assert artifact_paths == [tmp_path / "ppiflow/run-1/artifacts/artifact-1.json"]
    assert succeeded.status == NodeStatus.SUCCEEDED
    assert ledger.node_is_complete("design")


def test_node_is_not_complete_when_artifact_manifest_is_missing(
    tmp_path: Path,
) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))
    ledger.mark_node_running("design", "attempt-1")
    ledger.mark_node_succeeded("design", ["missing-artifact"])

    assert not ledger.node_is_complete("design")


def test_record_app_result_accepts_non_utf8_inline_bytes(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="demo", run_id="run-1"))
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="archive",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(data=b"\xff\x00", filename="archive.tar.zst"),
            )
        ],
    )

    path = ledger.record_app_result("node-1", "attempt-1", result)

    assert path.exists()
    data = cast(dict[str, Any], ledger._read_json(path))
    assert isinstance(data["outputs"][0]["storage"]["data"], str)
