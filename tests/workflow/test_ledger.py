"""Tests for the SQLite-backed workflow ledger."""

# ruff: noqa: D103

import sqlite3
from pathlib import Path

import pytest

from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    InlineBytes,
    NodeStatus,
    RunStatus,
    VolumePath,
    WorkflowArtifact,
    WorkflowRun,
)
from biomodals.workflow.core import ledger as ledger_module
from biomodals.workflow.core.ledger import LEDGER_TABLES, WorkflowLedger


def _connect(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "ppiflow" / "run-1" / "ledger.sqlite3")
    conn.row_factory = sqlite3.Row
    return conn


def test_create_run_initializes_sqlite_ledger_and_documented_tables(
    tmp_path: Path,
) -> None:
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
    assert tmp_path.joinpath("ppiflow", "run-1", "ledger.sqlite3").exists()

    with _connect(tmp_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", ("run-1",))
        row = run_row.fetchone()

    assert set(LEDGER_TABLES).issubset(tables)
    assert row["workflow_name"] == "ppiflow"
    assert row["dag_hash"] == "abc123"
    assert row["status"] == RunStatus.PENDING


def test_mark_run_status_updates_run_row(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))

    updated = ledger.mark_run_status(RunStatus.RUNNING)
    loaded = ledger.load_run("ppiflow", "run-1")

    assert updated.status == RunStatus.RUNNING
    assert loaded.status == RunStatus.RUNNING
    with _connect(tmp_path) as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id = 'run-1'").fetchone()
    assert row["status"] == RunStatus.RUNNING


def test_run_exists_closes_probe_connection(tmp_path: Path, monkeypatch) -> None:
    ledger_path = tmp_path / "ppiflow" / "run-1" / "ledger.sqlite3"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.touch()

    class FakeCursor:
        def fetchone(self):
            return (1,)

    class FakeConnection:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def execute(self, sql, params):
            return FakeCursor()

        def close(self) -> None:
            self.closed = True

    connection = FakeConnection()
    monkeypatch.setattr(
        ledger_module.sqlite3,
        "connect",
        lambda path: connection,
    )

    assert WorkflowLedger(tmp_path).run_exists("ppiflow", "run-1")
    assert connection.closed is True


def test_node_attempt_app_result_and_artifacts_are_debuggable_with_sql(
    tmp_path: Path,
) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))

    status = ledger.mark_node_running("design", "attempt-1")
    attempt = ledger.record_attempt_started("design", "attempt-1")
    ledger.record_app_result(
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
        files=[
            ArtifactFile(
                path="model.pdb",
                role="structure",
                media_type="chemical/x-pdb",
                size_bytes=12,
            )
        ],
    )
    ledger.record_artifacts([artifact])
    succeeded = ledger.mark_node_succeeded("design", ["artifact-1"])

    assert status.status == NodeStatus.RUNNING
    assert attempt.attempt_id == "attempt-1"
    assert succeeded.status == NodeStatus.SUCCEEDED
    assert ledger.node_is_complete("design")

    with _connect(tmp_path) as conn:
        node = conn.execute("SELECT * FROM nodes WHERE node_id = 'design'").fetchone()
        attempt_row = conn.execute(
            "SELECT * FROM attempts WHERE node_id = 'design'"
        ).fetchone()
        artifact_row = conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = 'artifact-1'"
        ).fetchone()
        file_row = conn.execute(
            "SELECT * FROM artifact_files WHERE artifact_id = 'artifact-1'"
        ).fetchone()
        output_row = conn.execute(
            "SELECT * FROM node_outputs WHERE node_id = 'design'"
        ).fetchone()

    assert node["status"] == NodeStatus.SUCCEEDED
    assert node["current_attempt_id"] == "attempt-1"
    assert (
        attempt_row["app_result_json"]
        == AppRunResult(status=AppRunStatus.SUCCEEDED).model_dump_json()
    )
    assert artifact_row["storage_path"] == "ppiflow/run-1/artifacts/artifact-1"
    assert file_row["path"] == "model.pdb"
    assert output_row["artifact_id"] == "artifact-1"


def test_node_is_not_complete_when_artifact_row_is_missing(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))
    ledger.mark_node_running("design", "attempt-1")
    ledger.mark_node_succeeded("design", ["missing-artifact"])

    assert not ledger.node_is_complete("design")


def test_record_app_result_rejects_unmaterialized_inline_bytes(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))
    ledger.mark_node_running("node-1", "attempt-1")
    ledger.record_attempt_started("node-1", "attempt-1")
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="archive",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(data=b"ok", filename="archive.txt"),
            )
        ],
    )

    with pytest.raises(ValueError, match="InlineBytes"):
        ledger.record_app_result("node-1", "attempt-1", result)


def test_next_attempt_id_uses_highest_numeric_suffix(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))
    ledger.record_attempt_started("design", "attempt-2")
    ledger.record_attempt_started("design", "attempt-uuid")

    assert ledger.next_attempt_id("design") == "attempt-3"


def test_remote_call_rows_are_human_debuggable(tmp_path: Path) -> None:
    ledger = WorkflowLedger(tmp_path)
    ledger.create_run(WorkflowRun(workflow_name="ppiflow", run_id="run-1"))
    ledger.mark_node_running("remote", "attempt-1")
    ledger.record_attempt_started("remote", "attempt-1")

    ledger.record_remote_call(
        call_id="fc-123",
        node_id="remote",
        attempt_id="attempt-1",
        function_name="direct_app_function",
        call_kind="node",
    )
    ledger.mark_remote_call_status("fc-123", "running")

    with _connect(tmp_path) as conn:
        row = conn.execute("SELECT * FROM remote_calls WHERE call_id = 'fc-123'")
        remote_call = row.fetchone()

    assert remote_call["node_id"] == "remote"
    assert remote_call["attempt_id"] == "attempt-1"
    assert remote_call["status"] == "running"
    assert remote_call["function_name"] == "direct_app_function"
