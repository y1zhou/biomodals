"""SQLite-backed durable ledger for Biomodals workflow runs."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from threading import RLock
from typing import Any

import orjson

from biomodals.schema import (
    AppRunResult,
    ArtifactSelector,
    AttemptRecord,
    NodeExecutionPolicy,
    NodePlacement,
    NodeStatus,
    NodeStatusRecord,
    RunStatus,
    VolumePath,
    WorkflowArtifact,
    WorkflowRun,
)

LEDGER_FILENAME = "ledger.sqlite3"
LEDGER_TABLES = (
    "runs",
    "nodes",
    "attempts",
    "remote_calls",
    "node_tasks",
    "artifacts",
    "artifact_files",
    "node_inputs",
    "node_outputs",
)


class WorkflowLedger:
    """SQLite-backed durable state for one workflow run."""

    def __init__(self, volume_root: str | Path):
        """Initialize a ledger rooted at a mounted workflow volume path."""
        self.volume_root = Path(volume_root)
        self.workflow_name: str | None = None
        self.run_id: str | None = None
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()

    @property
    def run_root(self) -> Path:
        """Return the root directory for the active workflow run."""
        if self.workflow_name is None or self.run_id is None:
            raise RuntimeError("Workflow run has not been initialized")
        return self.volume_root / self.workflow_name / self.run_id

    @property
    def ledger_path(self) -> Path:
        """Return the SQLite database path for the active workflow run."""
        return self.run_root / LEDGER_FILENAME

    def create_run(self, run: WorkflowRun) -> WorkflowRun:
        """Create a run ledger and initialize its SQLite schema."""
        self._activate(run.workflow_name, run.run_id)
        self._create_run_layout()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id,
                    workflow_name,
                    dag_hash,
                    status,
                    created_at,
                    updated_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.workflow_name,
                    run.dag_hash,
                    run.status.value,
                    _datetime_json(run.created_at),
                    _datetime_json(run.updated_at),
                    _json_dumps(run.metadata),
                ),
            )
        return run

    def load_run(self, workflow_name: str, run_id: str) -> WorkflowRun:
        """Load an existing run ledger."""
        self._activate(workflow_name, run_id)
        if not self.ledger_path.exists():
            raise FileNotFoundError(self.ledger_path)
        row = self._fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise FileNotFoundError(f"Workflow run not found: {workflow_name}/{run_id}")
        return _run_from_row(row)

    def run_exists(self, workflow_name: str, run_id: str) -> bool:
        """Return whether a SQLite ledger exists for one workflow run."""
        ledger_path = self.volume_root / workflow_name / run_id / LEDGER_FILENAME
        if not ledger_path.exists():
            return False
        try:
            with sqlite3.connect(ledger_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None

    def mark_run_status(self, status: RunStatus) -> WorkflowRun:
        """Update the durable status for the active workflow run."""
        now = _now_json()
        with self._transaction() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (status.value, now, self._require_run_id()),
            )
        return self.load_run(self._require_workflow_name(), self._require_run_id())

    def reset_run(self, workflow_name: str, run_id: str) -> None:
        """Remove all durable state for one workflow run."""
        if self.workflow_name == workflow_name and self.run_id == run_id:
            self._close()
        run_root = self.volume_root / workflow_name / run_id
        if run_root.exists():
            shutil.rmtree(run_root)

    def node_has_state(self, node_id: str) -> bool:
        """Return whether a node has any durable state in this run."""
        row = self._fetch_one(
            """
            SELECT 1 FROM nodes WHERE node_id = ?
            UNION ALL
            SELECT 1 FROM attempts WHERE node_id = ?
            UNION ALL
            SELECT 1 FROM artifacts WHERE producing_node_id = ?
            LIMIT 1
            """,
            (node_id, node_id, node_id),
        )
        return row is not None or (self.run_root / "nodes" / node_id).exists()

    def node_is_running(self, node_id: str) -> bool:
        """Return whether a node is currently marked as running."""
        row = self._fetch_one(
            "SELECT status FROM nodes WHERE node_id = ?",
            (node_id,),
        )
        return row is not None and row["status"] == NodeStatus.RUNNING.value

    def load_node_status(self, node_id: str) -> NodeStatusRecord:
        """Load a node status row or return the default pending state."""
        return self._load_node_status_or_default(node_id)

    def reset_node(self, node_id: str) -> None:
        """Remove durable state for one workflow node."""
        artifact_ids = [
            row["artifact_id"]
            for row in self._fetch_all(
                "SELECT artifact_id FROM artifacts WHERE producing_node_id = ?",
                (node_id,),
            )
        ]
        node_dir = self.run_root / "nodes" / node_id
        if node_dir.exists():
            shutil.rmtree(node_dir)
        for artifact_id in artifact_ids:
            manifest_path = self.run_root / "artifacts" / f"{artifact_id}.json"
            if manifest_path.exists():
                manifest_path.unlink()
            artifact_dir = self.run_root / "artifacts" / artifact_id
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)

        with self._transaction() as conn:
            for artifact_id in artifact_ids:
                conn.execute(
                    "DELETE FROM artifact_files WHERE artifact_id = ?",
                    (artifact_id,),
                )
                conn.execute(
                    "DELETE FROM node_outputs WHERE artifact_id = ?",
                    (artifact_id,),
                )
                conn.execute(
                    "DELETE FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                )
            conn.execute("DELETE FROM node_inputs WHERE node_id = ?", (node_id,))
            conn.execute("DELETE FROM node_outputs WHERE node_id = ?", (node_id,))
            conn.execute("DELETE FROM node_tasks WHERE node_id = ?", (node_id,))
            conn.execute("DELETE FROM remote_calls WHERE node_id = ?", (node_id,))
            conn.execute("DELETE FROM attempts WHERE node_id = ?", (node_id,))
            conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))

    def mark_node_pending(self, node_id: str) -> NodeStatusRecord:
        """Mark a node as pending."""
        self._upsert_node_status(node_id, NodeStatus.PENDING)
        return self._load_node_status_or_default(node_id)

    def mark_node_running(
        self,
        node_id: str,
        attempt_id: str,
        *,
        input_artifact_ids: list[str] | None = None,
        execution_policy: NodeExecutionPolicy | None = None,
        placement: NodePlacement | None = None,
    ) -> NodeStatusRecord:
        """Mark a node as running and record its attempt id."""
        now = _now_json()
        execution_policy = execution_policy or NodeExecutionPolicy.RERUN
        placement = placement or NodePlacement.ORCHESTRATOR
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id,
                    status,
                    execution_policy,
                    placement,
                    current_attempt_id,
                    error,
                    started_at,
                    completed_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    status = excluded.status,
                    execution_policy = excluded.execution_policy,
                    placement = excluded.placement,
                    current_attempt_id = excluded.current_attempt_id,
                    error = NULL,
                    started_at = COALESCE(nodes.started_at, excluded.started_at),
                    completed_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    node_id,
                    NodeStatus.RUNNING.value,
                    execution_policy.value,
                    placement.value,
                    attempt_id,
                    now,
                    now,
                ),
            )
            if input_artifact_ids is not None:
                conn.execute(
                    "DELETE FROM node_inputs WHERE node_id = ? AND input_name = ''",
                    (node_id,),
                )
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO node_inputs (
                        node_id,
                        input_name,
                        artifact_id
                    )
                    VALUES (?, '', ?)
                    """,
                    [(node_id, artifact_id) for artifact_id in input_artifact_ids],
                )
        return self._load_node_status_or_default(node_id)

    def record_node_inputs(
        self,
        node_id: str,
        inputs: dict[str, list[WorkflowArtifact]],
    ) -> None:
        """Record the named artifact inputs resolved for a node attempt."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM node_inputs WHERE node_id = ?", (node_id,))
            conn.executemany(
                """
                INSERT OR IGNORE INTO node_inputs (
                    node_id,
                    input_name,
                    artifact_id
                )
                VALUES (?, ?, ?)
                """,
                [
                    (node_id, input_name, artifact.artifact_id)
                    for input_name, artifacts in inputs.items()
                    for artifact in artifacts
                ],
            )

    def mark_node_succeeded(
        self, node_id: str, artifact_ids: list[str]
    ) -> NodeStatusRecord:
        """Mark a node as succeeded with output artifact ids."""
        now = _now_json()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id,
                    status,
                    execution_policy,
                    placement,
                    current_attempt_id,
                    error,
                    started_at,
                    completed_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    status = excluded.status,
                    error = NULL,
                    completed_at = excluded.completed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    node_id,
                    NodeStatus.SUCCEEDED.value,
                    NodeExecutionPolicy.RERUN.value,
                    NodePlacement.ORCHESTRATOR.value,
                    now,
                    now,
                    now,
                ),
            )
            conn.execute("DELETE FROM node_outputs WHERE node_id = ?", (node_id,))
            conn.executemany(
                """
                INSERT OR IGNORE INTO node_outputs (node_id, artifact_id)
                VALUES (?, ?)
                """,
                [(node_id, artifact_id) for artifact_id in artifact_ids],
            )
        return self._load_node_status_or_default(node_id)

    def mark_node_failed(self, node_id: str, error: str) -> NodeStatusRecord:
        """Mark a node as failed with an error message."""
        now = _now_json()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id,
                    status,
                    execution_policy,
                    placement,
                    current_attempt_id,
                    error,
                    started_at,
                    completed_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    status = excluded.status,
                    error = excluded.error,
                    completed_at = excluded.completed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    node_id,
                    NodeStatus.FAILED.value,
                    NodeExecutionPolicy.RERUN.value,
                    NodePlacement.ORCHESTRATOR.value,
                    error,
                    now,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT current_attempt_id FROM nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if row is not None and row["current_attempt_id"] is not None:
                conn.execute(
                    """
                    UPDATE attempts
                    SET status = ?,
                        completed_at = COALESCE(completed_at, ?),
                        error = COALESCE(error, ?)
                    WHERE node_id = ? AND attempt_id = ?
                    """,
                    (
                        NodeStatus.FAILED.value,
                        now,
                        error,
                        node_id,
                        row["current_attempt_id"],
                    ),
                )
        return self._load_node_status_or_default(node_id)

    def record_attempt_started(self, node_id: str, attempt_id: str) -> AttemptRecord:
        """Record that a node attempt started."""
        now = _now_json()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO attempts (
                    attempt_id,
                    node_id,
                    status,
                    started_at,
                    completed_at,
                    app_result_json,
                    error,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?)
                ON CONFLICT(attempt_id, node_id) DO UPDATE SET
                    status = excluded.status,
                    started_at = COALESCE(attempts.started_at, excluded.started_at),
                    completed_at = NULL,
                    error = NULL
                """,
                (
                    attempt_id,
                    node_id,
                    NodeStatus.RUNNING.value,
                    now,
                    _json_dumps({}),
                ),
            )
        return AttemptRecord(node_id=node_id, attempt_id=attempt_id)

    def record_app_result(
        self, node_id: str, attempt_id: str, result: AppRunResult
    ) -> Path:
        """Record the raw app result for one node attempt in SQLite."""
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE attempts
                SET app_result_json = ?
                WHERE node_id = ? AND attempt_id = ?
                """,
                (result.model_dump_json(), node_id, attempt_id),
            )
        return self.ledger_path

    def load_attempt_app_result(
        self, node_id: str, attempt_id: str
    ) -> AppRunResult | None:
        """Load the recorded app result for one attempt, if present."""
        row = self._fetch_one(
            """
            SELECT app_result_json
            FROM attempts
            WHERE node_id = ? AND attempt_id = ?
            """,
            (node_id, attempt_id),
        )
        if row is None or row["app_result_json"] is None:
            return None
        return AppRunResult.model_validate_json(row["app_result_json"])

    def record_attempt_completed(
        self,
        node_id: str,
        attempt_id: str,
        status: NodeStatus,
        *,
        result: AppRunResult | None = None,
        error: str | None = None,
    ) -> AttemptRecord:
        """Record terminal status for one node attempt."""
        now = _now_json()
        app_result_json = result.model_dump_json() if result is not None else None
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO attempts (
                    attempt_id,
                    node_id,
                    status,
                    started_at,
                    completed_at,
                    app_result_json,
                    error,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id, node_id) DO UPDATE SET
                    status = excluded.status,
                    completed_at = excluded.completed_at,
                    app_result_json = COALESCE(
                        excluded.app_result_json,
                        attempts.app_result_json
                    ),
                    error = excluded.error
                """,
                (
                    attempt_id,
                    node_id,
                    status.value,
                    now,
                    now,
                    app_result_json,
                    error,
                    _json_dumps({}),
                ),
            )
        return AttemptRecord(node_id=node_id, attempt_id=attempt_id, status=status)

    def record_remote_call(
        self,
        *,
        call_id: str,
        node_id: str,
        attempt_id: str,
        function_name: str,
        call_kind: str,
        status: str = "submitted",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a submitted Modal function call before waiting for it."""
        now = _now_json()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO remote_calls (
                    call_id,
                    node_id,
                    attempt_id,
                    function_name,
                    call_kind,
                    status,
                    submitted_at,
                    completed_at,
                    error,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    node_id = excluded.node_id,
                    attempt_id = excluded.attempt_id,
                    function_name = excluded.function_name,
                    call_kind = excluded.call_kind,
                    status = excluded.status,
                    metadata_json = excluded.metadata_json
                """,
                (
                    call_id,
                    node_id,
                    attempt_id,
                    function_name,
                    call_kind,
                    status,
                    now,
                    _json_dumps(metadata or {}),
                ),
            )

    def mark_remote_call_status(
        self,
        call_id: str,
        status: str,
        *,
        error: str | None = None,
        completed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update status for a recorded Modal function call."""
        completed_at = _now_json() if completed else None
        with self._transaction() as conn:
            if completed and metadata is not None:
                conn.execute(
                    """
                    UPDATE remote_calls
                    SET status = ?,
                        error = ?,
                        completed_at = ?,
                        metadata_json = ?
                    WHERE call_id = ?
                    """,
                    (status, error, completed_at, _json_dumps(metadata), call_id),
                )
            elif completed:
                conn.execute(
                    """
                    UPDATE remote_calls
                    SET status = ?,
                        error = ?,
                        completed_at = ?
                    WHERE call_id = ?
                    """,
                    (status, error, completed_at, call_id),
                )
            elif metadata is not None:
                conn.execute(
                    """
                    UPDATE remote_calls
                    SET status = ?,
                        error = ?,
                        metadata_json = ?
                    WHERE call_id = ?
                    """,
                    (status, error, _json_dumps(metadata), call_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE remote_calls
                    SET status = ?,
                        error = ?
                    WHERE call_id = ?
                    """,
                    (status, error, call_id),
                )

    def latest_remote_call(
        self,
        node_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return the latest remote call row for a node."""
        status_values: set[str] | None = None
        if statuses is not None:
            status_values = set(statuses)
            if not status_values:
                return None
        rows = self._fetch_all(
            """
            SELECT *
            FROM remote_calls
            WHERE node_id = ?
            ORDER BY submitted_at DESC, call_id DESC
            """,
            (node_id,),
        )
        for row in rows:
            if status_values is None or row["status"] in status_values:
                return _row_to_dict(row)
        return None

    def load_remote_call(self, call_id: str) -> dict[str, Any] | None:
        """Return one remote call row by id."""
        row = self._fetch_one(
            "SELECT * FROM remote_calls WHERE call_id = ?",
            (call_id,),
        )
        if row is None:
            return None
        return _row_to_dict(row)

    def record_artifacts(self, artifacts: list[WorkflowArtifact]) -> list[Path]:
        """Record workflow artifact manifests in SQLite rows."""
        paths: list[Path] = []
        with self._transaction() as conn:
            for artifact in artifacts:
                conn.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id,
                        producing_node_id,
                        kind,
                        volume_name,
                        storage_path,
                        source_app_output_name,
                        created_at,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        producing_node_id = excluded.producing_node_id,
                        kind = excluded.kind,
                        volume_name = excluded.volume_name,
                        storage_path = excluded.storage_path,
                        source_app_output_name = excluded.source_app_output_name,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        artifact.artifact_id,
                        artifact.producing_node_id,
                        artifact.kind.value,
                        artifact.storage.volume_name,
                        artifact.storage.path,
                        artifact.source_app_output_name,
                        _now_json(),
                        _json_dumps(artifact.metadata),
                    ),
                )
                conn.execute(
                    "DELETE FROM artifact_files WHERE artifact_id = ?",
                    (artifact.artifact_id,),
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO artifact_files (
                        artifact_id,
                        path,
                        role,
                        media_type,
                        size_bytes,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            artifact.artifact_id,
                            file.path,
                            file.role,
                            file.media_type,
                            file.size_bytes,
                            _json_dumps(file.metadata),
                        )
                        for file in artifact.files
                    ],
                )
                paths.append(
                    self.run_root / "artifacts" / f"{artifact.artifact_id}.json"
                )
        return paths

    def load_artifact(self, artifact_id: str) -> WorkflowArtifact:
        """Load one artifact manifest by id."""
        row = self._fetch_one(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if row is None:
            raise FileNotFoundError(f"Workflow artifact not found: {artifact_id}")
        return self._artifact_from_row(row)

    def select_artifacts(self, selector: ArtifactSelector) -> list[WorkflowArtifact]:
        """Return artifacts matching one upstream selector."""
        return [
            artifact
            for artifact in self._load_artifacts_if_any()
            if self._artifact_matches_selector(artifact, selector)
        ]

    def node_is_complete(self, node_id: str) -> bool:
        """Return whether a node has succeeded and all artifacts are recorded."""
        row = self._fetch_one(
            "SELECT status FROM nodes WHERE node_id = ?",
            (node_id,),
        )
        if row is None or row["status"] != NodeStatus.SUCCEEDED.value:
            return False
        output_ids = [
            output_row["artifact_id"]
            for output_row in self._fetch_all(
                "SELECT artifact_id FROM node_outputs WHERE node_id = ?",
                (node_id,),
            )
        ]
        if not output_ids:
            return True
        return all(
            self._fetch_one(
                "SELECT 1 FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            is not None
            for artifact_id in output_ids
        )

    def next_attempt_id(self, node_id: str) -> str:
        """Return the next deterministic attempt id for one node."""
        rows = self._fetch_all(
            "SELECT attempt_id FROM attempts WHERE node_id = ?",
            (node_id,),
        )
        max_suffix = 0
        prefix = "attempt-"
        for row in rows:
            attempt_id = str(row["attempt_id"])
            suffix = attempt_id.removeprefix(prefix)
            if suffix != attempt_id and suffix.isdecimal():
                max_suffix = max(max_suffix, int(suffix))
        return f"attempt-{max_suffix + 1}"

    def close(self) -> None:
        """Close the active SQLite connection, if one is open."""
        self._close()

    @staticmethod
    def _artifact_matches_selector(
        artifact: WorkflowArtifact,
        selector: ArtifactSelector,
    ) -> bool:
        if artifact.producing_node_id != selector.producing_node_id:
            return False
        if selector.kind is not None and artifact.kind != selector.kind:
            return False
        for key, expected in selector.metadata.items():
            if artifact.metadata.get(key) != expected:
                return False
        if selector.pattern is None and selector.role is None:
            return True
        return any(
            (selector.pattern is None or fnmatch(file.path, selector.pattern))
            and (selector.role is None or file.role == selector.role)
            for file in artifact.files
        )

    def _load_node_status_or_default(self, node_id: str) -> NodeStatusRecord:
        row = self._fetch_one("SELECT * FROM nodes WHERE node_id = ?", (node_id,))
        if row is None:
            return NodeStatusRecord(node_id=node_id, status=NodeStatus.PENDING)
        attempts = [
            attempt_row["attempt_id"]
            for attempt_row in self._fetch_all(
                """
                SELECT attempt_id
                FROM attempts
                WHERE node_id = ?
                ORDER BY started_at, attempt_id
                """,
                (node_id,),
            )
        ]
        input_artifact_ids = [
            input_row["artifact_id"]
            for input_row in self._fetch_all(
                """
                SELECT artifact_id
                FROM node_inputs
                WHERE node_id = ?
                ORDER BY input_name, artifact_id
                """,
                (node_id,),
            )
        ]
        output_artifact_ids = [
            output_row["artifact_id"]
            for output_row in self._fetch_all(
                """
                SELECT artifact_id
                FROM node_outputs
                WHERE node_id = ?
                ORDER BY artifact_id
                """,
                (node_id,),
            )
        ]
        return NodeStatusRecord(
            node_id=node_id,
            status=NodeStatus(row["status"]),
            execution_policy=NodeExecutionPolicy(row["execution_policy"]),
            placement=NodePlacement(row["placement"]),
            input_artifact_ids=input_artifact_ids,
            output_artifact_ids=output_artifact_ids,
            attempts=attempts,
            error=row["error"],
        )

    def _load_artifacts_if_any(self) -> list[WorkflowArtifact]:
        return [
            self._artifact_from_row(row)
            for row in self._fetch_all(
                "SELECT * FROM artifacts ORDER BY artifact_id",
            )
        ]

    def _artifact_from_row(self, row: sqlite3.Row) -> WorkflowArtifact:
        file_rows = self._fetch_all(
            """
            SELECT *
            FROM artifact_files
            WHERE artifact_id = ?
            ORDER BY path
            """,
            (row["artifact_id"],),
        )
        return WorkflowArtifact.model_validate({
            "artifact_id": row["artifact_id"],
            "producing_node_id": row["producing_node_id"],
            "kind": row["kind"],
            "storage": VolumePath(
                volume_name=row["volume_name"],
                path=row["storage_path"],
            ),
            "files": [
                {
                    "path": file_row["path"],
                    "role": file_row["role"],
                    "media_type": file_row["media_type"],
                    "size_bytes": file_row["size_bytes"],
                    "metadata": _json_loads(file_row["metadata_json"]),
                }
                for file_row in file_rows
            ],
            "source_app_output_name": row["source_app_output_name"],
            "metadata": _json_loads(row["metadata_json"]),
        })

    def _upsert_node_status(self, node_id: str, status: NodeStatus) -> None:
        now = _now_json()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id,
                    status,
                    execution_policy,
                    placement,
                    current_attempt_id,
                    error,
                    started_at,
                    completed_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    node_id,
                    status.value,
                    NodeExecutionPolicy.RERUN.value,
                    NodePlacement.ORCHESTRATOR.value,
                    now,
                ),
            )

    def _create_run_layout(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for name in ("inputs", "nodes", "artifacts", "final"):
            self.run_root.joinpath(name).mkdir(exist_ok=True)

    def _activate(self, workflow_name: str, run_id: str) -> None:
        if self.workflow_name == workflow_name and self.run_id == run_id:
            return
        self._close()
        self.workflow_name = workflow_name
        self.run_id = run_id

    @contextmanager
    def _transaction(self):
        with self._lock:
            conn = self._connect()
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def _fetch_one(
        self,
        sql: str,
        params: Iterable[Any] = (),
    ) -> sqlite3.Row | None:
        with self._lock:
            return self._connect().execute(sql, tuple(params)).fetchone()

    def _fetch_all(
        self,
        sql: str,
        params: Iterable[Any] = (),
    ) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connect().execute(sql, tuple(params)).fetchall())

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self.run_root.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.ledger_path,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._ensure_schema()
        return self._connection

    def _ensure_schema(self) -> None:
        if self._connection is None:
            raise RuntimeError("SQLite connection has not been opened")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                dag_hash TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                execution_policy TEXT NOT NULL,
                placement TEXT NOT NULL,
                current_attempt_id TEXT,
                error TEXT,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                app_result_json TEXT,
                error TEXT,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (attempt_id, node_id)
            );

            CREATE TABLE IF NOT EXISTS remote_calls (
                call_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                function_name TEXT NOT NULL,
                call_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                completed_at TEXT,
                error TEXT,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS node_tasks (
                task_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                attempt_id TEXT,
                status TEXT NOT NULL,
                input_artifact_id TEXT,
                output_artifact_id TEXT,
                remote_call_id TEXT,
                claimed_by TEXT,
                started_at TEXT,
                completed_at TEXT,
                error TEXT,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                producing_node_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                volume_name TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                source_app_output_name TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifact_files (
                artifact_id TEXT NOT NULL,
                path TEXT NOT NULL,
                role TEXT,
                media_type TEXT,
                size_bytes INTEGER,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (artifact_id, path)
            );

            CREATE TABLE IF NOT EXISTS node_inputs (
                node_id TEXT NOT NULL,
                input_name TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                PRIMARY KEY (node_id, input_name, artifact_id)
            );

            CREATE TABLE IF NOT EXISTS node_outputs (
                node_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                PRIMARY KEY (node_id, artifact_id)
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_status
                ON nodes(status);
            CREATE INDEX IF NOT EXISTS idx_attempts_node
                ON attempts(node_id);
            CREATE INDEX IF NOT EXISTS idx_remote_calls_node_status
                ON remote_calls(node_id, status);
            CREATE INDEX IF NOT EXISTS idx_node_tasks_node_status
                ON node_tasks(node_id, status);
            CREATE INDEX IF NOT EXISTS idx_artifacts_producing_node
                ON artifacts(producing_node_id);
            """
        )
        self._connection.commit()

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _require_workflow_name(self) -> str:
        if self.workflow_name is None:
            raise RuntimeError("Workflow run has not been initialized")
        return self.workflow_name

    def _require_run_id(self) -> str:
        if self.run_id is None:
            raise RuntimeError("Workflow run has not been initialized")
        return self.run_id

    @staticmethod
    def _read_json(path: Path) -> object:
        """Read JSON files written by artifact materialization helpers."""
        return orjson.loads(path.read_bytes())


def _run_from_row(row: sqlite3.Row) -> WorkflowRun:
    return WorkflowRun(
        workflow_name=row["workflow_name"],
        run_id=row["run_id"],
        status=RunStatus(row["status"]),
        dag_hash=row["dag_hash"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=_json_loads(row["metadata_json"]),
    )


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _json_dumps(value: object) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def _json_loads(value: str | bytes | None) -> dict[str, Any]:
    if not value:
        return {}
    return orjson.loads(value)


def _now_json() -> str:
    return _datetime_json(datetime.now(UTC))


def _datetime_json(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()
