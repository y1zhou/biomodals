"""Workflow-owned artifacts colocated with shared execution state."""

from __future__ import annotations

import sqlite3
from fnmatch import fnmatch

import orjson

from biomodals.schema import (
    AppRunResult,
    ArtifactSelector,
    InlineBytes,
    VolumePath,
    WorkflowArtifact,
)

WORKFLOW_ARTIFACT_TABLES = (
    "workflow_artifacts",
    "workflow_artifact_files",
    "workflow_node_inputs",
    "workflow_node_outputs",
    "workflow_node_results",
)


class WorkflowArtifactStore:
    """Persist workflow publications without owning execution state."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Bind a caller-owned connection without committing or closing it."""
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def initialize_schema(self) -> None:
        """Create only the workflow-specific artifact tables."""
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflow_artifacts (
                artifact_id TEXT PRIMARY KEY,
                producing_node_key TEXT NOT NULL,
                kind TEXT NOT NULL,
                volume_name TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                storage_media_type TEXT,
                source_app_output_name TEXT,
                created_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workflow_artifact_files (
                artifact_id TEXT NOT NULL
                    REFERENCES workflow_artifacts(artifact_id)
                    ON DELETE CASCADE,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                path TEXT NOT NULL,
                role TEXT,
                media_type TEXT,
                size_bytes INTEGER,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (artifact_id, path),
                UNIQUE (artifact_id, ordinal)
            );

            CREATE TABLE IF NOT EXISTS workflow_node_inputs (
                node_key TEXT NOT NULL,
                input_name TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                artifact_id TEXT NOT NULL
                    REFERENCES workflow_artifacts(artifact_id),
                PRIMARY KEY (node_key, input_name, artifact_id),
                UNIQUE (node_key, input_name, ordinal)
            );

            CREATE TABLE IF NOT EXISTS workflow_node_outputs (
                node_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                artifact_id TEXT NOT NULL UNIQUE
                    REFERENCES workflow_artifacts(artifact_id),
                PRIMARY KEY (node_key, ordinal)
            );

            CREATE TABLE IF NOT EXISTS workflow_node_results (
                node_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                completed_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS workflow_artifacts_node
                ON workflow_artifacts(producing_node_key);
            CREATE INDEX IF NOT EXISTS workflow_node_inputs_artifact
                ON workflow_node_inputs(artifact_id);
            """
        )

    def record_node_inputs(
        self,
        node_key: str,
        inputs: dict[str, list[WorkflowArtifact]],
    ) -> None:
        """Replace one Node's resolved artifact-input links."""
        self._connection.execute(
            "DELETE FROM workflow_node_inputs WHERE node_key = ?",
            (node_key,),
        )
        for input_name, selected in inputs.items():
            self._connection.executemany(
                """
                INSERT INTO workflow_node_inputs (
                    node_key,
                    input_name,
                    ordinal,
                    artifact_id
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (node_key, input_name, ordinal, artifact.artifact_id)
                    for ordinal, artifact in enumerate(selected)
                ],
            )

    def record_node_publication(
        self,
        node_key: str,
        *,
        result: AppRunResult,
        artifacts: tuple[WorkflowArtifact, ...],
        now: int,
    ) -> None:
        """Atomically stage one immutable Node result on the caller transaction."""
        _raise_for_inline_bytes_result(result)
        for artifact in artifacts:
            if artifact.producing_node_id != node_key:
                raise ValueError("Workflow artifact producer does not match Node")

        existing_result = self.load_node_result(node_key)
        if existing_result is not None:
            existing_artifacts = self.load_node_output_artifacts(node_key)
            if existing_result == result and existing_artifacts == artifacts:
                return
            raise ValueError(f"Workflow Node publication already exists: {node_key}")

        for artifact in artifacts:
            self._insert_artifact(artifact, now=now)
        self._connection.executemany(
            """
            INSERT INTO workflow_node_outputs (node_key, ordinal, artifact_id)
            VALUES (?, ?, ?)
            """,
            [
                (node_key, ordinal, artifact.artifact_id)
                for ordinal, artifact in enumerate(artifacts)
            ],
        )
        self._connection.execute(
            """
            INSERT INTO workflow_node_results (node_key, result_json, completed_at)
            VALUES (?, ?, ?)
            """,
            (node_key, result.model_dump_json(), now),
        )

    def load_node_result(self, node_key: str) -> AppRunResult | None:
        """Load a materialized Node result, if publication completed."""
        row = self._connection.execute(
            """
            SELECT result_json
            FROM workflow_node_results
            WHERE node_key = ?
            """,
            (node_key,),
        ).fetchone()
        if row is None:
            return None
        return AppRunResult.model_validate_json(row["result_json"])

    def discard_node_publication(self, node_key: str) -> None:
        """Remove one conclusively invalid publication before replacement work."""
        artifact_rows = self._connection.execute(
            """
            SELECT artifact_id
            FROM workflow_node_outputs
            WHERE node_key = ?
            """,
            (node_key,),
        ).fetchall()
        artifact_ids = tuple(str(row["artifact_id"]) for row in artifact_rows)
        self._connection.execute(
            "DELETE FROM workflow_node_results WHERE node_key = ?",
            (node_key,),
        )
        self._connection.execute(
            "DELETE FROM workflow_node_outputs WHERE node_key = ?",
            (node_key,),
        )
        self._connection.executemany(
            "DELETE FROM workflow_artifacts WHERE artifact_id = ?",
            [(artifact_id,) for artifact_id in artifact_ids],
        )

    def load_artifact(self, artifact_id: str) -> WorkflowArtifact:
        """Load one artifact manifest by stable identity."""
        row = self._connection.execute(
            """
            SELECT *
            FROM workflow_artifacts
            WHERE artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Workflow artifact not found: {artifact_id}")
        return self._artifact_from_row(row)

    def load_node_output_artifacts(
        self,
        node_key: str,
    ) -> tuple[WorkflowArtifact, ...]:
        """Load one Node publication's artifacts in output encounter order."""
        rows = self._connection.execute(
            """
            SELECT artifact_id
            FROM workflow_node_outputs
            WHERE node_key = ?
            ORDER BY ordinal
            """,
            (node_key,),
        ).fetchall()
        return tuple(self.load_artifact(str(row["artifact_id"])) for row in rows)

    def select_artifacts(
        self,
        selector: ArtifactSelector,
    ) -> tuple[WorkflowArtifact, ...]:
        """Select upstream artifacts using the workflow's typed contract."""
        return tuple(
            artifact
            for artifact in self.load_node_output_artifacts(selector.producing_node_id)
            if _artifact_matches_selector(artifact, selector)
        )

    def _insert_artifact(self, artifact: WorkflowArtifact, *, now: int) -> None:
        self._connection.execute(
            """
            INSERT INTO workflow_artifacts (
                artifact_id,
                producing_node_key,
                kind,
                volume_name,
                storage_path,
                storage_media_type,
                source_app_output_name,
                created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_id,
                artifact.producing_node_id,
                artifact.kind.value,
                artifact.storage.volume_name,
                artifact.storage.path,
                artifact.storage.media_type,
                artifact.source_app_output_name,
                now,
                _json_dumps(artifact.metadata),
            ),
        )
        self._connection.executemany(
            """
            INSERT INTO workflow_artifact_files (
                artifact_id,
                ordinal,
                path,
                role,
                media_type,
                size_bytes,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    artifact.artifact_id,
                    ordinal,
                    file.path,
                    file.role,
                    file.media_type,
                    file.size_bytes,
                    _json_dumps(file.metadata),
                )
                for ordinal, file in enumerate(artifact.files)
            ],
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> WorkflowArtifact:
        file_rows = self._connection.execute(
            """
            SELECT *
            FROM workflow_artifact_files
            WHERE artifact_id = ?
            ORDER BY ordinal
            """,
            (row["artifact_id"],),
        ).fetchall()
        return WorkflowArtifact.model_validate({
            "artifact_id": row["artifact_id"],
            "producing_node_id": row["producing_node_key"],
            "kind": row["kind"],
            "storage": VolumePath(
                volume_name=row["volume_name"],
                path=row["storage_path"],
                media_type=row["storage_media_type"],
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


def _raise_for_inline_bytes_result(result: AppRunResult) -> None:
    inline_outputs = [
        output.name
        for output in [*result.outputs, *result.logs]
        if isinstance(output.storage, InlineBytes)
    ]
    if inline_outputs:
        raise ValueError(
            "AppRunResult must be materialized before artifact storage; "
            f"InlineBytes outputs: {', '.join(sorted(inline_outputs))}"
        )


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


def _json_dumps(value: object) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()


def _json_loads(value: str | bytes | None) -> dict[str, object]:
    if not value:
        return {}
    loaded = orjson.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("Workflow artifact metadata must be an object")
    return loaded
