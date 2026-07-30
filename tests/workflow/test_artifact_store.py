"""Workflow-owned artifact persistence beside shared execution state."""

# ruff: noqa: D103

import sqlite3
from uuid import UUID

import pytest

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    SqliteExecutionRepository,
    TaskPlan,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    ArtifactSelector,
    InlineBytes,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow.core.artifact_store import (
    WORKFLOW_ARTIFACT_TABLES,
    WorkflowArtifactStore,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
EXPECTED_EXECUTION_TABLES = {
    "execution_dispatch_batches",
    "execution_node_dependencies",
    "execution_nodes",
    "execution_provider_calls",
    "execution_runs",
    "execution_schema",
    "execution_task_claim_requests",
    "execution_task_completion_requests",
    "execution_tasks",
    "execution_worker_assignments",
}


def _stores() -> tuple[
    sqlite3.Connection,
    SqliteExecutionRepository,
    WorkflowArtifactStore,
]:
    connection = sqlite3.connect(":memory:")
    execution = SqliteExecutionRepository(connection)
    execution.initialize_schema()
    artifacts = WorkflowArtifactStore(connection)
    artifacts.initialize_schema()
    connection.commit()
    return connection, execution, artifacts


def _start_task(execution: SqliteExecutionRepository) -> None:
    execution.create_run(
        execution_run_id=RUN_ID,
        plan=ExecutionPlan(
            workload_name="workflow:demo",
            nodes=(NodePlan(node_key="design"),),
        ),
        deployment=DeploymentIdentity("production", "DemoWorkflow", 3),
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=1,
        now=100,
    )
    execution.start_node(RUN_ID, "design", now=101)
    execution.discover_tasks(
        RUN_ID,
        "design",
        (
            TaskPlan(
                task_key="node",
                scientific_payload={"workflow_node_id": "design"},
            ),
        ),
        now=102,
    )
    execution.record_task_result_observation(
        RUN_ID,
        "design",
        "node",
        AvailabilityStatus.MISSING,
        now=103,
    )


def _publication() -> tuple[AppRunResult, WorkflowArtifact]:
    storage = VolumePath(
        volume_name="Workflow-outputs",
        path="demo/run/design/model.pdb",
    )
    artifact = WorkflowArtifact(
        artifact_id="design-structure",
        producing_node_id="design",
        kind=ArtifactKind.STRUCTURES,
        storage=storage,
        files=[
            ArtifactFile(
                path="model.pdb",
                role="structure",
                media_type="chemical/x-pdb",
                size_bytes=12,
            )
        ],
        metadata={"candidate_id": "candidate-1"},
    )
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="structure",
                kind=ArtifactKind.STRUCTURES,
                storage=storage,
            )
        ],
    )
    return result, artifact


def test_artifact_publication_and_task_completion_share_caller_transaction() -> None:
    connection, execution, artifacts = _stores()
    _start_task(execution)
    connection.commit()
    result, artifact = _publication()

    artifacts.record_node_publication(
        "design",
        result=result,
        artifacts=(artifact,),
        now=110,
    )
    execution.record_task_result_observation(
        RUN_ID,
        "design",
        "node",
        AvailabilityStatus.AVAILABLE,
        now=111,
    )
    connection.rollback()

    assert artifacts.load_node_result("design") is None
    assert artifacts.load_node_output_artifacts("design") == ()
    assert execution.get_task(RUN_ID, "design", "node").status.value == "pending"

    artifacts.record_node_publication(
        "design",
        result=result,
        artifacts=(artifact,),
        now=120,
    )
    execution.record_task_result_observation(
        RUN_ID,
        "design",
        "node",
        AvailabilityStatus.AVAILABLE,
        now=121,
    )
    connection.commit()

    assert artifacts.load_node_result("design") == result
    assert artifacts.load_node_output_artifacts("design") == (artifact,)
    assert artifacts.select_artifacts(
        ArtifactSelector(
            producing_node_id="design",
            kind=ArtifactKind.STRUCTURES,
            role="structure",
        )
    ) == (artifact,)
    assert execution.get_task(RUN_ID, "design", "node").status.value == "succeeded"


def test_schema_contains_shared_execution_and_workflow_artifacts_only() -> None:
    connection, _, _ = _stores()

    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert EXPECTED_EXECUTION_TABLES.issubset(tables)
    assert set(WORKFLOW_ARTIFACT_TABLES).issubset(tables)
    assert not {"runs", "nodes", "attempts", "remote_calls"} & tables


def test_publication_rejects_unmaterialized_inline_bytes() -> None:
    _, _, artifacts = _stores()
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="archive",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(data=b"result", filename="result.zip"),
            )
        ],
    )

    with pytest.raises(ValueError, match="InlineBytes"):
        artifacts.record_node_publication(
            "design",
            result=result,
            artifacts=(),
            now=100,
        )


def test_exact_publication_replay_is_idempotent_but_divergence_is_rejected() -> None:
    connection, _, artifacts = _stores()
    result, artifact = _publication()

    artifacts.record_node_publication(
        "design",
        result=result,
        artifacts=(artifact,),
        now=100,
    )
    connection.commit()
    artifacts.record_node_publication(
        "design",
        result=result,
        artifacts=(artifact,),
        now=200,
    )

    divergent = result.model_copy(update={"warnings": ["different"]})
    with pytest.raises(ValueError, match="publication already exists"):
        artifacts.record_node_publication(
            "design",
            result=divergent,
            artifacts=(artifact,),
            now=200,
        )


def test_task_publications_are_durable_idempotent_and_node_reusable() -> None:
    connection, _, artifacts = _stores()
    result, artifact = _publication()

    artifacts.record_task_publication(
        "design",
        "candidate-1",
        result=result,
        artifacts=(artifact,),
        now=100,
    )
    artifacts.record_task_publication(
        "design",
        "candidate-1",
        result=result,
        artifacts=(artifact,),
        now=200,
    )
    artifacts.record_node_publication(
        "design",
        result=result,
        artifacts=(artifact,),
        now=200,
    )
    connection.commit()

    assert artifacts.list_task_publication_keys("design") == ("candidate-1",)
    assert artifacts.load_task_result("design", "candidate-1") == result
    assert artifacts.load_task_output_artifacts(
        "design",
        "candidate-1",
    ) == (artifact,)
    assert artifacts.load_node_output_artifacts("design") == (artifact,)

    artifacts.discard_node_publication("design")
    connection.commit()
    assert artifacts.load_artifact(artifact.artifact_id) == artifact
    assert artifacts.load_task_result("design", "candidate-1") == result

    artifacts.discard_task_publication("design", "candidate-1")
    connection.commit()
    assert artifacts.load_task_result("design", "candidate-1") is None
    with pytest.raises(FileNotFoundError):
        artifacts.load_artifact(artifact.artifact_id)


def test_task_publication_rejects_divergent_replay() -> None:
    _, _, artifacts = _stores()
    result, artifact = _publication()
    artifacts.record_task_publication(
        "design",
        "candidate-1",
        result=result,
        artifacts=(artifact,),
        now=100,
    )

    with pytest.raises(ValueError, match="Task publication already exists"):
        artifacts.record_task_publication(
            "design",
            "candidate-1",
            result=result.model_copy(update={"warnings": ["changed"]}),
            artifacts=(artifact,),
            now=200,
        )


def test_invalid_copied_publication_can_be_discarded_before_replacement() -> None:
    connection, _, artifacts = _stores()
    result, artifact = _publication()
    artifacts.record_node_publication(
        "design",
        result=result,
        artifacts=(artifact,),
        now=100,
    )
    connection.commit()

    artifacts.discard_node_publication("design")
    connection.commit()

    assert artifacts.load_node_result("design") is None
    assert artifacts.load_node_output_artifacts("design") == ()
    with pytest.raises(FileNotFoundError):
        artifacts.load_artifact(artifact.artifact_id)

    replacement = result.model_copy(update={"warnings": ["recomputed"]})
    artifacts.record_node_publication(
        "design",
        result=replacement,
        artifacts=(artifact,),
        now=200,
    )
    connection.commit()
    assert artifacts.load_node_result("design") == replacement


def test_input_links_and_artifact_order_survive_round_trip() -> None:
    connection, _, artifacts = _stores()
    result, first = _publication()
    second = first.model_copy(
        update={
            "artifact_id": "design-scores",
            "kind": ArtifactKind.SCORES,
            "storage": VolumePath(
                volume_name="Workflow-outputs",
                path="demo/run/design/scores.csv",
                media_type="text/csv",
            ),
            "files": [
                ArtifactFile(
                    path="scores.csv",
                    role="scores",
                    media_type="text/csv",
                    metadata={"ranked": True},
                )
            ],
        }
    )
    artifacts.record_node_publication(
        "design",
        result=result,
        artifacts=(first, second),
        now=100,
    )
    artifacts.record_node_inputs(
        "summarize",
        {"structures": [first], "scores": [second]},
    )
    connection.commit()

    assert artifacts.load_node_output_artifacts("design") == (first, second)
    assert artifacts.load_artifact("design-scores") == second
    assert artifacts.select_artifacts(
        ArtifactSelector(
            producing_node_id="design",
            pattern="*.csv",
            metadata={"candidate_id": "candidate-1"},
        )
    ) == (second,)
    input_rows = connection.execute(
        """
        SELECT input_name, ordinal, artifact_id
        FROM workflow_node_inputs
        WHERE node_key = ?
        ORDER BY input_name, ordinal
        """,
        ("summarize",),
    ).fetchall()
    assert [tuple(row) for row in input_rows] == [
        ("scores", 0, "design-scores"),
        ("structures", 0, "design-structure"),
    ]
