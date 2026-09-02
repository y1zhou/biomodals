"""Tests for reusable execution Node helpers."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from uuid import UUID

import pytest

from biomodals.execution.nodes import (
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
    TaskDefinition,
)
from biomodals.schema import ArtifactKind, ExecutionArtifact, VolumePath


def test_app_backed_node_requires_caller_owned_remote_preparation(
    tmp_path: Path,
) -> None:
    node = ProviderNode()
    context = NodeRunContext(
        execution_run_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        workload_run_key="demo",
        node_id="remote",
        task_key="node",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={},
    )

    with pytest.raises(NotImplementedError):
        node.prepare_remote(context)


def test_remote_task_node_declares_data() -> None:
    task = TaskDefinition(
        task_key="candidate-a",
        scientific_payload={"candidate_id": "candidate-a"},
        execution_payload={"candidate_path": "inputs/candidate-a.pdb"},
    )

    assert task.task_key == "candidate-a"
    assert task.execution_payload == {"candidate_path": "inputs/candidate-a.pdb"}
    with pytest.raises(ValueError, match="cannot be empty"):
        TaskDefinition(
            task_key="",
            scientific_payload={},
        )
    with pytest.raises(ValueError, match="max_tasks_per_call must be positive"):
        ProviderCallSpec(
            function_name="run_candidate",
            uses_gpu=False,
            max_tasks_per_call=0,
        )
    with pytest.raises(ValueError, match="identity keyword"):
        ProviderCallSpec(
            function_name="run_candidate",
            uses_gpu=False,
            kwargs={"claim_owner": "duplicate"},
            provider_call_id_kwarg="claim_owner",
        )


def test_node_context_resolves_workflow_artifact(tmp_path: Path) -> None:
    context = NodeRunContext(
        execution_run_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        workload_run_key="demo",
        node_id="local",
        task_key="node",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={},
        volume_root=tmp_path,
        artifact_volume_name="workflow-volume",
    )
    artifact = ExecutionArtifact(
        artifact_id="input",
        producing_node_id="upstream",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(
            volume_name="workflow-volume",
            path="runs/demo/input.pdb",
        ),
    )

    assert (
        context.resolve_artifact(artifact)
        == (tmp_path / "runs/demo/input.pdb").resolve()
    )


@pytest.mark.parametrize("path", ["/absolute/input.pdb", "../outside.pdb"])
def test_node_context_rejects_uncontained_artifact(
    tmp_path: Path,
    path: str,
) -> None:
    context = NodeRunContext(
        execution_run_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        workload_run_key="demo",
        node_id="local",
        task_key="node",
        work_dir=tmp_path / "result",
        cache_dir=tmp_path / "cache",
        inputs={},
        volume_root=tmp_path,
        artifact_volume_name="workflow-volume",
    )
    artifact = ExecutionArtifact(
        artifact_id="input",
        producing_node_id="upstream",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath.model_construct(
            volume_name="workflow-volume",
            path=path,
        ),
    )

    with pytest.raises(ValueError, match="relative and contained"):
        context.resolve_artifact(artifact)
