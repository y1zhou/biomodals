"""Tests for reusable workflow node helpers."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from uuid import UUID

import pytest

from biomodals.schema import ArtifactKind, VolumePath, WorkflowArtifact
from biomodals.workflow.core.nodes import AppBackedNode, NodeRunContext


def test_app_backed_node_requires_caller_owned_remote_preparation(
    tmp_path: Path,
) -> None:
    node = AppBackedNode()
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


def test_app_backed_node_owns_no_modal_lookup_or_submission_api() -> None:
    assert not hasattr(AppBackedNode, "app_name")
    assert not hasattr(AppBackedNode, "function_name")
    assert not hasattr(AppBackedNode, "load_app_function")
    assert not hasattr(AppBackedNode, "invoke_app_function")
    assert not hasattr(AppBackedNode, "submit_remote")


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
        workflow_volume_name="workflow-volume",
    )
    artifact = WorkflowArtifact(
        artifact_id="input",
        producing_node_id="upstream",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(
            volume_name="workflow-volume",
            path="runs/demo/input.pdb",
        ),
    )

    assert (
        context.resolve_workflow_artifact(artifact)
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
        workflow_volume_name="workflow-volume",
    )
    artifact = WorkflowArtifact(
        artifact_id="input",
        producing_node_id="upstream",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath.model_construct(
            volume_name="workflow-volume",
            path=path,
        ),
    )

    with pytest.raises(ValueError, match="relative and contained"):
        context.resolve_workflow_artifact(artifact)
