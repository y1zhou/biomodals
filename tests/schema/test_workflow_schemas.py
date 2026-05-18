"""Tests for shared workflow schema contracts."""

# ruff: noqa: D103

from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    StorageKind,
    VolumePath,
    WorkflowArtifact,
)


def test_inline_bytes_round_trip() -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="packed",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=b"archive",
                    filename="packed.tar.zst",
                    archive_format="tar.zst",
                ),
            )
        ],
    )

    loaded = AppRunResult.model_validate(result.model_dump())

    assert loaded.outputs[0].storage.kind == StorageKind.INLINE_BYTES
    assert loaded.outputs[0].storage.filename == "packed.tar.zst"


def test_workflow_artifact_is_volume_backed() -> None:
    artifact = WorkflowArtifact(
        artifact_id="art-packed",
        producing_node_id="packed",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(
            volume_name="Workflow-outputs",
            path="ppiflow/run-1/artifacts/art-packed",
        ),
    )

    assert artifact.storage.path == "ppiflow/run-1/artifacts/art-packed"
