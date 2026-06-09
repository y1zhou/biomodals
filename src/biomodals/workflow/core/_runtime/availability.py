"""Read-only workflow artifact availability checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from biomodals.schema import WorkflowArtifact
from biomodals.workflow.core.artifacts import workflow_artifact_availability_errors


@dataclass(frozen=True)
class ArtifactAvailabilityError:
    """A private structured artifact availability error."""

    artifact_id: str
    reason: Literal["missing_manifest", "workflow_artifact_unavailable"]
    relative_path: str
    detail: str

    def format(self) -> str:
        """Return the user-facing error text."""
        if self.reason == "missing_manifest":
            return (
                f"{self.artifact_id}: missing workflow artifact manifest "
                f"{self.relative_path}"
            )
        return self.detail


def artifact_availability_errors(
    artifact: WorkflowArtifact,
    *,
    workflow_volume_name: str,
    volume_root: Path,
    run_root: Path,
) -> list[ArtifactAvailabilityError]:
    """Return read-only availability errors for one workflow artifact."""
    errors: list[ArtifactAvailabilityError] = []
    relative_manifest_path = f"artifacts/{artifact.artifact_id}.json"
    manifest_path = run_root / relative_manifest_path
    if not manifest_path.is_file():
        errors.append(
            ArtifactAvailabilityError(
                artifact_id=artifact.artifact_id,
                reason="missing_manifest",
                relative_path=relative_manifest_path,
                detail=(
                    f"{artifact.artifact_id}: missing workflow artifact manifest "
                    f"{relative_manifest_path}"
                ),
            )
        )
    errors.extend(
        ArtifactAvailabilityError(
            artifact_id=artifact.artifact_id,
            reason="workflow_artifact_unavailable",
            relative_path=artifact.storage.path,
            detail=error,
        )
        for error in workflow_artifact_availability_errors(
            artifact,
            workflow_volume_name=workflow_volume_name,
            volume_root=volume_root,
        )
    )
    return errors


def format_artifact_availability_errors(
    errors: list[ArtifactAvailabilityError],
) -> list[str]:
    """Format structured availability errors for logs and exceptions."""
    return [error.format() for error in errors]
