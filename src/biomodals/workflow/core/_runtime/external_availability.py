"""App-owned artifact availability checks for strict workflow recovery.

Workflow-volume artifact checks remain the default. Strict external checks use
one caller-provided checker so workflow-specific Modal code can mount the app
volumes it needs without importing Modal into the reusable runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from biomodals.schema import WorkflowArtifact
from biomodals.workflow.core.artifacts import workflow_artifact_availability_errors

ExternalArtifactChecker = Callable[[WorkflowArtifact], list[str]]


def check_external_artifact_availability(
    artifact: WorkflowArtifact,
    *,
    workflow_volume_name: str,
    volume_roots: Mapping[str, str | Path],
) -> list[str]:
    """Return availability errors for an artifact in an app-owned volume.

    This helper is intentionally pure Python. Workflow modules can call it from
    a lightweight Modal function that has the required app volumes mounted, then
    pass that function through the runtime's external checker hook.
    """
    if artifact.storage.volume_name == workflow_volume_name:
        return []

    volume_root = volume_roots.get(artifact.storage.volume_name)
    if volume_root is None:
        return [
            f"{artifact.artifact_id}: missing mounted volume root for "
            f"external volume {artifact.storage.volume_name!r}"
        ]
    return workflow_artifact_availability_errors(
        artifact,
        workflow_volume_name=artifact.storage.volume_name,
        volume_root=Path(volume_root),
    )


def mounted_volume_checker(
    *,
    workflow_volume_name: str,
    volume_roots: Mapping[str, str | Path],
) -> ExternalArtifactChecker:
    """Build a checker for already-mounted app-owned volume roots."""
    roots = {volume_name: Path(root) for volume_name, root in volume_roots.items()}

    def check(artifact: WorkflowArtifact) -> list[str]:
        return check_external_artifact_availability(
            artifact,
            workflow_volume_name=workflow_volume_name,
            volume_roots=roots,
        )

    return check
