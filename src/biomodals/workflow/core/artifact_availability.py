"""Typed workflow artifact availability checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from biomodals.schema import WorkflowArtifact
from biomodals.workflow.core.artifacts import workflow_artifact_availability_errors

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from enum import StrEnum  # type: ignore[no-redef]


class ArtifactAvailabilityStatus(StrEnum):
    """Artifact availability states understood by the workflow runtime."""

    AVAILABLE = "available"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArtifactAvailability:
    """Structured result for one artifact availability check."""

    artifact_id: str
    status: ArtifactAvailabilityStatus
    errors: tuple[str, ...] = field(default_factory=tuple)
    unknown_reason: str | None = None


ExternalArtifactChecker = Callable[
    [WorkflowArtifact], ArtifactAvailability | list[str] | tuple[str, ...]
]


def check_artifact_availability(
    artifact: WorkflowArtifact,
    *,
    workflow_volume_name: str,
    volume_root: str | Path,
    external_artifact_checker: ExternalArtifactChecker | None = None,
) -> ArtifactAvailability:
    """Return a typed availability state for one workflow artifact."""
    if artifact.storage.volume_name == workflow_volume_name:
        errors = workflow_artifact_availability_errors(
            artifact,
            workflow_volume_name=workflow_volume_name,
            volume_root=Path(volume_root),
        )
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=(
                ArtifactAvailabilityStatus.MISSING
                if errors
                else ArtifactAvailabilityStatus.AVAILABLE
            ),
            errors=tuple(errors),
        )

    if external_artifact_checker is None:
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=ArtifactAvailabilityStatus.UNKNOWN,
            unknown_reason=(
                f"external volume {artifact.storage.volume_name!r} was not checked"
            ),
        )

    try:
        return _normalize_external_check_result(artifact, external_artifact_checker)
    except Exception as exc:  # noqa: BLE001
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=ArtifactAvailabilityStatus.UNKNOWN,
            unknown_reason=(
                f"{artifact.artifact_id}: external artifact checker failed: {exc}"
            ),
        )


def check_external_artifact_availability(
    artifact: WorkflowArtifact,
    *,
    workflow_volume_name: str,
    volume_roots: Mapping[str, str | Path],
) -> list[str]:
    """Return missing-file errors for an artifact in an app-owned volume."""
    availability = check_external_artifact_status(
        artifact,
        workflow_volume_name=workflow_volume_name,
        volume_roots=volume_roots,
    )
    if availability.status == ArtifactAvailabilityStatus.MISSING:
        return list(availability.errors)
    if availability.status == ArtifactAvailabilityStatus.UNKNOWN:
        return [availability.unknown_reason or "external artifact availability unknown"]
    return []


def check_external_artifact_status(
    artifact: WorkflowArtifact,
    *,
    workflow_volume_name: str,
    volume_roots: Mapping[str, str | Path],
) -> ArtifactAvailability:
    """Return a typed availability state for an app-owned volume artifact."""
    if artifact.storage.volume_name == workflow_volume_name:
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=ArtifactAvailabilityStatus.AVAILABLE,
        )

    volume_root = volume_roots.get(artifact.storage.volume_name)
    if volume_root is None:
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=ArtifactAvailabilityStatus.UNKNOWN,
            unknown_reason=(
                f"{artifact.artifact_id}: missing mounted volume root for "
                f"external volume {artifact.storage.volume_name!r}"
            ),
        )

    errors = workflow_artifact_availability_errors(
        artifact,
        workflow_volume_name=artifact.storage.volume_name,
        volume_root=Path(volume_root),
    )
    return ArtifactAvailability(
        artifact_id=artifact.artifact_id,
        status=(
            ArtifactAvailabilityStatus.MISSING
            if errors
            else ArtifactAvailabilityStatus.AVAILABLE
        ),
        errors=tuple(errors),
    )


def mounted_volume_checker(
    *,
    workflow_volume_name: str,
    volume_roots: Mapping[str, str | Path],
) -> ExternalArtifactChecker:
    """Build a typed checker for already-mounted app-owned volume roots."""
    roots = {volume_name: Path(root) for volume_name, root in volume_roots.items()}

    def check(artifact: WorkflowArtifact) -> ArtifactAvailability:
        return check_external_artifact_status(
            artifact,
            workflow_volume_name=workflow_volume_name,
            volume_roots=roots,
        )

    return check


def _normalize_external_check_result(
    artifact: WorkflowArtifact,
    external_artifact_checker: ExternalArtifactChecker,
) -> ArtifactAvailability:
    result = external_artifact_checker(artifact)
    if isinstance(result, ArtifactAvailability):
        return result
    errors = tuple(result)
    return ArtifactAvailability(
        artifact_id=artifact.artifact_id,
        status=(
            ArtifactAvailabilityStatus.MISSING
            if errors
            else ArtifactAvailabilityStatus.AVAILABLE
        ),
        errors=errors,
    )
