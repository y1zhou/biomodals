"""Typed execution-artifact availability checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from biomodals.execution.artifacts import execution_artifact_availability_errors
from biomodals.execution.model import AvailabilityStatus
from biomodals.schema import ExecutionArtifact


@dataclass(frozen=True)
class ArtifactAvailability:
    """Structured result for one artifact availability check."""

    artifact_id: str
    status: AvailabilityStatus
    errors: tuple[str, ...] = field(default_factory=tuple)
    unknown_reason: str | None = None


ExternalArtifactChecker = Callable[[ExecutionArtifact], ArtifactAvailability]


def check_artifact_availability(
    artifact: ExecutionArtifact,
    *,
    artifact_volume_name: str,
    volume_root: str | Path,
    external_artifact_checker: ExternalArtifactChecker | None = None,
) -> ArtifactAvailability:
    """Return a typed availability state for one Execution Artifact."""
    if artifact.storage.volume_name == artifact_volume_name:
        errors = execution_artifact_availability_errors(
            artifact,
            artifact_volume_name=artifact_volume_name,
            volume_root=Path(volume_root),
        )
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=(
                AvailabilityStatus.MISSING if errors else AvailabilityStatus.AVAILABLE
            ),
            errors=tuple(errors),
        )

    if external_artifact_checker is None:
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=AvailabilityStatus.UNKNOWN,
            unknown_reason=(
                f"external volume {artifact.storage.volume_name!r} was not checked"
            ),
        )

    try:
        result = external_artifact_checker(artifact)
        if not isinstance(result, ArtifactAvailability):
            raise TypeError("external artifact checker returned an invalid result")
        return result
    except Exception as exc:  # noqa: BLE001
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=AvailabilityStatus.UNKNOWN,
            unknown_reason=(
                f"{artifact.artifact_id}: external artifact checker failed: {exc}"
            ),
        )


def check_external_artifact_status(
    artifact: ExecutionArtifact,
    *,
    artifact_volume_name: str,
    volume_roots: Mapping[str, str | Path],
) -> ArtifactAvailability:
    """Return a typed availability state for an app-owned volume artifact."""
    if artifact.storage.volume_name == artifact_volume_name:
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=AvailabilityStatus.AVAILABLE,
        )

    volume_root = volume_roots.get(artifact.storage.volume_name)
    if volume_root is None:
        return ArtifactAvailability(
            artifact_id=artifact.artifact_id,
            status=AvailabilityStatus.UNKNOWN,
            unknown_reason=(
                f"{artifact.artifact_id}: missing mounted volume root for "
                f"external volume {artifact.storage.volume_name!r}"
            ),
        )

    errors = execution_artifact_availability_errors(
        artifact,
        artifact_volume_name=artifact.storage.volume_name,
        volume_root=Path(volume_root),
    )
    return ArtifactAvailability(
        artifact_id=artifact.artifact_id,
        status=(AvailabilityStatus.MISSING if errors else AvailabilityStatus.AVAILABLE),
        errors=tuple(errors),
    )


def mounted_volume_checker(
    *,
    artifact_volume_name: str,
    volume_roots: Mapping[str, str | Path],
) -> ExternalArtifactChecker:
    """Build a typed checker for already-mounted app-owned volume roots."""
    roots = {volume_name: Path(root) for volume_name, root in volume_roots.items()}

    def check(artifact: ExecutionArtifact) -> ArtifactAvailability:
        return check_external_artifact_status(
            artifact,
            artifact_volume_name=artifact_volume_name,
            volume_roots=roots,
        )

    return check
