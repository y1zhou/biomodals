"""Provider-neutral schemas for execution artifacts and selectors."""

from __future__ import annotations

import sys
from typing import Any

from pydantic import BaseModel, Field, field_validator

from biomodals.schema.storage import VolumePath

# < Python 3.11 guards
if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    from backports.strenum import StrEnum  # noqa: UP035


class ArtifactKind(StrEnum):
    """Common artifact categories passed between Execution Nodes."""

    STRUCTURES = "structures"
    SCORES = "scores"
    REPORT = "report"
    ARCHIVE = "archive"
    DIRECTORY = "directory"
    TABLE = "table"
    LOGS = "logs"


class ArtifactFile(BaseModel):
    """One file recorded inside an Execution Artifact."""

    path: str
    role: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    content_sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content_sha256")
    @classmethod
    def _validate_content_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        return value


class ExecutionArtifact(BaseModel):
    """Durable manifest for data produced by an Execution Node."""

    artifact_id: str
    producing_node_id: str
    kind: ArtifactKind
    storage: VolumePath
    files: list[ArtifactFile] = Field(default_factory=list)
    source_app_output_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactSelector(BaseModel):
    """Reference to upstream artifacts consumed by an Execution Node."""

    producing_node_id: str
    kind: ArtifactKind | None = None
    pattern: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
