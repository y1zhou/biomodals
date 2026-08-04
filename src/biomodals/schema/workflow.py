"""Schemas for workflow artifacts and selectors."""

from __future__ import annotations

import sys
from typing import Any

from pydantic import BaseModel, Field

from biomodals.schema.storage import VolumePath

# < Python 3.11 guards
if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    from backports.strenum import StrEnum  # noqa: UP035


class ArtifactKind(StrEnum):
    """Common artifact categories passed between workflow nodes."""

    STRUCTURES = "structures"
    SCORES = "scores"
    REPORT = "report"
    ARCHIVE = "archive"
    DIRECTORY = "directory"
    TABLE = "table"
    LOGS = "logs"


class ArtifactFile(BaseModel):
    """One file recorded inside a workflow artifact."""

    path: str
    role: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowArtifact(BaseModel):
    """Durable manifest for data produced by a workflow node."""

    artifact_id: str
    producing_node_id: str
    kind: ArtifactKind
    storage: VolumePath
    files: list[ArtifactFile] = Field(default_factory=list)
    source_app_output_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactSelector(BaseModel):
    """Reference to upstream workflow artifacts consumed by a node input."""

    producing_node_id: str
    kind: ArtifactKind | None = None
    pattern: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
