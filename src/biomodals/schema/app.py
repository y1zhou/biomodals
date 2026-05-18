"""Schemas returned by workflow-compatible Biomodals app functions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from biomodals.schema.storage import InlineBytes, VolumePath
from biomodals.schema.workflow import ArtifactKind


class AppRunStatus(StrEnum):
    """Common completion states returned by workflow-compatible app functions."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


class AppOutput(BaseModel):
    """One output produced by a workflow-compatible app function."""

    name: str
    kind: ArtifactKind
    storage: InlineBytes | VolumePath = Field(discriminator="kind")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppRunResult(BaseModel):
    """Standard result returned by workflow-compatible app functions."""

    status: AppRunStatus
    outputs: list[AppOutput] = Field(default_factory=list)
    metrics: dict[str, str | int | float | bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    logs: list[AppOutput] = Field(default_factory=list)
