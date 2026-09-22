"""Reviewed preparation, immutable submission and retained scientific parents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from biomodals.helper.antibody import MAX_CHAIN_LENGTH
from biomodals.workflow.nanobody_humanization.preparation import (
    PREPARATION_VERSION,
    PreparationIssue,
    PreparedVH,
    VHInput,
    preparation_digest,
)
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings

MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024


class NanobodyOptions(BaseModel):
    """Authoritative website bounds and the complete flat native controls."""

    max_parents: int
    max_input_length: int = MAX_CHAIN_LENGTH
    max_csv_bytes: int = MAX_CSV_BYTES
    preparation_version: str = PREPARATION_VERSION
    defaults: NanobodySettings = Field(default_factory=NanobodySettings)
    settings_schema: dict[str, Any] = Field(
        default_factory=NanobodySettings.model_json_schema
    )


class NanobodyPreparationRequest(BaseModel):
    """Editable constructs, not yet eligible scientific parents."""

    model_config = ConfigDict(extra="forbid")
    parents: list[VHInput] = Field(min_length=1, max_length=200)


class PreparedVHPreview(BaseModel):
    """Display only the prepared baseline; native imputation evidence stays saved."""

    row_index: int
    id: str
    vh: str | None


class NanobodyPreparation(BaseModel):
    """All row outcomes, with a submission digest only for a valid whole batch."""

    preparation_version: str = PREPARATION_VERSION
    rows: list[PreparedVHPreview]
    errors: list[PreparationIssue]
    preparation_digest: str | None

    @classmethod
    def from_prepared(
        cls,
        originals: list[VHInput],
        parents: tuple[PreparedVH, ...],
        issues: list[PreparationIssue],
    ) -> NanobodyPreparation:
        """Retain positional correspondence even for invalid or duplicate IDs."""
        invalid = {issue.row_index for issue in issues if issue.field == "vhh"}
        prepared = iter(parents)
        return cls(
            rows=[
                PreparedVHPreview(
                    row_index=index,
                    id=record.id,
                    vh=None if index in invalid else next(prepared).sequence,
                )
                for index, record in enumerate(originals)
            ],
            errors=issues,
            preparation_digest=None if issues else preparation_digest(parents),
        )


class NanobodySubmission(NanobodyPreparationRequest):
    """Original inputs plus the exact reviewed preparation, never trusted outputs."""

    display_name: str = Field(default="Nanobody humanization", max_length=120)
    settings: NanobodySettings = Field(default_factory=NanobodySettings)
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("display_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Use the same stable whitespace/blank semantics as the other forms."""
        return " ".join(value.split()) or "Nanobody humanization"


class RetainedNanobodyInputs(BaseModel):
    """Original editable inputs and the saved baseline for candidate comparison."""

    display_name: str
    parents: list[VHInput]
    settings: NanobodySettings
    prepared_parents: list[PreparedVHPreview]
    preparation_version: str
