"""Typed website inputs and row-addressable validation diagnostics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from biomodals.helper.io import fasta_identifier
from biomodals.workflow.humanization.contracts import AntibodyPair
from biomodals.workflow.humanization.settings import HumanizationSettings


class PairInput(BaseModel):
    """One editable row, retained even when its biological values are invalid."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=200)
    vh: str = Field(max_length=4096)
    vl: str = Field(max_length=4096)

    @field_validator("vh", "vl")
    @classmethod
    def normalize_sequence(cls, value: str) -> str:
        """Normalize presentation, never delete unsupported residues."""
        return "".join(value.split()).upper()


class HumanizationSubmission(BaseModel):
    """One scientific configuration shared by every explicitly identified pair."""

    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(default="Antibody humanization", max_length=120)
    pairs: list[PairInput] = Field(min_length=1, max_length=200)
    settings: HumanizationSettings = Field(default_factory=HumanizationSettings)

    @field_validator("display_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Keep the default stable across explicit idempotent recovery."""
        return " ".join(value.split()) or "Antibody humanization"


class InputIssue(BaseModel):
    """Address one invalid field without echoing antibody sequence contents."""

    row_index: int | None
    field: str
    code: str
    message: str


class InputErrors(BaseModel):
    """Semantic validation errors for correction in the editable batch."""

    code: Literal["humanization_input_invalid"] = "humanization_input_invalid"
    detail: str = "Correct or remove invalid rows before submitting"
    errors: list[InputIssue]


class HumanizationOptions(BaseModel):
    """Server-owned admission limit and native scientific settings metadata."""

    max_pairs: int
    max_vh_length: int
    max_vl_length: int
    defaults: HumanizationSettings = Field(default_factory=HumanizationSettings)
    settings_schema: dict[str, Any]


def validate_pairs(
    submission: HumanizationSubmission, *, max_pairs: int
) -> tuple[tuple[AntibodyPair, ...], list[InputIssue]]:
    """Match the scientific parser's rules while reporting all invalid rows."""
    from biomodals.app.design.sapiens.app import MAX_VH_LENGTH, MAX_VL_LENGTH

    issues: list[InputIssue] = []
    if len(submission.pairs) > max_pairs:
        issues.append(
            InputIssue(
                row_index=None,
                field="pairs",
                code="batch_too_large",
                message=f"At most {max_pairs} pairs are allowed per job",
            )
        )
    seen: dict[str, int] = {}
    invalid_duplicates: set[int] = set()
    for index, pair in enumerate(submission.pairs):
        identifier = pair.id
        if (
            not identifier
            or identifier != identifier.strip()
            or any(ord(char) < 32 or char == ">" for char in identifier)
        ):
            issues.append(
                InputIssue(
                    row_index=index,
                    field="id",
                    code="id_invalid",
                    message="ID must be nonempty, unpadded, and contain no controls or >",
                )
            )
        normalized_id = fasta_identifier(identifier)
        if normalized_id in seen:
            invalid_duplicates.update((seen[normalized_id], index))
        else:
            seen[normalized_id] = index
        for field, max_length in (("vh", MAX_VH_LENGTH), ("vl", MAX_VL_LENGTH)):
            sequence = getattr(pair, field)
            if len(sequence) > max_length:
                issues.append(
                    InputIssue(
                        row_index=index,
                        field=field,
                        code="sequence_too_long",
                        message=f"{field.upper()} must be at most {max_length} residues; provide the variable domain only",
                    )
                )
            elif not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                issues.append(
                    InputIssue(
                        row_index=index,
                        field=field,
                        code="sequence_invalid",
                        message=f"Use 1–{max_length} canonical amino-acid residues",
                    )
                )
    issues.extend(
        InputIssue(
            row_index=index,
            field="id",
            code="id_duplicate",
            message="IDs must be unique after FASTA whitespace normalization",
        )
        for index in sorted(invalid_duplicates)
    )
    if issues:
        return (), issues
    return tuple(AntibodyPair(**pair.model_dump()) for pair in submission.pairs), []
