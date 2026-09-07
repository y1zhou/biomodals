"""Workflow-owned contracts for paired-antibody generation and cross-evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

HumanizationMethod = Literal["sapiens", "humatch", "pabnativ2", "hudiff_ab"]
HumanizationEvaluator = Literal["sapiens", "humatch", "pabnativ2"]


class AntibodyPair(BaseModel):
    """An explicitly associated pair of canonical, ungapped variable domains."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    vh: str = Field(min_length=1, max_length=200, pattern="^[ACDEFGHIKLMNPQRSTVWY]+$")
    vl: str = Field(min_length=1, max_length=200, pattern="^[ACDEFGHIKLMNPQRSTVWY]+$")


class CandidateOrigin(BaseModel):
    """One method's original result identity, including a sampling attempt if any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: HumanizationMethod
    source_id: str
    attempt_index: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)


class HumanizationCandidate(BaseModel):
    """One exact sequence pair within its parental context, never a chain remix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_id: str
    candidate_id: str
    vh: str = Field(min_length=1, max_length=200, pattern="^[ACDEFGHIKLMNPQRSTVWY]+$")
    vl: str = Field(min_length=1, max_length=200, pattern="^[ACDEFGHIKLMNPQRSTVWY]+$")
    is_parent: bool = False
    origins: tuple[CandidateOrigin, ...] = ()


class CandidateEvaluation(BaseModel):
    """Scalar summaries only; detailed native outputs are separate artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_id: str
    candidate_id: str
    evaluator: HumanizationEvaluator
    status: Literal["succeeded", "failed"]
    scores: dict[str, FiniteFloat | None] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> CandidateEvaluation:
        """Keep failure diagnostics separate from numeric evidence."""
        if self.status == "failed" and (not self.error or self.scores or self.labels):
            raise ValueError(
                "Failed evaluation requires an error and no scores or labels"
            )
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("Successful evaluation cannot have an error")
        return self


class CandidateAnnotation(BaseModel):
    """Common IMGT assessment, independent of native generation/score regions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_id: str
    candidate_id: str
    cdr_preservation: Literal["preserved", "changed", "unknown"]
    vh_mutations: int | None = Field(default=None, ge=0)
    vl_mutations: int | None = Field(default=None, ge=0)
    cdr_mutations: int | None = Field(default=None, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> CandidateAnnotation:
        """Require evidence for each preservation state."""
        if self.cdr_preservation == "unknown" and not self.error:
            raise ValueError("Unknown CDR preservation requires an explanation")
        if self.cdr_preservation == "preserved" and self.cdr_mutations != 0:
            raise ValueError("Preserved CDRs require zero CDR mutations")
        if self.cdr_preservation == "changed" and not self.cdr_mutations:
            raise ValueError("Changed CDRs require a positive mutation count")
        return self
