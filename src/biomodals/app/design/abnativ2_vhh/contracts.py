"""Explicit native controls and versioned exploration evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from biomodals.app.design.abnativ2_vhh.sampling import MAX_EXPLORATION_CANDIDATES

SCORE_BATCH_SIZE = 128


class HumanizationSettings(BaseModel):
    """Native defaults plus bounded optional exploration, not output truncation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    residue_score_threshold: float = Field(default=0.98, ge=0, le=1)
    rasa_threshold: float = Field(default=0.15, ge=0, le=1)
    max_relative_vhh_score_decrease: float = Field(default=0.05, ge=0, le=1)
    explore: bool = Field(default=False, strict=True)
    candidate_budget: int = Field(
        default=1000, ge=1, le=MAX_EXPLORATION_CANDIDATES, strict=True
    )
    sampling_seed: int = Field(default=0, ge=0, le=2**64 - 1, strict=True)


class SearchSummary(BaseModel):
    """Counts exclude the parent; arbitrary-size possible counts are decimal text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    possible_candidates: str = Field(pattern=r"^(0|[1-9][0-9]*)$")
    evaluated_candidates: int = Field(ge=0, le=MAX_EXPLORATION_CANDIDATES)
    accepted_candidates: int = Field(ge=0, le=MAX_EXPLORATION_CANDIDATES)
    coverage: Literal["complete", "sampled"]
    sampling_version: Literal["balanced-edit-count-v1"] = "balanced-edit-count-v1"

    @model_validator(mode="after")
    def validate_counts(self) -> "SearchSummary":
        """Distinguish exhausted spaces from budget-limited sampling exactly."""
        possible = int(self.possible_candidates)
        if not self.accepted_candidates <= self.evaluated_candidates <= possible:
            raise ValueError("Invalid exploration counts")
        if (self.coverage == "complete") != (self.evaluated_candidates == possible):
            raise ValueError("Exploration coverage does not match its counts")
        return self
