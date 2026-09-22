"""Bounded native controls, distinct from paired-antibody settings."""

from pydantic import BaseModel, ConfigDict, Field


class NanobodySettings(BaseModel):
    """Explicit native AbNatiV CLI defaults and fixed-batch HuDiff sampling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    hudiff_nb_candidate_count: int = Field(default=10, ge=1, le=25, strict=True)
    abnativ2_residue_score_threshold: float = Field(default=0.98, ge=0, le=1)
    abnativ2_rasa_threshold: float = Field(default=0.15, ge=0, le=1)
    abnativ2_max_relative_vhh_score_decrease: float = Field(default=0.05, ge=0, le=1)
