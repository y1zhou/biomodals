"""Native scientific controls; concurrency stays with the execution CLI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HumanizationSettings(BaseModel):
    """Namespaced controls map directly to the standalone generators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sapiens_iterations: int = Field(default=1, ge=1, le=5, strict=True)
    sapiens_numbering_scheme: Literal["kabat", "chothia", "imgt"] = "kabat"
    sapiens_cdr_definition: Literal["kabat", "chothia", "imgt", "north"] = "kabat"
    sapiens_mutate_cdrs: bool = False
    humatch_vh_target_family: str = "auto"
    humatch_vl_target_family: str = "auto"
    humatch_germline_likeness_target: float = Field(default=0.4, ge=0, le=1)
    humatch_vh_classifier_target: float = Field(default=0.95, ge=0, le=1)
    humatch_vl_classifier_target: float = Field(default=0.95, ge=0, le=1)
    humatch_pair_classifier_target: float = Field(default=0.95, ge=0, le=1)
    humatch_max_edits: int = Field(default=60, ge=0, le=400, strict=True)
    humatch_mutate_cdrs: bool = False
    humatch_fixed_vh_positions: str = ""
    humatch_fixed_vl_positions: str = ""
    pabnativ2_mutate_cdrs: bool = False
    pabnativ2_fixed_vh_positions: str = ""
    pabnativ2_fixed_vl_positions: str = ""
    pabnativ2_residue_score_threshold: float = Field(default=0.98, ge=0, le=1)
    pabnativ2_rasa_threshold: float = Field(default=0.15, ge=0, le=1)
    pabnativ2_max_relative_pairing_score_decrease: float = Field(
        default=0.1, ge=0, le=1
    )
    pabnativ2_forbidden_residues: str = "C,M"
    pabnativ2_seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    hudiff_ab_candidate_count: int = Field(default=10, ge=1, le=10, strict=True)
    hudiff_ab_seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    hudiff_ab_sampling_order: Literal["shuffle", "left_to_right"] = "shuffle"
    hudiff_ab_upstream_inference_dropout: bool = True

    def method_arguments(self, method: str) -> dict[str, Any]:
        """Strip only the owning method's prefix for its native app operation."""
        prefix = f"{method}_"
        return {
            key.removeprefix(prefix): value
            for key, value in self.model_dump().items()
            if key.startswith(prefix)
        }
