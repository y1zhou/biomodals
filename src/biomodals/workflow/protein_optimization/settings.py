"""Scientific website controls, separate from provider resource ceilings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from biomodals.app.design.mutation_ridge.inputs import AMINO_ACIDS

DEFAULT_ALLOWED_AMINO_ACIDS = "ADEFGHIKLNPQRSTVWY"
MAX_EXPLORATION_CANDIDATES = 20_000
MAX_COMBINATION_CANDIDATES = 1_000_000
MAX_DESIGN_POSITIONS = 1024
MAX_EXPLORATION_MUTATIONS = 10


class PositionChoices(BaseModel):
    """One raw parent-relative site; an empty residue set freezes this site."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chain_id: str = Field(min_length=1, max_length=100)
    position: int = Field(ge=1)
    amino_acids: str = DEFAULT_ALLOWED_AMINO_ACIDS

    @field_validator("amino_acids")
    @classmethod
    def valid_residues(cls, value: str) -> str:
        """Normalize presentation, rejecting unknown rather than deleting residues."""
        value = "".join(value.split()).upper()
        if set(value) - set(AMINO_ACIDS):
            raise ValueError("Use only the standard 20 amino acids")
        return "".join(sorted(set(value)))


class OptimizationSettings(BaseModel):
    """One selected scoring mode; absent site policy defaults to measured sites."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["combination", "exploration"] = "combination"
    direction: Literal["maximize", "minimize"] = "maximize"
    max_mutations: int = Field(default=2, ge=1, le=MAX_DESIGN_POSITIONS)
    candidate_budget: int = Field(
        default=1_000_000, ge=1, le=MAX_COMBINATION_CANDIDATES
    )
    max_new_mutations: int = Field(default=1, ge=1, le=MAX_EXPLORATION_MUTATIONS)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    positions: tuple[PositionChoices, ...] | None = Field(
        default=None, max_length=MAX_DESIGN_POSITIONS
    )

    def validate_mode_budget(self) -> None:
        """Report mode-specific errors explicitly, never scale a user's budget."""
        if self.mode == "exploration":
            if self.candidate_budget > MAX_EXPLORATION_CANDIDATES:
                raise ValueError(
                    f"Exploration permits at most {MAX_EXPLORATION_CANDIDATES} candidate evaluations"
                )
            if self.max_mutations > MAX_EXPLORATION_MUTATIONS:
                raise ValueError(
                    f"Exploration permits at most {MAX_EXPLORATION_MUTATIONS} total mutations"
                )


def mode_defaults() -> dict[str, OptimizationSettings]:
    """Return complete defaults so changing modes does not silently keep a million."""
    return {
        "combination": OptimizationSettings(),
        "exploration": OptimizationSettings(mode="exploration", candidate_budget=5000),
    }
