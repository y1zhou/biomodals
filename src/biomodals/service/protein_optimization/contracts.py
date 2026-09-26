"""Authoritative local-review contract for generic multichain protein design."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from biomodals.app.design.mutation_ridge.execution import (
    MAX_RESULT_BYTES,
    MAX_SUBSTITUTIONS,
    MAX_TOTAL_RESIDUES,
    MAX_VARIABLE_SITES,
)
from biomodals.app.design.mutation_ridge.inputs import (
    MAX_INPUT_BYTES,
    MAX_MEASUREMENT_ROWS,
    MAX_MUTATION_TOKENS,
    InputIssue,
)
from biomodals.workflow.protein_optimization.settings import (
    MAX_COMBINATION_CANDIDATES,
    MAX_EXPLORATION_CANDIDATES,
    MAX_EXPLORATION_MUTATIONS,
    OptimizationSettings,
    PositionChoices,
    mode_defaults,
)

REVIEW_VERSION = "1"
MAX_REQUEST_BYTES = 32 * 1024 * 1024


class ProteinOptimizationOptions(BaseModel):
    """Server limits and complete per-mode settings, not frontend-owned ceilings."""

    review_version: str = REVIEW_VERSION
    max_measurements_csv_bytes: int = MAX_INPUT_BYTES
    max_parental_fasta_bytes: int = MAX_INPUT_BYTES
    max_measurement_rows: int = MAX_MEASUREMENT_ROWS
    max_mutation_tokens: int = MAX_MUTATION_TOKENS
    max_chains: int = 16
    max_total_residues: int = MAX_TOTAL_RESIDUES
    max_exploration_chain_length: int = 2046
    max_design_positions: int = MAX_VARIABLE_SITES
    max_measured_substitutions: int = MAX_SUBSTITUTIONS
    max_combination_candidates: int = MAX_COMBINATION_CANDIDATES
    max_exploration_candidates: int = MAX_EXPLORATION_CANDIDATES
    max_exploration_mutations: int = MAX_EXPLORATION_MUTATIONS
    max_result_bytes: int = MAX_RESULT_BYTES
    defaults: dict[str, OptimizationSettings] = Field(default_factory=mode_defaults)
    settings_schema: dict[str, Any] = Field(
        default_factory=OptimizationSettings.model_json_schema
    )


class OptimizationReviewRequest(BaseModel):
    """Original editable inputs; FASTA may be absent during chain discovery."""

    model_config = ConfigDict(extra="forbid")
    measurements_csv: str = Field(min_length=1, max_length=MAX_INPUT_BYTES)
    parental_fasta: str | None = Field(default=None, max_length=MAX_INPUT_BYTES)
    settings: OptimizationSettings = Field(default_factory=OptimizationSettings)


class MeasurementPreview(BaseModel):
    """Every uploaded row, including invalid labels and mutation expressions."""

    row_index: int
    id: str
    mutations: str
    label: str
    canonical_mutations: str | None


class ParentalChain(BaseModel):
    """A complete, normalized user-supplied chain with no inferred role."""

    chain_id: str
    sequence: str


class OptimizationReview(BaseModel):
    """Local non-model review; only a valid complete request receives a digest."""

    review_version: str = REVIEW_VERSION
    required_chain_ids: list[str]
    rows: list[MeasurementPreview]
    chains: list[ParentalChain]
    errors: list[InputIssue]
    unique_variant_count: int | None = None
    replicate_rows: int | None = None
    positions: list[PositionChoices] = Field(default_factory=list)
    candidate_space_size: str | None = None
    evaluation_count: int | None = None
    warnings: list[str] = Field(default_factory=list)
    review_digest: str | None = None
