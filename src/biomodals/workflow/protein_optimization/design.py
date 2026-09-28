"""Strict scientific boundary shared by workflow execution and service admission."""

from __future__ import annotations

from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator

from biomodals.app.design.mutation_ridge.execution import (
    MAX_RESULT_BYTES,
    MAX_SUBSTITUTIONS,
    MAX_TOTAL_RESIDUES,
    MAX_VARIABLE_SITES,
    RidgeRequest,
)
from biomodals.app.design.mutation_ridge.execution import (
    RUNTIME_IDENTITY as RIDGE_IDENTITY,
)
from biomodals.app.design.mutation_ridge.inputs import (
    MAX_INPUT_BYTES,
    MutationDataset,
    build_dataset,
    combination_space,
)
from biomodals.app.misc.tabpfn.models import RUNTIME_IDENTITY as TABPFN_IDENTITY
from biomodals.workflow.protein_optimization.exploration import ExplorationSpace
from biomodals.workflow.protein_optimization.features import (
    EMBEDDING_IDENTITY,
    MAX_CHAIN_LENGTH,
)
from biomodals.workflow.protein_optimization.settings import OptimizationSettings

PCA_COMPONENTS = 1024
SCIENTIFIC_VERSIONS = {
    "workflow": "2",
    "result_schema": "1",
    "ridge": RIDGE_IDENTITY,
    "sampling": "1",
    "validation": "3",
    "esmc": EMBEDDING_IDENTITY,
    "tabpfn": TABPFN_IDENTITY,
    "projection": f"numeric_randomized_pca_max{PCA_COMPONENTS}_train_only_v1",
}


class OptimizationDesign(BaseModel):
    """Complete immutable input, distinct from row-preserving provisional review."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    measurements_csv: str = Field(min_length=1, max_length=MAX_INPUT_BYTES)
    parental_fasta: str | None = Field(
        default=None, min_length=1, max_length=MAX_INPUT_BYTES
    )
    settings: OptimizationSettings = Field(default_factory=OptimizationSettings)

    def dataset(self) -> MutationDataset:
        """Restore all validated observations, independent of proposal distance."""
        if self.settings.mode == "exploration" and self.parental_fasta is None:
            raise ValueError("Exploration requires parental chain sequences")
        return build_dataset(self.measurements_csv.encode(), self.parental_fasta)

    def candidate_count(self, dataset: MutationDataset) -> int:
        """Count exact admitted work before model initialization or allocations."""
        if self.settings.mode == "combination":
            return sum(combination_space(dataset, self.settings.max_mutations)[1])
        return min(
            sum(ExplorationSpace.build(dataset, self.settings).counts),
            self.settings.candidate_budget,
        )

    def digest(self) -> str:
        """Bind original input and complete settings without transport or display IDs."""
        return sha256(self.model_dump_json().encode()).hexdigest()

    def scientific_versions(self) -> dict[str, str]:
        """Only dependencies actually used by this mode enter compatibility checks."""
        if self.settings.mode == "combination":
            return {
                name: SCIENTIFIC_VERSIONS[name]
                for name in ("workflow", "result_schema", "ridge")
            }
        return {
            name: value
            for name, value in SCIENTIFIC_VERSIONS.items()
            if name != "ridge"
        }

    def ridge_request(self) -> RidgeRequest:
        """Use exactly the standalone app's scientific inputs and native defaults."""
        return RidgeRequest(
            measurements_csv=self.measurements_csv,
            parental_fasta=self.parental_fasta,
            max_mutations=self.settings.max_mutations,
            candidate_budget=self.settings.candidate_budget,
            seed=self.settings.seed,
            higher_is_better=self.settings.direction == "maximize",
        )

    @model_validator(mode="after")
    def validate_design(self) -> OptimizationDesign:
        """Repeat domain/resource admission at remote consumption, never trust review."""
        self.settings.validate_mode_budget()
        dataset = self.dataset()
        if (
            len(set(dataset.parents) | {m.chain for m in dataset.vocabulary}) > 16
            or sum(map(len, dataset.parents.values())) > MAX_TOTAL_RESIDUES
        ):
            raise ValueError(
                "Parental chain count or total residue count exceeds service limits"
            )
        if (
            len(dataset.vocabulary) > MAX_SUBSTITUTIONS
            or len({m.site for m in dataset.vocabulary}) > MAX_VARIABLE_SITES
        ):
            raise ValueError(
                "Measured substitution or site count exceeds service limits"
            )
        if self.settings.mode == "combination":
            self.ridge_request()
        else:
            if len(dataset.variants) < 2:
                raise ValueError(
                    "Exploration requires at least two unique measured variants"
                )
            if any(len(seq) > MAX_CHAIN_LENGTH for seq in dataset.parents.values()):
                raise ValueError("ESMC600M supports at most 2046 residues per chain")
            row_bytes = (
                sum(map(len, dataset.parents.values()))
                + self.settings.max_mutations * (max(map(len, dataset.parents)) + 20)
                + 256
            )
            if self.candidate_count(dataset) * row_bytes > MAX_RESULT_BYTES:
                raise ValueError(
                    "Estimated candidate CSV exceeds the 2 GiB output limit"
                )
        return self
