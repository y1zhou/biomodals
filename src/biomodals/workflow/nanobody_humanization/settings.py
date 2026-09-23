"""Bounded native controls, distinct from paired-antibody settings."""

from pydantic import BaseModel, ConfigDict, Field

from biomodals.app.design.abnativ2_vhh.contracts import HumanizationSettings
from biomodals.app.design.abnativ2_vhh.sampling import (
    MAX_EXPLORATION_CANDIDATES,
    parent_seed,
)

MAX_JOB_EXPLORATION_CANDIDATES = 10_000


class NanobodySettings(BaseModel):
    """Explicit native AbNatiV CLI defaults and fixed-batch HuDiff sampling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    hudiff_nb_candidate_count: int = Field(default=10, ge=1, le=25, strict=True)
    abnativ2_residue_score_threshold: float = Field(default=0.98, ge=0, le=1)
    abnativ2_rasa_threshold: float = Field(default=0.15, ge=0, le=1)
    abnativ2_max_relative_vhh_score_decrease: float = Field(default=0.05, ge=0, le=1)
    abnativ2_explore: bool = Field(default=False, strict=True)
    abnativ2_candidate_budget: int = Field(
        default=1000, ge=1, le=MAX_EXPLORATION_CANDIDATES, strict=True
    )

    def validate_budget(self, parent_count: int) -> None:
        """Bound worst-case evaluation/publication work before any scientific call."""
        if (
            self.abnativ2_explore
            and parent_count * self.abnativ2_candidate_budget
            > MAX_JOB_EXPLORATION_CANDIDATES
        ):
            raise ValueError(
                f"Parents × exploration budget must not exceed {MAX_JOB_EXPLORATION_CANDIDATES:,}. Reduce the batch or per-parent budget."
            )

    def abnativ_parameters(
        self, parent_id: str, sequence: str, protected: tuple[int, ...]
    ) -> HumanizationSettings:
        """Bind one private exploration stream without changing the HuDiff seed."""
        return HumanizationSettings(
            **{
                key.removeprefix("abnativ2_"): value
                for key, value in self.model_dump().items()
                if key.startswith("abnativ2_")
            },
            sampling_seed=parent_seed(self.root_seed, parent_id, sequence, protected),
        )
