"""Shared immutable residue policy consumed by the two single-domain apps."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class VHHInput(BaseModel):
    """An already prepared VH and protected zero-based sequence indices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: str = Field(
        min_length=1, max_length=152, pattern="^[ACDEFGHIKLMNPQRSTVWY]+$"
    )
    protected_indices: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def validate_protection(self) -> VHHInput:
        """Reject invalid indices and require preservation of parental cysteines."""
        protected = set(self.protected_indices)
        if len(protected) != len(self.protected_indices) or any(
            index < 0 or index >= len(self.sequence) for index in protected
        ):
            raise ValueError("Protected positions must be unique sequence indices")
        if any(
            aa == "C" and index not in protected
            for index, aa in enumerate(self.sequence)
        ):
            raise ValueError("All parental cysteines must be protected")
        return self

    def validate_candidate(self, sequence: str) -> None:
        """Validate policy after native generation, including a successful no-op."""
        if len(sequence) != len(self.sequence) or set(sequence) - set(
            "ACDEFGHIKLMNPQRSTVWY"
        ):
            raise ValueError("Generated sequence changed domain length or alphabet")
        if any(self.sequence[i] != sequence[i] for i in self.protected_indices):
            raise ValueError("Generated sequence changed a protected residue")
