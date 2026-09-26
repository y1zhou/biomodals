"""One reusable CPU execution node and its immutable scientific request."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator

from biomodals.app.design.mutation_ridge.inputs import (
    MAX_INPUT_BYTES,
    build_dataset,
    combination_space,
)
from biomodals.app.design.mutation_ridge.regression import (
    RIDGE_VERSION,
    SCIKIT_LEARN_VERSION,
)
from biomodals.execution import ExecutionGraph, ExecutionPlanMetadata
from biomodals.execution.nodes import NodeRunContext, ProviderCallSpec, ProviderNode

RUNTIME_IDENTITY = f"ridge={RIDGE_VERSION}|sklearn={SCIKIT_LEARN_VERSION}|numpy=2.5.1|scipy=1.18.1|polars=1.44.2"
OPERATION = "mutation_ridge_score"
MAX_VARIABLE_SITES = 1024
MAX_SUBSTITUTIONS = 4096
MAX_TOTAL_RESIDUES = 16_384
MAX_RESULT_BYTES = 2 * 1024**3


class RidgeRequest(BaseModel):
    """Bounded fit-and-score input; no uploaded models or output paths."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    measurements_csv: str = Field(min_length=1, max_length=MAX_INPUT_BYTES)
    parental_fasta: str = Field(min_length=1, max_length=MAX_INPUT_BYTES)
    max_mutations: int = Field(default=2, ge=1)
    candidate_budget: int = Field(default=1_000_000, ge=1)
    alpha: float = Field(default=1.0, gt=0)
    higher_is_better: bool = True
    seed: int = Field(default=0, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_scientific_input(self) -> RidgeRequest:
        """Reject invalid or oversized exhaustive work before dispatch."""
        dataset = build_dataset(self.measurements_csv.encode(), self.parental_fasta)
        if not dataset.vocabulary:
            raise ValueError("At least one measured substitution is required")
        if (
            len(dataset.vocabulary) > MAX_SUBSTITUTIONS
            or len({m.site for m in dataset.vocabulary}) > MAX_VARIABLE_SITES
        ):
            raise ValueError("Input exceeds the measured substitution/site limit")
        if (
            len(dataset.parents) > 16
            or sum(map(len, dataset.parents.values())) > MAX_TOTAL_RESIDUES
        ):
            raise ValueError(
                "Provide at most 16 chains and 16384 total parental residues"
            )
        count = sum(combination_space(dataset, self.max_mutations)[1])
        if count > self.candidate_budget:
            raise ValueError(
                f"Novel combination count {count} exceeds candidate budget {self.candidate_budget}"
            )
        longest_tokens = sorted(
            (len(m.token.encode()) for m in dataset.vocabulary), reverse=True
        )
        row_bytes = (
            sum(map(len, dataset.parents.values()))
            + sum(longest_tokens[: self.max_mutations])
            + 256
        )
        if count * row_bytes > MAX_RESULT_BYTES:
            raise ValueError(
                "Estimated candidate CSV exceeds the 2 GiB output limit; reduce maximum mutations"
            )
        return self


@dataclass
class MutationRidgeNode(ProviderNode):
    """Use the caller's execution Run; never launch a child coordinator."""

    request: RidgeRequest
    runtime_identity: str = RUNTIME_IDENTITY

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Derive a run-owned output namespace, not a cross-run fitted cache."""
        return ProviderCallSpec(
            function_name=OPERATION,
            uses_gpu=False,
            runtime_image_key="mutation-ridge-cpu",
            kwargs={
                "request_json": self.request.model_dump_json(),
                "runtime_identity": self.runtime_identity,
                "output_key": sha256(
                    f"{context.execution_run_id}/{context.node_id}/{context.task_key}".encode()
                ).hexdigest(),
            },
        )


def ridge_graph(request: RidgeRequest) -> ExecutionGraph:
    """Build the same scientific node used by the composed optimization workflow."""
    graph = ExecutionGraph(
        "mutation_ridge",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="mutation_ridge",
            scientific_payload={
                "request_sha256": sha256(request.model_dump_json().encode()).hexdigest()
            },
            scientific_versions={"mutation_ridge": RUNTIME_IDENTITY},
        ),
    )
    graph.add_node(MutationRidgeNode(request), id="fit_score_combinations")
    return graph
