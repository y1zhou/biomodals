"""Cheap stateless input review; no model weights, fitting or remote work."""

from __future__ import annotations

from hashlib import sha256

from biomodals.app.design.mutation_ridge.execution import MAX_RESULT_BYTES, RidgeRequest
from biomodals.app.design.mutation_ridge.inputs import (
    InputIssue,
    combination_space,
    parent_issues,
    parse_parents,
    review_measurements,
    validated_dataset,
)
from biomodals.service.protein_optimization.contracts import (
    OptimizationReview,
    OptimizationReviewRequest,
    ParentalChain,
    ProteinOptimizationOptions,
)
from biomodals.workflow.protein_optimization.exploration import (
    ExplorationSpace,
    resolved_positions,
)


def review_inputs(body: OptimizationReviewRequest) -> OptimizationReview:
    """Keep raw rows and errors; bind only an error-free reviewed scientific intent."""
    review = review_measurements(body.measurements_csv.encode())
    response = OptimizationReview(
        required_chain_ids=list(review.required_chains),
        rows=review.observations.select(
            "row_index", "id", "mutations", "label", "canonical_mutations"
        ).to_dicts(),
        chains=[],
        errors=list(review.issues),
    )
    if body.parental_fasta is None:
        return response
    try:
        parents = parse_parents(body.parental_fasta)
    except ValueError as exc:
        response.errors.append(
            InputIssue(None, "parental_fasta", "invalid_parents", str(exc))
        )
        return response
    response.chains = [
        ParentalChain(chain_id=chain, sequence=sequence)
        for chain, sequence in parents.items()
    ]
    response.errors.extend(parent_issues(review, parents))
    if response.errors:
        return response
    dataset = validated_dataset(review, parents)
    response.unique_variant_count = len(dataset.variants)
    response.replicate_rows = review.observations.height - len(dataset.variants)
    options = ProteinOptimizationOptions()
    if (
        len(parents) > options.max_chains
        or sum(map(len, parents.values())) > options.max_total_residues
    ):
        response.errors.append(
            InputIssue(
                None,
                "parental_fasta",
                "parent_limit_exceeded",
                "Parental chain count or total residue count exceeds the service limits",
            )
        )
        return response
    if (
        len(dataset.vocabulary) > options.max_measured_substitutions
        or len({m.site for m in dataset.vocabulary}) > options.max_design_positions
    ):
        response.errors.append(
            InputIssue(
                None,
                "mutations",
                "feature_limit_exceeded",
                "Measured substitution or site count exceeds service limits",
            )
        )
        return response
    try:
        body.settings.validate_mode_budget()
        response.positions = list(resolved_positions(dataset, body.settings))
        if body.settings.mode == "combination":
            count = sum(combination_space(dataset, body.settings.max_mutations)[1])
            response.candidate_space_size = str(count)
            if count > body.settings.candidate_budget:
                raise ValueError(
                    f"All {count} novel combinations require a budget of at least {count}; reduce maximum mutations or raise the budget"
                )
            response.evaluation_count = count
            # Enforce the standalone app's identical scientific/resource boundary.
            RidgeRequest(
                measurements_csv=body.measurements_csv,
                parental_fasta=body.parental_fasta,
                max_mutations=body.settings.max_mutations,
                candidate_budget=body.settings.candidate_budget,
            )
        else:
            if any(
                len(sequence) > options.max_exploration_chain_length
                for sequence in parents.values()
            ):
                raise ValueError(
                    "ESMC600M supports at most 2046 residues per chain; sequences are never truncated"
                )
            space = ExplorationSpace.build(dataset, body.settings)
            count = sum(space.counts)
            response.candidate_space_size = str(count)
            response.evaluation_count = min(count, body.settings.candidate_budget)
            longest_chain_id = max(len(chain) for chain in parents)
            row_bytes = (
                sum(map(len, parents.values()))
                + body.settings.max_mutations * (longest_chain_id + 20)
                + 256
            )
            if response.evaluation_count * row_bytes > MAX_RESULT_BYTES:
                raise ValueError(
                    "Estimated candidate CSV exceeds the 2 GiB output limit; reduce the evaluation budget"
                )
    except ValueError as exc:
        response.errors.append(
            InputIssue(None, "settings", "invalid_design_space", str(exc))
        )
        return response
    if response.evaluation_count == 0:
        response.warnings.append("No novel candidates satisfy this design space.")
    if () not in dataset.variants:
        response.warnings.append(
            "No parental measurement was supplied; model predictions are not measured improvement over the parent."
        )
    response.review_digest = sha256(body.model_dump_json().encode()).hexdigest()
    return response
