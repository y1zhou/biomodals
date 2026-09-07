"""Independent IMGT checks; never graft, filter, or repair generated sequences."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow.humanization.artifacts import json_output
from biomodals.workflow.humanization.contracts import (
    AntibodyPair,
    CandidateAnnotation,
    HumanizationCandidate,
)


def annotate_humanization_candidate(
    parent: dict[str, Any], candidate: dict[str, Any]
) -> AppRunResult:
    """Remote operation bound to a numbering-enabled image by the workflow root."""
    annotation, mutations = annotate_candidate(
        AntibodyPair.model_validate(parent),
        HumanizationCandidate.model_validate(candidate),
    )
    if annotation.cdr_preservation == "unknown":
        return AppRunResult(
            status=AppRunStatus.FAILED,
            warnings=[annotation.error or "IMGT annotation unavailable"],
        )
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            json_output("annotation", annotation.model_dump()),
            json_output("imgt_mutations", mutations),
        ],
    )


def annotate_candidate(
    parent: AntibodyPair,
    candidate: HumanizationCandidate,
    *,
    number: Callable[..., Any] | None = None,
) -> tuple[CandidateAnnotation, list[dict[str, Any]]]:
    """Count changed IMGT positions, including insertions/deletions, per chain.

    CDR boundaries are IMGT 27–38, 56–65 and 105–117, including insertion
    positions. Numbering failure is explicit missing evidence, never a pass.
    """
    if candidate.parent_id != parent.id:
        raise ValueError("Candidate does not belong to the supplied parent")
    mutations = []
    try:
        if number is None:
            import anarci  # noqa: PLC0415  # type: ignore[ty:unresolved-import]

            number = anarci.number
        for chain, roles in (("vh", {"H"}), ("vl", {"K", "L"})):
            endpoints = []
            for sequence in (getattr(parent, chain), getattr(candidate, chain)):
                result = number(sequence, scheme="imgt")
                if not result or result[0] is None or result[1] not in roles:
                    raise ValueError(
                        f"{chain}: IMGT numbering failed or wrong chain role"
                    )
                positions = {
                    (int(position[0]), str(position[1]).strip()): residue
                    for position, residue in result[0]
                    if residue != "-"
                }
                if "".join(positions.values()) != sequence:
                    raise ValueError(
                        f"{chain}: numbering does not preserve the complete sequence"
                    )
                endpoints.append(positions)
            before, after = endpoints
            for position in sorted(before.keys() | after.keys()):
                original, final = before.get(position, "-"), after.get(position, "-")
                if original == final:
                    continue
                is_cdr = any(
                    start <= position[0] <= end
                    for start, end in ((27, 38), (56, 65), (105, 117))
                )
                mutations.append({
                    "parent_id": parent.id,
                    "candidate_id": candidate.candidate_id,
                    "chain": chain,
                    "numbering_scheme": "imgt",
                    "cdr_definition": "imgt",
                    "position": position[0],
                    "insertion_code": position[1],
                    "parent_residue": original,
                    "candidate_residue": final,
                    "region": "cdr" if is_cdr else "framework",
                    "change_type": "insertion"
                    if original == "-"
                    else "deletion"
                    if final == "-"
                    else "substitution",
                })
    except Exception as exc:
        # Partial numbering is not a complete mutation or preservation assessment.
        return CandidateAnnotation(
            parent_id=parent.id,
            candidate_id=candidate.candidate_id,
            cdr_preservation="unknown",
            error=f"{type(exc).__name__}: {exc}",
        ), []
    cdr_changes = sum(row["region"] == "cdr" for row in mutations)
    return CandidateAnnotation(
        parent_id=parent.id,
        candidate_id=candidate.candidate_id,
        cdr_preservation="changed" if cdr_changes else "preserved",
        vh_mutations=sum(row["chain"] == "vh" for row in mutations),
        vl_mutations=sum(row["chain"] == "vl" for row in mutations),
        cdr_mutations=cdr_changes,
    ), mutations
