"""Deterministic candidate union and sortable, nullable selection tables."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import orjson
import polars as pl

from biomodals.app.design.hudiff_ab.validation import parse_hudiff_ab_csv
from biomodals.workflow.humanization.contracts import (
    AntibodyPair,
    CandidateAnnotation,
    CandidateEvaluation,
    CandidateOrigin,
    HumanizationCandidate,
    HumanizationEvaluator,
)

SCORE_COLUMNS: dict[HumanizationEvaluator, tuple[str, ...]] = {
    "sapiens": ("vh_mean_probability", "vl_mean_probability"),
    "humatch": (
        "vh_target_probability",
        "vl_target_probability",
        "pairing_score",
        "vh_germline_likeness",
        "vl_germline_likeness",
        "vh_best_family_probability",
        "vl_best_family_probability",
    ),
    "pabnativ2": ("pair_nativeness", "vh_nativeness", "vl_nativeness", "pairing_score"),
}
FAMILY_COLUMNS = (
    "vh_target_family",
    "vl_target_family",
    "vh_best_family",
    "vl_best_family",
)


def parse_parents(content: bytes) -> tuple[AntibodyPair, ...]:
    """Reuse the bounded shared-format parser without HuDiff numbering checks."""
    return tuple(AntibodyPair(**row) for row in parse_hudiff_ab_csv(content).to_dicts())


def candidate_identity(parent_id: str, vh: str, vl: str) -> str:
    """Content identity independent of yield order, method, and scheduling."""
    return hashlib.sha256(orjson.dumps([parent_id, vh, vl])).hexdigest()


def candidate_union(
    parents: Sequence[AntibodyPair],
    generated: Sequence[tuple[str, str, str, CandidateOrigin]],
) -> tuple[HumanizationCandidate, ...]:
    """Include baselines and merge exact pairs, preserving every original result."""
    parent_by_id = {parent.id: parent for parent in parents}
    if len(parent_by_id) != len(parents):
        raise ValueError("Duplicate parent IDs")
    origins: dict[tuple[str, str, str], set[CandidateOrigin]] = {
        (parent.id, parent.vh, parent.vl): set() for parent in parents
    }
    for parent_id, vh, vl, origin in generated:
        if parent_id not in parent_by_id:
            raise ValueError(f"Unknown parent: {parent_id}")
        origins.setdefault((parent_id, vh, vl), set()).add(origin)
    candidates = []
    for (parent_id, vh, vl), sources in sorted(origins.items()):
        parent = parent_by_id[parent_id]
        ordered_sources = sorted(sources, key=lambda source: source.model_dump_json())
        candidates.append(
            HumanizationCandidate(
                parent_id=parent_id,
                candidate_id=candidate_identity(parent_id, vh, vl),
                vh=vh,
                vl=vl,
                is_parent=(vh, vl) == (parent.vh, parent.vl),
                origins=tuple(ordered_sources),
            )
        )
    return tuple(candidates)


def selection_table(
    candidates: Sequence[HumanizationCandidate],
    evaluations: Sequence[CandidateEvaluation],
    annotations: Sequence[CandidateAnnotation],
) -> pl.DataFrame:
    """Join every candidate explicitly; absent evaluations are missing, never zero."""
    keys = {(candidate.parent_id, candidate.candidate_id) for candidate in candidates}
    if len(keys) != len(candidates):
        raise ValueError("Duplicate candidate identities")
    baselines = {}
    for candidate in candidates:
        if candidate.is_parent:
            if candidate.parent_id in baselines:
                raise ValueError("Multiple parental baselines")
            baselines[candidate.parent_id] = candidate.candidate_id
    if set(baselines) != {candidate.parent_id for candidate in candidates}:
        raise ValueError("Every parent requires a baseline")
    scored = {}
    for evaluation in evaluations:
        key = (evaluation.parent_id, evaluation.candidate_id, evaluation.evaluator)
        if key[:2] not in keys or key in scored:
            raise ValueError(f"Unknown or duplicate evaluation: {key}")
        if set(evaluation.scores) - set(SCORE_COLUMNS[evaluation.evaluator]):
            raise ValueError(
                "Detailed or unknown metrics do not belong in the selection table"
            )
        if set(evaluation.labels) - (
            set(FAMILY_COLUMNS) if evaluation.evaluator == "humatch" else set()
        ):
            raise ValueError("Unknown summary labels")
        scored[key] = evaluation
    annotated = {}
    for annotation in annotations:
        key = (annotation.parent_id, annotation.candidate_id)
        if key not in keys or key in annotated:
            raise ValueError(f"Unknown or duplicate annotation: {key}")
        annotated[key] = annotation

    schema = {
        "parent_id": pl.String,
        "candidate_id": pl.String,
        "vh": pl.String,
        "vl": pl.String,
        "is_parent": pl.Boolean,
        "generating_methods": pl.String,
        "cdr_preservation": pl.String,
        "annotation_error": pl.String,
        "vh_mutations": pl.Int64,
        "vl_mutations": pl.Int64,
        "cdr_mutations": pl.Int64,
        "evaluation_complete": pl.Boolean,
    }
    for method, metrics in SCORE_COLUMNS.items():
        schema[f"{method}_status"] = pl.String
        schema[f"{method}_error"] = pl.String
        for metric in metrics:
            schema[f"{method}_{metric}"] = pl.Float64
            # Best-family scores can change reference and have no comparable delta.
            if "best_family" not in metric:
                schema[f"{method}_{metric}_delta"] = pl.Float64
    schema.update({f"humatch_{name}": pl.String for name in FAMILY_COLUMNS})
    rows = []
    for candidate in candidates:
        key = (candidate.parent_id, candidate.candidate_id)
        annotation = annotated.get(key)
        row = {name: None for name in schema}
        row.update({
            "parent_id": candidate.parent_id,
            "candidate_id": candidate.candidate_id,
            "vh": candidate.vh,
            "vl": candidate.vl,
            "is_parent": candidate.is_parent,
            "generating_methods": ";".join(
                sorted({origin.method for origin in candidate.origins})
            ),
            "cdr_preservation": annotation.cdr_preservation
            if annotation
            else "unknown",
            "annotation_error": annotation.error
            if annotation
            else "Annotation unavailable",
            "evaluation_complete": annotation is not None
            and annotation.cdr_preservation != "unknown",
        })
        if annotation:
            for name in ("vh_mutations", "vl_mutations", "cdr_mutations"):
                row[name] = getattr(annotation, name)
        for method, metrics in SCORE_COLUMNS.items():
            result = scored.get((*key, method))
            parent = scored.get((
                candidate.parent_id,
                baselines[candidate.parent_id],
                method,
            ))
            row[f"{method}_status"] = result.status if result else "missing"
            row[f"{method}_error"] = (
                result.error if result else "Evaluation unavailable"
            )
            if result is None or result.status != "succeeded":
                row["evaluation_complete"] = False
                continue
            for name, label in result.labels.items():
                row[f"{method}_{name}"] = label
            for metric in metrics:
                value = result.scores.get(metric)
                row[f"{method}_{metric}"] = value
                if value is None:
                    row["evaluation_complete"] = False
                if (
                    "best_family" in metric
                    or parent is None
                    or parent.status != "succeeded"
                ):
                    continue
                reference = parent.scores.get(metric)
                if method == "humatch" and metric != "pairing_score":
                    family = f"{metric[:2]}_target_family"
                    if not result.labels.get(family) or result.labels.get(
                        family
                    ) != parent.labels.get(family):
                        raise ValueError(
                            "Humatch parental deltas require identical target families"
                        )
                if value is not None and reference is not None:
                    row[f"{method}_{metric}_delta"] = value - reference
        rows.append(row)
    return pl.DataFrame(rows, schema=schema).sort(
        "parent_id", "is_parent", "candidate_id", descending=[False, True, False]
    )
