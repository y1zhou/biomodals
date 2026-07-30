"""AlphaFold3 Task identities for the shared execution kernel."""

from __future__ import annotations

import re
from collections.abc import Mapping

from biomodals.app.fold.alphafold3.inference_inputs import PreparedInferenceRun
from biomodals.app.fold.alphafold3.invocation_cache import PreparedInvocation
from biomodals.app.fold.alphafold3.msa_search import (
    MsaAssemblyTask,
    RawSearchTask,
    sequence_hash,
)
from biomodals.app.fold.alphafold3.template_search import TemplateTask
from biomodals.execution import TaskPlan

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _digest(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def raw_search_task_plan(
    task: RawSearchTask,
    *,
    search_identity: str,
) -> TaskPlan:
    """Describe one canonical sequence-by-database search."""
    identity = _digest(search_identity, field_name="search_identity")
    query_identity = sequence_hash(task.sequence)
    return TaskPlan(
        task_key=f"{task.database_id}:{query_identity}:{identity}",
        scientific_payload={
            "database_id": task.database_id,
            "polymer": task.polymer,
            "search_identity": identity,
            "sequence_sha256": query_identity,
        },
        execution_payload={
            "database_id": task.database_id,
            "sequence_sha256": query_identity,
        },
    )


def stage_request_task_plan(invocation: PreparedInvocation) -> TaskPlan:
    """Describe validation of the immutable staged request input."""
    return TaskPlan(
        task_key="request",
        scientific_payload={"invocation_id": invocation.invocation_id},
        execution_payload={"mode": "local"},
    )


def combined_msa_task_plan(
    task: MsaAssemblyTask,
    *,
    raw_search_identities: Mapping[str, str],
) -> TaskPlan:
    """Describe one combined-MSA publication over fixed raw-search identities."""
    query_identity = sequence_hash(task.sequence)
    dependencies = {
        database_id: _digest(
            identity,
            field_name=f"raw_search_identities[{database_id!r}]",
        )
        for database_id, identity in sorted(raw_search_identities.items())
    }
    return TaskPlan(
        task_key=(
            f"{task.polymer}:{query_identity}:"
            f"u{int(task.include_unpaired)}:p{int(task.include_paired)}"
        ),
        scientific_payload={
            "include_paired": task.include_paired,
            "include_unpaired": task.include_unpaired,
            "polymer": task.polymer,
            "raw_search_identities": dependencies,
            "sequence_sha256": query_identity,
        },
        execution_payload={
            "include_paired": task.include_paired,
            "include_unpaired": task.include_unpaired,
            "polymer": task.polymer,
            "sequence_sha256": query_identity,
        },
    )


def template_search_task_plan(task: TemplateTask) -> TaskPlan:
    """Describe one canonical or request-local protein template search."""
    query_identity = sequence_hash(task.sequence)
    publication_kind = "canonical" if task.publish_canonical else "request-local"
    return TaskPlan(
        task_key=f"{query_identity}:{task.template_identity}:{publication_kind}",
        scientific_payload={
            "max_template_date": task.max_template_date,
            "publish_canonical": task.publish_canonical,
            "sequence_sha256": query_identity,
            "template_identity": task.template_identity,
            "unpaired_msa_sha256": task.unpaired_msa_sha256,
        },
        execution_payload={
            "publish_canonical": task.publish_canonical,
            "sequence_sha256": query_identity,
            "template_identity": task.template_identity,
        },
    )


def seed_prediction_task_plan(
    prepared: PreparedInferenceRun,
    seed: int,
) -> TaskPlan:
    """Describe one independently publishable AlphaFold3 seed prediction."""
    if seed not in prepared.normalized_seeds:
        raise ValueError(f"Seed {seed} is not part of the prepared request")
    return TaskPlan(
        task_key=f"seed:{seed}",
        scientific_payload={
            "request_id": prepared.request_id,
            "run_id": prepared.run_id,
            "sample_count": prepared.sample_count,
            "seed": seed,
        },
        execution_payload={
            "request_id": prepared.request_id,
            "run_id": prepared.run_id,
            "seed": seed,
        },
    )


def staged_inference_task_plan(prepared: PreparedInferenceRun) -> TaskPlan:
    """Describe publication of one marker-complete staged inference input."""
    return TaskPlan(
        task_key="staged-input",
        scientific_payload={
            "request_id": prepared.request_id,
            "run_id": prepared.run_id,
        },
        execution_payload={"mode": "local"},
    )


def inference_summary_task_plan(prepared: PreparedInferenceRun) -> TaskPlan:
    """Describe the accumulated summary required by this seed request."""
    return TaskPlan(
        task_key="summary",
        scientific_payload={
            "normalized_seeds": list(prepared.normalized_seeds),
            "run_id": prepared.run_id,
            "sample_count": prepared.sample_count,
        },
    )


def request_publication_task_plan(prepared: PreparedInferenceRun) -> TaskPlan:
    """Describe the immutable request view and exact invocation receipt."""
    return TaskPlan(
        task_key="request-view",
        scientific_payload={
            "display_name": prepared.display_name,
            "normalized_seeds": list(prepared.normalized_seeds),
            "request_id": prepared.request_id,
            "run_id": prepared.run_id,
            "submitted_seeds": list(prepared.submitted_seeds),
        },
    )
