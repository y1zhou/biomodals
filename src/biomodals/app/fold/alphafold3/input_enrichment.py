"""Pure AlphaFold 3 MSA and template planning and result reduction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from uniaf3.schema.alphafold3 import AF3Config, AF3Template

from biomodals.app.fold.alphafold3.msa_search import (
    ChainMsaState,
    MsaArtifactReference,
    MsaAssemblyTask,
    Polymer,
    RawSearchTask,
    field_is_populated,
    sequence_cache_relpath,
    sequence_hash,
)
from biomodals.app.fold.alphafold3.template_search import TemplateTask

SequenceKey = tuple[Polymer, str]


@dataclass(frozen=True, slots=True)
class MsaAssemblyResolution:
    """Validated assembled MSA fields keyed by polymer and sequence."""

    fields_by_sequence: dict[SequenceKey, dict[str, str]]
    unpaired_references: dict[SequenceKey, MsaArtifactReference]


@dataclass(frozen=True, slots=True)
class TemplateSearchPlan:
    """Deduplicated template tasks and the chains that consume each result."""

    tasks: tuple[TemplateTask, ...]
    chain_indices_by_identity: dict[str, tuple[int, ...]]

    @property
    def canonical_tasks(self) -> tuple[TemplateTask, ...]:
        """Return tasks whose MSA evidence can be published canonically."""
        return tuple(task for task in self.tasks if task.publish_canonical)

    @property
    def request_local_tasks(self) -> tuple[TemplateTask, ...]:
        """Return tasks derived from caller-supplied MSA evidence."""
        return tuple(task for task in self.tasks if not task.publish_canonical)


@dataclass(frozen=True, slots=True)
class TemplateCacheResolution:
    """Validated cache hits plus canonical tasks that still need workers."""

    templates_by_identity: dict[str, tuple[AF3Template, ...]]
    missing_tasks: tuple[TemplateTask, ...]


def fill_missing_msa_for_inference(config: AF3Config) -> AF3Config:
    """Mark bare chains as explicit single-sequence inference inputs."""
    for entry in config.sequences:
        if (protein := entry.protein) is not None:
            if not field_is_populated(
                protein.unpairedMsa,
                protein.unpairedMsaPath,
            ):
                protein.unpairedMsa = ""
            if not field_is_populated(
                protein.pairedMsa,
                protein.pairedMsaPath,
            ):
                protein.pairedMsa = ""
            if not protein.templates:
                protein.templates = []
        elif (rna := entry.rna) is not None and not field_is_populated(
            rna.unpairedMsa,
            rna.unpairedMsaPath,
        ):
            rna.unpairedMsa = ""
    return config


def chain_msa_states(config: AF3Config) -> tuple[ChainMsaState, ...]:
    """Describe caller-supplied MSA fields without sharing caller evidence."""
    states: list[ChainMsaState] = []
    for index, entry in enumerate(config.sequences):
        if (protein := entry.protein) is not None:
            states.append(
                ChainMsaState(
                    chain_index=index,
                    polymer="protein",
                    sequence=protein.sequence,
                    unpaired_present=field_is_populated(
                        protein.unpairedMsa,
                        protein.unpairedMsaPath,
                    ),
                    paired_present=field_is_populated(
                        protein.pairedMsa,
                        protein.pairedMsaPath,
                    ),
                )
            )
        elif (rna := entry.rna) is not None:
            states.append(
                ChainMsaState(
                    chain_index=index,
                    polymer="rna",
                    sequence=rna.sequence,
                    unpaired_present=field_is_populated(
                        rna.unpairedMsa,
                        rna.unpairedMsaPath,
                    ),
                    paired_present=False,
                )
            )
    return tuple(states)


def missing_raw_searches(
    tasks: Sequence[RawSearchTask],
    statuses: Sequence[Mapping[str, object]],
) -> tuple[RawSearchTask, ...]:
    """Validate cache inspection results and return missing database tasks."""
    if len(statuses) != len(tasks):
        raise RuntimeError("MSA cache inspection returned the wrong result count")
    missing: list[RawSearchTask] = []
    for task, status in zip(tasks, statuses, strict=True):
        if (
            status.get("database_id") != task.database_id
            or status.get("sequence_sha256") != task.sequence_hash
            or status.get("status") not in {"missing", "reused"}
        ):
            raise RuntimeError(f"Invalid MSA cache inspection result: {status}")
        if status["status"] == "missing":
            missing.append(task)
    return tuple(missing)


def reduce_msa_assembly_results(
    tasks: Sequence[MsaAssemblyTask],
    outcomes: Sequence[Mapping[str, object]],
) -> MsaAssemblyResolution:
    """Validate assembled MSA results and reduce them by unique sequence."""
    if len(outcomes) != len(tasks):
        raise RuntimeError("MSA assembly returned the wrong result count")
    fields_by_sequence: dict[SequenceKey, dict[str, str]] = {}
    unpaired_references: dict[SequenceKey, MsaArtifactReference] = {}
    for task, outcome in zip(tasks, outcomes, strict=True):
        status = outcome.get("status")
        if (
            status not in {"published", "reused", "request-local"}
            or outcome.get("polymer") != task.polymer
            or outcome.get("sequence_sha256") != sequence_hash(task.sequence)
        ):
            raise RuntimeError(f"Invalid MSA assembly result: {outcome!r}")
        raw_fields = outcome.get("fields")
        if not isinstance(raw_fields, dict):
            raise RuntimeError(f"Invalid MSA assembly result: {outcome!r}")
        expected_fields = {
            field
            for field, include in (
                ("unpairedMsa", task.include_unpaired),
                ("pairedMsa", task.include_paired),
            )
            if include
        }
        if set(raw_fields) != expected_fields:
            raise RuntimeError(f"Invalid MSA assembly fields: {raw_fields!r}")
        fields: dict[str, str] = {}
        for field, value in raw_fields.items():
            if not isinstance(field, str) or not isinstance(value, str) or not value:
                raise RuntimeError(f"Invalid MSA assembly fields: {raw_fields!r}")
            fields[field] = value
        key = (task.polymer, task.sequence)
        if status in {"published", "reused"}:
            combined_identity = outcome.get("combined_identity")
            if (
                not isinstance(combined_identity, str)
                or re.fullmatch(r"[0-9a-f]{64}", combined_identity) is None
            ):
                raise RuntimeError(f"Invalid MSA assembly result: {outcome!r}")
            sequence_root = sequence_cache_relpath(task.polymer, task.sequence)
            try:
                reference = MsaArtifactReference.from_record(
                    outcome.get("unpaired_msa_reference"),
                    expected_path=sequence_root / "unpaired.a3m",
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid unpaired MSA reference: {outcome!r}"
                ) from exc
            if not reference.matches_content(fields["unpairedMsa"].encode()):
                raise RuntimeError(
                    "Unpaired MSA reference does not match the returned field"
                )
            unpaired_references[key] = reference
        fields_by_sequence[key] = fields
    return MsaAssemblyResolution(
        fields_by_sequence=fields_by_sequence,
        unpaired_references=unpaired_references,
    )


def apply_msa_resolution(
    config: AF3Config,
    states: Sequence[ChainMsaState],
    resolution: MsaAssemblyResolution,
    *,
    search_protein_templates: bool,
) -> AF3Config:
    """Apply resolved MSA fields and explicit no-template semantics to chains."""
    for state in states:
        fields = resolution.fields_by_sequence.get(
            (state.polymer, state.sequence),
            {},
        )
        entry = config.sequences[state.chain_index]
        if (protein := entry.protein) is not None:
            if not state.unpaired_present:
                protein.unpairedMsa = fields["unpairedMsa"]
                protein.unpairedMsaPath = None
            if not state.paired_present:
                protein.pairedMsa = fields["pairedMsa"]
                protein.pairedMsaPath = None
            if not search_protein_templates and not protein.templates:
                protein.templates = []
        elif (rna := entry.rna) is not None and not state.unpaired_present:
            rna.unpairedMsa = fields["unpairedMsa"]
            rna.unpairedMsaPath = None
    return config


def _resolved_msa_text(
    inline_value: str | None,
    path_value: str | None,
    *,
    field_name: str,
) -> str:
    if inline_value and path_value:
        raise ValueError(f"{field_name} cannot set both inline and path forms")
    if inline_value:
        return inline_value
    if path_value:
        value = Path(path_value).read_text()
        if not value:
            raise ValueError(f"{field_name} path is empty: {path_value}")
        return value
    raise ValueError(f"{field_name} is unresolved")


def plan_template_searches(
    config: AF3Config,
    states: Sequence[ChainMsaState],
    resolution: MsaAssemblyResolution,
) -> TemplateSearchPlan:
    """Deduplicate missing templates without publishing caller MSA evidence."""
    tasks: dict[str, TemplateTask] = {}
    chain_indices: dict[str, list[int]] = {}
    for state in states:
        if state.polymer != "protein":
            continue
        protein = config.sequences[state.chain_index].protein
        if protein is None:
            raise RuntimeError("Protein MSA state no longer matches its chain")
        if protein.templates:
            continue
        reference = resolution.unpaired_references.get(("protein", state.sequence))
        publish_canonical = not state.unpaired_present and reference is not None
        unpaired_msa = (
            None
            if publish_canonical
            else _resolved_msa_text(
                protein.unpairedMsa,
                protein.unpairedMsaPath,
                field_name=f"sequences[{state.chain_index}].protein.unpairedMsa",
            )
        )
        candidate = TemplateTask(
            sequence=state.sequence,
            unpaired_msa=unpaired_msa,
            unpaired_msa_reference=reference if publish_canonical else None,
            publish_canonical=publish_canonical,
        )
        identity = candidate.template_identity
        if existing := tasks.get(identity):
            if (
                existing.sequence != candidate.sequence
                or existing.unpaired_msa_sha256 != candidate.unpaired_msa_sha256
            ):
                raise RuntimeError("Protein template identity collision")
            if publish_canonical and not existing.publish_canonical:
                tasks[identity] = candidate
        else:
            tasks[identity] = candidate
        chain_indices.setdefault(identity, []).append(state.chain_index)
    return TemplateSearchPlan(
        tasks=tuple(tasks.values()),
        chain_indices_by_identity={
            identity: tuple(indices) for identity, indices in chain_indices.items()
        },
    )


def validate_template_result(
    task: TemplateTask,
    outcome: Mapping[str, object],
    *,
    allowed_statuses: frozenset[str],
) -> tuple[AF3Template, ...]:
    """Validate one cache or worker result before applying it to input chains."""
    if (
        outcome.get("status") not in allowed_statuses
        or outcome.get("sequence_sha256") != sequence_hash(task.sequence)
        or outcome.get("unpaired_msa_sha256") != task.unpaired_msa_sha256
        or outcome.get("template_identity") != task.template_identity
    ):
        raise RuntimeError(f"Invalid protein template result: {outcome!r}")
    raw_templates = outcome.get("templates")
    if not isinstance(raw_templates, list) or not all(
        isinstance(template, dict) for template in raw_templates
    ):
        raise RuntimeError(f"Invalid protein template payload: {outcome!r}")
    return tuple(AF3Template.model_validate(template) for template in raw_templates)


def reduce_template_cache_results(
    tasks: Sequence[TemplateTask],
    statuses: Sequence[Mapping[str, object]],
) -> TemplateCacheResolution:
    """Validate canonical template cache results and identify missing tasks."""
    if len(statuses) != len(tasks):
        raise RuntimeError(
            "Protein template cache inspection returned the wrong result count"
        )
    templates_by_identity: dict[str, tuple[AF3Template, ...]] = {}
    missing: list[TemplateTask] = []
    for task, status in zip(tasks, statuses, strict=True):
        if status.get("status") == "missing":
            if (
                status.get("sequence_sha256") != sequence_hash(task.sequence)
                or status.get("unpaired_msa_sha256") != task.unpaired_msa_sha256
                or status.get("template_identity") != task.template_identity
            ):
                raise RuntimeError(f"Invalid protein template cache result: {status!r}")
            missing.append(task)
        else:
            templates_by_identity[task.template_identity] = validate_template_result(
                task,
                status,
                allowed_statuses=frozenset({"reused"}),
            )
    return TemplateCacheResolution(
        templates_by_identity=templates_by_identity,
        missing_tasks=tuple(missing),
    )


def apply_template_results(
    config: AF3Config,
    plan: TemplateSearchPlan,
    templates_by_identity: Mapping[str, tuple[AF3Template, ...]],
) -> AF3Config:
    """Apply each deduplicated template result to all consuming protein chains."""
    for identity, chain_indices in plan.chain_indices_by_identity.items():
        templates = templates_by_identity.get(identity)
        if templates is None:
            raise RuntimeError(f"Protein template task produced no result: {identity}")
        for chain_index in chain_indices:
            protein = config.sequences[chain_index].protein
            if protein is None:
                raise RuntimeError("Protein template plan no longer matches its chain")
            protein.templates = [
                template.model_copy(deep=True) for template in templates
            ]
    return config
