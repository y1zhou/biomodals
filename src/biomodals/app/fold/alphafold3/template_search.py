"""Resumable AlphaFold 3 protein-template search.

Template search is deliberately separate from the sharded MSA phase. It reads
the fixed PDB sequence/mmCIF store directly and publishes only canonical
sequence-plus-unpaired-MSA results.
"""

from __future__ import annotations

import datetime
import inspect
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, cast

import orjson

from biomodals.app.fold.alphafold3.artifacts import (
    VolumeHandle,
    append_log,
    artifact_record,
    json_bytes,
    load_artifact_bytes,
    require_regular_file,
    sha256_bytes,
    utc_now,
    write_json_atomic,
)
from biomodals.app.fold.alphafold3.generation_claims import (
    ActiveGenerationError,
    ClaimStore,
    GenerationClaim,
    acquire_generation_claim,
    assert_generation_current,
    finish_generation_claim,
)
from biomodals.app.fold.alphafold3.inference_inputs import MAX_LOCAL_MSA_BYTES
from biomodals.app.fold.alphafold3.msa_search import (
    MsaArtifactReference,
    SearchRuntime,
    sequence_cache_relpath,
    sequence_hash,
    validate_query,
    validate_remote_search_task_count,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    HMMER_VERSION,
    SOURCE_DB_VOLUME_NAME,
    resolve_database_profile,
)

DEFAULT_MAX_TEMPLATE_DATE = "2021-09-30"
PDB_SEQRES_FILENAME = "pdb_seqres_2022_09_28.fasta"
MMCIF_DIRECTORY_NAME = "mmcif_files"
HMMSEARCH_BINARY_PATH = "/hmmer/bin/hmmsearch"
HMMBUILD_BINARY_PATH = "/hmmer/bin/hmmbuild"
HMMSEARCH_N_CPU = 8

TEMPLATE_RESULT_SCHEMA_VERSION = 1
TEMPLATE_IDENTITY_SCHEMA_VERSION = 1
TEMPLATE_ADAPTER_VERSION = "af3-protein-template-v1"


@dataclass(frozen=True, slots=True)
class TemplateRuntime:
    """Mounted resources and coordination state for one template worker."""

    SOURCE_MOUNT: ClassVar[str] = f"/{SOURCE_DB_VOLUME_NAME}"
    CACHE_MOUNT: ClassVar[str] = SearchRuntime.CACHE_MOUNT
    CACHE_VOLUME_SUBPATH: ClassVar[str] = SearchRuntime.CACHE_VOLUME_SUBPATH

    source_volume: VolumeHandle
    cache_volume: VolumeHandle
    claims: ClaimStore
    container_id: str
    maximum_age_seconds: int | float
    wait_timeout_seconds: int | float
    claim_poll_seconds: float = 5.0
    source_root: Path = Path(SOURCE_MOUNT)
    cache_root: Path = Path(CACHE_MOUNT)


@dataclass(frozen=True, slots=True)
class TemplateTask:
    """One deduplicated protein template-search input."""

    sequence: str
    unpaired_msa: str | None
    publish_canonical: bool
    max_template_date: str = DEFAULT_MAX_TEMPLATE_DATE
    unpaired_msa_reference: MsaArtifactReference | None = None
    _unpaired_msa_sha256: str = field(init=False, repr=False, compare=False)
    _template_identity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Require canonical references or request-local inline evidence."""
        has_inline = self.unpaired_msa is not None
        has_reference = self.unpaired_msa_reference is not None
        if has_inline == has_reference:
            raise ValueError(
                "TemplateTask requires exactly one unpaired MSA representation"
            )
        if self.publish_canonical != has_reference:
            raise ValueError(
                "Only canonical template tasks may use an MSA artifact reference"
            )
        digest = (
            self.unpaired_msa_reference.sha256
            if self.unpaired_msa_reference is not None
            else sha256_bytes(cast(str, self.unpaired_msa).encode())
        )
        object.__setattr__(self, "_unpaired_msa_sha256", digest)
        object.__setattr__(
            self,
            "_template_identity",
            template_search_identity(
                self.sequence,
                digest,
                self.max_template_date,
            ),
        )

    @property
    def unpaired_msa_sha256(self) -> str:
        """Return the content identity of the resolved unpaired MSA."""
        return self._unpaired_msa_sha256

    @property
    def template_identity(self) -> str:
        """Return the scientific identity of this template search."""
        return self._template_identity


@dataclass(frozen=True, slots=True)
class TemplateContext:
    """Expected identity and flat cache location for one template result."""

    sequence: str
    sequence_hash: str
    unpaired_msa_sha256: str
    template_identity: str
    provenance: dict[str, object]
    sequence_root: Path


@dataclass(frozen=True, slots=True)
class TemplateEntry:
    """One validated canonical template publication."""

    context: TemplateContext
    templates: list[dict[str, object]]
    done_sha256: str

    def summary(self, status: str) -> dict[str, object]:
        """Return a reusable result for the local coordinator."""
        return {
            "status": status,
            "sequence_sha256": self.context.sequence_hash,
            "unpaired_msa_sha256": self.context.unpaired_msa_sha256,
            "template_identity": self.context.template_identity,
            "done_sha256": self.done_sha256,
            "templates": self.templates,
        }


def validate_max_template_date(value: str) -> str:
    """Validate and normalize an ISO template-date cutoff."""
    if not isinstance(value, str):
        raise TypeError("max_template_date must be a string")
    try:
        parsed = datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("max_template_date must use YYYY-MM-DD") from exc
    return parsed.isoformat()


def _hmmsearch_parameters() -> dict[str, object]:
    """Return result-affecting HMMER arguments shared by identity and runtime."""
    return {
        "filter_f1": 0.1,
        "filter_f2": 0.1,
        "filter_f3": 0.1,
        "e_value": 100,
        "inc_e": 100,
        "dom_e": 100,
        "incdom_e": 100,
        "alphabet": "amino",
        "filter_max": False,
    }


def _template_filter_parameters() -> dict[str, object]:
    """Return result-affecting template filters shared by identity and runtime."""
    return {
        "max_subsequence_ratio": 0.95,
        "min_align_ratio": 0.1,
        "min_hit_length": 10,
        "deduplicate_sequences": True,
        "max_hits": 4,
    }


def template_search_parameters(max_template_date: str) -> dict[str, object]:
    """Return every result-affecting upstream template parameter."""
    selected_date = validate_max_template_date(max_template_date)
    return {
        "tool": "hmmsearch",
        "hmmsearch_n_cpu": HMMSEARCH_N_CPU,
        "max_template_date": selected_date,
        "max_a3m_query_sequences": None,
        "hmmsearch": _hmmsearch_parameters(),
        "filter": _template_filter_parameters(),
    }


def template_search_identity(
    sequence: str,
    unpaired_msa_sha256: str,
    max_template_date: str,
) -> str:
    """Hash the scientific identity of one protein template search."""
    validate_query(resolve_database_profile("uniref90"), sequence)
    if re.fullmatch(r"[0-9a-f]{64}", unpaired_msa_sha256) is None:
        raise ValueError("unpaired_msa_sha256 must be a lowercase SHA-256 digest")
    return sha256_bytes(
        json_bytes({
            "schema_version": TEMPLATE_IDENTITY_SCHEMA_VERSION,
            "adapter_version": TEMPLATE_ADAPTER_VERSION,
            "sequence": sequence,
            "unpaired_msa_sha256": unpaired_msa_sha256,
            "parameters": template_search_parameters(max_template_date),
            "alphafold_commit": ALPHAFOLD3_COMMIT,
            "hmmer_version": HMMER_VERSION,
        })
    )


def build_template_context(
    cache_root: Path,
    sequence: str,
    unpaired_msa_sha256: str,
    max_template_date: str,
) -> TemplateContext:
    """Build one expected flat template-cache identity."""
    selected_date = validate_max_template_date(max_template_date)
    query = validate_query(resolve_database_profile("uniref90"), sequence)
    template_identity = template_search_identity(
        query,
        unpaired_msa_sha256,
        selected_date,
    )
    query_hash = sequence_hash(query)
    provenance: dict[str, object] = {
        "sequence_sha256": query_hash,
        "sequence_length": len(query),
        "unpaired_msa_sha256": unpaired_msa_sha256,
        "template_identity": template_identity,
        "parameters": template_search_parameters(selected_date),
        "adapter_version": TEMPLATE_ADAPTER_VERSION,
        "alphafold_commit": ALPHAFOLD3_COMMIT,
        "hmmer_version": HMMER_VERSION,
    }
    return TemplateContext(
        sequence=query,
        sequence_hash=query_hash,
        unpaired_msa_sha256=unpaired_msa_sha256,
        template_identity=template_identity,
        provenance=provenance,
        sequence_root=cache_root / sequence_cache_relpath("protein", query),
    )


def load_template_entry(context: TemplateContext) -> TemplateEntry | None:
    """Validate and load one marker-complete template publication."""
    done_path = context.sequence_root / "templates.done.json"
    if not done_path.is_file():
        return None
    try:
        done_bytes = done_path.read_bytes()
        done = orjson.loads(done_bytes)
    except (OSError, orjson.JSONDecodeError):
        return None
    if (
        not isinstance(done, dict)
        or done.get("schema_version") != TEMPLATE_RESULT_SCHEMA_VERSION
        or done.get("status") != "complete"
        or done.get("provenance") != context.provenance
    ):
        return None
    artifact = done.get("templates")
    templates_bytes = load_artifact_bytes(
        context.sequence_root,
        artifact,
        "templates.json",
    )
    if templates_bytes is None:
        return None
    try:
        templates = orjson.loads(templates_bytes)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(templates, list) or not all(
        isinstance(template, dict) for template in templates
    ):
        return None
    return TemplateEntry(
        context=context,
        templates=cast(list[dict[str, object]], templates),
        done_sha256=sha256_bytes(done_bytes),
    )


def inspect_template_entries(
    cache_root: Path,
    inputs: tuple[tuple[str, str, str], ...],
) -> list[dict[str, object]]:
    """Inspect canonical template markers and return reusable payloads."""
    validate_remote_search_task_count(len(inputs))
    statuses: list[dict[str, object]] = []
    for sequence, unpaired_msa_sha256, max_template_date in inputs:
        context = build_template_context(
            cache_root,
            sequence,
            unpaired_msa_sha256,
            max_template_date,
        )
        entry = load_template_entry(context)
        if entry is None:
            statuses.append({
                "status": "missing",
                "sequence_sha256": context.sequence_hash,
                "unpaired_msa_sha256": context.unpaired_msa_sha256,
                "template_identity": context.template_identity,
            })
        else:
            statuses.append(entry.summary("reused"))
    return statuses


def assert_pinned_template_contract() -> dict[str, str]:
    """Bind the adapter to the pinned upstream template call and conversion."""
    from importlib import import_module

    pipeline = import_module("alphafold3.data.pipeline")
    templates_module = import_module("alphafold3.data.templates")
    pipeline_source = inspect.getsource(pipeline._get_protein_templates)  # noqa: SLF001
    compact_pipeline = re.sub(r"\s+", "", pipeline_source)
    required_pipeline = (
        "Templates.from_seq_and_a3m(",
        "max_a3m_query_sequences=None",
        "chain_poly_type=mmcif_names.PROTEIN_CHAIN",
        "structure_store=structure_stores.StructureStore(pdb_database_path)",
    )
    if not all(
        pattern.replace(" ", "") in compact_pipeline for pattern in required_pipeline
    ):
        raise RuntimeError("Pinned protein template-search contract changed")
    structures_source = inspect.getsource(
        templates_module.Templates.get_hits_with_structures
    )
    return {
        "get_protein_templates_sha256": sha256_bytes(pipeline_source.encode()),
        "get_hits_with_structures_sha256": sha256_bytes(structures_source.encode()),
    }


def _validate_template_msa(sequence: str, unpaired_msa: str) -> str:
    """Bound and parse one protein A3M, including its required query row."""
    query = validate_query(resolve_database_profile("uniref90"), sequence)
    if not isinstance(unpaired_msa, str) or not unpaired_msa:
        raise ValueError("unpaired_msa must be a non-empty A3M string")
    if not unpaired_msa.isascii():
        raise ValueError("unpaired_msa must contain only ASCII A3M text")
    if len(unpaired_msa) > MAX_LOCAL_MSA_BYTES:
        raise ValueError(f"unpaired_msa exceeds the {MAX_LOCAL_MSA_BYTES}-byte limit")

    records: list[str] = []
    sequence_lines: list[str] = []
    saw_header = False
    for line in unpaired_msa.splitlines():
        if line.startswith(">"):
            if len(line) == 1:
                raise ValueError("unpaired_msa contains an empty FASTA header")
            if saw_header:
                if not sequence_lines:
                    raise ValueError("unpaired_msa contains an empty FASTA record")
                records.append("".join(sequence_lines))
                sequence_lines.clear()
            saw_header = True
        elif line:
            if not saw_header or any(char.isspace() for char in line):
                raise ValueError("unpaired_msa is not valid FASTA/A3M text")
            sequence_lines.append(line)
    if not saw_header or not sequence_lines:
        raise ValueError("unpaired_msa contains no complete FASTA/A3M record")
    records.append("".join(sequence_lines))
    if records[0] != query:
        raise ValueError("unpaired_msa query row does not match the protein sequence")
    for record in records:
        if any(not char.isalpha() and char != "-" for char in record):
            raise ValueError("unpaired_msa contains invalid A3M sequence characters")
        if sum(not char.islower() for char in record) != len(query):
            raise ValueError("unpaired_msa rows do not match the query alignment width")
    return unpaired_msa


def _resolve_template_msa(runtime: TemplateRuntime, task: TemplateTask) -> str:
    """Load canonical evidence from the cache Volume or use caller text."""
    reference = task.unpaired_msa_reference
    if reference is None:
        if task.unpaired_msa is None:
            raise RuntimeError("Template task has no unpaired MSA")
        return _validate_template_msa(task.sequence, task.unpaired_msa)
    if reference.size_bytes > MAX_LOCAL_MSA_BYTES:
        raise ValueError(f"unpaired_msa exceeds the {MAX_LOCAL_MSA_BYTES}-byte limit")
    runtime.cache_volume.reload()
    content = load_artifact_bytes(
        runtime.cache_root,
        reference.to_record(),
        reference.relative_path.as_posix(),
    )
    if content is None:
        raise RuntimeError("Canonical unpaired MSA reference failed validation")
    try:
        unpaired_msa = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("unpaired_msa must contain only ASCII A3M text") from exc
    return _validate_template_msa(task.sequence, unpaired_msa)


def _execute_template_search(
    sequence: str,
    unpaired_msa: str,
    source_root: Path,
    max_template_date: str,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Run the exact pinned eight-CPU template search and serialize its hits."""
    from importlib import import_module

    query = validate_query(resolve_database_profile("uniref90"), sequence)
    selected_date = datetime.date.fromisoformat(
        validate_max_template_date(max_template_date)
    )
    seqres_path = source_root / PDB_SEQRES_FILENAME
    mmcif_path = source_root / MMCIF_DIRECTORY_NAME
    require_regular_file(seqres_path)
    if not mmcif_path.is_dir():
        raise FileNotFoundError(f"Expected mmCIF directory: {mmcif_path}")

    contract = assert_pinned_template_contract()
    msa_config = import_module("alphafold3.data.msa_config")
    mmcif_names = import_module("alphafold3.constants.mmcif_names")
    pipeline = import_module("alphafold3.data.pipeline")
    template_config = msa_config.TemplatesConfig(
        template_tool_config=msa_config.TemplateToolConfig(
            database_path=str(seqres_path),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            hmmsearch_config=msa_config.HmmsearchConfig(
                hmmsearch_binary_path=HMMSEARCH_BINARY_PATH,
                hmmbuild_binary_path=HMMBUILD_BINARY_PATH,
                **_hmmsearch_parameters(),
            ),
        ),
        filter_config=msa_config.TemplateFilterConfig(
            **_template_filter_parameters(),
            max_template_date=selected_date,
        ),
    )
    template_hits = pipeline._get_protein_templates(  # noqa: SLF001
        sequence=query,
        input_msa_a3m=unpaired_msa,
        run_template_search=True,
        templates_config=template_config,
        pdb_database_path=str(mmcif_path),
    )
    templates: list[dict[str, object]] = []
    for hit, structure in template_hits.get_hits_with_structures():
        mapping = list(hit.query_to_hit_mapping.items())
        templates.append({
            "mmcif": structure.to_mmcif(),
            "queryIndices": [query_index for query_index, _ in mapping],
            "templateIndices": [template_index for _, template_index in mapping],
        })
    return templates, contract


def _template_claim_scope(context: TemplateContext) -> str:
    return f"template:Protein:{context.sequence_hash}"


def _wait_for_template_claim(
    runtime: TemplateRuntime,
    context: TemplateContext,
) -> tuple[TemplateEntry | None, GenerationClaim | None]:
    generation_id = uuid.uuid4().hex
    deadline = time.monotonic() + float(runtime.wait_timeout_seconds)
    while True:
        runtime.cache_volume.reload()
        if entry := load_template_entry(context):
            return entry, None
        try:
            claim = acquire_generation_claim(
                runtime.claims,
                scope_key=_template_claim_scope(context),
                generation_id=generation_id,
                identity=context.provenance,
                container_id=runtime.container_id,
                maximum_age_seconds=runtime.maximum_age_seconds,
            )
            return None, claim
        except ActiveGenerationError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                owner = exc.owner["generation_id"]
                raise TimeoutError(
                    f"Timed out waiting for template owner {owner!r}: "
                    f"{context.sequence_hash}"
                ) from exc
            time.sleep(min(runtime.claim_poll_seconds, remaining))


def run_template_search(
    runtime: TemplateRuntime,
    task: TemplateTask,
) -> dict[str, object]:
    """Run a request-local search or publish one canonical template result."""
    if not isinstance(task.publish_canonical, bool):
        raise TypeError("publish_canonical must be a boolean")
    validate_query(resolve_database_profile("uniref90"), task.sequence)
    unpaired_msa = _resolve_template_msa(runtime, task)
    runtime.source_volume.reload()
    context = build_template_context(
        runtime.cache_root,
        task.sequence,
        task.unpaired_msa_sha256,
        task.max_template_date,
    )
    if not task.publish_canonical:
        templates, contract = _execute_template_search(
            task.sequence,
            unpaired_msa,
            runtime.source_root,
            task.max_template_date,
        )
        return {
            "status": "request-local",
            "sequence_sha256": context.sequence_hash,
            "unpaired_msa_sha256": context.unpaired_msa_sha256,
            "template_identity": context.template_identity,
            "templates": templates,
            "contract": contract,
        }

    runtime.cache_volume.reload()
    if entry := load_template_entry(context):
        return entry.summary("reused")
    raced_entry, claim = _wait_for_template_claim(runtime, context)
    if raced_entry is not None:
        return raced_entry.summary("reused")
    if claim is None:
        raise RuntimeError("Template claim election returned no owner")

    generation_root = (
        context.sequence_root
        / ".staging"
        / "templates"
        / context.template_identity
        / claim.generation_id
    )
    log_path = generation_root / "run.log"
    terminal_status = "failed"
    terminal_detail: dict[str, object] = {}
    try:
        runtime.cache_volume.reload()
        if entry := load_template_entry(context):
            terminal_status = "complete"
            terminal_detail = {
                "publication": "raced",
                "done_sha256": entry.done_sha256,
            }
            return entry.summary("reused")
        append_log(
            log_path,
            f"Searching protein templates for {context.sequence_hash}",
        )
        templates, contract = _execute_template_search(
            task.sequence,
            unpaired_msa,
            runtime.source_root,
            task.max_template_date,
        )
        append_log(
            log_path,
            f"Completed template search with {len(templates)} hits",
        )
        templates_path = generation_root / "templates.json"
        write_json_atomic(templates_path, templates)
        artifact = artifact_record(templates_path, generation_root)
        runtime.cache_volume.commit()
        assert_generation_current(runtime.claims, claim)
        context.sequence_root.mkdir(parents=True, exist_ok=True)
        os.replace(
            templates_path,
            context.sequence_root / "templates.json",
        )
        runtime.cache_volume.commit()
        write_json_atomic(
            context.sequence_root / "templates.done.json",
            {
                "schema_version": TEMPLATE_RESULT_SCHEMA_VERSION,
                "status": "complete",
                "completed_at": utc_now(),
                "generation_id": claim.generation_id,
                "provenance": context.provenance,
                "contract": contract,
                "templates": artifact,
            },
        )
        runtime.cache_volume.commit()
        entry = load_template_entry(context)
        if entry is None:
            raise RuntimeError("Published template result failed validation")
        shutil.rmtree(generation_root, ignore_errors=True)
        runtime.cache_volume.commit()
        terminal_status = "complete"
        terminal_detail = {
            "publication": "published",
            "done_sha256": entry.done_sha256,
        }
        return entry.summary("published")
    except Exception as exc:
        append_log(log_path, f"Failed with {type(exc).__name__}: {exc}")
        write_json_atomic(
            generation_root / "failure.json",
            {
                "failed_at": utc_now(),
                "sequence_sha256": context.sequence_hash,
                "template_identity": context.template_identity,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        runtime.cache_volume.commit()
        terminal_detail = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        finish_generation_claim(
            runtime.claims,
            claim,
            status=terminal_status,
            detail=terminal_detail,
        )
