"""Resumable sharded AlphaFold 3 MSA searches and assembly.

This module owns the scientific search adapter and durable cache protocol.
Modal decorators and named resources remain in the app composition roots.
"""

from __future__ import annotations

import hashlib
import heapq
import inspect
import itertools
import os
import re
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import orjson

from biomodals.app.fold.alphafold3.generation_claims import (
    ActiveGenerationError,
    ClaimStore,
    GenerationClaim,
    acquire_generation_claim,
    assert_generation_current,
    finish_generation_claim,
)
from biomodals.app.fold.alphafold3.profile_builder import (
    validate_profile_manifest,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    HMMER_VERSION,
    JACKHMMER_PATCH_SHA256,
    DatabaseProfileSpec,
    profile_root,
    resolve_database_profile,
    shard_names,
)
from biomodals.app.fold.alphafold3.sharding import (
    append_log,
    load_json_object,
    require_regular_file,
    sha256_file,
    utc_now,
    write_json_atomic,
)

Polymer = Literal["protein", "rna"]

MSA_SEARCH_CLAIM_DICT_NAME = "AlphaFold3-msa-search-claims"
SEARCH_IDENTITY_SCHEMA_VERSION = 1
RAW_RESULT_SCHEMA_VERSION = 1
COMBINED_RESULT_SCHEMA_VERSION = 1
SEARCH_ADAPTER_VERSION = "af3-sharded-msa-v1"
NHMMER_SHARDED_MERGE_ORDER = "reported-evalue-descending-bit-score-name-v1"

JACKHMMER_BINARY_PATH = "/hmmer/bin/jackhmmer"
NHMMER_BINARY_PATH = "/hmmer/bin/nhmmer"
HMMALIGN_BINARY_PATH = "/hmmer/bin/hmmalign"
HMMBUILD_BINARY_PATH = "/hmmer/bin/hmmbuild"

JACKHMMER_N_ITER = 1
JACKHMMER_E_VALUE = 1e-4
JACKHMMER_FILTER_F1 = 5e-4
JACKHMMER_FILTER_F2 = 5e-5
JACKHMMER_FILTER_F3 = 5e-7
NHMMER_E_VALUE = 1e-3
NHMMER_FILTER_F3 = 1e-5
NHMMER_SHORT_SEQUENCE_FILTER_F3 = 0.02

SEARCH_N_CPU = 2
SEARCH_MAX_PARALLEL_SHARDS = 16

PROTEIN_UNPAIRED_DATABASES = ("uniref90", "small_bfd", "mgnify")
PROTEIN_PAIRED_DATABASES = ("uniprot",)
RNA_UNPAIRED_DATABASES = ("rfam", "rnacentral", "ntrna")

_JSON_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE


class VolumeHandle(Protocol):
    """Persistence barriers required from a mounted Modal Volume."""

    def reload(self) -> None:
        """Reload commits made by other containers."""
        ...

    def commit(self) -> None:
        """Commit this container's changes."""
        ...


@dataclass(frozen=True, slots=True)
class SearchRuntime:
    """Mounted resources and scheduling state for one search worker."""

    sharded_root: Path
    cache_root: Path
    sharded_volume: VolumeHandle
    cache_volume: VolumeHandle
    claims: ClaimStore
    container_id: str
    maximum_age_seconds: int | float
    wait_timeout_seconds: int | float
    claim_poll_seconds: float = 5.0
    function_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class RawSearchTask:
    """One canonical sequence-by-database search."""

    database_id: str
    sequence: str

    @property
    def spec(self) -> DatabaseProfileSpec:
        """Return the fixed database profile."""
        return resolve_database_profile(self.database_id)

    @property
    def polymer(self) -> Polymer:
        """Return the fixed polymer class."""
        return self.spec.polymer

    @property
    def sequence_hash(self) -> str:
        """Return the sequence-only cache hash."""
        return sequence_hash(self.sequence)


@dataclass(frozen=True, slots=True)
class ChainMsaState:
    """Missing-field state for one input chain."""

    chain_index: int
    polymer: Polymer
    sequence: str
    unpaired_present: bool
    paired_present: bool


@dataclass(frozen=True, slots=True)
class MsaAssemblyTask:
    """Canonical fields needed for one unique polymer and sequence."""

    polymer: Polymer
    sequence: str
    include_unpaired: bool
    include_paired: bool


@dataclass(frozen=True, slots=True)
class MsaResolutionPlan:
    """Deduplicated raw searches and sequence-level assemblies."""

    raw_searches: tuple[RawSearchTask, ...]
    assemblies: tuple[MsaAssemblyTask, ...]


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Validated immutable profile and cache identity for one raw search."""

    spec: DatabaseProfileSpec
    sequence: str
    sequence_hash: str
    profile_root: Path
    manifest_sha256: str
    search_identity: str
    provenance: dict[str, object]
    result_root: Path


@dataclass(frozen=True, slots=True)
class RawMsaEntry:
    """One validated raw database MSA publication."""

    context: SearchContext
    a3m: str
    done_sha256: str
    result_record: dict[str, object]

    def summary(self, status: str) -> dict[str, object]:
        """Return a compact worker result without transferring the A3M."""
        return {
            "status": status,
            "database_id": self.context.spec.database_id,
            "profile_id": self.context.spec.profile_id,
            "polymer": self.context.spec.polymer,
            "sequence_sha256": self.context.sequence_hash,
            "search_identity": self.context.search_identity,
            "done_sha256": self.done_sha256,
            "result": self.result_record,
        }


def _json_bytes(value: object) -> bytes:
    return orjson.dumps(value, option=_JSON_OPTIONS)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sequence_hash(sequence: str) -> str:
    """Hash sequence text only for the shared cache namespace."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    return _sha256_bytes(sequence.encode())


def validate_query(spec: DatabaseProfileSpec, sequence: str) -> str:
    """Validate one query before invoking a pinned HMMER wrapper."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    if not 1 <= len(sequence) <= 10_000:
        raise ValueError("sequence length must be between 1 and 10,000")
    pattern = r"[A-Z]+" if spec.polymer == "protein" else r"[ACGU]+"
    if re.fullmatch(pattern, sequence) is None:
        raise ValueError(f"sequence contains invalid {spec.polymer} characters")
    return sequence


def scientific_search_parameters(
    spec: DatabaseProfileSpec,
) -> dict[str, object]:
    """Return only result-affecting parameters from the pinned pipeline."""
    common: dict[str, object] = {
        "database_id": spec.database_id,
        "polymer": spec.polymer,
        "max_sequences": spec.max_sequences,
        "z_value": spec.search_space_value,
    }
    if spec.polymer == "protein":
        return common | {
            "tool": "jackhmmer",
            "n_iter": JACKHMMER_N_ITER,
            "e_value": JACKHMMER_E_VALUE,
            "dom_z_value": spec.search_space_value,
            "filter_f1": JACKHMMER_FILTER_F1,
            "filter_f2": JACKHMMER_FILTER_F2,
            "filter_f3": JACKHMMER_FILTER_F3,
        }
    return common | {
        "tool": "nhmmer",
        "e_value": NHMMER_E_VALUE,
        "filter_f3": NHMMER_FILTER_F3,
        "alphabet": "rna",
        "short_sequence_filter_f3": NHMMER_SHORT_SEQUENCE_FILTER_F3,
        "sharded_merge_order": NHMMER_SHARDED_MERGE_ORDER,
    }


def operational_search_parameters(spec: DatabaseProfileSpec) -> dict[str, object]:
    """Return scientific parameters plus non-identity worker topology."""
    return scientific_search_parameters(spec) | {
        "n_cpu": SEARCH_N_CPU,
        "max_parallel_shards": SEARCH_MAX_PARALLEL_SHARDS,
    }


def production_search_identity(
    spec: DatabaseProfileSpec,
    sequence: str,
    manifest_sha256: str,
) -> str:
    """Hash the complete scientific identity of one production search."""
    query = validate_query(spec, sequence)
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    return _sha256_bytes(
        _json_bytes({
            "schema_version": SEARCH_IDENTITY_SCHEMA_VERSION,
            "adapter_version": SEARCH_ADAPTER_VERSION,
            "profile_id": spec.profile_id,
            "profile_manifest_sha256": manifest_sha256,
            "sequence": query,
            "parameters": scientific_search_parameters(spec),
            "alphafold_commit": ALPHAFOLD3_COMMIT,
            "hmmer_version": HMMER_VERSION,
            "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
        })
    )


def polymer_cache_dir(polymer: Polymer) -> str:
    """Return the collision-proof top-level cache namespace."""
    if polymer == "protein":
        return "Protein"
    if polymer == "rna":
        return "RNA"
    raise ValueError(f"Unsupported polymer: {polymer!r}")


def sequence_cache_relpath(polymer: Polymer, sequence: str) -> Path:
    """Return one sequence root relative to the cache mount."""
    digest = sequence_hash(sequence)
    return Path(polymer_cache_dir(polymer), digest[:2], digest)


def raw_result_relpath(
    spec: DatabaseProfileSpec,
    sequence: str,
    search_identity: str,
) -> Path:
    """Return one production Raw Database MSA directory."""
    if re.fullmatch(r"[0-9a-f]{64}", search_identity) is None:
        raise ValueError("search_identity must be a lowercase SHA-256 digest")
    return (
        sequence_cache_relpath(spec.polymer, sequence)
        / "raw-msa"
        / spec.database_id
        / search_identity
    )


def field_is_populated(inline_value: str | None, path_value: str | None) -> bool:
    """Return whether a caller supplied a non-empty inline or path value."""
    return bool(inline_value) or bool(path_value)


def plan_msa_resolution(chains: tuple[ChainMsaState, ...]) -> MsaResolutionPlan:
    """Plan independent fields while sharing canonical duplicate-sequence work."""
    raw_searches: dict[tuple[Polymer, str, str], RawSearchTask] = {}
    assemblies: dict[tuple[Polymer, str], MsaAssemblyTask] = {}
    for chain in chains:
        if (
            isinstance(chain.chain_index, bool)
            or not isinstance(chain.chain_index, int)
            or chain.chain_index < 0
        ):
            raise ValueError("chain_index must be a non-negative integer")
        if chain.polymer not in {"protein", "rna"}:
            raise ValueError(f"Unsupported polymer: {chain.polymer!r}")
        if not isinstance(chain.unpaired_present, bool) or not isinstance(
            chain.paired_present, bool
        ):
            raise TypeError("field-presence flags must be booleans")
        query_spec = resolve_database_profile(
            "uniref90" if chain.polymer == "protein" else "rfam"
        )
        validate_query(query_spec, chain.sequence)
        include_unpaired = not chain.unpaired_present
        include_paired = chain.polymer == "protein" and not chain.paired_present
        if chain.polymer == "rna" and chain.paired_present:
            raise ValueError("RNA chains cannot declare a paired MSA field")
        if not include_unpaired and not include_paired:
            continue

        key = (chain.polymer, chain.sequence)
        existing = assemblies.get(key)
        assemblies[key] = MsaAssemblyTask(
            polymer=chain.polymer,
            sequence=chain.sequence,
            include_unpaired=include_unpaired
            or (existing.include_unpaired if existing else False),
            include_paired=include_paired
            or (existing.include_paired if existing else False),
        )
        database_ids: tuple[str, ...] = ()
        if include_unpaired:
            database_ids += (
                PROTEIN_UNPAIRED_DATABASES
                if chain.polymer == "protein"
                else RNA_UNPAIRED_DATABASES
            )
        if include_paired:
            database_ids += PROTEIN_PAIRED_DATABASES
        for database_id in database_ids:
            task = RawSearchTask(database_id=database_id, sequence=chain.sequence)
            if task.polymer != chain.polymer:
                raise RuntimeError("Database registry polymer mismatch")
            raw_searches.setdefault(
                (chain.polymer, chain.sequence, database_id),
                task,
            )
    return MsaResolutionPlan(
        raw_searches=tuple(raw_searches.values()),
        assemblies=tuple(assemblies.values()),
    )


def load_search_context(
    sharded_root: Path,
    cache_root: Path,
    database_id: str,
    sequence: str,
) -> SearchContext:
    """Validate a fixed manifest and derive its production search identity."""
    spec = resolve_database_profile(database_id)
    query = validate_query(spec, sequence)
    selected_profile_root = profile_root(sharded_root, spec)
    manifest_path = selected_profile_root / "manifest.json"
    require_regular_file(manifest_path)
    manifest = load_json_object(manifest_path)
    validate_profile_manifest(manifest, spec)
    manifest_sha256 = sha256_file(manifest_path)
    search_identity = production_search_identity(
        spec,
        query,
        manifest_sha256,
    )
    query_hash = sequence_hash(query)
    provenance: dict[str, object] = {
        "database_id": spec.database_id,
        "profile_id": spec.profile_id,
        "polymer": spec.polymer,
        "sequence_sha256": query_hash,
        "sequence_length": len(query),
        "search_identity": search_identity,
        "profile_manifest_sha256": manifest_sha256,
        "parameters": scientific_search_parameters(spec),
        "adapter_version": SEARCH_ADAPTER_VERSION,
        "alphafold_commit": ALPHAFOLD3_COMMIT,
        "hmmer_version": HMMER_VERSION,
        "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
    }
    return SearchContext(
        spec=spec,
        sequence=query,
        sequence_hash=query_hash,
        profile_root=selected_profile_root,
        manifest_sha256=manifest_sha256,
        search_identity=search_identity,
        provenance=provenance,
        result_root=cache_root / raw_result_relpath(spec, query, search_identity),
    )


def _load_artifact(
    root: Path,
    record: object,
    expected_path: str,
) -> bytes | None:
    if not isinstance(record, dict) or record.get("path") != expected_path:
        return None
    path = root / expected_path
    if not path.is_file():
        return None
    try:
        value = path.read_bytes()
    except OSError:
        return None
    if (
        isinstance(record.get("size_bytes"), bool)
        or record.get("size_bytes") != len(value)
        or record.get("sha256") != _sha256_bytes(value)
    ):
        return None
    return value


def load_raw_msa(context: SearchContext) -> RawMsaEntry | None:
    """Validate and load one marker-complete Raw Database MSA."""
    done_path = context.result_root / "done.json"
    if not done_path.is_file():
        return None
    try:
        done_bytes = done_path.read_bytes()
        done = orjson.loads(done_bytes)
    except (OSError, orjson.JSONDecodeError):
        return None
    if (
        not isinstance(done, dict)
        or done.get("schema_version") != RAW_RESULT_SCHEMA_VERSION
        or done.get("status") != "complete"
        or done.get("provenance") != context.provenance
    ):
        return None
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    result_bytes = _load_artifact(
        context.result_root,
        artifacts.get("result"),
        "result.a3m",
    )
    metrics_bytes = _load_artifact(
        context.result_root,
        artifacts.get("metrics"),
        "metrics.json",
    )
    log_bytes = _load_artifact(
        context.result_root,
        artifacts.get("log"),
        "run.log",
    )
    if result_bytes is None or metrics_bytes is None or log_bytes is None:
        return None
    try:
        a3m = result_bytes.decode()
    except UnicodeDecodeError:
        return None
    return RawMsaEntry(
        context=context,
        a3m=a3m,
        done_sha256=_sha256_bytes(done_bytes),
        result_record=cast(dict[str, object], artifacts["result"]),
    )


def inspect_raw_searches(
    sharded_root: Path,
    cache_root: Path,
    tasks: tuple[RawSearchTask, ...],
) -> list[dict[str, object]]:
    """Inspect fixed manifests and marker-valid cache entries without searching."""
    statuses: list[dict[str, object]] = []
    for task in tasks:
        context = load_search_context(
            sharded_root,
            cache_root,
            task.database_id,
            task.sequence,
        )
        entry = load_raw_msa(context)
        statuses.append({
            "status": "reused" if entry is not None else "missing",
            "database_id": context.spec.database_id,
            "profile_id": context.spec.profile_id,
            "polymer": context.spec.polymer,
            "sequence_sha256": context.sequence_hash,
            "search_identity": context.search_identity,
        })
    return statuses


def nhmmer_reported_score_sort_key(
    row: tuple[str, str, str, str],
) -> tuple[float, float, str]:
    """Rank an Nhmmer hit by printed E-value, score, then target ID."""
    _, _, tblout_line, name = row
    fields = tblout_line.split(maxsplit=15)
    if len(fields) < 14:
        raise ValueError(f"Invalid Nhmmer tblout row: {tblout_line!r}")
    return float(fields[12]), -float(fields[13]), name


def merge_nhmmer_results_by_reported_score(
    module: Any,
    results: tuple[Any, ...],
    max_sequences: int,
) -> Any:
    """Merge Nhmmer shards with the scientifically validated score ordering."""
    if max_sequences <= 1:
        raise ValueError("max_sequences must be greater than one")
    if not results:
        raise ValueError("At least one Nhmmer result is required")
    if len({result.target_sequence for result in results}) != 1:
        raise ValueError("Nhmmer shard results have different target sequences")
    if len({result.e_value for result in results}) != 1:
        raise ValueError("Nhmmer shard results have different E-value thresholds")

    tblout_by_id: dict[str, str] = {}
    for result in results:
        if result.tblout is None:
            raise ValueError("Nhmmer shard result is missing tblout")
        for line in result.tblout.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=15)
            if len(fields) < 14:
                raise ValueError(f"Invalid Nhmmer tblout row: {line!r}")
            hit_id = f"{fields[0]}/{fields[6]}-{fields[7]}"
            tblout_by_id[hit_id] = line

    def iter_shard_rows(a3m: str):
        records = iter(module.parsers.lazy_parse_fasta_string(a3m))
        next(records)
        for aligned_sequence, description in records:
            name = description.partition(" ")[0]
            if tblout_line := tblout_by_id.get(name):
                yield aligned_sequence, description, tblout_line, name

    top_rows = heapq.nsmallest(
        max_sequences - 1,
        itertools.chain.from_iterable(
            iter_shard_rows(result.a3m) for result in results
        ),
        key=nhmmer_reported_score_sort_key,
    )
    merged_a3m = [f">query\n{results[0].target_sequence}"]
    merged_a3m.extend(
        f">{description}\n{aligned_sequence}"
        for aligned_sequence, description, _, _ in top_rows
    )
    return module.msa_tool.MsaToolResult(
        target_sequence=results[0].target_sequence,
        a3m="\n".join(merged_a3m),
        e_value=results[0].e_value,
        tblout=None,
    )


def execute_profile_database_search(
    spec: DatabaseProfileSpec,
    sequence: str,
    *,
    selected_profile_root: Path,
    sharded_n_cpu: int = SEARCH_N_CPU,
    max_parallel_shards: int = SEARCH_MAX_PARALLEL_SHARDS,
) -> str:
    """Run one pinned search over every shard in an immutable profile."""
    from importlib import import_module

    query = validate_query(spec, sequence)
    database_path = (
        selected_profile_root / "shards" / spec.source_filename
    ).as_posix() + f"@{spec.shard_count}"
    search_paths = tuple(
        selected_profile_root / "shards" / name for name in shard_names(spec)
    )

    if spec.polymer == "protein":
        module = import_module("alphafold3.data.tools.jackhmmer")
        tool = module.Jackhmmer(
            binary_path=JACKHMMER_BINARY_PATH,
            database_path=database_path,
            n_cpu=sharded_n_cpu,
            n_iter=JACKHMMER_N_ITER,
            e_value=JACKHMMER_E_VALUE,
            z_value=spec.search_space_value,
            dom_z_value=spec.search_space_value,
            max_sequences=spec.max_sequences,
            filter_f1=JACKHMMER_FILTER_F1,
            filter_f2=JACKHMMER_FILTER_F2,
            filter_f3=JACKHMMER_FILTER_F3,
            max_threads=max_parallel_shards,
        )
    else:
        module = import_module("alphafold3.data.tools.nhmmer")
        tool = module.Nhmmer(
            binary_path=NHMMER_BINARY_PATH,
            hmmalign_binary_path=HMMALIGN_BINARY_PATH,
            hmmbuild_binary_path=HMMBUILD_BINARY_PATH,
            database_path=database_path,
            n_cpu=sharded_n_cpu,
            e_value=NHMMER_E_VALUE,
            z_value=spec.search_space_value,
            max_sequences=spec.max_sequences,
            filter_f3=NHMMER_FILTER_F3,
            alphabet="rna",
            max_threads=max_parallel_shards,
        )

    global_temp_dir = tempfile.mkdtemp(
        prefix=f"af3-{spec.database_id}-sharded-",
    )

    def query_one(search_path: Path) -> Any:
        return tool._query_db_shard(  # noqa: SLF001
            target_sequence=query,
            db_shard_path=str(search_path),
            get_tblout=True,
            global_temp_dir=global_temp_dir,
        )

    try:
        with ThreadPoolExecutor(max_workers=max_parallel_shards) as executor:
            results = tuple(executor.map(query_one, search_paths))
    finally:
        shutil.rmtree(global_temp_dir, ignore_errors=True)

    for search_path, result in zip(search_paths, results, strict=True):
        if result.tblout is None:
            raise ValueError(f"{search_path.name} search did not return tblout")
    if spec.polymer == "protein":
        merged = module._merge_jackhmmer_results(  # noqa: SLF001
            results,
            spec.max_sequences,
        )
    else:
        merged = merge_nhmmer_results_by_reported_score(
            module,
            results,
            spec.max_sequences,
        )
    return merged.a3m


def materialize_upstream_profile_msa(
    spec: DatabaseProfileSpec,
    sequence: str,
    raw_a3m: str,
) -> str:
    """Apply pinned ``get_msa`` query-row and empty-result behavior."""
    if not isinstance(raw_a3m, str):
        raise ValueError("Pinned MSA wrapper returned a non-string A3M")
    from importlib import import_module

    msa = import_module("alphafold3.data.msa")
    mmcif_names = import_module("alphafold3.constants.mmcif_names")
    chain_poly_type = (
        mmcif_names.PROTEIN_CHAIN
        if spec.polymer == "protein"
        else mmcif_names.RNA_CHAIN
    )
    return msa.Msa.from_a3m(
        query_sequence=sequence,
        chain_poly_type=chain_poly_type,
        a3m=raw_a3m,
        deduplicate=False,
    ).to_a3m()


def assert_pinned_msa_assembly_contract() -> dict[str, str]:
    """Bind the assembly adapter to the pinned upstream function bodies."""
    from importlib import import_module

    pipeline = import_module("alphafold3.data.pipeline")
    msa_module = import_module("alphafold3.data.msa")
    protein_source = inspect.getsource(
        pipeline._get_protein_msa_and_templates  # noqa: SLF001
    )
    rna_source = inspect.getsource(pipeline._get_rna_msa)  # noqa: SLF001
    compact_protein = re.sub(r"\s+", "", protein_source)
    compact_rna = re.sub(r"\s+", "", rna_source)
    required_protein = (
        "msas=[uniref90_msa,small_bfd_msa,mgnify_msa],deduplicate=True",
        "msas=[uniprot_msa],deduplicate=False",
    )
    required_rna = "msas=[rfam_msa,rnacentral_msa,nt_rna_msa],deduplicate=True"
    if not all(pattern in compact_protein for pattern in required_protein):
        raise RuntimeError("Pinned protein MSA assembly contract changed")
    if required_rna not in compact_rna:
        raise RuntimeError("Pinned RNA MSA assembly contract changed")
    get_msa_source = inspect.getsource(msa_module.get_msa)
    deduplicate_parameter = inspect.signature(msa_module.get_msa).parameters.get(
        "deduplicate"
    )
    if (
        deduplicate_parameter is None
        or deduplicate_parameter.default is not False
        or "deduplicate=deduplicate" not in re.sub(r"\s+", "", get_msa_source)
    ):
        raise RuntimeError("Pinned per-database MSA deduplication contract changed")
    return {
        "protein_function_sha256": _sha256_bytes(protein_source.encode()),
        "rna_function_sha256": _sha256_bytes(rna_source.encode()),
        "get_msa_function_sha256": _sha256_bytes(get_msa_source.encode()),
    }


def assemble_msa_fields(
    polymer: Polymer,
    database_a3ms: dict[str, str],
    *,
    include_unpaired: bool,
    include_paired: bool,
) -> dict[str, str]:
    """Assemble requested fields in exact pinned database order."""
    from importlib import import_module

    if not include_unpaired and not include_paired:
        raise ValueError("At least one MSA field must be requested")
    if polymer == "rna" and include_paired:
        raise ValueError("RNA does not have a paired MSA")
    msa = import_module("alphafold3.data.msa")
    mmcif_names = import_module("alphafold3.constants.mmcif_names")
    fields: dict[str, str] = {}
    if include_unpaired:
        database_ids = (
            PROTEIN_UNPAIRED_DATABASES
            if polymer == "protein"
            else RNA_UNPAIRED_DATABASES
        )
        missing = [name for name in database_ids if name not in database_a3ms]
        if missing:
            raise ValueError(f"Missing unpaired MSA databases: {missing}")
        chain_poly_type = (
            mmcif_names.PROTEIN_CHAIN if polymer == "protein" else mmcif_names.RNA_CHAIN
        )
        fields["unpairedMsa"] = msa.Msa.from_multiple_a3ms(
            a3ms=[database_a3ms[name] for name in database_ids],
            chain_poly_type=chain_poly_type,
            deduplicate=True,
        ).to_a3m()
    if include_paired:
        if "uniprot" not in database_a3ms:
            raise ValueError("Missing paired MSA database: uniprot")
        fields["pairedMsa"] = msa.Msa.from_multiple_a3ms(
            a3ms=[database_a3ms["uniprot"]],
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            deduplicate=False,
        ).to_a3m()
    return fields


def _artifact_record(path: Path, root: Path) -> dict[str, object]:
    require_regular_file(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _raw_claim_scope(context: SearchContext) -> str:
    polymer_dir = polymer_cache_dir(context.spec.polymer)
    return (
        f"raw:{polymer_dir}:{context.sequence_hash}:"
        f"{context.spec.database_id}:{context.search_identity}"
    )


def _wait_for_raw_claim(
    runtime: SearchRuntime,
    context: SearchContext,
) -> tuple[RawMsaEntry | None, GenerationClaim | None]:
    generation_id = uuid.uuid4().hex
    deadline = time.monotonic() + float(runtime.wait_timeout_seconds)
    while True:
        runtime.cache_volume.reload()
        if entry := load_raw_msa(context):
            return entry, None
        try:
            claim = acquire_generation_claim(
                runtime.claims,
                scope_key=_raw_claim_scope(context),
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
                    f"Timed out waiting for raw MSA owner {owner!r}: "
                    f"{context.spec.database_id} {context.sequence_hash}"
                ) from exc
            time.sleep(min(runtime.claim_poll_seconds, remaining))


def run_database_search(
    runtime: SearchRuntime,
    database_id: str,
    sequence: str,
) -> dict[str, object]:
    """Validate/reuse or publish one resumable Raw Database MSA."""
    runtime.sharded_volume.reload()
    runtime.cache_volume.reload()
    context = load_search_context(
        runtime.sharded_root,
        runtime.cache_root,
        database_id,
        sequence,
    )
    if entry := load_raw_msa(context):
        return entry.summary("reused")
    raced_entry, claim = _wait_for_raw_claim(runtime, context)
    if raced_entry is not None:
        return raced_entry.summary("reused")
    if claim is None:
        raise RuntimeError("Raw MSA claim election returned no owner")

    generation_root = (
        runtime.cache_root
        / sequence_cache_relpath(
            context.spec.polymer,
            context.sequence,
        )
        / ".staging"
        / "raw-msa"
        / context.spec.database_id
        / context.search_identity
        / claim.generation_id
    )
    log_path = generation_root / "run.log"
    terminal_status = "failed"
    terminal_detail: dict[str, object] = {}
    started_at = utc_now()
    search_started = time.perf_counter()
    try:
        runtime.cache_volume.reload()
        if entry := load_raw_msa(context):
            terminal_status = "complete"
            terminal_detail = {
                "publication": "raced",
                "done_sha256": entry.done_sha256,
            }
            return entry.summary("reused")
        append_log(
            log_path,
            f"Searching {context.spec.profile_id} for a "
            f"{len(context.sequence)}-residue query",
        )
        raw_a3m = execute_profile_database_search(
            context.spec,
            context.sequence,
            selected_profile_root=context.profile_root,
        )
        a3m = materialize_upstream_profile_msa(
            context.spec,
            context.sequence,
            raw_a3m,
        )
        elapsed_seconds = time.perf_counter() - search_started
        append_log(
            log_path,
            f"Completed search in {elapsed_seconds:.3f} seconds",
        )
        result_path = generation_root / "result.a3m"
        _write_bytes_atomic(result_path, a3m.encode())
        metrics = {
            "schema_version": RAW_RESULT_SCHEMA_VERSION,
            "status": "published",
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "provenance": context.provenance,
            "operational_parameters": operational_search_parameters(context.spec),
            "container": {
                "container_id": runtime.container_id,
                "function_call_id": runtime.function_call_id,
            },
        }
        metrics_path = generation_root / "metrics.json"
        write_json_atomic(metrics_path, metrics)
        artifacts = {
            "result": _artifact_record(result_path, generation_root),
            "metrics": _artifact_record(metrics_path, generation_root),
            "log": _artifact_record(log_path, generation_root),
        }
        runtime.cache_volume.commit()
        assert_generation_current(runtime.claims, claim)

        context.result_root.mkdir(parents=True, exist_ok=True)
        for filename in ("result.a3m", "metrics.json", "run.log"):
            os.replace(generation_root / filename, context.result_root / filename)
        runtime.cache_volume.commit()
        done = {
            "schema_version": RAW_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at": utc_now(),
            "generation_id": claim.generation_id,
            "provenance": context.provenance,
            "artifacts": artifacts,
        }
        write_json_atomic(context.result_root / "done.json", done)
        runtime.cache_volume.commit()
        entry = load_raw_msa(context)
        if entry is None:
            raise RuntimeError("Published Raw Database MSA failed validation")
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
                "database_id": context.spec.database_id,
                "profile_id": context.spec.profile_id,
                "search_identity": context.search_identity,
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


def _required_database_ids(task: MsaAssemblyTask) -> tuple[str, ...]:
    database_ids: tuple[str, ...] = ()
    if task.include_unpaired:
        database_ids += (
            PROTEIN_UNPAIRED_DATABASES
            if task.polymer == "protein"
            else RNA_UNPAIRED_DATABASES
        )
    if task.include_paired:
        database_ids += PROTEIN_PAIRED_DATABASES
    return database_ids


def _combined_provenance(
    task: MsaAssemblyTask,
    entries: dict[str, RawMsaEntry],
    assembly_contract: dict[str, str],
) -> dict[str, object]:
    dependencies = {
        database_id: {
            "search_identity": entry.context.search_identity,
            "raw_done_sha256": entry.done_sha256,
            "result_sha256": entry.result_record["sha256"],
        }
        for database_id, entry in entries.items()
    }
    identity_view = {
        "schema_version": COMBINED_RESULT_SCHEMA_VERSION,
        "polymer": task.polymer,
        "sequence_sha256": sequence_hash(task.sequence),
        "fields": {
            "unpairedMsa": task.include_unpaired,
            "pairedMsa": task.include_paired,
        },
        "dependencies": dependencies,
        "assembly_contract": assembly_contract,
    }
    return identity_view | {
        "combined_identity": _sha256_bytes(_json_bytes(identity_view))
    }


def _load_combined_msa(
    sequence_root: Path,
    provenance: dict[str, object],
    task: MsaAssemblyTask,
) -> dict[str, str] | None:
    done_path = sequence_root / "combined.done.json"
    if not done_path.is_file():
        return None
    try:
        done = orjson.loads(done_path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    if (
        not isinstance(done, dict)
        or done.get("schema_version") != COMBINED_RESULT_SCHEMA_VERSION
        or done.get("status") != "complete"
        or done.get("provenance") != provenance
    ):
        return None
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    fields: dict[str, str] = {}
    expected = []
    if task.include_unpaired:
        expected.append(("unpairedMsa", "unpaired.a3m"))
    if task.include_paired:
        expected.append(("pairedMsa", "paired.a3m"))
    for field, filename in expected:
        value = _load_artifact(sequence_root, artifacts.get(field), filename)
        if value is None:
            return None
        try:
            fields[field] = value.decode()
        except UnicodeDecodeError:
            return None
    return fields


def _combined_claim_scope(task: MsaAssemblyTask) -> str:
    return f"combined:{polymer_cache_dir(task.polymer)}:{sequence_hash(task.sequence)}"


def assemble_and_publish_msas(
    runtime: SearchRuntime,
    task: MsaAssemblyTask,
) -> dict[str, object]:
    """Assemble requested fields and publish complete canonical combinations."""
    runtime.sharded_volume.reload()
    runtime.cache_volume.reload()
    entries: dict[str, RawMsaEntry] = {}
    for database_id in _required_database_ids(task):
        context = load_search_context(
            runtime.sharded_root,
            runtime.cache_root,
            database_id,
            task.sequence,
        )
        entry = load_raw_msa(context)
        if entry is None:
            raise RuntimeError(
                "Required Raw Database MSA is incomplete: "
                f"{database_id} {context.sequence_hash}"
            )
        entries[database_id] = entry
    assembly_contract = assert_pinned_msa_assembly_contract()
    fields = assemble_msa_fields(
        task.polymer,
        {database_id: entry.a3m for database_id, entry in entries.items()},
        include_unpaired=task.include_unpaired,
        include_paired=task.include_paired,
    )

    complete_canonical = task.include_unpaired and (
        task.polymer == "rna" or task.include_paired
    )
    if not complete_canonical:
        return {
            "status": "request-local",
            "polymer": task.polymer,
            "sequence_sha256": sequence_hash(task.sequence),
            "fields": fields,
        }

    sequence_root = runtime.cache_root / sequence_cache_relpath(
        task.polymer,
        task.sequence,
    )
    provenance = _combined_provenance(task, entries, assembly_contract)
    if reusable := _load_combined_msa(sequence_root, provenance, task):
        return {
            "status": "reused",
            "polymer": task.polymer,
            "sequence_sha256": sequence_hash(task.sequence),
            "combined_identity": provenance["combined_identity"],
            "fields": reusable,
        }

    generation_id = uuid.uuid4().hex
    deadline = time.monotonic() + float(runtime.wait_timeout_seconds)
    claim: GenerationClaim | None = None
    while claim is None:
        runtime.cache_volume.reload()
        if reusable := _load_combined_msa(sequence_root, provenance, task):
            return {
                "status": "reused",
                "polymer": task.polymer,
                "sequence_sha256": sequence_hash(task.sequence),
                "combined_identity": provenance["combined_identity"],
                "fields": reusable,
            }
        try:
            claim = acquire_generation_claim(
                runtime.claims,
                scope_key=_combined_claim_scope(task),
                generation_id=generation_id,
                identity=provenance,
                container_id=runtime.container_id,
                maximum_age_seconds=runtime.maximum_age_seconds,
            )
        except ActiveGenerationError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                owner = exc.owner["generation_id"]
                raise TimeoutError(
                    f"Timed out waiting for combined MSA owner {owner!r}: "
                    f"{task.polymer} {sequence_hash(task.sequence)}"
                ) from exc
            time.sleep(min(runtime.claim_poll_seconds, remaining))

    generation_root = (
        sequence_root
        / ".staging"
        / "combined"
        / cast(str, provenance["combined_identity"])
        / claim.generation_id
    )
    terminal_status = "failed"
    terminal_detail: dict[str, object] = {}
    try:
        runtime.cache_volume.reload()
        if reusable := _load_combined_msa(sequence_root, provenance, task):
            terminal_status = "complete"
            terminal_detail = {"publication": "raced"}
            return {
                "status": "reused",
                "polymer": task.polymer,
                "sequence_sha256": sequence_hash(task.sequence),
                "combined_identity": provenance["combined_identity"],
                "fields": reusable,
            }
        filenames = {
            "unpairedMsa": "unpaired.a3m",
            "pairedMsa": "paired.a3m",
        }
        for field, value in fields.items():
            _write_bytes_atomic(generation_root / filenames[field], value.encode())
        artifacts = {
            field: _artifact_record(generation_root / filenames[field], generation_root)
            for field in fields
        }
        runtime.cache_volume.commit()
        assert_generation_current(runtime.claims, claim)
        sequence_root.mkdir(parents=True, exist_ok=True)
        for field in fields:
            filename = filenames[field]
            os.replace(generation_root / filename, sequence_root / filename)
        runtime.cache_volume.commit()
        write_json_atomic(
            sequence_root / "combined.done.json",
            {
                "schema_version": COMBINED_RESULT_SCHEMA_VERSION,
                "status": "complete",
                "completed_at": utc_now(),
                "generation_id": claim.generation_id,
                "provenance": provenance,
                "artifacts": artifacts,
            },
        )
        runtime.cache_volume.commit()
        reusable = _load_combined_msa(sequence_root, provenance, task)
        if reusable is None:
            raise RuntimeError("Published combined MSA failed validation")
        shutil.rmtree(generation_root, ignore_errors=True)
        runtime.cache_volume.commit()
        terminal_status = "complete"
        terminal_detail = {
            "publication": "published",
            "combined_identity": provenance["combined_identity"],
        }
        return {
            "status": "published",
            "polymer": task.polymer,
            "sequence_sha256": sequence_hash(task.sequence),
            "combined_identity": provenance["combined_identity"],
            "fields": reusable,
        }
    except Exception as exc:
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
