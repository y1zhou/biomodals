"""Production contracts for AlphaFold3 preparation, search, and inference."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import orjson
import pytest
from uniaf3.schema.alphafold3 import (
    AF3Config,
    AF3Protein,
    AF3SequenceEntry,
    AF3Template,
)

from biomodals.app.fold.alphafold3 import template_search
from biomodals.app.fold.alphafold3.artifacts import (
    artifact_record,
    json_bytes,
    load_artifact_bytes,
    validate_artifact_record,
    write_bytes_atomic,
)
from biomodals.app.fold.alphafold3.generation_claims import (
    ActiveGenerationError,
    GenerationClaim,
    abandon_generation_claim,
    acquire_generation_claim,
    finish_generation_claim,
    generation_status,
    latest_generation_owner,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    LocalTemplateFile,
    PreparedInferenceRun,
    VolumeUpload,
    hash_sequences,
    load_staged_inference_input,
    prepare_inference_run,
    sanitize_af3_name,
    serialize_af3_input,
)
from biomodals.app.fold.alphafold3.inference_pipeline import (
    InferenceBatchOutcome,
    coordinate_seed_predictions,
)
from biomodals.app.fold.alphafold3.input_enrichment import (
    apply_msa_resolution,
    chain_msa_states,
    plan_template_searches,
    reduce_msa_assembly_results,
)
from biomodals.app.fold.alphafold3.msa_search import (
    RAW_RESULT_SCHEMA_VERSION,
    ChainMsaState,
    MsaAssemblyTask,
    RawMsaEntry,
    RawSearchTask,
    SearchContext,
    load_raw_msa,
    merge_nhmmer_results_by_reported_score,
    plan_msa_resolution,
    scientific_search_parameters,
    sequence_hash,
)
from biomodals.app.fold.alphafold3.profile_builder import (
    ShardBuildEvidence,
    SourceProfileEvidence,
    build_profile_manifest,
    plan_missing_profile_builds,
)
from biomodals.app.fold.alphafold3.profile_manifest import validate_profile_manifest
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
    COMPOSABLE_LEGACY_VALIDATION_RELPATHS,
    COMPOSABLE_MULTISET_RECIPE_VERSION,
    DATABASE_PROFILE_SPECS,
    DEFAULT_SEQKIT_THREADS,
    HMMER_VERSION,
    JACKHMMER_PATCH_SHA256,
    ORDINAL_SHUFFLER_PREFETCH_BYTES,
    ORDINAL_SHUFFLER_PREFETCH_RECORDS,
    ORDINAL_SHUFFLER_SOURCE_SHA256,
    ORDINAL_SHUFFLER_VERSION,
    PROFILE_SCHEMA_VERSION,
    SEQKIT_VERSION,
    SHARD_RANDOM_SEED,
    SOURCE_DB_VOLUME_NAME,
    VALIDATION_RELPATHS,
    record_multiset_identity,
    resolve_database_profile,
    shard_names,
)
from biomodals.app.fold.alphafold3.request_results import (
    REQUEST_MANIFEST_SCHEMA_VERSION,
    create_request_archive,
    request_view_id,
)
from biomodals.app.fold.alphafold3.search_pipeline import (
    resolve_msa_and_templates,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    SEED_MARKER_SCHEMA_VERSION,
    ClaimedSeed,
    SeedClaimPlan,
    canonical_output_name,
    load_seed_marker,
)
from biomodals.app.fold.alphafold3.template_search import (
    TEMPLATE_RESULT_SCHEMA_VERSION,
    TemplateTask,
    build_template_context,
    load_template_entry,
)
from biomodals.helper.shell import run_command


def _artifact(path: str, content: bytes = b"x") -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _request_manifest(
    *,
    run_id: str,
    submitted_seeds: list[int],
    display_name: str,
    artifacts: list[dict[str, object]],
    sample_count: int = 1,
) -> dict[str, object]:
    normalized_seeds = sorted(set(submitted_seeds))
    request_id = hash_sequences(run_id, normalized_seeds)
    view_id = request_view_id(request_id, tuple(submitted_seeds), display_name)
    canonical_name = canonical_output_name(run_id)
    presentation_name = sanitize_af3_name(display_name)
    return {
        "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "run_id": run_id,
        "request_id": request_id,
        "view_id": view_id,
        "canonical_name": canonical_name,
        "sample_count": sample_count,
        "submitted_display_name": display_name,
        "presentation_name": presentation_name,
        "name_mapping": {
            "canonical": canonical_name,
            "presentation": presentation_name,
        },
        "submitted_seeds": submitted_seeds,
        "normalized_seeds": normalized_seeds,
        "duplicates_removed": [
            seed
            for index, seed in enumerate(submitted_seeds)
            if seed in submitted_seeds[:index]
        ],
        "artifacts": artifacts,
        "manifest_volume_path": (
            f"{run_id[:2]}/{run_id}/requests/{request_id}/views/{view_id}/manifest.json"
        ),
    }


def _materialize_prepared_run(
    output_root: Path,
    prepared: PreparedInferenceRun,
) -> None:
    output_root.mkdir()
    for upload in (*prepared.payload_uploads, prepared.staged_input):
        path = output_root / Path(upload.relative_path.as_posix())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(upload.content)


def test_durable_artifact_helpers_detect_changed_bytes(tmp_path: Path) -> None:
    payload_path = tmp_path / "nested" / "payload.txt"
    write_bytes_atomic(payload_path, b"expected")
    record = artifact_record(payload_path, tmp_path)

    assert load_artifact_bytes(tmp_path, record, "nested/payload.txt") == b"expected"
    assert validate_artifact_record(tmp_path, record) == record

    payload_path.write_bytes(b"changed!")
    assert load_artifact_bytes(tmp_path, record, "nested/payload.txt") is None
    assert validate_artifact_record(tmp_path, record) is None


def _profile_manifest(database_id: str) -> dict[str, Any]:
    spec = resolve_database_profile(database_id)
    num_seqs = spec.expected_num_seqs or 1
    sum_len = spec.expected_sum_len or 1
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": spec.profile_id,
        "database_id": spec.database_id,
        "polymer": spec.polymer,
        "shard_count": spec.shard_count,
        "shard_prefix": f"shards/{spec.source_filename}",
        "search_space_value": spec.search_space_value,
        "search_space_unit": spec.search_space_unit,
        "source": {
            "volume": SOURCE_DB_VOLUME_NAME,
            "path": spec.source_filename,
            "size_bytes": 1,
            "sha256": "a" * 64,
            "num_seqs": num_seqs,
            "sum_len": sum_len,
        },
        "shards": [_artifact(f"shards/{name}") for name in shard_names(spec)],
        "compatibility": {
            "alphafold_repository": ALPHAFOLD3_REPOSITORY,
            "alphafold_commit": ALPHAFOLD3_COMMIT,
            "hmmer_version": HMMER_VERSION,
            "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
        },
        "recipe": {
            "version": COMPOSABLE_MULTISET_RECIPE_VERSION,
            "seqkit_version": SEQKIT_VERSION,
            "seqkit_threads": DEFAULT_SEQKIT_THREADS,
            "random_seed": SHARD_RANDOM_SEED,
            "shuffle": [
                "two-pass",
                "first-pass-stage-local-source",
                "source-occurrence-offset-index",
                "splitmix64-fisher-yates-u32",
                "bounded-concurrent-local-pread",
                "ordered-write",
            ],
            "shuffler": {
                "version": ORDINAL_SHUFFLER_VERSION,
                "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
                "record_identity": "source-occurrence",
                "offset_index": "uint64-source-occurrence-offsets-v1",
                "permutation": "splitmix64-fisher-yates-u32-v1",
                "staging": "first-pass-tee-to-container-local-v1",
                "read": "bounded-concurrent-local-pread-ordered-write-v2",
                "ordered_output": True,
            },
            "execution": {
                "worker_threads": DEFAULT_SEQKIT_THREADS,
                "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
                "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
            },
            "duplicate_recovery": {
                "warning_source": None,
                "record_identity": "source-occurrence",
                "append_after_shuffle": False,
                "strip_after_split": False,
            },
            "record_multiset": record_multiset_identity()
            | {"shard_threads": DEFAULT_SEQKIT_THREADS},
            "split": ["--by-part", spec.shard_count],
        },
        "validation": {
            "passed": True,
            "temporary_recovery_prefix_absent": True,
            "num_seqs": num_seqs,
            "sum_len": sum_len,
            "record_occurrences_preserved": True,
            "recovered_records": 0,
            "recovered_residues": 0,
            "first_recovered_byte_offset": None,
            "last_recovered_byte_offset": None,
            "canonical_record_multiset_match": True,
            "record_multiset_signature_sha256": "b" * 64,
            "artifacts": [_artifact(path) for path in VALIDATION_RELPATHS],
        },
    }


class FakeClaimStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def put(
        self,
        key: str,
        value: object,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)


class FakeVolumeReader:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def read_file(self, path: str):
        value = self.files[path]
        midpoint = len(value) // 2
        yield value[:midpoint]
        yield value[midpoint:]


def test_profile_manifest_and_missing_build_plan_are_fixed() -> None:
    manifest = _profile_manifest("small_bfd")
    source, shards, validation = validate_profile_manifest(
        manifest,
        resolve_database_profile("small_bfd"),
    )

    assert source["num_seqs"] == 65_984_053
    assert len(shards) == 64
    assert [record["path"] for record in validation] == list(VALIDATION_RELPATHS)

    inventory: dict[str, object] = {
        "invalid_profiles": {},
        "missing_database_ids": ["uniref90", "small_bfd"],
    }
    assert plan_missing_profile_builds(
        inventory,
        seqkit_threads=4,
        source_policy="compress",
    ) == (
        ("small_bfd", 4, "compress"),
        ("uniref90", 4, "compress"),
    )

    manifest["shards"][0], manifest["shards"][1] = (
        manifest["shards"][1],
        manifest["shards"][0],
    )
    with pytest.raises(ValueError, match="shard order"):
        validate_profile_manifest(manifest, resolve_database_profile("small_bfd"))


def test_composable_profile_accepts_legacy_timing_artifact() -> None:
    assert "validation/shuffle-stderr.log" not in VALIDATION_RELPATHS
    assert "validation/shuffle-stderr.log" in COMPOSABLE_LEGACY_VALIDATION_RELPATHS
    manifest = _profile_manifest("small_bfd")
    validation = cast(dict[str, object], manifest["validation"])
    validation["artifacts"] = [
        _artifact(path) for path in COMPOSABLE_LEGACY_VALIDATION_RELPATHS
    ]

    _, _, artifacts = validate_profile_manifest(
        manifest,
        resolve_database_profile("small_bfd"),
    )

    assert [record["path"] for record in artifacts] == list(
        COMPOSABLE_LEGACY_VALIDATION_RELPATHS
    )


def test_profile_manifest_builder_preserves_the_fixed_recipe(tmp_path: Path) -> None:
    spec = resolve_database_profile("small_bfd")
    staging_root = tmp_path / "profile"
    for name in shard_names(spec):
        path = staging_root / "shards" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"shard")
    for relative in VALIDATION_RELPATHS:
        path = staging_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"validation")
    source = SourceProfileEvidence(
        size_bytes=1,
        sha256="a" * 64,
        num_seqs=spec.expected_num_seqs or 1,
        stats_path=staging_root / "validation" / "source-stats.tsv",
    )
    shards = ShardBuildEvidence(
        shard_paths=tuple(staging_root / "shards" / name for name in shard_names(spec)),
        statistics={
            "num_seqs": spec.expected_num_seqs or 1,
            "sum_len": spec.expected_sum_len or 1,
            "maximum_residue_imbalance": 0.01,
        },
        recovery_metrics={
            "recovered_records": 0,
            "recovered_residues": 0,
            "first_byte_offset": None,
            "last_byte_offset": None,
            "temporary_namespace": None,
        },
        record_multiset_signature_sha256="b" * 64,
    )

    manifest = build_profile_manifest(
        spec,
        "generation",
        source,
        shards,
        staging_root,
        seqkit_threads=8,
    )

    validate_profile_manifest(manifest, spec)
    recipe = cast(dict[str, object], manifest["recipe"])
    validation = cast(dict[str, object], manifest["validation"])
    assert recipe == _profile_manifest("small_bfd")["recipe"]
    assert recipe["version"] == COMPOSABLE_MULTISET_RECIPE_VERSION
    assert validation["canonical_record_multiset_match"] is True
    artifacts = cast(list[dict[str, object]], validation["artifacts"])
    assert str(artifacts[-1]["path"]).endswith("shuffler-evidence.json")


def test_msa_resolution_deduplicates_queries_across_input_chains() -> None:
    sequence = "ACDEFG"
    plan = plan_msa_resolution((
        ChainMsaState(
            chain_index=0,
            polymer="protein",
            sequence=sequence,
            unpaired_present=False,
            paired_present=False,
        ),
        ChainMsaState(
            chain_index=1,
            polymer="protein",
            sequence=sequence,
            unpaired_present=True,
            paired_present=False,
        ),
        ChainMsaState(
            chain_index=2,
            polymer="rna",
            sequence="ACGU",
            unpaired_present=False,
            paired_present=False,
        ),
    ))

    assert [(task.database_id, task.sequence) for task in plan.raw_searches] == [
        ("uniref90", sequence),
        ("small_bfd", sequence),
        ("mgnify", sequence),
        ("uniprot", sequence),
        ("rfam", "ACGU"),
        ("rnacentral", "ACGU"),
        ("ntrna", "ACGU"),
    ]
    assert [
        (
            task.polymer,
            task.sequence,
            task.include_unpaired,
            task.include_paired,
        )
        for task in plan.assemblies
    ] == [
        ("protein", sequence, True, True),
        ("rna", "ACGU", True, False),
    ]


def test_input_enrichment_reuses_one_result_across_identical_chains() -> None:
    config = AF3Config(
        name="homodimer",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
            AF3SequenceEntry(protein=AF3Protein(id="B", sequence="ACDE")),
        ],
    )
    states = chain_msa_states(config)
    assembly_task = plan_msa_resolution(states).assemblies[0]
    resolution = reduce_msa_assembly_results(
        (assembly_task,),
        (
            {
                "status": "published",
                "polymer": "protein",
                "sequence_sha256": sequence_hash("ACDE"),
                "combined_identity": "c" * 64,
                "fields": {
                    "unpairedMsa": ">query\nACDE\n",
                    "pairedMsa": ">query\nACDE\n",
                },
            },
        ),
    )

    apply_msa_resolution(
        config,
        states,
        resolution,
        search_protein_templates=True,
    )
    template_plan = plan_template_searches(
        config, states, resolution.canonical_sequences
    )

    assert len(template_plan.tasks) == 1
    assert template_plan.tasks[0].publish_canonical is True
    assert template_plan.chain_indices_by_identity == {
        template_plan.tasks[0].template_identity: (0, 1)
    }


def test_search_pipeline_coordinates_cache_assembly_and_templates() -> None:
    """The deep search seam should hide remote topology from its caller."""

    class FakeSearchExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def inspect_raw(
            self,
            tasks: tuple[RawSearchTask, ...],
        ) -> tuple[dict[str, object], ...]:
            self.calls.append("inspect-raw")
            return tuple(
                {
                    "status": "reused",
                    "database_id": task.database_id,
                    "sequence_sha256": task.sequence_hash,
                }
                for task in tasks
            )

        def run_raw(
            self,
            tasks: tuple[RawSearchTask, ...],
            *,
            max_parallel: int,
        ) -> tuple[dict[str, object] | Exception, ...]:
            del tasks, max_parallel
            self.calls.append("run-raw")
            return ()

        def run_assemblies(
            self,
            tasks: tuple[MsaAssemblyTask, ...],
            *,
            max_parallel: int,
        ) -> tuple[dict[str, object] | Exception, ...]:
            del max_parallel
            self.calls.append("assemble")
            return tuple(
                {
                    "status": "published",
                    "polymer": task.polymer,
                    "sequence_sha256": sequence_hash(task.sequence),
                    "combined_identity": "a" * 64,
                    "fields": {
                        "unpairedMsa": ">query\nACDE\n",
                        "pairedMsa": ">query\nACDE\n",
                    },
                }
                for task in tasks
            )

        def inspect_templates(
            self,
            tasks: tuple[TemplateTask, ...],
        ) -> tuple[dict[str, object], ...]:
            self.calls.append("inspect-templates")
            return tuple(
                {
                    "status": "missing",
                    "sequence_sha256": sequence_hash(task.sequence),
                    "unpaired_msa_sha256": task.unpaired_msa_sha256,
                    "template_identity": task.template_identity,
                }
                for task in tasks
            )

        def run_templates(
            self,
            tasks: tuple[TemplateTask, ...],
            *,
            max_parallel: int,
        ) -> tuple[dict[str, object] | Exception, ...]:
            del max_parallel
            self.calls.append("search-templates")
            return tuple(
                {
                    "status": "published",
                    "sequence_sha256": sequence_hash(task.sequence),
                    "unpaired_msa_sha256": task.unpaired_msa_sha256,
                    "template_identity": task.template_identity,
                    "templates": [],
                }
                for task in tasks
            )

    config = AF3Config(
        name="coordinated-search",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                )
            )
        ],
    )
    executor = FakeSearchExecutor()

    resolved = resolve_msa_and_templates(
        config,
        executor,
        max_parallel_search_workers=2,
    )

    protein = resolved.sequences[0].protein
    assert protein is not None
    assert protein.unpairedMsa == ">query\nACDE\n"
    assert protein.pairedMsa == ">query\nACDE\n"
    assert protein.templates == []
    assert executor.calls == [
        "inspect-raw",
        "assemble",
        "inspect-templates",
        "search-templates",
    ]


def test_rna_shards_merge_by_reported_score_with_deterministic_ties() -> None:
    assert (
        scientific_search_parameters(resolve_database_profile("rfam"))[
            "sharded_merge_order"
        ]
        == "reported-evalue-descending-bit-score-name-v1"
    )

    @dataclass
    class Result:
        target_sequence: str
        a3m: str
        e_value: float
        tblout: str | None

    def lazy_parse_fasta_string(value: str):
        header: str | None = None
        sequence_parts: list[str] = []
        for line in value.splitlines():
            if line.startswith(">"):
                if header is not None:
                    yield "".join(sequence_parts), header
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        if header is not None:
            yield "".join(sequence_parts), header

    def tblout(name: str, score: float, e_value: str = "1e-5") -> str:
        return f"{name} x x x x x 1 4 x x x x {e_value} {score} x x"

    module = SimpleNamespace(
        parsers=SimpleNamespace(lazy_parse_fasta_string=lazy_parse_fasta_string),
        msa_tool=SimpleNamespace(MsaToolResult=Result),
    )
    results = (
        Result(
            target_sequence="ACGU",
            a3m=">query\nACGU\n>hitB/1-4 second\nACGU\n",
            e_value=1e-3,
            tblout=tblout("hitB", 50.0),
        ),
        Result(
            target_sequence="ACGU",
            a3m=(">query\nACGU\n>hitC/1-4 third\nACGU\n>hitA/1-4 first\nACGU\n"),
            e_value=1e-3,
            tblout="\n".join((
                tblout("hitC", 10.0, "1e-2"),
                tblout("hitA", 50.0),
            )),
        ),
    )

    merged = merge_nhmmer_results_by_reported_score(
        module,
        results,
        max_sequences=3,
    )

    assert merged.a3m.splitlines() == [
        ">query",
        "ACGU",
        ">hitA/1-4 first",
        "ACGU",
        ">hitB/1-4 second",
        "ACGU",
    ]


def test_raw_msa_cache_requires_a_valid_completion_marker(tmp_path: Path) -> None:
    result_root = tmp_path / "raw"
    result_root.mkdir()
    spec = resolve_database_profile("small_bfd")
    provenance: dict[str, object] = {"identity": "fixture"}
    context = SearchContext(
        spec=spec,
        sequence="ACDE",
        sequence_hash=sequence_hash("ACDE"),
        profile_root=tmp_path / "profile",
        search_identity="b" * 64,
        provenance=provenance,
        result_root=result_root,
    )
    files = {"result": ("result.a3m", b">query\nACDE\n")}
    artifacts = {}
    for role, (name, content) in files.items():
        (result_root / name).write_bytes(content)
        artifacts[role] = _artifact(name, content)
    (result_root / "done.json").write_bytes(
        orjson.dumps({
            "schema_version": RAW_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "provenance": provenance,
            "artifacts": artifacts,
        })
    )

    entry = load_raw_msa(context)
    assert isinstance(entry, RawMsaEntry)
    assert entry.a3m == ">query\nACDE\n"

    (result_root / "result.a3m").write_bytes(b">query\nCHANGED\n")
    assert load_raw_msa(context) is None


def test_template_cache_rejects_changed_template_bytes(tmp_path: Path) -> None:
    context = build_template_context(
        tmp_path,
        "ACDE",
        "a" * 64,
        "2021-09-30",
    )
    context.sequence_root.mkdir(parents=True)
    templates = [
        {
            "mmcif": "data_template\n#\n",
            "queryIndices": [0],
            "templateIndices": [0],
        }
    ]
    template_bytes = orjson.dumps(templates)
    (context.sequence_root / "templates.json").write_bytes(template_bytes)
    (context.sequence_root / "templates.done.json").write_bytes(
        orjson.dumps({
            "schema_version": TEMPLATE_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "provenance": context.provenance,
            "templates": _artifact("templates.json", template_bytes),
        })
    )

    entry = load_template_entry(context)
    assert entry is not None
    assert entry.templates == templates

    (context.sequence_root / "templates.json").write_bytes(b"[]")
    assert load_template_entry(context) is None


def test_template_search_validates_remote_a3m_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = ">query\nACDE\n>hit\nACd-E\n"

    assert template_search._validate_template_msa("ACDE", valid) == valid
    with pytest.raises(ValueError, match="valid FASTA/A3M"):
        template_search._validate_template_msa("ACDE", "ACDE\n")
    with pytest.raises(ValueError, match="query row"):
        template_search._validate_template_msa("ACDE", ">query\nAAAA\n")
    with pytest.raises(ValueError, match="alignment width"):
        template_search._validate_template_msa(
            "ACDE",
            ">query\nACDE\n>hit\nACD\n",
        )

    monkeypatch.setattr(template_search, "MAX_LOCAL_MSA_BYTES", len(valid) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        template_search._validate_template_msa("ACDE", valid)


def test_generation_claims_fence_active_and_terminal_writers() -> None:
    store = FakeClaimStore()
    first = acquire_generation_claim(
        store,
        scope_key="raw:Protein:sequence:small_bfd",
        generation_id="first",
        identity={"search": "one"},
        container_id="container-a",
        maximum_age_seconds=100,
        now_epoch_seconds=1_000,
        now_text="first-start",
    )
    with pytest.raises(ActiveGenerationError):
        acquire_generation_claim(
            store,
            scope_key=first.scope_key,
            generation_id="second",
            identity={"search": "one"},
            container_id="container-b",
            maximum_age_seconds=100,
            now_epoch_seconds=1_001,
            now_text="second-start",
        )

    finish_generation_claim(
        store,
        first,
        status="complete",
        detail={"publication": "published"},
        now_text="first-finish",
    )
    second = acquire_generation_claim(
        store,
        scope_key=first.scope_key,
        generation_id="second",
        identity={"search": "one"},
        container_id="container-b",
        maximum_age_seconds=100,
        now_epoch_seconds=1_002,
        now_text="second-start",
    )

    assert latest_generation_owner(store, first.scope_key) == second.owner
    assert second.owner["predecessor_status"] == "complete"
    abandon_generation_claim(
        store,
        second,
        detail={"cleanup_recovery": True},
        now_text="second-abandoned",
    )
    assert generation_status(store, second.scope_key, second.generation_id) == {
        "status": "abandoned",
        "finished_at": "second-abandoned",
        "cleanup_recovery": True,
    }


def test_generation_claims_adapt_legacy_owners() -> None:
    """A stage may preserve an append-only chain created before canonical owners."""
    store = FakeClaimStore()
    scope_key = "small-bfd-64-v2"
    store.put(
        f"claim:{scope_key}:root",
        {
            "profile_id": scope_key,
            "database_id": "small_bfd",
            "generation_id": "legacy",
            "container_id": "old-container",
            "started_at": "legacy-start",
            "started_at_epoch_seconds": 1_000,
            "maximum_age_seconds": 100,
        },
    )

    def adapt_profile_owner(
        selected_scope: str,
        value: object,
    ) -> dict[str, object]:
        assert isinstance(value, dict)
        legacy = cast(dict[str, object], value)
        return {
            "scope_key": selected_scope,
            "generation_id": legacy["generation_id"],
            "identity": {
                "profile_id": legacy["profile_id"],
                "database_id": legacy["database_id"],
            },
            "container_id": legacy["container_id"],
            "started_at": legacy["started_at"],
            "started_at_epoch_seconds": legacy["started_at_epoch_seconds"],
            "maximum_age_seconds": legacy["maximum_age_seconds"],
        }

    successor = acquire_generation_claim(
        store,
        scope_key=scope_key,
        generation_id="canonical",
        identity={"profile_id": scope_key, "database_id": "small_bfd"},
        container_id="new-container",
        maximum_age_seconds=100,
        now_epoch_seconds=1_101,
        now_text="canonical-start",
        owner_adapter=adapt_profile_owner,
    )

    assert successor.owner["predecessor_generation_id"] == "legacy"
    assert successor.owner["predecessor_status"] == "abandoned"
    assert (
        latest_generation_owner(
            store,
            scope_key,
            owner_adapter=adapt_profile_owner,
        )
        == successor.owner
    )


def test_seed_marker_is_the_prediction_reuse_boundary(tmp_path: Path) -> None:
    run_id = "c" * 64
    marker_path = tmp_path / ".markers" / "seeds" / "42.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_bytes(
        orjson.dumps({
            "schema_version": SEED_MARKER_SCHEMA_VERSION,
            "status": "complete",
            "run_id": run_id,
            "seed": 42,
            "sample_count": 2,
            "generation_id": "generation",
            "rankings": [
                {"seed": 42, "sample_index": 1, "ranking_score": 0.7},
                {"seed": 42, "sample_index": 0, "ranking_score": 0.9},
            ],
        })
    )

    marker = load_seed_marker(tmp_path, run_id, 42, sample_count=2)
    assert marker is not None
    assert [row.sample_index for row in marker.rankings] == [0, 1]
    assert load_seed_marker(tmp_path, run_id, 42, sample_count=1) is None


def test_staged_input_rederives_identity_and_preserves_inline_templates(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    prepared = prepare_inference_run(
        AF3Config(
            name="inline-template",
            modelSeeds=[2, 1, 2],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        templates=[
                            AF3Template(
                                mmcif="data_inline\n#\n",
                                queryIndices=[0],
                                templateIndices=[0],
                            )
                        ],
                    )
                )
            ],
        ),
        (),
        output_mount_root=output_root,
        recycle=3,
        sample=2,
    )
    _materialize_prepared_run(output_root, prepared)

    loaded = load_staged_inference_input(
        output_root,
        run_id=prepared.run_id,
        request_id=prepared.request_id,
        staged_input_record=prepared.staged_input.to_record(),
    )

    assert loaded.recycle == 3
    assert loaded.sample_count == 2
    assert loaded.config.modelSeeds == [1, 2]
    template = loaded.config.sequences[0].protein.templates[0]
    assert template.mmcif == "data_inline\n#\n"
    assert template.mmcifPath is None
    assert not any(
        upload.relative_path.parent.name == "custom-templates"
        for upload in prepared.payload_uploads
    )

    marker = orjson.loads(prepared.staged_input.content)
    marker["request_id"] = "f" * 64
    marker_bytes = json_bytes(marker)
    marker_path = output_root / Path(prepared.staged_input.relative_path.as_posix())
    marker_path.write_bytes(marker_bytes)
    with pytest.raises(ValueError, match="marker identity"):
        load_staged_inference_input(
            output_root,
            run_id=prepared.run_id,
            request_id=prepared.request_id,
            staged_input_record=VolumeUpload(
                prepared.staged_input.relative_path,
                marker_bytes,
            ).to_record(),
        )


def test_staged_input_confines_path_backed_templates(tmp_path: Path) -> None:
    template_path = tmp_path / "template.cif"
    template_content = b"data_path_backed\n#\n"
    template_path.write_bytes(template_content)
    output_root = tmp_path / "output"
    prepared = prepare_inference_run(
        AF3Config(
            name="path-template",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        templates=[
                            AF3Template(
                                mmcifPath=str(template_path),
                                queryIndices=[0],
                                templateIndices=[0],
                            )
                        ],
                    )
                )
            ],
        ),
        (
            LocalTemplateFile(
                source_path=template_path,
                content=template_content,
                sha256=hashlib.sha256(template_content).hexdigest(),
            ),
        ),
        output_mount_root=output_root,
        recycle=1,
        sample=1,
    )
    _materialize_prepared_run(output_root, prepared)

    loaded = load_staged_inference_input(
        output_root,
        run_id=prepared.run_id,
        request_id=prepared.request_id,
        staged_input_record=prepared.staged_input.to_record(),
    )
    template = loaded.config.sequences[0].protein.templates[0]
    assert template.mmcif is None
    assert Path(template.mmcifPath).is_relative_to(output_root)

    input_upload = next(
        upload
        for upload in prepared.payload_uploads
        if upload.relative_path.name == "input.json"
    )
    escaped_config = AF3Config.model_validate_json(input_upload.content)
    escaped_config.sequences[0].protein.templates[0].mmcifPath = str(
        tmp_path / "escape.cif"
    )
    escaped_input = serialize_af3_input(escaped_config)
    input_path = output_root / Path(input_upload.relative_path.as_posix())
    input_path.write_bytes(escaped_input)

    marker = orjson.loads(prepared.staged_input.content)
    marker["input"] = VolumeUpload(
        input_upload.relative_path,
        escaped_input,
    ).to_record()
    marker_bytes = json_bytes(marker)
    marker_path = output_root / Path(prepared.staged_input.relative_path.as_posix())
    marker_path.write_bytes(marker_bytes)
    with pytest.raises(ValueError, match="escapes the output Volume"):
        load_staged_inference_input(
            output_root,
            run_id=prepared.run_id,
            request_id=prepared.request_id,
            staged_input_record=VolumeUpload(
                prepared.staged_input.relative_path,
                marker_bytes,
            ).to_record(),
        )


def test_inference_pipeline_coordinates_seed_reuse_and_publication() -> None:
    """The deep inference seam should expose one request-level operation."""

    class FakeInferenceExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def claim_seeds(
            self,
            run_id: str,
            seeds: tuple[int, ...],
            *,
            sample_count: int,
        ) -> SeedClaimPlan:
            del run_id, sample_count
            self.calls.append("claim")
            assert seeds == (1, 2)
            return SeedClaimPlan(
                reused_seeds=(1,),
                owned=(
                    ClaimedSeed(
                        seed=2,
                        claim=GenerationClaim(
                            scope_key="seed:test:2",
                            generation_id="generation",
                            owner={},
                        ),
                    ),
                ),
                active=(),
            )

        def inspect_seeds(
            self,
            run_id: str,
            seeds: tuple[int, ...],
            *,
            sample_count: int,
        ) -> tuple[dict[str, object], ...]:
            del sample_count
            self.calls.append(f"inspect:{','.join(map(str, seeds))}")
            return tuple(
                {
                    "status": "reused",
                    "run_id": run_id,
                    "seed": seed,
                }
                for seed in seeds
            )

        def run_claimed(
            self,
            prepared,
            claimed_seeds: tuple[ClaimedSeed, ...],
            *,
            recycle: int,
            sample_count: int,
            max_workers: int,
            poll_timeout_seconds: int,
        ) -> InferenceBatchOutcome:
            del prepared, recycle, sample_count, max_workers, poll_timeout_seconds
            self.calls.append("run")
            assert tuple(item.seed for item in claimed_seeds) == (2,)
            return InferenceBatchOutcome(
                published_seeds=frozenset({2}),
                reused_seeds=frozenset(),
                failures=(),
            )

        def finalize_summary(
            self,
            prepared,
            *,
            sample_count: int,
        ) -> dict[str, object]:
            del prepared, sample_count
            self.calls.append("summary")
            return {"status": "complete"}

        def finalize_request(
            self,
            prepared,
            *,
            sample_count: int,
        ) -> dict[str, object]:
            del prepared, sample_count
            self.calls.append("request")
            return {"status": "complete"}

    prepared = prepare_inference_run(
        AF3Config(
            name="coordinated-inference",
            modelSeeds=[1, 2],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        unpairedMsa="",
                        pairedMsa="",
                        templates=[],
                    )
                )
            ],
        ),
        (),
        output_mount_root=Path("/outputs"),
        recycle=1,
        sample=1,
    )
    executor = FakeInferenceExecutor()

    result = coordinate_seed_predictions(
        prepared,
        executor,
        recycle=1,
        sample=1,
        num_containers=2,
        active_wait_timeout_seconds=60,
    )

    assert result["reused_seeds"] == [1]
    assert result["published_seeds"] == [2]
    assert result["completed_seeds"] == [1, 2]
    assert result["summary"] == {"status": "complete"}
    assert result["request"] == {"status": "complete"}
    assert executor.calls == [
        "claim",
        "run",
        "inspect:2",
        "inspect:1,2",
        "summary",
        "request",
    ]


def test_request_view_identity_preserves_invocation_presentation() -> None:
    request_id = hash_sequences("a" * 64, [1, 2])
    view_id = request_view_id(request_id, (2, 1, 1), "Readable Name")

    assert request_view_id(request_id, (2, 1, 1), "Readable Name") == view_id
    assert request_view_id(request_id, (1, 2), "Readable Name") != view_id
    assert request_view_id(request_id, (2, 1, 1), "Another Name") != view_id


def test_request_archive_downloads_exact_manifest_view(tmp_path: Path) -> None:
    run_id = "d" * 64
    normalized_seeds = [7]
    request_id = hash_sequences(run_id, normalized_seeds)
    canonical_name = canonical_output_name(run_id)
    input_bytes = serialize_af3_input(
        AF3Config(
            name=canonical_name,
            modelSeeds=normalized_seeds,
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        unpairedMsa="",
                        pairedMsa="",
                        templates=[],
                    )
                )
            ],
        )
    )
    volume_path = f"{run_id[:2]}/{run_id}/requests/{request_id}/input.json"
    manifest = _request_manifest(
        run_id=run_id,
        submitted_seeds=normalized_seeds,
        display_name="Readable Name",
        artifacts=[
            {
                "role": "input",
                "volume_path": volume_path,
                "archive_path": f"{canonical_name}_data.json",
                "size_bytes": len(input_bytes),
                "sha256": hashlib.sha256(input_bytes).hexdigest(),
            }
        ],
    )
    view_id = cast(str, manifest["view_id"])

    archive = create_request_archive(
        FakeVolumeReader({volume_path: input_bytes}),
        manifest,
        output_dir=tmp_path,
        display_name="Readable Name",
    )

    assert archive.name == f"Readable_Name_{view_id[:12]}_AlphaFold3.tar.zst"
    archived_input = "\n".join(
        run_command(
            [
                "tar",
                "-I",
                "zstd",
                "-xOf",
                str(archive),
                "Readable_Name/Readable_Name_data.json",
            ],
            output_mode="capture",
            show_command=False,
        )
    )
    assert orjson.loads(archived_input)["name"] == "Readable Name"

    assert (
        create_request_archive(
            FakeVolumeReader({volume_path: input_bytes}),
            manifest,
            output_dir=tmp_path,
            display_name="Readable Name",
        )
        == archive
    )

    changed_manifest = cast(
        dict[str, object],
        orjson.loads(orjson.dumps(manifest)),
    )
    changed_manifest["published_at"] = "changed"
    with pytest.raises(RuntimeError, match="does not match the current request"):
        create_request_archive(
            FakeVolumeReader({volume_path: input_bytes}),
            changed_manifest,
            output_dir=tmp_path,
            display_name="Readable Name",
        )

    unpacked_root = tmp_path / "unpacked"
    unpacked_root.mkdir()
    run_command(
        [
            "tar",
            "-I",
            "zstd",
            "-xf",
            str(archive),
            "-C",
            str(unpacked_root),
        ],
        output_mode="capture",
        show_command=False,
    )
    archived_input_path = unpacked_root / "Readable_Name" / "Readable_Name_data.json"
    corrupted_input = orjson.loads(archived_input_path.read_bytes())
    corrupted_input["name"] = "Corrupted Name"
    archived_input_path.write_bytes(orjson.dumps(corrupted_input))
    embedded_manifest_path = unpacked_root / "Readable_Name" / "request_manifest.json"
    embedded_manifest = orjson.loads(embedded_manifest_path.read_bytes())
    embedded_artifact = embedded_manifest["artifacts"][0]
    embedded_artifact["archive_size_bytes"] = archived_input_path.stat().st_size
    embedded_artifact["archive_sha256"] = hashlib.sha256(
        archived_input_path.read_bytes()
    ).hexdigest()
    embedded_manifest_path.write_bytes(orjson.dumps(embedded_manifest))
    corrupted_archive = tmp_path / "corrupted.tar.zst"
    run_command(
        [
            "tar",
            "-I",
            "zstd -T0",
            "-cf",
            str(corrupted_archive),
            "--",
            "Readable_Name",
        ],
        output_mode="capture",
        show_command=False,
        cwd=unpacked_root,
    )
    corrupted_archive.replace(archive)
    with pytest.raises(RuntimeError, match="does not match the current request"):
        create_request_archive(
            FakeVolumeReader({volume_path: input_bytes}),
            manifest,
            output_dir=tmp_path,
            display_name="Readable Name",
        )


def test_request_archive_rejects_a_partial_volume_download(tmp_path: Path) -> None:
    run_id = "e" * 64
    normalized_seeds = [9]
    request_id = hash_sequences(run_id, normalized_seeds)
    canonical_name = canonical_output_name(run_id)
    volume_path = f"{run_id[:2]}/{run_id}/requests/{request_id}/input.json"
    manifest = _request_manifest(
        run_id=run_id,
        submitted_seeds=normalized_seeds,
        display_name="partial",
        artifacts=[
            {
                "role": "input",
                "volume_path": volume_path,
                "archive_path": f"{canonical_name}_data.json",
                "size_bytes": 10,
                "sha256": hashlib.sha256(b"expected!!").hexdigest(),
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Downloaded size mismatch"):
        create_request_archive(
            FakeVolumeReader({volume_path: b"partial"}),
            manifest,
            output_dir=tmp_path,
            display_name="partial",
        )


def test_request_archive_rejects_same_size_changed_bytes(tmp_path: Path) -> None:
    run_id = "f" * 64
    normalized_seeds = [10]
    request_id = hash_sequences(run_id, normalized_seeds)
    canonical_name = canonical_output_name(run_id)
    volume_path = f"{run_id[:2]}/{run_id}/requests/{request_id}/input.json"
    expected = b'{"ok":true}'
    manifest = _request_manifest(
        run_id=run_id,
        submitted_seeds=normalized_seeds,
        display_name="changed",
        artifacts=[
            {
                "role": "input",
                "volume_path": volume_path,
                "archive_path": f"{canonical_name}_data.json",
                "size_bytes": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Downloaded SHA-256 mismatch"):
        create_request_archive(
            FakeVolumeReader({volume_path: b'{"ok":null}'}),
            manifest,
            output_dir=tmp_path,
            display_name="changed",
        )


def test_every_fixed_profile_has_one_missing_build_input() -> None:
    inventory: dict[str, object] = {
        "invalid_profiles": {},
        "missing_database_ids": [
            spec.database_id for spec in reversed(DATABASE_PROFILE_SPECS)
        ],
    }

    planned = plan_missing_profile_builds(
        inventory,
        seqkit_threads=DEFAULT_SEQKIT_THREADS,
        source_policy="keep",
    )

    assert [database_id for database_id, _, _ in planned] == [
        spec.database_id for spec in DATABASE_PROFILE_SPECS
    ]
