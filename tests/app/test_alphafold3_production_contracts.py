"""Production contracts for AlphaFold3 preparation, search, and inference."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import datetime
import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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

from biomodals.app.fold import alphafold3_app
from biomodals.app.fold.alphafold3 import (
    inference_inputs,
    msa_search,
    request_results,
    template_search,
)
from biomodals.app.fold.alphafold3.artifacts import (
    artifact_record,
    json_bytes,
    load_artifact_bytes,
    read_volume_bytes,
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
from biomodals.app.fold.alphafold3.invocation_cache import (
    build_invocation_receipt,
    load_invocation_manifest,
    prepare_invocation,
)
from biomodals.app.fold.alphafold3.msa_search import (
    RAW_RESULT_SCHEMA_VERSION,
    ChainMsaState,
    MsaArtifactReference,
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
from biomodals.app.fold.alphafold3.profile_manifest import (
    current_profile_recipe,
    profile_search_identity,
    validate_profile_manifest,
)
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
    RequestPublication,
    create_request_archive,
    load_request_manifest,
    publish_request_results,
    request_manifest_path,
    request_view_id,
)
from biomodals.app.fold.alphafold3.search_pipeline import (
    resolve_msa_and_templates,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    CORE_OUTPUT_SUFFIXES,
    SEED_MARKER_SCHEMA_VERSION,
    ClaimedSeed,
    InferenceRuntime,
    SeedClaimPlan,
    canonical_output_name,
    claim_seed_predictions,
    inspect_seed_predictions,
    load_seed_marker,
)
from biomodals.app.fold.alphafold3.template_search import (
    TEMPLATE_RESULT_SCHEMA_VERSION,
    TemplateRuntime,
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


def _unpaired_msa_reference(
    polymer: str,
    sequence: str,
    unpaired_msa: str,
) -> dict[str, object]:
    root = f"{'Protein' if polymer == 'protein' else 'RNA'}"
    digest = sequence_hash(sequence)
    return _artifact(
        f"{root}/{digest[:2]}/{digest}/unpaired.a3m",
        unpaired_msa.encode(),
    )


def _combined_outcome(
    task: MsaAssemblyTask,
    *,
    status: str = "published",
) -> dict[str, object]:
    fields = {
        field: f">query\n{task.sequence}\n"
        for field, include in (
            ("unpairedMsa", task.include_unpaired),
            ("pairedMsa", task.include_paired),
        )
        if include
    }
    return {
        "status": status,
        "polymer": task.polymer,
        "sequence_sha256": sequence_hash(task.sequence),
        "combined_identity": "a" * 64,
        "fields": fields,
        "unpaired_msa_reference": _unpaired_msa_reference(
            task.polymer,
            task.sequence,
            fields["unpairedMsa"],
        ),
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
    ranking = [
        {
            "seed": seed,
            "sample_index": sample_index,
            "ranking_score": float(len(normalized_seeds) - seed_index)
            - sample_index / max(sample_count, 1),
        }
        for seed_index, seed in enumerate(normalized_seeds)
        for sample_index in range(sample_count)
    ]
    ranking.sort(
        key=lambda row: (
            -cast(float, row["ranking_score"]),
            cast(int, row["seed"]),
            cast(int, row["sample_index"]),
        )
    )
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
        "ranking": ranking,
        "best": ranking[0],
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


def test_bounded_volume_read_distinguishes_missing_and_oversized_files() -> None:
    """Local Volume reads should stream within their explicit metadata budget."""
    reader = FakeVolumeReader({"present": b"payload"})

    assert read_volume_bytes(reader, "present", max_bytes=7) == b"payload"
    assert read_volume_bytes(reader, "missing", max_bytes=7) is None
    with pytest.raises(ValueError, match="exceeds the 6-byte limit"):
        read_volume_bytes(reader, "present", max_bytes=6)


def test_artifact_loader_rejects_size_mismatch_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"oversized")
    record = _artifact("payload.bin", b"x")
    path_open = Path.open

    def reject_payload_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == payload_path:
            pytest.fail("size-mismatched artifact was opened")
        return path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_payload_open)

    assert load_artifact_bytes(tmp_path, record, "payload.bin") is None


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
        if path not in self.files:
            raise FileNotFoundError(path)
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


def test_profile_search_identity_excludes_build_execution_metadata() -> None:
    spec = resolve_database_profile("small_bfd")
    original = _profile_manifest("small_bfd")
    rebuilt = cast(
        dict[str, Any],
        orjson.loads(orjson.dumps(original)),
    )
    rebuilt["created_at"] = "another-time"
    rebuilt["generation_id"] = "another-generation"
    rebuilt["recipe"] = current_profile_recipe(spec, 4)

    assert profile_search_identity(rebuilt, spec) == profile_search_identity(
        original,
        spec,
    )

    changed_shards = cast(
        dict[str, Any],
        orjson.loads(orjson.dumps(rebuilt)),
    )
    shards = cast(list[dict[str, object]], changed_shards["shards"])
    shards[0]["sha256"] = "f" * 64
    assert profile_search_identity(changed_shards, spec) != profile_search_identity(
        original,
        spec,
    )


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
        (_combined_outcome(assembly_task),),
    )

    apply_msa_resolution(
        config,
        states,
        resolution,
        search_protein_templates=True,
    )
    template_plan = plan_template_searches(config, states, resolution)

    assert len(template_plan.tasks) == 1
    assert template_plan.tasks[0].publish_canonical is True
    assert template_plan.tasks[0].unpaired_msa is None
    assert template_plan.tasks[0].unpaired_msa_reference is not None
    assert template_plan.chain_indices_by_identity == {
        template_plan.tasks[0].template_identity: (0, 1)
    }


def test_template_task_computes_immutable_identities_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpaired_msa = ">query\nACDE\n"
    hashed_values: list[bytes] = []
    sha256_bytes = template_search.sha256_bytes

    def count_hash(value: bytes) -> str:
        hashed_values.append(value)
        return sha256_bytes(value)

    monkeypatch.setattr(template_search, "sha256_bytes", count_hash)
    task = TemplateTask(
        sequence="ACDE",
        unpaired_msa=unpaired_msa,
        publish_canonical=False,
    )

    for _ in range(3):
        assert task.unpaired_msa_sha256
        assert task.template_identity

    assert hashed_values.count(unpaired_msa.encode()) == 1
    assert len(hashed_values) == 2


def test_template_task_rejects_invalid_inline_msa_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(template_search, "MAX_LOCAL_MSA_BYTES", 4)
    monkeypatch.setattr(
        template_search,
        "sha256_bytes",
        lambda _: pytest.fail("invalid inline MSA was hashed"),
    )

    with pytest.raises(ValueError, match="byte limit"):
        TemplateTask(
            sequence="ACDE",
            unpaired_msa="12345",
            publish_canonical=False,
        )
    with pytest.raises(TypeError, match="must be a string"):
        TemplateTask(
            sequence="ACDE",
            unpaired_msa=cast(Any, b"1234"),
            publish_canonical=False,
        )


def test_msa_resolution_binds_references_to_returned_fields() -> None:
    task = MsaAssemblyTask(
        polymer="protein",
        sequence="ACDE",
        include_unpaired=True,
        include_paired=True,
    )
    fields = {
        "unpairedMsa": ">query\nACDE\n",
        "pairedMsa": ">query\nACDE\n",
    }
    reference = _unpaired_msa_reference("protein", "ACDE", fields["unpairedMsa"])
    reference = _artifact(
        cast(str, reference["path"]),
        b">query\nCHANGED\n",
    )

    with pytest.raises(RuntimeError, match="does not match the returned field"):
        reduce_msa_assembly_results(
            (task,),
            (
                {
                    "status": "reused",
                    "polymer": "protein",
                    "sequence_sha256": sequence_hash("ACDE"),
                    "combined_identity": "a" * 64,
                    "fields": fields,
                    "unpaired_msa_reference": reference,
                },
            ),
        )


def test_search_pipeline_coordinates_cache_assembly_and_templates() -> None:
    """The deep search seam should hide remote topology from its caller."""

    class FakeSearchExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def inspect_msa(
            self,
            raw_tasks: tuple[RawSearchTask, ...],
            assembly_tasks: tuple[MsaAssemblyTask, ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
            self.calls.append("inspect-msa")
            raw_statuses = tuple(
                {
                    "status": "reused",
                    "database_id": task.database_id,
                    "sequence_sha256": task.sequence_hash,
                }
                for task in raw_tasks
            )
            combined_statuses = tuple(
                {
                    "status": "missing",
                    "polymer": task.polymer,
                    "sequence_sha256": sequence_hash(task.sequence),
                }
                for task in assembly_tasks
            )
            return raw_statuses, combined_statuses

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
            return tuple(_combined_outcome(task) for task in tasks)

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
        "inspect-msa",
        "assemble",
        "inspect-templates",
        "search-templates",
    ]


def test_search_pipeline_reuses_combined_msa_without_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CombinedHitExecutor:
        def inspect_msa(self, raw_tasks, assembly_tasks):
            return (
                tuple(
                    {
                        "status": "reused",
                        "database_id": task.database_id,
                        "sequence_sha256": task.sequence_hash,
                    }
                    for task in raw_tasks
                ),
                tuple(
                    _combined_outcome(task, status="reused") for task in assembly_tasks
                ),
            )

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"Unexpected executor method: {name}")

    config = AF3Config(
        name="combined-hit",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )
    matches_content = MsaArtifactReference.matches_content
    match_calls = 0

    def count_matches(
        reference: MsaArtifactReference,
        content: bytes,
    ) -> bool:
        nonlocal match_calls
        match_calls += 1
        return matches_content(reference, content)

    monkeypatch.setattr(MsaArtifactReference, "matches_content", count_matches)

    resolved = resolve_msa_and_templates(
        config,
        cast(Any, CombinedHitExecutor()),
        search_protein_templates=False,
    )

    protein = resolved.sequences[0].protein
    assert protein is not None
    assert protein.unpairedMsa == ">query\nACDE\n"
    assert protein.pairedMsa == ">query\nACDE\n"
    assert match_calls == 1


def test_search_pipeline_validates_combined_hits_before_scheduling() -> None:
    class MalformedHitExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def inspect_msa(self, raw_tasks, assembly_tasks):
            outcomes = []
            for task in assembly_tasks:
                outcome = _combined_outcome(task, status="reused")
                reference = cast(
                    dict[str, object],
                    outcome["unpaired_msa_reference"],
                )
                outcome["unpaired_msa_reference"] = _artifact(
                    cast(str, reference["path"]),
                    b">query\nCHANGED\n",
                )
                outcomes.append(outcome)
            return (
                tuple(
                    {
                        "status": "missing",
                        "database_id": task.database_id,
                        "sequence_sha256": task.sequence_hash,
                    }
                    for task in raw_tasks
                ),
                tuple(outcomes),
            )

        def run_raw(self, *args, **kwargs):
            self.calls.append("run-raw")
            return ()

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"Unexpected executor method: {name}")

    executor = MalformedHitExecutor()
    config = AF3Config(
        name="malformed-combined-hit",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )

    with pytest.raises(RuntimeError, match="does not match the returned field"):
        resolve_msa_and_templates(
            config,
            cast(Any, executor),
            search_protein_templates=False,
        )

    assert executor.calls == []


def test_search_pipeline_bounds_derived_tasks_before_remote_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NeverCalledExecutor:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"Remote executor method was accessed: {name}")

    monkeypatch.setattr(msa_search, "MAX_REMOTE_SEARCH_TASKS", 1)
    config = AF3Config(
        name="bounded-search",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )

    with pytest.raises(ValueError, match="1 remote search tasks"):
        resolve_msa_and_templates(config, cast(Any, NeverCalledExecutor()))


def test_remote_cache_inspectors_repeat_task_bounds() -> None:
    with pytest.raises(ValueError, match="512 remote search tasks"):
        alphafold3_app.inspect_msa_search_cache.get_raw_f()(
            [("small_bfd", "ACDE")] * 513,
            [],
        )
    with pytest.raises(ValueError, match="512 remote search tasks"):
        alphafold3_app.inspect_protein_template_cache.get_raw_f()(
            [("ACDE", "a" * 64, "2021-09-30")] * 513,
        )


def test_remote_msa_inspector_validates_assembly_shape_before_volume_access() -> None:
    inspect = alphafold3_app.inspect_msa_search_cache.get_raw_f()
    with pytest.raises(TypeError, match="include_unpaired must be a boolean"):
        inspect([], [("protein", "ACDE", 1, True)])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="complete canonical MSAs"):
        inspect([], [("protein", "ACDE", True, False)])


def test_remote_search_repeats_query_length_bound() -> None:
    with pytest.raises(ValueError, match="between 1 and 5,120"):
        alphafold3_app.search_database_msa.get_raw_f()(
            "small_bfd",
            "A" * 5_121,
        )


@pytest.mark.parametrize(
    ("database_id", "module_name", "constructor_name"),
    [
        ("small_bfd", "alphafold3.data.tools.jackhmmer", "Jackhmmer"),
        ("rfam", "alphafold3.data.tools.nhmmer", "Nhmmer"),
    ],
)
def test_search_identity_matches_upstream_constructor_arguments(
    database_id: str,
    module_name: str,
    constructor_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = resolve_database_profile(database_id)
    captured: dict[str, object] = {}

    class Tool:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def _query_db_shard(self, **kwargs):
            return SimpleNamespace(
                target_sequence=kwargs["target_sequence"],
                a3m=f">query\n{kwargs['target_sequence']}\n",
                e_value=0.0,
                tblout="# no hits",
            )

    module = SimpleNamespace(**{constructor_name: Tool})
    if spec.polymer == "protein":
        module._merge_jackhmmer_results = lambda results, max_sequences: (  # noqa: SLF001
            SimpleNamespace(a3m=results[0].a3m)
        )
    else:
        monkeypatch.setattr(
            msa_search,
            "merge_nhmmer_results_by_reported_score",
            lambda module, results, max_sequences: SimpleNamespace(a3m=results[0].a3m),
        )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module if name == module_name else None,
    )

    msa_search.execute_profile_database_search(
        spec,
        "ACDE" if spec.polymer == "protein" else "ACGU",
        selected_profile_root=tmp_path,
        sharded_n_cpu=3,
        max_parallel_shards=1,
    )

    hmmer = {
        "max_sequences": spec.max_sequences,
        "z_value": spec.search_space_value,
        **(
            {
                "n_iter": 1,
                "e_value": 1e-4,
                "dom_e": None,
                "dom_z_value": spec.search_space_value,
                "filter_f1": 5e-4,
                "filter_f2": 5e-5,
                "filter_f3": 5e-7,
            }
            if spec.polymer == "protein"
            else {
                "e_value": 1e-3,
                "filter_f3": 1e-5,
                "alphabet": "rna",
                "strand": None,
            }
        ),
    }
    binary_arguments = (
        {"binary_path": msa_search.JACKHMMER_BINARY_PATH}
        if spec.polymer == "protein"
        else {
            "binary_path": msa_search.NHMMER_BINARY_PATH,
            "hmmalign_binary_path": msa_search.HMMALIGN_BINARY_PATH,
            "hmmbuild_binary_path": msa_search.HMMBUILD_BINARY_PATH,
        }
    )
    assert captured == {
        **binary_arguments,
        "database_path": (tmp_path / "shards" / spec.source_filename).as_posix()
        + f"@{spec.shard_count}",
        "n_cpu": 3,
        "max_threads": 1,
        **hmmer,
    }
    expected_identity: dict[str, object] = {
        "database_id": spec.database_id,
        "polymer": spec.polymer,
        "tool": "jackhmmer" if spec.polymer == "protein" else "nhmmer",
        "hmmer": hmmer,
    }
    if spec.polymer == "rna":
        expected_identity |= {
            "short_sequence": {
                "length_cutoff": 50,
                "filter_f3": 0.02,
            },
            "sharded_merge_order": "reported-evalue-descending-bit-score-name-v1",
        }
    assert scientific_search_parameters(spec) == expected_identity


def test_rna_identity_captures_upstream_short_query_override() -> None:
    assert scientific_search_parameters(resolve_database_profile("rfam"))[
        "short_sequence"
    ] == {
        "length_cutoff": 50,
        "filter_f3": 0.02,
    }


def test_template_identity_matches_upstream_constructor_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, object]] = {}

    def constructor(name: str):
        def build(**kwargs):
            captured[name] = kwargs
            return SimpleNamespace(**kwargs)

        return build

    def get_templates(**kwargs):
        captured["pipeline"] = kwargs
        return SimpleNamespace(get_hits_with_structures=lambda: ())

    msa_config = SimpleNamespace(
        HmmsearchConfig=constructor("hmmsearch"),
        TemplateToolConfig=constructor("tool"),
        TemplateFilterConfig=constructor("filter"),
        TemplatesConfig=constructor("templates"),
    )
    modules = {
        "alphafold3.data.msa_config": msa_config,
        "alphafold3.constants.mmcif_names": SimpleNamespace(PROTEIN_CHAIN="protein"),
        "alphafold3.data.pipeline": SimpleNamespace(
            _get_protein_templates=get_templates
        ),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        template_search,
        "assert_pinned_template_contract",
        lambda: {"contract": "pinned"},
    )
    (tmp_path / template_search.PDB_SEQRES_FILENAME).write_text(
        ">template\nACDE\n",
        encoding="ascii",
    )
    (tmp_path / template_search.MMCIF_DIRECTORY_NAME).mkdir()

    templates, contract = template_search._execute_template_search(
        "ACDE",
        ">query\nACDE\n",
        tmp_path,
        "2025-01-01",
    )

    hmmsearch = {
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
    template_filter = {
        "max_subsequence_ratio": 0.95,
        "min_align_ratio": 0.1,
        "min_hit_length": 10,
        "deduplicate_sequences": True,
        "max_hits": 4,
    }
    assert templates == []
    assert contract == {"contract": "pinned"}
    assert captured["hmmsearch"] == {
        "hmmsearch_binary_path": template_search.HMMSEARCH_BINARY_PATH,
        "hmmbuild_binary_path": template_search.HMMBUILD_BINARY_PATH,
        **hmmsearch,
    }
    assert captured["filter"] == {
        **template_filter,
        "max_template_date": datetime.date(2025, 1, 1),
    }
    assert captured["tool"] == {
        "database_path": str(tmp_path / template_search.PDB_SEQRES_FILENAME),
        "chain_poly_type": "protein",
        "hmmsearch_config": cast(
            SimpleNamespace,
            captured["templates"]["template_tool_config"],
        ).hmmsearch_config,
    }
    templates_config = cast(
        SimpleNamespace,
        captured["pipeline"]["templates_config"],
    )
    assert vars(templates_config.filter_config) == captured["filter"]
    assert captured["pipeline"] == {
        "sequence": "ACDE",
        "input_msa_a3m": ">query\nACDE\n",
        "run_template_search": True,
        "templates_config": templates_config,
        "pdb_database_path": str(tmp_path / template_search.MMCIF_DIRECTORY_NAME),
    }
    assert template_search.template_search_parameters("2025-01-01") == {
        "tool": "hmmsearch",
        "hmmsearch_n_cpu": 8,
        "max_template_date": "2025-01-01",
        "max_a3m_query_sequences": None,
        "hmmsearch": hmmsearch,
        "filter": template_filter,
    }


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


def test_cache_inspection_prefers_combined_msa_and_recovers_raw_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = MsaAssemblyTask(
        polymer="protein",
        sequence="ACDE",
        include_unpaired=True,
        include_paired=True,
    )
    contexts = {
        database_id: SearchContext(
            spec=resolve_database_profile(database_id),
            sequence=task.sequence,
            sequence_hash=sequence_hash(task.sequence),
            profile_root=tmp_path / "profiles" / database_id,
            search_identity=hashlib.sha256(database_id.encode()).hexdigest(),
            provenance={},
            result_root=tmp_path / "raw" / database_id,
        )
        for database_id in (
            *msa_search.PROTEIN_UNPAIRED_DATABASES,
            *msa_search.PROTEIN_PAIRED_DATABASES,
        )
    }
    metadata = {
        database_id: msa_search.RawMsaMetadata(
            context=context,
            done_sha256=hashlib.sha256(f"done:{database_id}".encode()).hexdigest(),
            result_record={
                "path": "result.a3m",
                "size_bytes": 10,
                "sha256": hashlib.sha256(f"result:{database_id}".encode()).hexdigest(),
            },
        )
        for database_id, context in contexts.items()
    }
    raw_tasks = tuple(
        RawSearchTask(database_id=database_id, sequence=task.sequence)
        for database_id in contexts
    )
    monkeypatch.setattr(
        msa_search,
        "load_search_context",
        lambda sharded_root, cache_root, database_id, sequence: contexts[database_id],
    )
    monkeypatch.setattr(
        msa_search,
        "load_raw_msa_metadata",
        lambda context: metadata[context.spec.database_id],
    )
    monkeypatch.setattr(
        msa_search,
        "assert_pinned_msa_assembly_contract",
        lambda: {"contract": "pinned"},
    )
    unpaired_msa = ">query\nACDE\n"
    unpaired_reference = MsaArtifactReference.from_content(
        msa_search.sequence_cache_relpath("protein", task.sequence) / "unpaired.a3m",
        unpaired_msa.encode(),
    )
    monkeypatch.setattr(
        msa_search,
        "_load_combined_msa",
        lambda sequence_root, provenance, selected_task: (
            {
                "unpairedMsa": unpaired_msa,
                "pairedMsa": unpaired_msa,
            },
            unpaired_reference,
        ),
    )
    monkeypatch.setattr(
        msa_search,
        "load_raw_msa",
        lambda context: pytest.fail("combined hit read a raw A3M"),
    )
    sha256_bytes = msa_search.sha256_bytes

    def reject_rehash(value: bytes) -> str:
        if value == unpaired_msa.encode():
            pytest.fail("combined hit re-hashed its validated unpaired MSA")
        return sha256_bytes(value)

    monkeypatch.setattr(msa_search, "sha256_bytes", reject_rehash)

    raw_statuses, combined_statuses = msa_search.inspect_msa_cache(
        tmp_path / "profiles",
        tmp_path / "cache",
        raw_tasks,
        (task,),
    )

    assert {status["status"] for status in raw_statuses} == {"reused"}
    assert combined_statuses[0]["status"] == "reused"
    assert combined_statuses[0]["polymer"] == "protein"
    assert combined_statuses[0]["sequence_sha256"] == sequence_hash("ACDE")
    assert combined_statuses[0]["fields"] == {
        "unpairedMsa": ">query\nACDE\n",
        "pairedMsa": ">query\nACDE\n",
    }
    assert len(cast(str, combined_statuses[0]["combined_identity"])) == 64
    monkeypatch.setattr(
        msa_search,
        "MAX_MSA_INSPECTION_BYTES",
        len(unpaired_msa) * 2 + 1,
    )
    with pytest.raises(ValueError, match="MSA cache inspection exceeds"):
        msa_search.inspect_msa_cache(
            tmp_path / "profiles",
            tmp_path / "cache",
            raw_tasks,
            (task, task),
        )

    monkeypatch.setattr(msa_search, "_load_combined_msa", lambda *args: None)
    monkeypatch.setattr(
        msa_search,
        "load_raw_msa",
        lambda context: (
            None if context.spec.database_id == "small_bfd" else cast(Any, object())
        ),
    )
    raw_statuses, combined_statuses = msa_search.inspect_msa_cache(
        tmp_path / "profiles",
        tmp_path / "cache",
        raw_tasks,
        (task,),
    )

    assert [status["status"] for status in raw_statuses] == [
        "reused",
        "missing",
        "reused",
        "reused",
    ]
    assert combined_statuses == [
        {
            "status": "missing",
            "polymer": "protein",
            "sequence_sha256": sequence_hash("ACDE"),
        }
    ]


def test_combined_msa_cache_bounds_declared_fields_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = MsaAssemblyTask(
        polymer="protein",
        sequence="ACDE",
        include_unpaired=True,
        include_paired=True,
    )
    provenance = {"combined_identity": "a" * 64}
    (tmp_path / "combined.done.json").write_bytes(
        orjson.dumps({
            "schema_version": msa_search.COMBINED_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "provenance": provenance,
            "artifacts": {
                "unpairedMsa": _artifact("unpaired.a3m", b"12345"),
                "pairedMsa": _artifact("paired.a3m", b"x"),
            },
        })
    )
    monkeypatch.setattr(msa_search, "MAX_MSA_FIELD_BYTES", 4)
    monkeypatch.setattr(
        msa_search,
        "load_artifact_bytes",
        lambda *args, **kwargs: pytest.fail("oversized MSA field was read"),
    )

    assert msa_search._load_combined_msa(tmp_path, provenance, task) is None


def test_combined_msa_rejects_oversized_fields_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = MsaAssemblyTask(
        polymer="protein",
        sequence="ACDE",
        include_unpaired=True,
        include_paired=True,
    )
    contexts = {
        database_id: SearchContext(
            spec=resolve_database_profile(database_id),
            sequence=task.sequence,
            sequence_hash=sequence_hash(task.sequence),
            profile_root=tmp_path / "profiles" / database_id,
            search_identity=hashlib.sha256(database_id.encode()).hexdigest(),
            provenance={},
            result_root=tmp_path / "raw" / database_id,
        )
        for database_id in (
            *msa_search.PROTEIN_UNPAIRED_DATABASES,
            *msa_search.PROTEIN_PAIRED_DATABASES,
        )
    }
    metadata = {
        database_id: msa_search.RawMsaMetadata(
            context=context,
            done_sha256=hashlib.sha256(f"done:{database_id}".encode()).hexdigest(),
            result_record=_artifact("result.a3m"),
        )
        for database_id, context in contexts.items()
    }
    entries = {
        database_id: RawMsaEntry(
            context=context,
            a3m=">query\nACDE\n",
            done_sha256=metadata[database_id].done_sha256,
            result_record=metadata[database_id].result_record,
        )
        for database_id, context in contexts.items()
    }
    monkeypatch.setattr(
        msa_search,
        "load_search_context",
        lambda sharded_root, cache_root, database_id, sequence: contexts[database_id],
    )
    monkeypatch.setattr(
        msa_search,
        "load_raw_msa_metadata",
        lambda context: metadata[context.spec.database_id],
    )
    monkeypatch.setattr(
        msa_search,
        "load_raw_msa",
        lambda context: entries[context.spec.database_id],
    )
    monkeypatch.setattr(
        msa_search,
        "assert_pinned_msa_assembly_contract",
        lambda: {"contract": "pinned"},
    )
    monkeypatch.setattr(msa_search, "_load_combined_msa", lambda *args: None)
    monkeypatch.setattr(
        msa_search,
        "assemble_msa_fields",
        lambda *args, **kwargs: {
            "unpairedMsa": "12345",
            "pairedMsa": "x",
        },
    )
    monkeypatch.setattr(msa_search, "MAX_MSA_FIELD_BYTES", 4)
    runtime = msa_search.SearchRuntime(
        sharded_volume=cast(
            Any,
            SimpleNamespace(reload=lambda: None, commit=lambda: None),
        ),
        cache_volume=cast(
            Any,
            SimpleNamespace(reload=lambda: None, commit=lambda: None),
        ),
        claims=FakeClaimStore(),
        container_id="test",
        maximum_age_seconds=100,
        wait_timeout_seconds=100,
        sharded_root=tmp_path / "profiles",
        cache_root=tmp_path / "cache",
    )

    with pytest.raises(ValueError, match="MSA field exceeds"):
        msa_search.assemble_and_publish_msas(runtime, task)

    sequence_root = runtime.cache_root / msa_search.sequence_cache_relpath(
        task.polymer,
        task.sequence,
    )
    assert not (sequence_root / "combined.done.json").exists()


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


def test_template_cache_bounds_declared_payload_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = build_template_context(
        tmp_path,
        "ACDE",
        "a" * 64,
        "2021-09-30",
    )
    context.sequence_root.mkdir(parents=True)
    (context.sequence_root / "templates.done.json").write_bytes(
        orjson.dumps({
            "schema_version": TEMPLATE_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "provenance": context.provenance,
            "templates": _artifact("templates.json", b"12345"),
        })
    )
    monkeypatch.setattr(template_search, "MAX_TEMPLATE_INSPECTION_BYTES", 4)
    monkeypatch.setattr(
        template_search,
        "load_artifact_bytes",
        lambda *args, **kwargs: pytest.fail("oversized template payload was read"),
    )

    assert load_template_entry(context) is None


def test_template_search_rejects_oversized_result_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    templates = [{"mmcif": "12345"}]
    monkeypatch.setattr(
        template_search,
        "MAX_TEMPLATE_INSPECTION_BYTES",
        len(json_bytes(templates)) - 1,
    )
    monkeypatch.setattr(
        template_search,
        "_resolve_template_msa",
        lambda runtime, task: ">query\nACDE\n",
    )
    monkeypatch.setattr(
        template_search,
        "_execute_template_search",
        lambda *args: (templates, {"contract": "pinned"}),
    )
    monkeypatch.setattr(
        template_search,
        "_wait_for_template_claim",
        lambda runtime, context: (
            None,
            SimpleNamespace(generation_id="generation"),
        ),
    )
    monkeypatch.setattr(
        template_search, "assert_generation_current", lambda *args: None
    )
    monkeypatch.setattr(
        template_search, "finish_generation_claim", lambda *args, **kwargs: None
    )
    runtime = TemplateRuntime(
        source_volume=cast(
            Any,
            SimpleNamespace(reload=lambda: None, commit=lambda: None),
        ),
        cache_volume=cast(
            Any,
            SimpleNamespace(reload=lambda: None, commit=lambda: None),
        ),
        claims=FakeClaimStore(),
        container_id="test",
        maximum_age_seconds=100,
        wait_timeout_seconds=100,
        source_root=tmp_path / "source",
        cache_root=tmp_path,
    )
    reference = MsaArtifactReference.from_content(
        msa_search.sequence_cache_relpath("protein", "ACDE") / "unpaired.a3m",
        b">query\nACDE\n",
    )
    task = TemplateTask(
        sequence="ACDE",
        unpaired_msa=None,
        unpaired_msa_reference=reference,
        publish_canonical=True,
    )

    with pytest.raises(ValueError, match="template search result exceeds"):
        template_search.run_template_search(runtime, task)

    context = build_template_context(
        tmp_path,
        task.sequence,
        task.unpaired_msa_sha256,
        task.max_template_date,
    )
    assert not (context.sequence_root / "templates.done.json").exists()


def test_template_cache_inspection_bounds_aggregate_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "status": "reused",
        "sequence_sha256": "a" * 64,
        "unpaired_msa_sha256": "b" * 64,
        "template_identity": "c" * 64,
        "done_sha256": "d" * 64,
        "templates": [{"mmcif": "12345"}],
    }
    context = SimpleNamespace(
        sequence_hash="a" * 64,
        unpaired_msa_sha256="b" * 64,
        template_identity="c" * 64,
    )
    entry = SimpleNamespace(summary=lambda result_status: status)
    monkeypatch.setattr(
        template_search,
        "build_template_context",
        lambda *args: context,
    )
    monkeypatch.setattr(template_search, "load_template_entry", lambda _: entry)
    monkeypatch.setattr(
        template_search,
        "MAX_TEMPLATE_INSPECTION_BYTES",
        2 + len(json_bytes(status)),
    )

    with pytest.raises(ValueError, match="template cache inspection exceeds"):
        template_search.inspect_template_entries(
            tmp_path,
            (
                ("ACDE", "b" * 64, "2021-09-30"),
                ("ACDE", "b" * 64, "2021-09-30"),
            ),
        )


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


def test_template_search_loads_canonical_msa_from_its_volume_reference(
    tmp_path: Path,
) -> None:
    unpaired_msa = ">query\nACDE\n"
    relative_path = (
        msa_search.sequence_cache_relpath("protein", "ACDE") / "unpaired.a3m"
    )
    msa_path = tmp_path / relative_path
    msa_path.parent.mkdir(parents=True)
    msa_path.write_text(unpaired_msa, encoding="ascii")
    reference = MsaArtifactReference.from_record(
        _artifact(relative_path.as_posix(), unpaired_msa.encode()),
        expected_path=relative_path,
    )
    reloads: list[None] = []
    runtime = TemplateRuntime(
        source_volume=cast(
            Any, SimpleNamespace(reload=lambda: None, commit=lambda: None)
        ),
        cache_volume=cast(
            Any,
            SimpleNamespace(
                reload=lambda: reloads.append(None),
                commit=lambda: None,
            ),
        ),
        claims=FakeClaimStore(),
        container_id="test",
        maximum_age_seconds=100,
        wait_timeout_seconds=100,
        source_root=tmp_path / "source",
        cache_root=tmp_path,
    )
    task = TemplateTask(
        sequence="ACDE",
        unpaired_msa=None,
        unpaired_msa_reference=reference,
        publish_canonical=True,
    )

    assert template_search._resolve_template_msa(runtime, task) == unpaired_msa
    assert reloads == [None]


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


def test_seed_claims_reload_the_volume_once_per_reconciliation(
    tmp_path: Path,
) -> None:
    class CountingVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1

        def commit(self) -> None:
            pass

    volume = CountingVolume()
    plan = claim_seed_predictions(
        InferenceRuntime(
            output_root=tmp_path,
            volume=volume,
            claims=FakeClaimStore(),
            container_id="test",
            maximum_age_seconds=100,
            summary_maximum_age_seconds=100,
            wait_timeout_seconds=100,
        ),
        "a" * 64,
        (1, 2, 3),
        sample_count=1,
    )

    assert tuple(item.seed for item in plan.owned) == (1, 2, 3)
    assert volume.reload_count == 2


@pytest.mark.parametrize("boundary", ["inspect", "claim", "publish"])
@pytest.mark.parametrize(
    ("seeds", "sample_count", "message"),
    [
        ((-1,), 1, "32-bit unsigned"),
        ((2**32,), 1, "32-bit unsigned"),
        ((1,), 101, "between 1 and"),
        (tuple(range(501)), 2, "modelSeeds × sample"),
    ],
)
def test_downstream_inference_boundaries_repeat_request_limits(
    tmp_path: Path,
    boundary: str,
    seeds: tuple[int, ...],
    sample_count: int,
    message: str,
) -> None:
    runtime = InferenceRuntime(
        output_root=tmp_path,
        volume=cast(
            Any,
            SimpleNamespace(
                reload=lambda: pytest.fail("invalid request reached the Volume"),
                commit=lambda: None,
            ),
        ),
        claims=FakeClaimStore(),
        container_id="test",
        maximum_age_seconds=100,
        summary_maximum_age_seconds=100,
        wait_timeout_seconds=100,
    )
    run_id = "a" * 64

    with pytest.raises(ValueError, match=message):
        if boundary == "inspect":
            inspect_seed_predictions(
                runtime,
                run_id,
                seeds,
                sample_count=sample_count,
            )
        elif boundary == "claim":
            claim_seed_predictions(
                runtime,
                run_id,
                seeds,
                sample_count=sample_count,
            )
        else:
            publish_request_results(
                runtime,
                RequestPublication(
                    run_id=run_id,
                    request_id=hash_sequences(run_id, list(seeds)),
                    submitted_seeds=seeds,
                    normalized_seeds=seeds,
                    sample_count=sample_count,
                    display_name="bounded",
                ),
            )


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
    protein = loaded.config.sequences[0].protein
    assert protein is not None
    template = protein.templates[0]
    assert template.mmcif == "data_inline\n#\n"
    assert template.mmcifPath is None
    assert not [
        upload
        for upload in prepared.payload_uploads
        if upload.relative_path.parent.name == "custom-templates"
    ]
    assert orjson.loads(prepared.staged_input.content)["custom_templates"] == []

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


def test_staged_input_accepts_a_symlinked_volume_mount(tmp_path: Path) -> None:
    volume_root = tmp_path / "volume"
    mount_root = tmp_path / "mount"
    mount_root.symlink_to(volume_root, target_is_directory=True)
    prepared = prepare_inference_run(
        AF3Config(
            name="symlinked-mount",
            modelSeeds=[1],
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
        output_mount_root=mount_root,
        recycle=1,
        sample=1,
    )
    _materialize_prepared_run(volume_root, prepared)

    loaded = load_staged_inference_input(
        mount_root,
        run_id=prepared.run_id,
        request_id=prepared.request_id,
        staged_input_record=prepared.staged_input.to_record(),
    )

    assert loaded.config.modelSeeds == [1]


def test_staged_input_rechecks_the_serialized_input_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    prepared = prepare_inference_run(
        AF3Config(
            name="bounded-reload",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        unpairedMsa=">query\nACDE\n",
                        pairedMsa="",
                        templates=[],
                    )
                )
            ],
        ),
        (),
        output_mount_root=output_root,
        recycle=1,
        sample=1,
    )
    _materialize_prepared_run(output_root, prepared)
    input_upload = next(
        upload
        for upload in prepared.payload_uploads
        if upload.relative_path.name == "input.json"
    )
    monkeypatch.setattr(
        inference_inputs,
        "MAX_STAGED_INPUT_BYTES",
        len(input_upload.content) - 1,
    )

    with pytest.raises(ValueError, match="Staged artifact is too large"):
        load_staged_inference_input(
            output_root,
            run_id=prepared.run_id,
            request_id=prepared.request_id,
            staged_input_record=prepared.staged_input.to_record(),
        )


def test_staged_input_rechecks_the_run_identity_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    prepared = prepare_inference_run(
        AF3Config(
            name="bounded-identity-reload",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        unpairedMsa=">query\nACDE\n",
                        pairedMsa="",
                        templates=[],
                    )
                )
            ],
        ),
        (),
        output_mount_root=output_root,
        recycle=1,
        sample=1,
    )
    _materialize_prepared_run(output_root, prepared)
    identity_upload = next(
        upload
        for upload in prepared.payload_uploads
        if upload.relative_path.name == "identity.json"
    )
    monkeypatch.setattr(
        inference_inputs,
        "MAX_STAGED_INPUT_BYTES",
        len(identity_upload.content) - 1,
    )

    with pytest.raises(ValueError, match=r"inputs/identity\.json"):
        load_staged_inference_input(
            output_root,
            run_id=prepared.run_id,
            request_id=prepared.request_id,
            staged_input_record=prepared.staged_input.to_record(),
        )


def test_inference_staging_bounds_all_custom_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_paths = [tmp_path / "first.cif", tmp_path / "second.cif"]
    template_contents = [b"12345", b"67890"]
    for path, content in zip(template_paths, template_contents, strict=True):
        path.write_bytes(content)
    monkeypatch.setattr(inference_inputs, "MAX_CUSTOM_TEMPLATE_TOTAL_BYTES", 8)
    config = AF3Config(
        name="bounded-template-staging",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                    templates=[
                        AF3Template(
                            mmcifPath=str(path),
                            queryIndices=[index],
                            templateIndices=[index],
                        )
                        for index, path in enumerate(template_paths)
                    ],
                )
            )
        ],
    )
    custom_templates = tuple(
        LocalTemplateFile(
            source_path=path,
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for path, content in zip(template_paths, template_contents, strict=True)
    )

    with pytest.raises(ValueError, match="custom templates exceed the 8-byte limit"):
        prepare_inference_run(
            config,
            custom_templates,
            output_mount_root=tmp_path / "output",
            recycle=1,
            sample=1,
        )


def test_staged_input_rechecks_the_custom_template_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_paths = [tmp_path / "first.cif", tmp_path / "second.cif"]
    template_contents = [b"12345", b"67890"]
    for path, content in zip(template_paths, template_contents, strict=True):
        path.write_bytes(content)
    output_root = tmp_path / "output"
    prepared = prepare_inference_run(
        AF3Config(
            name="bounded-template-reload",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        templates=[
                            AF3Template(
                                mmcifPath=str(path),
                                queryIndices=[index],
                                templateIndices=[index],
                            )
                            for index, path in enumerate(template_paths)
                        ],
                    )
                )
            ],
        ),
        tuple(
            LocalTemplateFile(
                source_path=path,
                content=content,
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for path, content in zip(template_paths, template_contents, strict=True)
        ),
        output_mount_root=output_root,
        recycle=1,
        sample=1,
    )
    _materialize_prepared_run(output_root, prepared)
    monkeypatch.setattr(inference_inputs, "MAX_CUSTOM_TEMPLATE_TOTAL_BYTES", 8)

    with pytest.raises(ValueError, match="custom templates exceed the 8-byte limit"):
        load_staged_inference_input(
            output_root,
            run_id=prepared.run_id,
            request_id=prepared.request_id,
            staged_input_record=prepared.staged_input.to_record(),
        )


def test_staging_canonicalizes_equivalent_inline_and_path_templates(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "template.cif"
    template_content = b"data_equivalent\n#\n"
    template_path.write_bytes(template_content)

    def config(template: AF3Template) -> AF3Config:
        return AF3Config(
            name="equivalent-template",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        templates=[template],
                    )
                )
            ],
        )

    inline = prepare_inference_run(
        config(
            AF3Template(
                mmcif=template_content.decode(),
                queryIndices=[0],
                templateIndices=[0],
            )
        ),
        (),
        output_mount_root=tmp_path / "output",
        recycle=1,
        sample=1,
    )
    path_backed = prepare_inference_run(
        config(
            AF3Template(
                mmcifPath=str(template_path),
                queryIndices=[0],
                templateIndices=[0],
            )
        ),
        (
            LocalTemplateFile(
                source_path=template_path,
                content=template_content,
                sha256=hashlib.sha256(template_content).hexdigest(),
            ),
        ),
        output_mount_root=tmp_path / "output",
        recycle=1,
        sample=1,
    )

    assert inline.run_id == path_backed.run_id
    assert inline.request_id == path_backed.request_id
    inline_identity = next(
        upload
        for upload in inline.payload_uploads
        if upload.relative_path.name == "identity.json"
    )
    path_identity = next(
        upload
        for upload in path_backed.payload_uploads
        if upload.relative_path.name == "identity.json"
    )
    assert inline_identity == path_identity
    assert not [
        upload
        for upload in inline.payload_uploads
        if upload.relative_path.parent.name == "custom-templates"
    ]
    assert (
        len([
            upload
            for upload in path_backed.payload_uploads
            if upload.relative_path.parent.name == "custom-templates"
        ])
        == 1
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
    protein = loaded.config.sequences[0].protein
    assert protein is not None
    template = protein.templates[0]
    assert template.mmcif is None
    assert template.mmcifPath is not None
    assert Path(template.mmcifPath).is_relative_to(output_root)

    input_upload = next(
        upload
        for upload in prepared.payload_uploads
        if upload.relative_path.name == "input.json"
    )
    escaped_config = AF3Config.model_validate_json(input_upload.content)
    escaped_protein = escaped_config.sequences[0].protein
    assert escaped_protein is not None
    escaped_protein.templates[0].mmcifPath = str(tmp_path / "escape.cif")
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
            max_workers: int,
            poll_timeout_seconds: int,
        ) -> InferenceBatchOutcome:
            del prepared, max_workers, poll_timeout_seconds
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
        ) -> dict[str, object]:
            del prepared
            self.calls.append("summary")
            return {"status": "complete"}

        def finalize_request(
            self,
            prepared,
        ) -> dict[str, object]:
            del prepared
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


def test_completed_request_manifest_loads_without_a_remote_worker() -> None:
    """A locally known request identity should resolve its durable manifest."""
    run_id = "c" * 64
    submitted_seeds = [2, 1, 1]
    publication = RequestPublication(
        run_id=run_id,
        request_id=hash_sequences(run_id, [1, 2]),
        submitted_seeds=tuple(submitted_seeds),
        normalized_seeds=(1, 2),
        sample_count=1,
        display_name="Readable Name",
    )
    input_path = f"{run_id[:2]}/{run_id}/requests/{publication.request_id}/input.json"
    input_bytes = b"input"
    manifest = _request_manifest(
        run_id=run_id,
        submitted_seeds=submitted_seeds,
        display_name=publication.display_name,
        artifacts=[
            {
                "role": "input",
                "volume_path": input_path,
                "archive_path": f"{canonical_output_name(run_id)}_data.json",
                "size_bytes": len(input_bytes),
                "sha256": hashlib.sha256(input_bytes).hexdigest(),
            }
        ],
    )
    manifest_path = request_manifest_path(publication).as_posix()

    assert (
        load_request_manifest(
            FakeVolumeReader({manifest_path: json_bytes(manifest)}),
            publication,
        )
        == manifest
    )
    assert load_request_manifest(FakeVolumeReader({}), publication) is None


def test_invocation_identity_covers_result_and_presentation_options() -> None:
    """Only an exact scientific request and presentation should share a receipt."""

    def invocation(
        *,
        name: str = "Readable Name",
        seeds: list[int] | None = None,
        search_msa: bool = True,
        search_templates: bool = True,
        recycle: int = 1,
        sample: int = 1,
    ):
        return prepare_invocation(
            AF3Config(
                name=name,
                modelSeeds=seeds or [2, 1, 1],
                sequences=[
                    AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE"))
                ],
            ),
            (),
            search_msa=search_msa,
            search_protein_templates=search_templates,
            recycle=recycle,
            sample=sample,
        )

    baseline = invocation()
    assert invocation().invocation_id == baseline.invocation_id
    assert (
        len({
            baseline.invocation_id,
            invocation(name="Another Name").invocation_id,
            invocation(seeds=[1, 2]).invocation_id,
            invocation(search_msa=False).invocation_id,
            invocation(search_templates=False).invocation_id,
            invocation(recycle=2).invocation_id,
            invocation(sample=2).invocation_id,
        })
        == 7
    )


def test_invocation_receipt_resolves_and_binds_the_manifest() -> None:
    """A receipt should be deterministic and reject changed manifest bytes."""
    submitted = AF3Config(
        name="Readable Name",
        modelSeeds=[2, 1, 1],
        sequences=[AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE"))],
    )
    invocation = prepare_invocation(
        submitted,
        (),
        search_msa=True,
        search_protein_templates=True,
        recycle=1,
        sample=1,
    )
    enriched = submitted.model_copy(deep=True)
    protein = enriched.sequences[0].protein
    assert protein is not None
    protein.unpairedMsa = ""
    protein.pairedMsa = ""
    protein.templates = []
    prepared = prepare_inference_run(
        enriched,
        (),
        output_mount_root=Path("/outputs"),
        recycle=1,
        sample=1,
    )
    run_id = prepared.run_id
    input_bytes = b"input"
    request_id = prepared.request_id
    manifest = _request_manifest(
        run_id=run_id,
        submitted_seeds=[2, 1, 1],
        display_name="Readable Name",
        artifacts=[
            {
                "role": "input",
                "volume_path": (
                    f"{run_id[:2]}/{run_id}/requests/{request_id}/input.json"
                ),
                "archive_path": f"{canonical_output_name(run_id)}_data.json",
                "size_bytes": len(input_bytes),
                "sha256": hashlib.sha256(input_bytes).hexdigest(),
            }
        ],
    )
    receipt = build_invocation_receipt(invocation, prepared, manifest)
    assert build_invocation_receipt(invocation, prepared, manifest) == receipt
    manifest_path = cast(str, manifest["manifest_volume_path"])
    manifest_bytes = json_bytes(manifest)
    files = {
        receipt.relative_path.as_posix(): receipt.content,
        manifest_path: manifest_bytes,
    }

    assert load_invocation_manifest(FakeVolumeReader({}), invocation) is None
    assert load_invocation_manifest(FakeVolumeReader(files), invocation) == manifest

    corrupted = manifest_bytes.replace(b'"complete"', b'"corruptx"', 1)
    with pytest.raises(RuntimeError, match="digest is invalid"):
        load_invocation_manifest(
            FakeVolumeReader(files | {manifest_path: corrupted}),
            invocation,
        )


def test_request_publication_persists_only_a_manifest_view(tmp_path: Path) -> None:
    """Request aliases should reference canonical outputs without Volume copies."""
    run_id = "d" * 64
    seed = 7
    request_id = hash_sequences(run_id, [seed])
    canonical_name = canonical_output_name(run_id)
    run_root = tmp_path / run_id[:2] / run_id
    input_path = run_root / "requests" / request_id / "input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(
        serialize_af3_input(
            AF3Config(
                name=canonical_name,
                modelSeeds=[seed],
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
    )
    outputs_root = run_root / "outputs"
    sample_root = outputs_root / f"seed-{seed}_sample-0"
    sample_root.mkdir(parents=True)
    prefix = f"{canonical_name}_seed-{seed}_sample-0"
    for suffix in CORE_OUTPUT_SUFFIXES:
        (sample_root / f"{prefix}_{suffix}").write_text(
            f"{suffix}\n",
            encoding="utf-8",
        )
    (outputs_root / "TERMS_OF_USE.md").write_text("terms\n", encoding="utf-8")
    marker_path = run_root / ".markers" / "seeds" / f"{seed}.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_bytes(
        orjson.dumps({
            "schema_version": SEED_MARKER_SCHEMA_VERSION,
            "status": "complete",
            "run_id": run_id,
            "seed": seed,
            "sample_count": 1,
            "generation_id": "generation",
            "rankings": [{"seed": seed, "sample_index": 0, "ranking_score": 0.9}],
        })
    )

    class CountingVolume:
        def __init__(self) -> None:
            self.reload_count = 0
            self.commit_count = 0

        def reload(self) -> None:
            self.reload_count += 1

        def commit(self) -> None:
            self.commit_count += 1

    volume = CountingVolume()
    manifest = publish_request_results(
        InferenceRuntime(
            output_root=tmp_path,
            volume=cast(Any, volume),
            claims=FakeClaimStore(),
            container_id="test",
            maximum_age_seconds=100,
            summary_maximum_age_seconds=100,
            wait_timeout_seconds=100,
        ),
        RequestPublication(
            run_id=run_id,
            request_id=request_id,
            submitted_seeds=(seed,),
            normalized_seeds=(seed,),
            sample_count=1,
            display_name="Readable Name",
        ),
    )

    view_id = cast(str, manifest["view_id"])
    view_root = run_root / "requests" / request_id / "views" / view_id
    assert [path.name for path in view_root.iterdir()] == ["manifest.json"]
    artifacts = cast(list[dict[str, object]], manifest["artifacts"])
    assert "request_ranking" not in {artifact["role"] for artifact in artifacts}
    best_artifacts = [
        artifact
        for artifact in artifacts
        if cast(str, artifact["role"]).startswith("request_best_")
    ]
    assert len(best_artifacts) == len(CORE_OUTPUT_SUFFIXES)
    assert all(
        f"/outputs/seed-{seed}_sample-0/" in f"/{artifact['volume_path']}"
        for artifact in best_artifacts
    )
    assert volume.reload_count == 1
    assert volume.commit_count == 1


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
    ranking_csv = "\n".join(
        run_command(
            [
                "tar",
                "-I",
                "zstd",
                "-xOf",
                str(archive),
                "Readable_Name/Readable_Name_ranking_scores.csv",
            ],
            output_mode="capture",
            show_command=False,
        )
    )
    assert ranking_csv.splitlines() == [
        "seed,sample,ranking_score",
        "7,0,1.0",
    ]

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


def test_artifact_download_rejects_overflow_before_writing_the_chunk(
    tmp_path: Path,
) -> None:
    class OverflowReader:
        def read_file(self, path: str):
            yield b"1234"
            yield b"56"
            pytest.fail("download continued after the first overflowing chunk")

    destination = tmp_path / "artifact.bin"
    artifact = {
        "volume_path": "artifact.bin",
        "size_bytes": 5,
        "sha256": hashlib.sha256(b"12345").hexdigest(),
    }

    with pytest.raises(RuntimeError, match="Downloaded size mismatch"):
        request_results._download_artifact(
            cast(Any, OverflowReader()),
            artifact,
            destination,
        )

    assert destination.read_bytes() == b"1234"


def test_artifact_download_hashes_while_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"streamed"
    destination = tmp_path / "artifact.bin"
    artifact = {
        "volume_path": "artifact.bin",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(
        request_results,
        "sha256_file",
        lambda path: pytest.fail("downloaded artifact was reread to hash it"),
    )

    request_results._download_artifact(
        FakeVolumeReader({"artifact.bin": content}),
        artifact,
        destination,
    )

    assert destination.read_bytes() == content


def test_archive_manifest_rehashes_only_rewritten_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "data.json"
    output_path = tmp_path / "model.cif"
    input_path.write_bytes(b"rewritten")
    output_path.write_bytes(b"unchanged")
    output_sha256 = hashlib.sha256(b"unchanged").hexdigest()
    local_manifest: dict[str, object] = {
        "artifacts": [
            {"role": "input"},
            {
                "role": "seed_output",
                "size_bytes": len(b"unchanged"),
                "sha256": output_sha256,
            },
        ]
    }
    transformed = [
        (
            {
                "role": "input",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
            PurePosixPath("data.json"),
        ),
        (
            {
                "role": "seed_output",
                "size_bytes": len(b"unchanged"),
                "sha256": output_sha256,
            },
            PurePosixPath("model.cif"),
        ),
    ]
    sha256_file = request_results.sha256_file
    hashed_paths: list[Path] = []

    def track_hash(path: Path) -> str:
        hashed_paths.append(path)
        return sha256_file(path)

    monkeypatch.setattr(request_results, "sha256_file", track_hash)

    request_results._record_archive_artifacts(
        local_manifest,
        transformed,
        tmp_path,
    )

    assert hashed_paths == [input_path]
    local_output = cast(list[dict[str, object]], local_manifest["artifacts"])[1]
    assert local_output["archive_size_bytes"] == len(b"unchanged")
    assert local_output["archive_sha256"] == output_sha256


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
