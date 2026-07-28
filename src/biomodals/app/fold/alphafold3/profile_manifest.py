"""Immutable AlphaFold 3 database-profile manifest contract.

Both the one-time builder and every search worker cross this seam. Keeping
manifest parsing and publication validation here prevents runtime search code
from depending on the database builder implementation.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

from biomodals.app.fold.alphafold3.artifacts import (
    json_bytes,
    load_json_object,
    require_regular_file,
    sha256_bytes,
    sha256_file,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
    COMPOSABLE_LEGACY_VALIDATION_RELPATHS,
    COMPOSABLE_MULTISET_RECIPE_VERSION,
    HMMER_VERSION,
    JACKHMMER_PATCH_SHA256,
    LEGACY_PROFILE_RECIPE_VERSION,
    LEGACY_VALIDATION_RELPATHS,
    ORDINAL_SHUFFLER_PREFETCH_BYTES,
    ORDINAL_SHUFFLER_PREFETCH_RECORDS,
    ORDINAL_SHUFFLER_RECIPE_VERSION,
    ORDINAL_SHUFFLER_SOURCE_SHA256,
    ORDINAL_SHUFFLER_VERSION,
    ORDINAL_VALIDATION_RELPATHS,
    PROFILE_SCHEMA_VERSION,
    SEQKIT_VERSION,
    SHARD_RANDOM_SEED,
    SOURCE_DB_VOLUME_NAME,
    VALIDATION_RELPATHS,
    DatabaseProfileSpec,
    record_multiset_identity,
    shard_names,
    validate_seqkit_threads,
)


def profile_compatibility_identity() -> dict[str, str]:
    """Return the exact upstream and adapter identity accepted by profiles."""
    return {
        "alphafold_repository": ALPHAFOLD3_REPOSITORY,
        "alphafold_commit": ALPHAFOLD3_COMMIT,
        "hmmer_version": HMMER_VERSION,
        "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
    }


def current_profile_recipe(
    spec: DatabaseProfileSpec,
    seqkit_threads: int,
) -> dict[str, object]:
    """Return the one recipe emitted for newly built database profiles."""
    threads = validate_seqkit_threads(seqkit_threads)
    return {
        "version": COMPOSABLE_MULTISET_RECIPE_VERSION,
        "seqkit_version": SEQKIT_VERSION,
        "seqkit_threads": threads,
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
            "worker_threads": threads,
            "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
            "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
        },
        "duplicate_recovery": {
            "warning_source": None,
            "record_identity": "source-occurrence",
            "append_after_shuffle": False,
            "strip_after_split": False,
        },
        "record_multiset": record_multiset_identity() | {"shard_threads": threads},
        "split": ["--by-part", spec.shard_count],
    }


def _validate_recipe(
    recipe: dict[str, Any],
    spec: DatabaseProfileSpec,
) -> tuple[int, tuple[tuple[str, ...], ...]]:
    """Validate one supported immutable sharding recipe."""
    if recipe.get("seqkit_version") != SEQKIT_VERSION:
        raise ValueError("Unexpected profile SeqKit version")
    if recipe.get("random_seed") != SHARD_RANDOM_SEED:
        raise ValueError("Unexpected profile shuffle seed")
    if recipe.get("split") != ["--by-part", spec.shard_count]:
        raise ValueError("Unexpected profile split recipe")
    raw_threads = recipe.get("seqkit_threads")
    if isinstance(raw_threads, bool) or not isinstance(raw_threads, int):
        raise ValueError("Invalid profile SeqKit threads")
    try:
        seqkit_threads = validate_seqkit_threads(raw_threads)
    except ValueError as exc:
        raise ValueError("Invalid profile SeqKit threads") from exc

    recipe_version = recipe.get("version")
    if recipe_version == LEGACY_PROFILE_RECIPE_VERSION:
        if recipe.get("shuffle") != [
            "--two-pass",
            "--update-faidx",
            "--tmp-dir=/tmp",
        ]:
            raise ValueError("Unexpected legacy profile shuffle recipe")
        if recipe.get("duplicate_recovery") != {
            "warning_source": "seqkit-fai-sequence-byte-offset",
            "temporary_header_identity": "generation-unique-uuid",
            "append_after_shuffle": True,
            "strip_after_split": True,
        }:
            raise ValueError("Unexpected legacy duplicate-recovery recipe")
        return recipe_version, (LEGACY_VALIDATION_RELPATHS,)

    if recipe_version not in {
        ORDINAL_SHUFFLER_RECIPE_VERSION,
        COMPOSABLE_MULTISET_RECIPE_VERSION,
    }:
        raise ValueError("Unexpected profile recipe version")
    current_recipe = current_profile_recipe(spec, seqkit_threads)
    if recipe.get("shuffle") != current_recipe["shuffle"]:
        raise ValueError("Unexpected occurrence-indexed shuffle recipe")
    if recipe.get("shuffler") != current_recipe["shuffler"]:
        raise ValueError("Unexpected native shuffler identity")
    if recipe.get("execution") != current_recipe["execution"]:
        raise ValueError("Unexpected native shuffler execution plan")
    if recipe.get("duplicate_recovery") != current_recipe["duplicate_recovery"]:
        raise ValueError("Unexpected occurrence-indexed duplicate policy")
    if recipe_version == ORDINAL_SHUFFLER_RECIPE_VERSION:
        return recipe_version, (ORDINAL_VALIDATION_RELPATHS,)
    if recipe.get("record_multiset") != current_recipe["record_multiset"]:
        raise ValueError("Unexpected composable record-multiset validator")
    return recipe_version, (
        VALIDATION_RELPATHS,
        COMPOSABLE_LEGACY_VALIDATION_RELPATHS,
    )


def validate_profile_manifest(
    manifest: dict[str, Any],
    spec: DatabaseProfileSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one profile manifest without filesystem access."""
    if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unexpected profile schema version")
    if manifest.get("profile_id") != spec.profile_id:
        raise ValueError("Unexpected profile ID")
    if manifest.get("database_id") != spec.database_id:
        raise ValueError("Unexpected database ID")
    if manifest.get("polymer") != spec.polymer:
        raise ValueError("Unexpected profile polymer")
    if manifest.get("shard_count") != spec.shard_count:
        raise ValueError("Unexpected profile shard count")
    if manifest.get("shard_prefix") != f"shards/{spec.source_filename}":
        raise ValueError("Unexpected profile shard prefix")
    if manifest.get("search_space_value") != spec.search_space_value:
        raise ValueError("Unexpected profile search-space value")
    if manifest.get("search_space_unit") != spec.search_space_unit:
        raise ValueError("Unexpected profile search-space unit")

    source = manifest.get("source")
    shards = manifest.get("shards")
    validation = manifest.get("validation")
    recipe = manifest.get("recipe")
    compatibility = manifest.get("compatibility")
    if not isinstance(source, dict):
        raise ValueError("Profile source must be an object")
    if source.get("volume") != SOURCE_DB_VOLUME_NAME:
        raise ValueError("Profile source Volume is invalid")
    if source.get("path") != spec.source_filename:
        raise ValueError("Profile source path is invalid")
    if not isinstance(source.get("size_bytes"), int) or source["size_bytes"] <= 0:
        raise ValueError("Profile source size is invalid")
    if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
        raise ValueError("Profile source SHA-256 is invalid")
    if (
        not isinstance(source.get("num_seqs"), int)
        or source["num_seqs"] <= 0
        or not isinstance(source.get("sum_len"), int)
        or source["sum_len"] <= 0
    ):
        raise ValueError("Profile source statistics are invalid")
    if (
        spec.expected_num_seqs is not None
        and source["num_seqs"] != spec.expected_num_seqs
    ):
        raise ValueError("Profile source sequence count is invalid")
    if spec.expected_sum_len is not None and source["sum_len"] != spec.expected_sum_len:
        raise ValueError("Profile source residue count is invalid")

    if not isinstance(shards, list) or len(shards) != spec.shard_count:
        raise ValueError(f"Profile must declare {spec.shard_count} shards")
    if not isinstance(recipe, dict):
        raise ValueError("Profile recipe must be an object")
    if compatibility != profile_compatibility_identity():
        raise ValueError("Unexpected profile compatibility pin")
    recipe_version, allowed_validation_relpaths = _validate_recipe(recipe, spec)

    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Profile does not declare passed validation")
    if validation.get("temporary_recovery_prefix_absent") is not True:
        raise ValueError("Profile may retain recovery prefixes")
    if validation.get("num_seqs") != source["num_seqs"]:
        raise ValueError("Profile validation sequence count is invalid")
    if validation.get("sum_len") != source["sum_len"]:
        raise ValueError("Profile validation residue count is invalid")
    if recipe_version in {
        ORDINAL_SHUFFLER_RECIPE_VERSION,
        COMPOSABLE_MULTISET_RECIPE_VERSION,
    }:
        if validation.get("record_occurrences_preserved") is not True:
            raise ValueError("Profile does not preserve record occurrences")
        if (
            validation.get("recovered_records") != 0
            or validation.get("recovered_residues") != 0
            or validation.get("first_recovered_byte_offset") is not None
            or validation.get("last_recovered_byte_offset") is not None
        ):
            raise ValueError("Occurrence-indexed profile declares FAI recovery")
    if recipe_version == COMPOSABLE_MULTISET_RECIPE_VERSION:
        if validation.get("canonical_record_multiset_match") is not True:
            raise ValueError("Canonical source and shard record multisets differ")
        signature_sha256 = validation.get("record_multiset_signature_sha256")
        if (
            not isinstance(signature_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", signature_sha256) is None
        ):
            raise ValueError("Invalid canonical record-multiset signature")
        if "seqkit_sum" in validation:
            raise ValueError(
                "Composable-multiset profile unexpectedly declares SeqKit sum"
            )
    validation_artifacts = validation.get("artifacts")
    if not isinstance(validation_artifacts, list):
        raise ValueError("Profile validation artifacts must be a list")

    expected_shard_paths = [f"shards/{name}" for name in shard_names(spec)]
    actual_shard_paths: list[str] = []
    for record in [*shards, *validation_artifacts]:
        if not isinstance(record, dict):
            raise ValueError("Profile artifact record must be an object")
        relative = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Profile artifact path must be relative")
        if ".." in PurePosixPath(relative).parts:
            raise ValueError(f"Profile artifact escapes root: {relative}")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(f"Profile artifact is empty: {relative}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid profile artifact digest: {relative}")
        if relative.startswith("shards/"):
            actual_shard_paths.append(relative)
    if actual_shard_paths != expected_shard_paths:
        raise ValueError("Profile shard order or names are invalid")
    actual_validation_relpaths = tuple(
        str(record["path"]) for record in validation_artifacts
    )
    if actual_validation_relpaths not in allowed_validation_relpaths:
        raise ValueError("Profile validation artifact paths are invalid")
    return source, shards, validation_artifacts


def profile_search_identity(
    manifest: dict[str, Any],
    spec: DatabaseProfileSpec,
) -> str:
    """Hash only profile content that can affect database-search results."""
    source, shards, _ = validate_profile_manifest(manifest, spec)
    return sha256_bytes(
        json_bytes({
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_id": spec.profile_id,
            "database_id": spec.database_id,
            "polymer": spec.polymer,
            "source": {
                key: source[key] for key in ("path", "sha256", "num_seqs", "sum_len")
            },
            "shards": [
                {key: shard[key] for key in ("path", "size_bytes", "sha256")}
                for shard in shards
            ],
            "search_space_value": manifest["search_space_value"],
            "search_space_unit": manifest["search_space_unit"],
            "compatibility": manifest["compatibility"],
        })
    )


def validate_published_profile(
    root: Path,
    spec: DatabaseProfileSpec,
    *,
    verify_digests: bool,
) -> dict[str, Any]:
    """Validate a manifest-last profile publication and its artifacts."""
    manifest_path = root / "manifest.json"
    require_regular_file(manifest_path)
    if (root / "source").exists():
        raise ValueError("Profile must not contain a source copy")
    manifest = load_json_object(manifest_path)
    _, shards, validation_artifacts = validate_profile_manifest(manifest, spec)
    resolved_root = root.resolve()
    for record in [*shards, *validation_artifacts]:
        relative = str(record["path"])
        artifact_path = (root / relative).resolve()
        if not artifact_path.is_relative_to(resolved_root):
            raise ValueError(f"Profile artifact escapes root: {relative}")
        require_regular_file(artifact_path)
        if artifact_path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Profile artifact size mismatch: {relative}")
        if verify_digests and sha256_file(artifact_path) != record["sha256"]:
            raise ValueError(f"Profile artifact digest mismatch: {relative}")
    return manifest
