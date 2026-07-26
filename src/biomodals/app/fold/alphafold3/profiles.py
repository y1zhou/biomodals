"""Fixed AlphaFold 3 genetic-database sharding profiles.

This module owns the production profile identities and the pure helpers used
by both the temporary MSA validation app and the production AlphaFold 3 app.
It deliberately contains no Modal objects or benchmark campaign state.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

ALPHAFOLD3_REPOSITORY = "https://github.com/y1zhou/alphafold3"
ALPHAFOLD3_COMMIT = "987ad1cb7d7028b6d35908cf63fe7d951d98d6b6"

SOURCE_DB_VOLUME_NAME = "AlphaFold3-msa-db"
SHARDED_DB_VOLUME_NAME = "AlphaFold3-msa-db-sharded"
PROFILE_BUILD_CLAIM_DICT_NAME = "AlphaFold3-msa-profile-build-claims"

PROFILE_SCHEMA_VERSION = 2
LEGACY_PROFILE_RECIPE_VERSION = 3
ORDINAL_SHUFFLER_RECIPE_VERSION = 4
COMPOSABLE_MULTISET_RECIPE_VERSION = 5
PROFILE_ROOT = "profiles"
DEFAULT_SEQKIT_THREADS = 8
MAX_SEQKIT_THREADS = 32
SEQKIT_VERSION = "2.13.0"
SHARD_RANDOM_SEED = 23
MAX_PROFILE_IMBALANCE = 0.05
BUILD_TIMEOUT_SECONDS = 86_400
BUILD_MEMORY_MIB = (1024, 262_144)
PROFILE_STALE_SECONDS = BUILD_TIMEOUT_SECONDS + 900
SCRATCH_ROOT = Path(tempfile.gettempdir())

HMMER_VERSION = "3.4"
JACKHMMER_PATCH_SHA256 = (
    "df9e3ae35ad1659921d96ebfca67a9616a7a467ddde2be18a56f9bd3edb38c41"
)

LEGACY_VALIDATION_RELPATHS = (
    "validation/source-stats.tsv",
    "validation/shard-stats.tsv",
    "validation/shard-summary.parquet",
    "validation/source-sum.tsv",
    "validation/shard-sum.tsv",
    "validation/seqkit-sum.json",
    "validation/shuffle-stderr.log",
    "validation/duplicate-recovery.jsonl",
)
ORDINAL_VALIDATION_RELPATHS = (
    *LEGACY_VALIDATION_RELPATHS,
    "validation/shuffler-metrics.json",
)
VALIDATION_RELPATHS = (
    "validation/source-stats.tsv",
    "validation/shard-stats.tsv",
    "validation/shard-summary.parquet",
    "validation/record-multiset.json",
    "validation/shuffle-stderr.log",
    "validation/duplicate-recovery.jsonl",
    "validation/shuffler-metrics.json",
)

SourcePolicy = Literal["keep", "compress", "delete"]
SOURCE_POLICIES: tuple[SourcePolicy, ...] = ("keep", "compress", "delete")


@dataclass(frozen=True, slots=True)
class DatabaseProfileSpec:
    """One code-owned immutable database-sharding specification."""

    database_id: str
    profile_id: str
    source_filename: str
    shard_count: int
    polymer: Literal["protein", "rna"]
    expected_num_seqs: int | None
    expected_sum_len: int | None
    max_sequences: int

    @property
    def search_space_value(self) -> int | float:
        """Return the full-database HMMER search-space value."""
        if self.polymer == "protein":
            if self.expected_num_seqs is None:
                raise RuntimeError(
                    f"{self.database_id} lacks an expected sequence count"
                )
            return self.expected_num_seqs
        if self.expected_sum_len is None:
            raise RuntimeError(f"{self.database_id} lacks an expected residue count")
        return self.expected_sum_len / 1_000_000

    @property
    def search_space_unit(self) -> str:
        """Return the unit expected by the pinned HMMER wrapper."""
        return "sequences" if self.polymer == "protein" else "megabases"


DATABASE_PROFILE_SPECS = (
    DatabaseProfileSpec(
        database_id="small_bfd",
        profile_id="small-bfd-64-v2",
        source_filename="bfd-first_non_consensus_sequences.fasta",
        shard_count=64,
        polymer="protein",
        expected_num_seqs=65_984_053,
        expected_sum_len=None,
        max_sequences=5_000,
    ),
    DatabaseProfileSpec(
        database_id="mgnify",
        profile_id="mgnify-512-v1",
        source_filename="mgy_clusters_2022_05.fa",
        shard_count=512,
        polymer="protein",
        expected_num_seqs=623_796_864,
        expected_sum_len=None,
        max_sequences=5_000,
    ),
    DatabaseProfileSpec(
        database_id="uniprot",
        profile_id="uniprot-384-v1",
        source_filename="uniprot_all_2021_04.fa",
        shard_count=384,
        polymer="protein",
        expected_num_seqs=225_619_586,
        expected_sum_len=None,
        max_sequences=50_000,
    ),
    DatabaseProfileSpec(
        database_id="uniref90",
        profile_id="uniref90-128-v1",
        source_filename="uniref90_2022_05.fa",
        shard_count=128,
        polymer="protein",
        expected_num_seqs=153_742_194,
        expected_sum_len=None,
        max_sequences=10_000,
    ),
    DatabaseProfileSpec(
        database_id="ntrna",
        profile_id="nt-rna-256-v1",
        source_filename="nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
        shard_count=256,
        polymer="rna",
        expected_num_seqs=None,
        expected_sum_len=76_752_808_514,
        max_sequences=10_000,
    ),
    DatabaseProfileSpec(
        database_id="rfam",
        profile_id="rfam-16-v1",
        source_filename="rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
        shard_count=16,
        polymer="rna",
        expected_num_seqs=None,
        expected_sum_len=138_115_553,
        max_sequences=10_000,
    ),
    DatabaseProfileSpec(
        database_id="rnacentral",
        profile_id="rnacentral-64-v1",
        source_filename="rnacentral_active_seq_id_90_cov_80_linclust.fasta",
        shard_count=64,
        polymer="rna",
        expected_num_seqs=None,
        expected_sum_len=13_271_415_730,
        max_sequences=10_000,
    ),
)

DATABASE_PROFILES_BY_ID = MappingProxyType({
    spec.database_id: spec for spec in DATABASE_PROFILE_SPECS
})
PROFILE_BUILD_MAX_CONTAINERS = len(DATABASE_PROFILE_SPECS)


def resolve_database_profile(database_id: str) -> DatabaseProfileSpec:
    """Resolve one fixed database ID without accepting free-form paths."""
    if not isinstance(database_id, str):
        raise TypeError("database_id must be a string")
    try:
        return DATABASE_PROFILES_BY_ID[database_id]
    except KeyError:
        choices = ", ".join(DATABASE_PROFILES_BY_ID)
        raise ValueError(
            f"Unknown database_id {database_id!r}; expected one of {choices}"
        ) from None


def validate_seqkit_threads(seqkit_threads: int) -> int:
    """Validate the SeqKit/native-helper concurrency argument."""
    if isinstance(seqkit_threads, bool) or not isinstance(seqkit_threads, int):
        raise TypeError("seqkit_threads must be an integer")
    if not 1 <= seqkit_threads <= MAX_SEQKIT_THREADS:
        raise ValueError(
            f"seqkit_threads must be between 1 and {MAX_SEQKIT_THREADS}, "
            f"got {seqkit_threads}"
        )
    return seqkit_threads


def validate_source_policy(source_policy: str) -> SourcePolicy:
    """Validate the post-publication source-retirement policy."""
    if not isinstance(source_policy, str):
        raise TypeError("source_policy must be a string")
    if source_policy not in SOURCE_POLICIES:
        choices = ", ".join(SOURCE_POLICIES)
        raise ValueError(
            f"Unknown source_policy {source_policy!r}; expected one of {choices}"
        )
    return source_policy


def profile_build_slot_budget(
    builder_count: int,
    seqkit_threads: int,
) -> dict[str, int]:
    """Describe the bounded process-worker fanout for one setup run."""
    if (
        isinstance(builder_count, bool)
        or not isinstance(builder_count, int)
        or not 0 <= builder_count <= PROFILE_BUILD_MAX_CONTAINERS
    ):
        raise ValueError(
            "builder_count must be between 0 and "
            f"{PROFILE_BUILD_MAX_CONTAINERS}, got {builder_count!r}"
        )
    threads = validate_seqkit_threads(seqkit_threads)
    source_validator_threads = 1
    return {
        "builder_containers": builder_count,
        "container_cap": PROFILE_BUILD_MAX_CONTAINERS,
        "local_worker_threads_per_builder": threads,
        "overlapping_source_validator_threads_per_builder": (source_validator_threads),
        "maximum_effective_worker_slots": builder_count
        * (threads + source_validator_threads),
    }


def shard_filename(spec: DatabaseProfileSpec, index: int) -> str:
    """Return one fixed AlphaFold-compatible shard filename."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("shard index must be an integer")
    if not 0 <= index < spec.shard_count:
        raise ValueError(f"shard index must be in [0, {spec.shard_count}), got {index}")
    return f"{spec.source_filename}-{index:05d}-of-{spec.shard_count:05d}"


def shard_names(spec: DatabaseProfileSpec) -> tuple[str, ...]:
    """Return every expected shard name in AlphaFold order."""
    return tuple(shard_filename(spec, index) for index in range(spec.shard_count))


def profile_root(sharded_root: Path, spec: DatabaseProfileSpec) -> Path:
    """Return one immutable profile root below the sharded Volume mount."""
    return sharded_root / PROFILE_ROOT / spec.profile_id
