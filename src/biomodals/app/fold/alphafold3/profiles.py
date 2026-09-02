"""Fixed AlphaFold 3 genetic-database sharding profiles.

This module owns production profile identities and pure preparation helpers.
It deliberately contains no Modal objects or application orchestration.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

ALPHAFOLD3_REPOSITORY = "https://github.com/y1zhou/alphafold3"
ALPHAFOLD3_COMMIT = "8f8abfedb88024c631f641e9f8a282e50afb7146"

SOURCE_DB_VOLUME_NAME = "AlphaFold3-msa-db"
SHARDED_DB_VOLUME_NAME = "AlphaFold3-msa-db-sharded"

PROFILE_SCHEMA_VERSION = 2
COMPOSABLE_MULTISET_RECIPE_VERSION = 5
PROFILE_ROOT = "profiles"
DEFAULT_SEQKIT_THREADS = 8
MAX_SEQKIT_THREADS = 32
SEQKIT_VERSION = "2.13.0"
SHARD_RANDOM_SEED = 23
MAX_PROFILE_IMBALANCE = 0.05
BUILD_TIMEOUT_SECONDS = 86_400
PROFILE_BUILD_CPU = (0.125, 32.125)
BUILD_MEMORY_MIB = (1024, 262_144)
PROFILE_STALE_SECONDS = BUILD_TIMEOUT_SECONDS + 900
SCRATCH_ROOT = Path(tempfile.gettempdir())

ORDINAL_SHUFFLER_VERSION = "af3-fasta-two-pass-v2"
ORDINAL_SHUFFLER_PREFETCH_RECORDS = 65_536
ORDINAL_SHUFFLER_PREFETCH_BYTES = 256 * 1024 * 1024
ORDINAL_SHUFFLER_SOURCE_SHA256 = (
    "ea9318fd9382d54321081cc1862fa636cd5a2868d1f1ee40a5d211106c8b6f9c"
)
RECORD_MULTISET_VERSION = "af3-fasta-record-multiset-v1"
RECORD_MULTISET_SOURCE_SHA256 = (
    "52556f6181a42d17a9db19d0ca745b371bf84e57349075c65ce8a911f8ad722e"
)
RECORD_MULTISET_CANONICALIZATION = (
    "full-header-and-sequence-case-sensitive-line-ending-independent-v1"
)
RECORD_MULTISET_AGGREGATE = "sha256-lane-sum-xor-and-square-sum-with-counts-v1"

HMMER_VERSION = "3.4"
JACKHMMER_PATCH_SHA256 = (
    "df9e3ae35ad1659921d96ebfca67a9616a7a467ddde2be18a56f9bd3edb38c41"
)

VALIDATION_RELPATHS = (
    "validation/source-stats.tsv",
    "validation/shard-stats.tsv",
    "validation/shard-summary.parquet",
    "validation/record-multiset.json",
    "validation/duplicate-recovery.jsonl",
    "validation/shuffler-evidence.json",
)


def record_multiset_identity() -> dict[str, str]:
    """Return the canonical-record multiset algorithm identity."""
    return {
        "version": RECORD_MULTISET_VERSION,
        "source_code_sha256": RECORD_MULTISET_SOURCE_SHA256,
        "canonicalization": RECORD_MULTISET_CANONICALIZATION,
        "digest": "sha256",
        "aggregate": RECORD_MULTISET_AGGREGATE,
    }


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
        profile_id="uniref90-256-v1",
        source_filename="uniref90_2022_05.fa",
        shard_count=256,
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
