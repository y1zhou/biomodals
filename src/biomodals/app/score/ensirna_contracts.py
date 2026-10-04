"""Canonical ENsiRNA worker plans without app or execution-runtime imports."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnsirnaPdbChunkSpec:
    """One CPU Rosetta PDB preparation chunk."""

    chunk_name: str
    csv_path: str
    json_path: str
    pdb_dir: str


@dataclass(frozen=True, slots=True)
class EnsirnaPreparationPlan:
    """Volume-backed prepared-input contract for ENsiRNA inference."""

    cache_key: str
    prepared_dir: str
    json_path: str
    processed_dir: str
    candidate_count: int
    chunk_count: int
    chunks: list[EnsirnaPdbChunkSpec]
    cached: bool
