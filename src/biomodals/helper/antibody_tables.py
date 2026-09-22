"""Typed germline-assignment evidence used by paired and single-domain designs."""

import polars as pl

GERMLINE_EVIDENCE = pl.Struct({
    "reference_id": pl.String,
    "species": pl.String,
    "gene": pl.String,
    "allele": pl.String,
    "accession": pl.String,
    "score": pl.Float64,
    "known_pairs": pl.Int64,
    "known_matches": pl.Int64,
    "known_fr4_pairs": pl.Int64,
    "reference_coverage": pl.Float64,
    "query_coverage": pl.Float64,
    "query_input_start": pl.Int64,
    "imgt_span": pl.List(pl.Int64),
})
GERMLINE_ASSIGNMENT_SCHEMA = {
    "chain_type": pl.String,
    "v_gene": pl.String,
    "j_gene": pl.String,
    "v": pl.List(GERMLINE_EVIDENCE),
    "j": pl.List(GERMLINE_EVIDENCE),
    "diagnostics": pl.List(pl.String),
    "error": pl.String,
}
GERMLINE_TABLE_SCHEMA = {
    "parent_id": pl.String,
    "candidate_id": pl.String,
    "chain": pl.String,
    "sequence_sha256": pl.String,
    **GERMLINE_ASSIGNMENT_SCHEMA,
}
