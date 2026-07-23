# Persist only the merged database MSA

Status: accepted.

Each production Raw Database MSA cache entry contains the database-level
`result.a3m`, compact `metrics.json` and `run.log` provenance, and a
digest-validating `done.json` completion marker written last. It lives at
`/{polymer}/{prefix}/{sequence_hash}/raw-msa/{database_id}/{search_identity}/`.

Per-shard `tblout` files are transient worker scratch. The pinned Jackhmmer and
Nhmmer implementations use them to rank and merge shard hits, then discard
them; AlphaFold's cross-database assembly consumes only the resulting A3M.
Retaining shard tables in production would require upstream instrumentation and
additional Volume storage without improving the chosen database-level retry
boundary. Benchmark evidence remains separate and may retain shard tables when
needed for scientific comparison.
