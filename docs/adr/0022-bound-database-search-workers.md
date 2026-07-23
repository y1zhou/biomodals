# Bound database search workers

Status: superseded by ADR 0028.

The production AlphaFold3 entrypoint replaces `search_chains_in_parallel` and
`max_parallel_data_pipelines` with `max_parallel_msa_workers`, defaulting to 4.
This MSA Worker Budget applies across every missing sequence-by-database search
for the request after duplicate sequences and cache hits are removed. One
protein sequence can therefore run its four database searches together, while
additional work waits for capacity.

Each database worker initially uses Modal CPU `(0.125, 32.125)`, HMMER
`n_cpu=2`, and at most 16 concurrently active shards, matching the accepted
small-BFD layout and upstream topology. These resource and scheduling settings
are operational and do not alter Search Identity. The CPU floor may be raised
later if measurements show that non-search phases are starved, without
invalidating scientifically identical results.

The worker budget prevents the number of input chains from multiplying
32-CPU-capable containers without a request-wide bound. Chain parallelism is
now an outcome of database-work scheduling rather than a separate user-facing
mode.
