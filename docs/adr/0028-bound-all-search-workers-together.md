# Bound all search workers together

Status: accepted.

The production AlphaFold3 entrypoint replaces `search_chains_in_parallel` and
`max_parallel_data_pipelines` with `max_parallel_search_workers`, defaulting to
4. This Search Worker Budget applies request-wide after duplicate sequences and
validated cache hits are removed.

The MSA phase schedules one Modal worker per missing sequence-by-database
search. Each database worker initially uses Modal CPU `(0.125, 32.125)`, HMMER
`n_cpu=2`, and at most 16 concurrently active shards, preserving the worker
layout selected in ADR 0022. Once the request's MSA phase has completed, the
template phase schedules one separate worker per unique protein that still
needs templates. Upstream template search uses its fixed eight-CPU
`hmmsearch`; it is not sharded.

Both phases use the same worker limit and do not overlap. There is no separate
template-concurrency option initially. The phase barrier and shared budget keep
peak CPU-container fanout predictable without introducing another operational
control. These resource settings do not alter scientific cache identities.
