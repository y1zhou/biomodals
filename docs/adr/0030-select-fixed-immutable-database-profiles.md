# Select fixed immutable database profiles

Status: accepted.

Every Supported Database Specification names one code-owned Profile ID.
Production profiles live under `/profiles/{profile_id}/`, with shards below
`shards/`, validation evidence below `validation/`, and `manifest.json`
published last. The profile records the source FASTA identity but does not
retain a duplicate source FASTA.

Searches resolve this fixed registry path directly. There is no mutable
`current` pointer, directory discovery, or automatic promotion. Rebuilding the
identical specification reuses its valid publication and never overwrites it.

During construction, the builder reads the source FASTA from the original
database Volume but writes the shuffled FASTA and its SeqKit index under the
worker's ephemeral `/tmp`. It preflights available local space before starting.
Only generation-scoped raw/final shards and compact validation evidence are
written to `AlphaFold3-msa-db-sharded`.

After `split2` succeeds, the builder deletes the local shuffled payload, then
rewrites and deletes one raw shard at a time. On failure it preserves compact
diagnostics but removes that generation's partial shard payload. Existing
published profiles are never touched by staging cleanup.

A changed source database generation, shard count, or build recipe requires a
new Profile ID and a corresponding registry and app deployment change. The old
profile remains available so existing Search Identities and cached results
retain their scientific provenance. This trades automatic database promotion
for explicit, reviewable production selection.
