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

A changed source database generation, shard count, or build recipe requires a
new Profile ID and a corresponding registry and app deployment change. The old
profile remains available so existing Search Identities and cached results
retain their scientific provenance. This trades automatic database promotion
for explicit, reviewable production selection.
