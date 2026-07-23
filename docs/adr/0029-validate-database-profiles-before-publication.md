# Validate database profiles before publication

Status: accepted.

Database-profile construction is the scientific trust boundary. Before
publishing, the builder verifies source and aggregate-shard SeqKit statistics,
the order-independent `seqkit sum --all` result, recovered duplicate records,
temporary-prefix removal, shard balance, and every declared artifact digest.
It writes `manifest.json` only after those checks pass.

After publication, search workers use the configured sharded database prefix
directly. They may read the small immutable manifest or its digest to identify
the profile, but they do not walk, stat, hash, or run SeqKit over shard
artifacts. A missing or unreadable shard therefore fails the affected search
rather than triggering automatic profile validation.

Full digest revalidation is an explicit Profile Audit, not part of query
execution or cache lookup. This avoids adding a database-scale read before
every search while preserving a deliberate way to check stored artifacts.
