# Staged Caches and Durable Publication

Treat every reusable or resumable cache stage as a versioned data contract.

## Identity

Give each reusable stage a semantic identity covering content digests, normalized
record identity, exact upstream code, wrapper/patch version, model/reference
identity, relevant policy, and schema. Exclude result-neutral scheduling knobs
and downstream-only options. Record components in the manifest.

## Readiness contract

- Store a versioned manifest with expected artifacts/counts and format-specific
  facts: sizes/digests, table schema/rows, or shard names.
- Parse and validate manifest schema, identity, and every artifact before reuse;
  existence alone is insufficient.
- Put result-affecting model/reference identity in key and manifest. URLs or
  repository commits do not identify mutable weights/data.

## Single-writer publication

For one cache key:

1. Acquire a lease/coordinator-owned slot before expensive work.
2. Write a unique staging or generation directory.
3. Validate artifacts, close files, and commit the Volume.
4. Publish the completion manifest last and commit again.
5. Make readers `reload()` and revalidate before reuse.
6. Delete transients only after compact evidence and its marker are durable.

A queue distributes work; it does not prevent two builders for one key. Put it
behind one coordinator or lease. Give leases owner, expiry, stale recovery, and
`finally` release.

Never let `force` delete a normal/active entry. Write and return an isolated
generation; update a shared pointer only by policy. Garbage-collect separately.

## Modal Volume v2 semantics

- Give concurrent writers exclusive shard paths and unique sibling temporaries;
  same-file writes are last-write-wins, not locking.
- Writers close and `commit()` before consumption; readers `reload()`.
- Enforce barriers: publish inputs/commit/fan out; children publish distinct
  shards/commit; consolidators reload, verify the expected set, then merge.
- Commit final artifacts and marker before cleanup; do not rely on implicit or
  background commits for correctness.
