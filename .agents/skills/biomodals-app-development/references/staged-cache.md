# Staged Caches and Durable Publication

Treat every reusable or resumable cache stage as a versioned data contract.

## Identity

Give each reusable stage a semantic identity covering content digests, normalized
record identity, exact upstream code, wrapper/patch version, model/reference
identity, relevant policy, and schema. Exclude result-neutral scheduling knobs
and downstream-only options. Record components in the manifest.

## Readiness contract

- Define the minimal authoritative publication for the workload. Store a
  versioned manifest with every artifact/count needed to establish that result,
  plus format-specific facts such as sizes/digests, table schemas/rows, or shard
  names. It need not inventory unrelated debugging or optional outputs.
- Parse and validate manifest schema, identity, and every declared artifact
  before reuse; existence alone is insufficient.
- Put result-affecting model/reference identity in key and manifest. URLs or
  repository commits do not identify mutable weights/data.

## Single-writer publication

For one cache key with concurrent publishers:

1. Let the execution kernel own Task and Provider Call admission. Acquire a
   workload publication claim only when multiple containers may write the same
   scientific key.
2. Write a unique staging or generation directory.
3. Validate artifacts, close files, and publish the completion manifest last.
4. Commit once before another container consumes the completed publication.
   Add an earlier commit only when a real intermediate cross-container
   consumer must observe the artifacts before the manifest is published.
5. Make the consuming container `reload()` and revalidate before reuse.
6. Delete transients only after compact evidence and its marker are durable.

A claim coordinates publishers; it is never completion evidence or permission
to replace active/unknown execution ownership. Do not infer staleness from a
timeout. Replace a predecessor claim only through workload policy after its
owner is conclusively terminal and the Successor Run still observes the
publication as missing.

Never let `force` delete a normal/active entry. Write and return an isolated
generation; update a shared pointer only by policy. Garbage-collect separately.

## Modal Volume v2 semantics

- Give concurrent writers exclusive shard paths and unique sibling temporaries;
  same-file writes are last-write-wins, not locking.
- Use explicit `commit()`/`reload()` only at cross-container visibility or
  ownership barriers. Code in one container sees its own writes without a
  commit, and Modal periodically commits Volume changes automatically.
- Enforce cross-container barriers: publish inputs/commit/fan out; children
  publish distinct shards/commit; consolidators reload, verify the expected
  set, then merge.
- Commit final artifacts and marker before another container consumes them or
  cleanup begins. Do not commit after every same-container mutation.
