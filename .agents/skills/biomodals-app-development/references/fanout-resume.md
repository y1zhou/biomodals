# Fanout, Concurrency, and Resumption

Start with static fanout. Add queues only for measured skew, dynamic discovery,
or fine-grained retry.

## Choose the primitive

- Use `map()`/`starmap()` when the caller consumes one-/multi-argument results.
  Use `spawn_map()` only when workers persist results; detached work needs a
  persistent app context and separate consolidation.
- Pass run-specific container fanout, shard sizes, and local worker counts as
  validated function arguments. Keep a broad static CPU range on the function
  when those values only control in-container concurrency; this avoids creating
  a separate function pool for every tuning combination.
- Use `with_options(...)` only when the run needs a distinct Modal resource or
  autoscaling pool, not merely to choose local worker counts.
- Use `@modal.concurrent(...)` only for thread-safe synchronous work or
  nonblocking async work. Do not stack it on a resource-saturating subprocess
  pool.
- Use queues for work stealing, skew, discovery, or retry. Keep a durable task
  manifest: a queue is neither source of truth nor cache lock.

## Set one run-wide budget

Account for nested concurrency across simultaneous branches:

```text
total process slots = Σ(
  branch containers × concurrent inputs/container × local workers/input
  × subprocess fanout/worker
)
```

- Allocate the budget among simultaneous branches before launch; per-function
  caps do not cap a run.
- Keep operational topology out of scientific cache identities. Do not forward
  tuning environment variables through image construction: changing them
  fingerprints the image. Prefer CLI arguments bundled into a small serializable
  execution config, validate them against declared resource limits, reject
  nonpositive values, and reserve headroom.
- Log task counts/batches, container caps, concurrent inputs, local workers, and
  effective slots.
- Batch fine tasks into bounded per-container work. Split along every large
  dimension: reference tiles and candidate batches, not only one of them.
- Include tool-internal threads/helper commands. Measure CPU, memory, storage,
  and queue behavior before raising defaults.

## Make fanout resumable

- Give tasks deterministic IDs and exclusive artifact/marker paths.
- Reuse only validated completion. Publish via unique temporary, atomic replace,
  then marker.
- After fanout, reload, compare expected/completed sets, and retry only missing
  or transiently failed tasks.
- Return compact statuses or counts rather than large deterministic path lists.
- Call `warmup_directory(...)` just before bulk reads, not metadata traversal.
- Preserve failed diagnostics and successful siblings. Consolidate only durable
  evidence; defer/distribute large cleanup until after final publication.

Use Polars for tabular consolidation: prefer lazy scans/streaming sinks or k-way
merge of sorted shards. Add multiprocessing only for measured CPU bottlenecks.
