# Fanout, Concurrency, and Resumption

Start with local or static fanout. Use the execution kernel for durable
multi-call scheduling, work stealing, and recovery; do not build a second
scheduler inside the app.

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
- For durable work stealing, persist the complete Task set and use the kernel's
  SQLite pull-worker dispatch. Ready Tasks and Worker Assignments are the queue;
  Modal Queue, Dict, locks, and markers are not execution state.

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
- Reuse only validated completion. Publish through a unique temporary and atomic
  replace, then publish the completion marker or manifest last.
- Report cache/publication state as `available`, `missing`, or `unknown`. Only
  conclusive `missing` permits work; `unknown` never permits replacement.
- Reconcile attached work on `resume`. Do not retry conclusive Task failure or
  create a second owner in the same Execution Run; use a compatible Successor
  Run to retry missing work.
- Return compact statuses or counts rather than large deterministic path lists.
- Call `warmup_directory(...)` just before bulk reads, not metadata traversal.
- Preserve failed diagnostics and successful siblings. Consolidate only durable
  evidence; defer/distribute large cleanup until after final publication.

Use Polars for tabular consolidation: prefer lazy scans/streaming sinks or k-way
merge of sorted shards. Add multiprocessing only for measured CPU bottlenecks.
