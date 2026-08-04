# Execution-Kernel Integration

Use `biomodals.execution` for generic durable scheduling. Keep workload science
and host persistence outside it. Read
[ADR 0006](../../../../docs/adr/0006-unified-execution-kernel.md) and the
[scheduler specification](../../../../docs/specs/unified-task-scheduler.md)
before changing statuses, ownership, restart, durability, or coordinator
semantics.

## Choose the execution boundary

- Keep a simple app simple when it has one ordinary remote call and needs no
  durable DAG, fan-out, recovery, or cross-process lifecycle.
- Use the kernel when an app durably schedules multiple Tasks or Provider Calls,
  needs result-driven recovery, or exposes a remotely recoverable direct CLI
  Run.
- Give a top-level direct CLI App Run one remote run-scoped coordinator and
  per-Run ledger. Keep its Local Entrypoint a thin client that validates and
  stages local inputs, targets an exact deployed version, and retrieves results.
- Treat source-backed ephemeral execution as explicit development mode. Do not
  wrap the normal CLI client in another Modal App.
- When a service or workflow calls an app function, keep that call in the
  parent's Execution Run. Do not create a nested coordinator or SQLite ledger.

## Preserve ownership

The kernel owns durable Run, Node, Task, dispatch, Worker Assignment, Provider
Call, Result Envelope, single-submission, and Run-level call-limit mechanics.

The app owns:

- immutable scientific plan and Task construction;
- input parsing, validation, staging, and content digests;
- cache and publication probes returning `available`, `missing`, or `unknown`;
- Modal function names, runtime-image keys, compatibility, and arguments;
- Result Envelope encoding/decoding and Task-specific outcome mapping;
- output paths, manifests, markers, claims, and scientific publication.

The host owns the SQLite location and transaction boundary, request files, and
Volume synchronization. Reuse the existing execution runtime and app-execution
helpers; do not add a workload-handler hierarchy, callback registry, provider
plugin layer, or universal coordinator deployment.

## Schedule and recover safely

- Give each Task a stable Node-local key and normalized scientific payload.
  Exclude concurrency, batching, resources, staging paths, and deployment from
  scientific identity.
- Record publication observations before authorizing work. Only conclusive
  `missing` permits execution; `unknown` stops admission.
- Admit a Task at most once in one Execution Run. `resume` reconciles that Run
  and never retries conclusive failure. Retry through an explicit compatible
  Successor Execution Run.
- Persist Provider Call ownership before spawn and attach the returned call ID.
  Never replace active or outcome-unknown work.
- Let provider redelivery re-execute the same call and Task identity. A worker
  must be idempotent and must not open the coordinator's SQLite database.
- Use fixed-batch dispatch for bounded compatible Tasks and the kernel's SQLite
  pull-worker queue for work stealing. Do not recreate generic scheduling with
  Modal Queue, Dict, file locks, leases, or output markers.
- Keep workload claims when multiple containers may publish the same scientific
  key, but never treat a claim as completion or scheduler authority.

## Cross Volume boundaries deliberately

- Commit a Volume explicitly when another container can act on or consume the
  new state. Reload before that other container reads it.
- Do not commit merely so later code in the same container can see its own
  writes, and do not commit after every file mutation. Modal performs periodic
  commits automatically.
- Close SQLite around a required Volume synchronization boundary. Before a
  provider spawn, claim response, or other cross-container ownership handoff,
  checkpoint all preceding discovery and ownership state together.
