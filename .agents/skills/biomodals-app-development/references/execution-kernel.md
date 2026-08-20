# Execution-Kernel Integration

Use `biomodals.execution` for provider-neutral durable orchestration and
`biomodals.execution.modal` for Modal hosting. Keep workload science app-owned.
Read
[ADR 0006](../../../../docs/adr/0006-unified-execution-kernel.md) and the
[scheduler specification](../../../../docs/specs/unified-task-scheduler.md)
before changing statuses, ownership, restart, durability, or coordinator
semantics.

The implemented
[execution-kernel consolidation](../../../../docs/specs/execution-kernel-consolidation.md)
gives apps and workflows one `ExecutionDefinition`. Generic graph, artifact,
runtime, repository, and scheduling behavior belongs to
`biomodals.execution`; Modal request, coordinator, Volume, and CLI-submission
behavior belongs to `biomodals.execution.modal`. The removed
`helper.app_execution` and `workflow.core` packages have no compatibility
aliases.

## Choose the execution boundary

- Keep a simple app simple when it has one ordinary remote call and needs no
  durable DAG, fan-out, recovery, or cross-process lifecycle. That
  uncoordinated path is development-only under the current CLI.
- Use the kernel when an app durably schedules multiple Tasks or Provider Calls,
  needs result-driven recovery, or exposes a remotely recoverable direct CLI
  Run.
- Give a top-level direct CLI App Run one remote run-scoped coordinator and
  per-Run ledger. Keep its Local Entrypoint a thin client that validates and
  stages local inputs, targets an exact deployed version, and retrieves results.
- Treat source-backed ephemeral execution as explicit development mode. Do not
  wrap the normal CLI client in another Modal App.
- A normal deployed `biomodals app run` must declare a coordinator-aware Local
  Entrypoint and expose a Deployment Coordinator Adapter.
- When a service or workflow calls an app function, keep that call in the
  parent's Execution Run. Do not create a nested coordinator or SQLite ledger.

## Preserve ownership

The kernel owns durable Run, Node, Task, dispatch, Worker Assignment, Provider
Call, Result Envelope, single-submission, and Run-level call-limit mechanics.

Map the outer CLI's `max_containers` and `max_gpu_containers` values to the
kernel's `max_active_provider_calls` and
`max_active_gpu_provider_calls`. Use `resolve_provider_call_limits(...)` to
apply workload defaults and validate the GPU-subset invariant. Do not expose
the kernel field names or additional remote-container caps as app-specific CLI
arguments.

The app owns:

- immutable scientific plan and Task construction;
- input parsing, validation, staging, and content digests;
- cache and publication probes returning `available`, `missing`, or `unknown`;
- Modal function names, runtime-image keys, compatibility, and arguments;
- Result Envelope encoding/decoding and Task-specific outcome mapping;
- output paths, manifests, markers, claims, and scientific publication.

The provider host owns the SQLite location and transaction boundary, request
files, and provider-specific durability. Reuse the execution runtime and Modal
host; add scientific behavior through Execution Nodes rather than copying or
subclassing lifecycle loops. Keep deployment-local decorated composition roots;
do not add a universal coordinator deployment.

## Reuse the host lifecycle

Use the supported `biomodals.execution.modal` host interface for run-scoped
persistence, request staging, coordinator driving, restart, and CLI submission.
Use `drive_pull_worker` for pull-worker loops. Apps supply an
`ExecutionDefinition` and scientific Node hooks; they do not assemble stores,
locks, checkpoints, or lifecycle classes.

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
