# ADR 0007: API Jobs use deployed Tool coordinators

Status: accepted

Decision date: 2026-08-23

Last amended: 2026-08-27

## Context

The first API-service implementation made FastAPI the Execution Coordinator
for service-owned Jobs and stored kernel Runs, Nodes, Tasks, and Provider Calls
inside `service.sqlite3`. Apps and workflows now have a shared execution
kernel and deployed run-scoped coordinators with their own per-run SQLite
ledgers. Keeping the service as another coordinator would describe the same
scientific execution at two scheduling levels and require synchronization
between them.

The service still needs durable ownership, admission, idempotency, browser
status, cancellation, result delivery, and administration. It does not need to
decide which scientific Task should run next.

## Decision

An API Job submits one top-level Execution Run to the exact deployed
coordinator of its registered Tool. That coordinator and its Remote Run Ledger
are the sole authorities for the Run, Nodes, Tasks, Provider Calls, scheduling,
recovery, and cancellation.

The service stores an immutable Execution Locator consisting of the Execution
Run ID and exact Deployment Identity. It also stores the root coordinator
Function Call ID when submission returns one. The Function Call ID is launch
evidence and a diagnostic rather than the Run's primary identity.

`service.sqlite3` contains service data only:

- Users, Sessions, Service Jobs, idempotency, and admission;
- the Execution Locator and root submission state;
- a bounded Job State Projection;
- result-finalization, cache, download, and cleanup state; and
- Administrator-owned Tool configuration.

It does not contain kernel execution tables or copied Task and Provider Call
records. The Job State Projection contains the most recently observed Run
status, semantic stages, per-stage aggregate Task counts, timestamps, and
observation time. It is a cache for presentation and admission accounting, not
an execution authority.

The existing Remote Run Ledger schema remains authoritative and does not need
service-specific columns. The shared Modal coordinator contract gains bounded
read methods:

- `status()` returns Run and Node state, aggregate Task status counts per Node,
  active Provider Call counts, and representative calls; and
- a cursor-paginated, Node-filtered Provider Call query returns call IDs,
  statuses, and timestamps for logs and diagnostics without returning Task
  payloads or provider bindings.

FastAPI never opens a Tool's SQLite file or treats its Modal Volume path as a
public interface. It reconstructs the exact deployed coordinator from the
Execution Locator and calls the shared coordinator contract.

Scientific execution ends when the remote coordinator reaches its scientific
publication boundary. The service then owns browser archive staging,
download-cache metadata, cleanup, and the `finalizing` and `blocked` Job states.
These delivery states are not added to the execution kernel.

Each Tool has one Administrator-configurable active-Job limit. At admission,
the service snapshots a per-Run total Provider Call ceiling equal to eight
times that limit and a GPU Provider Call ceiling equal to that limit, then
supplies both to the remote coordinator. Later Administrator changes affect
new Jobs only.

GROMACS and AlphaFold3 adopt this topology together. Future coordinator-aware
apps and workflows can be registered through the same service boundary without
adding a service-local scheduler.

### AlphaFold3 publication-scope resubmission

The service keeps the full request digest for HTTP idempotency and separately
stores an AlphaFold3 publication-scope digest. The latter excludes the Job
name and selected model seeds while retaining the biological input, search
choices, recycles, and samples per seed. This lets requests with overlapping
seed publications coordinate without treating presentation changes as new
scientific work. A later Job remains `queued` with
`state_reason=waiting_for_shared_publication` while any earlier matching Job
may still be executing. `state_unknown` is treated as possibly active until
the recorded root Function Call conclusively completes. A completed root with
a durable nonterminal Run does not make that Run failed: `suspended` projects
as recoverable `blocked`, while `state_unknown` remains unknown. Either state
continues to block overlapping publication repair until an explicit refresh or
Administrator action resumes the exact pinned Run and obtains a new root call,
or the Run reaches a conclusive terminal state. No second coordinator is
launched automatically during genuine uncertainty.

After every earlier matching Job is conclusively terminal, the new Job launches
with its own current operational arguments. Failed, partial, and cancelled
predecessor Job IDs are included as narrow workload claim-repair authorization.
AlphaFold3 derives stable generations from the Execution Run ID, Node key, and
Task key, independent of deployment-plan fingerprints, and may fence only
those generations. Unrelated claims remain protected, and validated completion
markers remain the sole reuse evidence.

This is not a kernel Successor and does not copy or reopen a predecessor Remote
Run Ledger. It is an independent root Run with workload-owned repair metadata.
The service stores no Task rows and creates no cross-Run Modal coordination
store. Coordination is intentionally limited to publication-compatible API
requests visible in `service.sqlite3`; independently submitted CLI Runs are not
inferred.

The service calls this user-facing registration a Tool. `tool` replaces
`workload` in its HTTP and persistence vocabulary; Workload remains the
execution kernel's scientific-plan term. Each registration consists of a
static Tool Definition and a narrow Tool Adapter. The adapter validates and
stages scientific requests and retrieves published Results. A shared Remote
Execution Client owns deployment preflight, launch, status, cancellation,
Provider Call discovery, logs, and root-call polling.

Tools retain typed submission routes, while Job inspection, cancellation,
refresh, and download routes remain shared. A Tool Definition maps one or more
kernel Nodes into ordered semantic stages and aggregates their Task counts for
presentation. Raw Nodes and Provider Calls remain diagnostics rather than the
regular user contract.

Service Result metadata is archive-format-neutral and records the filename,
media type, byte size, and digest. GROMACS may therefore publish a ZIP while
AlphaFold3 publishes its established `.tar.zst` without conversion.

AlphaFold3 Regular and Expert modes create the same validated native JSON
resource. Regular Mode constructs an `AF3Config` containing Protein, DNA, RNA,
and simple Ligand entities; Ligands accept CCD codes or a SMILES string. Expert
Mode loads a complete self-contained native AlphaFold3 JSON configuration for
advanced fields such as modifications, covalent bonds, custom CCD definitions,
templates, and embedded MSAs. Job creation consumes that retained validation
without uploading the document again. The backend validates both through the
same existing AlphaFold3 request path and does not maintain a second scientific
input interpretation.

Each Tool has one Administrator-configurable active-Job limit. Admission
derives and snapshots the corresponding total and GPU Provider Call ceilings
as described above. The deployed Modal App name remains startup configuration;
Administrators may pin the exact deployment version and choose Job-log
visibility.

Submission preflights the exact deployment, durably admits a queued Service
Job and its retained Input reference, then returns `202` without waiting for
remote staging or launch. Admission wakes the existing bounded background
reconciler immediately. The configured 60-second interval remains a retry and
repair fallback rather than routine submission latency. This keeps HTTP
acceptance independent of Modal launch time while preserving one launch owner
and the reconciler's four-Job concurrency bound.

## Observation and polling

The background service polls the root Function Call every 60 seconds with
`FunctionCall.get(timeout=0)`. This is a nonblocking provider control-plane
query and does not invoke coordinator code. A completed call returns the final
execution overview and triggers service-owned Result finalization in that same
background pass. One reconciliation pass advances at most four independent
Jobs concurrently. An interactive observation may persist `finalizing` and
return promptly; background reconciliation owns the potentially long archive
work.

Detailed progress is refreshed from `ExecutionCoordinator.status()` only while
a Job is being viewed, when an explicit refresh is requested, or when another
operation requires current remote state. During an active Run this request can
share the already-running concurrent coordinator container. The service avoids
cold-starting a coordinator merely to poll a terminal Job.

Cancellation, resume, and paginated Provider Call diagnostics intentionally
invoke coordinator methods because they read or modify authoritative remote
state.

If the exact pinned deployment cannot be resolved, the Service Job becomes
`state_unknown`. The service does not mark the scientific Run failed, bind to a
newer deployment, or infer execution state from cached projections. Refreshes
for one Job are serialized before updating its local projection so responses
cannot be applied out of order. Reconciliation still polls a recorded root
Function Call without waking coordinator code. A conclusive root result
preserves the durable Run status: `suspended` becomes a recoverable `blocked`
Service Job and `state_unknown` remains `state_unknown`. Background
reconciliation never resumes either state blindly. An explicit authenticated
refresh or Administrator resume action spawns `ExecutionCoordinator.resume()`
on the exact pinned deployment and records the replacement root Function Call
before normal observation resumes.
Before calling Modal, a launch attempt is durably fenced as `state_unknown`;
a process interruption therefore cannot cause an automatic duplicate spawn.
An Administrator may then attach the known root Function Call and resume,
requeue only after confirming that no spawn occurred, or request cancellation.
Launch and cancellation use the same per-Job service lock.

Result restoration is likewise service-owned. Concurrent preparations for the
same Job join one cancellation-shielded restoration task, while different Jobs
remain independent. A rebuilt archive is published only when its size, digest,
and archive schema match the recorded Result identity. A mismatch discards the
candidate and moves the Job to `blocked/result_integrity` without repeatedly
performing the same expensive automatic rebuild. A later explicit preparation
can restore the Job after the exact bytes become available again.

Public failure messages are fixed service copy. Raw coordinator failures,
provider identifiers, and storage paths remain in Administrator-authorized
diagnostics; streamed log records redact the selected provider-call handle even
when it is split across SDK chunks.

## Consequences

- The service can restart or be unavailable without stalling scientific work.
- CLI and API submissions use the same deployed coordinator, recovery rules,
  and execution ledger.
- `GromacsExecutionCoordinator`, service-owned execution tables, local graph
  driving, and async-to-sync Modal scheduling bridges can be removed.
- Job list and admission queries remain local and bounded.
- Detailed progress requires a remote coordinator request when the cached
  projection is stale.
- Each API Tool Run incurs the same small coordinator compute as an equivalent
  deployed CLI run.
- An unavailable pinned deployment temporarily prevents authoritative status
  discovery and is represented as `state_unknown`.

## Superseded decisions

This ADR supersedes the API-service repository row and API-service execution
ownership described by ADR 0006 and `docs/specs/unified-task-scheduler.md`.
Their kernel, CLI, workflow, and remote-ledger decisions remain in force. The
service architecture and scheduler specification must be amended during the
implementation so they no longer describe FastAPI as an Execution
Coordinator.
