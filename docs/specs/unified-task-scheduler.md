<!-- markdownlint-disable MD013 -->

# Unified task scheduler refactor plan

Status: accepted architecture; execution topology under review.

This plan consolidates the execution and recovery findings from the API
service, reusable workflow runtime, PPIFlow fan-out, and AlphaFold3 search and
inference pipelines. It proposes a narrow `biomodals.execution` kernel rather
than a new all-purpose `TaskManager`.

The target is one place to reason about:

- fixed DAG construction and readiness;
- runtime-discovered tasks inside fixed semantic nodes;
- single-submission Task state and its relationship to Provider Calls;
- safe submission, attachment, polling, cancellation, and recovery;
- cache observations and the decision to reuse or compute;
- input preparation and output collection boundaries;
- batching and run-level resource budgets;
- reusable direct-fan-out and SQLite-backed pull worker-pool dispatch.

It is not one place to encode every workload's scientific semantics or persist
every kind of state.

## Success definition

The refactor is complete when GROMACS service jobs, reusable workflows,
PPIFlow candidate fan-out, AlphaFold3 search and inference, BoltzGen direct
fan-out, and Rosetta work stealing all use the same execution-state vocabulary
and scheduling primitives without changing their public behavior, scientific
identities, scientific publication layouts, or cost-safety authority. Legacy
execution-ledger and attempt-directory layouts are intentionally replaced.

In particular:

1. A cache hit never starts a provider call.
2. A failed cache check is `unknown`, not evidence that work is missing.
3. Unknown submission or completion state never authorizes automatic paid
   replacement work.
4. Every submitted call is either durably attached or leaves an explicit
   outcome-unknown record.
5. A successful call is not terminally committed before its decoded result and
   workload publication are durably recoverable.
6. Parallel tasks run when their dependencies and resource budget allow it.
7. One provider call may represent a batch of independently identified tasks.
8. Several provider calls may claim work from one SQLite-backed Dispatch Batch
   without opening or writing the repository themselves.
9. Service, workflow, and app CLI entrypoints remain functionally testable;
   internal imports and unfinished schemas carry no compatibility promise.

## Current execution authorities

| Area | Current authority | Reusable strength | Boundary to preserve |
| --- | --- | --- | --- |
| API service | `ServiceStore`, `ModalJobSubmitter`, GROMACS coordinator and plan | API admission, per-Job locking, preclaim-before-spawn, call attachment, cancellation, user-visible operations | User/auth/config state and global service transactions stay service-owned |
| Workflow runtime | `Workflow`, `WorkflowRuntime`, `WorkflowLedger`, `RemoteCallManager` | Static DAG validation, node state, artifacts, per-run recovery, terminal pruning | Per-run ledger and workflow artifact contracts stay workflow-owned |
| PPIFlow | Fixed workflow nodes plus candidate manifests and bounded coordinator loops | Runtime candidate identity, partial outcomes, stage-specific fan-out | Scientific candidate schemas and joins stay PPIFlow-owned |
| AlphaFold3 | Pure search/inference plans, Modal adapters, generation claims, Volume markers | Fine-grained cache identity, multi-writer claims, per-seed reuse, batched inference, publication validation | Markers and validated publications remain scientific completion authority |
| BoltzGen | Incomplete-run discovery, bounded direct fan-out, output-directory locks | One independently reusable run per Task and resumable upstream output | BoltzGen inputs, completion validation, and cross-coordinator publication claims stay workload-owned |
| Rosetta | Modal Queue, multi-pod work stealing, per-pod thread pools | Efficient balancing of many independent CPU Tasks | Rosetta commands, Task payloads, and output validation stay workload-owned |
| Helper layer | `bounded_map` and `batches_for_total_concurrency` | Small deterministic local concurrency helpers | Thread pools are not durable schedulers or global limits |

The duplication is not merely similar function names. Each area independently
answers some combination of:

- what work is ready;
- whether cached work is reusable;
- whether a call may be started;
- how a call is attached and recovered;
- how partial or batched outcomes map back to work items;
- how concurrency is bounded.

The common algorithms should be extracted. The different durability and
scientific ownership rules should not be flattened.

## Confirmed constraints

These existing decisions remain binding during the refactor:

- The semantic DAG is fixed after plan construction. Runtime cardinality
  changes create Tasks inside a Node, not new Nodes.
- A claim coordinates writers; it is never proof of completion. A validated
  marker or publication remains authoritative.
- A Task receives at most one scheduler submission in an Execution Run.
  Retrying failed paid work requires a Successor Execution Run, and active or
  unknown predecessor ownership blocks replacement.
- Only the repository call that creates a durable `submitting` Provider Call
  may invoke spawn. Duplicate or recovered callers never resubmit it.
- Scientific and cache identity excludes operational concurrency, placement,
  and resource allocation.
- A successor requires the same Workload Plan Fingerprint over normalized
  result-affecting inputs and declared scientific versions. Changed science
  requires a new root Run.
- Workflow node parallelism and child-call task budgets are different limits.
- AlphaFold3 raw searches, assemblies, templates, and seeds retain their
  current identities and Volume layouts.
- GROMACS continues to call the deployed app's established functions. The
  service does not rewrite the app or replace its CLI entrypoint.
- PPIFlow keeps a fixed stage DAG while candidate work fans out inside a stage.
- Provider workers never write the coordinator's SQLite repository.
- Ready Task rows and Worker Assignments are the durable pull-work queue.
- A remote SQLite coordinator is routed by Execution Run and pinned deployment
  version to a provider pool capped at one container.
- The coordinator binding ships with each app or workflow deployment; there
  is no universal execution-coordinator deployment or workload registry.
- `biomodals app run` and `biomodals workflow run` target exact deployed
  versions by default; source-backed ephemeral execution is explicit
  development mode without cross-invocation resume.
- A CLI version override is optional. Without one, deployment history is
  resolved once and the resulting exact version is persisted before work.
- An incomplete run never changes Deployment Identity. If its version becomes
  unavailable, it fails with reason `deployment_unavailable`; an explicit
  restart creates a linked successor run.
- Direct CLI App Run ledgers live in a reserved namespace in that deployment's
  configured durable Volume; there is no cross-app execution-state Volume.
- Terminal remote CLI and workflow ledgers have no automatic TTL or garbage
  collection. A future explicit host cleanup may remove terminal execution
  state but never scientific outputs or publications.
- Every Execution Run has a kernel-generated UUID distinct from workload run
  names, scientific identities, Service Job IDs, and user-provided paths.
- Launch commands print the explicit deployment and run identity fields;
  generic lifecycle commands accept those values as repeated CLI flags and
  need no encoded reference, local registry, or app-specific implementation.
- Repeating an app or workflow launch without predecessor identity creates a
  new root Run. `--restart-from <execution-run-id>` explicitly creates a
  successor through the same operation as the generic restart command.
- Every Direct CLI App Run uses that remote coordinator and stores no execution
  database on the user's machine.
- Each remote top-level CLI run has one detached coordinator loop that advances
  it to a terminal state without client polling, then returns so its container
  may scale to zero.
- Child App Calls use their service, workflow, or app-run parent's execution
  repository rather than creating nested execution databases.
- Coordinator Interruption suspends scheduling and preserves attached child
  Provider Calls; only explicit cancellation terminates them.
- Provider redelivery may recover infrastructure interruption, but an
  uncaught coordinator application exception is not automatically retried. It
  leaves the Run incomplete with a visible diagnostic until explicit resume.
- Worker lifecycle callbacks are advisory: they may checkpoint or diagnose an
  interruption but never fail or reassign a Task.
- Provider redelivery may re-execute the same call and Task identity; it never
  creates a second kernel submission. `resume` does not retry failed Tasks.
- No paid Modal calls run in CI.

## Proposed domain model

```text
Service Job (optional API envelope)
  └── Execution Run
        └── Node
              ├── contains zero or more Tasks
              └── Dispatch Batch
                    ├── groups Tasks from this Node
                    ├── records call-bound Worker Assignments
                    └── is served by one or more Provider Calls
```

The three levels answer different questions:

- Node: what semantic stage is running?
- Task: which independently cacheable and verifiable item is being processed?
- Provider Call: where and how is that work executing remotely?

Their strict relationships are:

- a cache hit completes a Task without a call;
- every Task, Dispatch Batch, and Provider Call belongs to exactly one Node;
- one Node may use zero, one, or many Provider Calls;
- one Provider Call may own zero or many Tasks from its Node;
- a Task has at most one durable Provider Call or Worker Assignment owner in
  its Execution Run;
- a Provider Call never spans Nodes, and there is no generic many-to-many
  Task-to-call association;
- GROMACS usually has one Task per Modal call;
- one AlphaFold3 inference worker call may serve several seed Tasks;
- several Rosetta worker calls may steal Tasks from one Dispatch Batch;
- provider redelivery retains the same Provider Call identity;
- a failed Task can be represented again only in a Successor Execution Run.

For example, an `alphafold3-inference` Node might contain one Task per seed.
Changing its GPU concurrency repartitions those Tasks across Provider Calls
without changing the Node, its dependencies, or the Task identities. If every
seed publication is already reusable, the same Node succeeds without creating
any Provider Call.

Provider Calls are internal runtime records rather than workload modeling
objects. Workloads define Nodes and Tasks; the kernel creates calls while
dispatching them. The remote Modal invocation hosting the scheduler is instead
a Coordinator Attempt because it coordinates Tasks rather than executing
them.

### Proposed terms

**Execution Run** is one invocation of an immutable execution plan. An API Job
may own one run, while a CLI workflow or app invocation can create a run
without an API Job.

**Execution Run ID** is a kernel-generated UUID that keys execution state,
coordinator routing, lineage, and ledger location.

**Workload Run Key** is an optional workload-owned name or scientific key in
the immutable plan. It may be reused by successor runs and publications but is
never an execution primary key or ledger path.

**Workload Plan Fingerprint** is a stable digest over normalized
result-affecting inputs and declared scientific tool, model, adapter, and
schema versions. File inputs contribute content digests rather than paths.
Operational concurrency, batching, resource allocation, and Deployment
Identity are excluded.

**Execution Node** is a fixed semantic DAG stage. Its identity does not change
with batching or concurrency. It replaces neither `WorkflowNode` nor
user-facing service stages immediately; adapters map those concepts to it
during migration.

**Task** is the smallest independently identified unit whose cache and outcome
can be reasoned about. Every Task belongs to exactly one Node, and Tasks may be
discovered only when their containing Node starts.

**Single-Submission Rule** means the kernel schedules each Task once and
creates at most one Provider Call submission or Worker Assignment for it in
one Execution Run. Provider redelivery can re-execute that same call, so this
is not an exactly-once execution claim.

**Provider Call** is one concrete remote worker invocation, including its
durable provider call ID and observed lifecycle. It belongs to one Node and
can cover zero or many Tasks from that Node. A Modal Function Call is its
current provider implementation.

**Dispatch Batch** is a durable grouping of Tasks from one Node offered to one
Provider Call or a shared worker pool. Exact Task-to-call attribution is
recorded only when it is observed.

**Worker Assignment** is a durable, call-bound SQLite record electing the
worker allowed to execute one Task from a shared pull work pool. It is
checkpointed before the Task payload is returned.

**Task Claim Request** is an idempotent request for a bounded set of ready
Tasks. Repeating its stable request ID returns the same Worker Assignments.

**Deployment Identity** is the Modal Environment, deployed app or workflow
name, and exact numeric deployment version fixed before run admission.

**Successor Execution Run** is a new, explicitly authorized run linked to a
terminal predecessor. It is the retry boundary for failed work, uses a new
Deployment Identity, reuses validated Workload Publications, and submits only
missing Tasks whose predecessor ownership is conclusively terminal.

**App Run Ledger** is the physical per-run SQLite repository stored at
`.biomodals/execution/runs/<execution-run-id>/ledger.sqlite3` in an app
deployment's durable Volume.

**Publication** is workload-owned durable evidence that a Task's scientific
output is complete. The kernel records the observation but does not prescribe
its file or marker format.

These accepted terms are reconciled in `CONTEXT.md`; adapters should use them
instead of preserving legacy workflow attempt terminology.

### Execution Run statuses

The kernel uses exactly nine Run statuses:

| Status | Terminal | Meaning |
| --- | --- | --- |
| `pending` | No | The immutable plan is persisted and no Task has been dispatched |
| `running` | No | The DAG is advancing or waiting on dependencies, permits, publications, or attached calls |
| `cancel_requested` | No | Explicit cancellation is durable and attached work is being reconciled |
| `suspended` | No | A coordinator application error stopped admission until explicit resume |
| `state_unknown` | No | Provider submission, call state, or cancellation outcome cannot be established; replacement is forbidden |
| `succeeded` | Yes | All required work and publications completed successfully |
| `partial` | Yes | The declared aggregation policy produced a usable partial result |
| `failed` | Yes | Required work conclusively failed or the Run cannot continue |
| `cancelled` | Yes | Cancellation completed conclusively |

Two nullable fields refine the current status:

- `status_reason` is a stable, machine-readable kernel code;
- `status_message` is a human-readable diagnostic and is never control-flow
  input.

A repository transition atomically replaces or clears both fields with
`status`. The first version has a closed, kernel-owned `RunStatusReason`:

| Status | Required `status_reason` |
| --- | --- |
| `suspended` | `coordinator_error` |
| `state_unknown` | `submission_outcome_unknown`, `provider_outcome_unknown`, or `cancellation_outcome_unknown` |
| `failed` | `required_work_failed` or `deployment_unavailable` |
| Every other status | `NULL` |

The repository rejects missing, unknown, and status-incompatible reason codes.
Provider-call `outcome_unknown` projects to Run-level `state_unknown` with
`provider_outcome_unknown`. Task- and Node-specific errors remain canonical on
those records. The Run fields summarize the lifecycle change without copying
stack traces or detailed workload diagnostics. There are no status-specific
reason columns.

The primary transitions are:

```text
pending
  -> running | cancel_requested | suspended | state_unknown
  -> failed | cancelled

running
  -> cancel_requested | suspended | state_unknown
  -> succeeded | partial | failed

cancel_requested
  -> cancelled | state_unknown | succeeded | partial | failed

suspended
  -> running through explicit resume
  -> cancel_requested | state_unknown | failed

state_unknown
  -> running | cancel_requested
  -> succeeded | partial | failed | cancelled
```

Modal preemption does not create a Run transition. Result preparation is an
ordinary running Node. `queued` and `finalizing` may remain service-facing Job
labels derived from the execution projection; `blocked`, `interrupted`,
`retrying`, `expired`, `cached`, and `skipped` are not Run statuses.

### Execution Node statuses

The kernel uses exactly seven Node statuses:

| Status | Terminal | Meaning |
| --- | --- | --- |
| `pending` | No | Waiting for dependencies or Task discovery |
| `running` | No | Discovering, scheduling, or executing Tasks |
| `succeeded` | Yes | The required Node output is complete |
| `partial` | Yes | The Node aggregation policy accepted incomplete output |
| `failed` | Yes | Required work conclusively failed |
| `cancelled` | Yes | Run cancellation stopped the Node |
| `skipped` | Yes | An upstream terminal outcome made the Node unreachable |

Readiness is derived from dependencies and their accepted outcomes; it is not
a persisted `ready` status. A Node satisfied entirely by reusable
publications becomes `succeeded`, while its Tasks retain cache provenance.
Run-level `cancel_requested`, `suspended`, and `state_unknown` are not
duplicated onto Nodes. A workload excludes an optional branch from its
immutable plan rather than creating a Node that is already `skipped`.

### Task statuses

The kernel uses exactly six Task statuses:

| Status | Terminal | Meaning |
| --- | --- | --- |
| `pending` | No | Discovered but not yet cache-satisfied or assigned |
| `running` | No | Durably owned by local execution, a Provider Call, or a Worker Assignment |
| `succeeded` | Yes | The required Workload Publication was validated |
| `failed` | Yes | Execution or publication validation conclusively failed |
| `cancelled` | Yes | Explicit Run cancellation stopped the Task |
| `skipped` | Yes | The Node's `fail_fast` policy stopped it before admission |

Provider submission and attachment phases belong to the Provider Call rather
than the Task. If the owner's outcome becomes unknown, the Task remains
`running`, the owner retains it, and the Run becomes `state_unknown`; the Task
does not become eligible for replacement work. Task success records whether it
came from cache validation or execution as provenance, not as a `cached`
status. Individual Tasks are never `partial`; the Node aggregation policy
derives partiality.

## Responsibility boundary

| Concern | Execution kernel owns | Workload or host owns |
| --- | --- | --- |
| DAG | Validation, topological readiness, terminal reachability | Nodes, dependencies, semantic labels |
| Task planning | Immutable task records, fingerprints, dependency links | Task discovery and scientific identity payload |
| Cache | `available` / `missing` / `unknown` vocabulary and scheduling policy | Validation logic, markers, manifests, content checks |
| Inputs | Calling preparation hooks and recording normalized fingerprints | Parsing, validation, staging, provider kwargs |
| Calls | Claim, submit, attach, resolve, poll, cancel, recover state machine | Function selection and provider adapter binding |
| Dispatch | Durable batches, direct fan-out, pull claims, worker-call tracking, returned outcome routing | Task payloads and batch compatibility |
| Outputs | Calling decode/validate/publish hooks and committing outcome ordering | Schemas, scientific validation, paths, publication |
| Batching | Mapping Tasks to call batches and distributing outcomes | Batch compatibility and workload-specific limits |
| Resources | Coordinator-scoped run budget and persisted permit accounting | Service admission, Modal resources, deployment limits, cross-coordinator policy |
| Persistence | State schema, legal transitions, and atomic repository operations | Repository location, transaction integration, Volume synchronization |
| Presentation | Stable snapshots/events for adapters | HTTP Jobs, CLI output, timelines, logs, admin policy |

“Centralized input/output handling” therefore means one lifecycle for invoking
typed workload hooks. It does not mean moving PDB, A3M, Parquet, archive, or
scientific result parsers into the scheduler.

## Minimal kernel shape

Add the package incrementally. Do not scaffold empty modules in advance.

```text
src/biomodals/execution/
  __init__.py             # deliberately small supported surface
  model.py                # immutable plan and state value objects
  plan.py                 # graph validation and readiness
  availability.py         # tri-state observations
  sqlite.py               # schema and transitions on a host connection
  ports.py                # provider and workload protocols
  runtime.py              # composition facade after primitives stabilize
  modal.py                # reusable remote-coordinator mechanics, no app globals
  _internal/
    scheduler.py          # ready-node/task selection
    submission.py         # paid-call lifecycle
    batching.py           # task-to-call grouping
    resources.py          # budgets and permit accounting
```

The initial internal interface should be no larger than:

- `ExecutionPlan`
- `NodePlan`
- `TaskPlan`
- `TaskResult`
- `ExecutionRuntime`

Each composition root supplies its SQLite connection and transaction,
provider implementation, and workload implementation explicitly. Do not add
global registries, plugin discovery, YAML workflows, or import-time Modal
app or Volume bindings. Each app and workflow declares its own thin decorated
coordinator wrapper over `execution.modal`.

### Coordinator loop model

The pure transition and readiness functions are shared. The API service keeps
an async coordinator loop around them; remote workflow and app-run
coordinators use a sync loop. Local Entrypoints are thin clients of those
remote coordinators rather than execution hosts. By default they address a
named deployment version; local source-backed execution is an explicit
development path. Maintaining two small host loops is preferable to infecting
every consumer with an async abstraction or running nested event loops.

Each remote top-level CLI run submits one detached coordinator-loop input. The
loop reconciles durable state, schedules ready work, observes attached calls,
and advances the DAG until the Execution Run becomes terminal. The launching
CLI may observe it but need not stay connected or poll to make progress.
Lifecycle methods and pull-worker claims or completions may enter the same
Run-Scoped Coordinator Pool concurrently; they all use its serialized writer.
After preemption, a replacement Coordinator Attempt reloads the ledger and
continues reconciliation. After a terminal transition, the loop returns and
the container may scale to zero. A later `status` call may start a fresh
container and read the retained ledger. “Coordinator loop” is an internal
activity, not a public `drive` CLI command or a new kernel domain type.

Provider redelivery of an interrupted coordinator input is recovery, not an
application-level retry. An uncaught exception from coordinator code stops
admission, leaves attached Provider Calls running, and records its diagnostic
when possible. It sets the Run to `suspended`; the adapter does not
automatically retry that exception. Explicit `resume` reloads the durable
ledger, reconciles attached calls, and then returns the same Execution Run to
`running`.

A workload or Provider Call failure is different: it terminally fails the
affected Task, and the Node aggregation policy derives the Node and Run
outcome. The coordinator reports that outcome and returns when the DAG is
terminal. It does not convert a known Task failure into a resumable coordinator
error.

Every app and workflow deployment exports its thin class under the standard
name `ExecutionCoordinator`. Its version-pinned, run-parameterized instance
provides the common `status`, `cancel`, and `resume` lifecycle methods.
`restart` resolves a new Deployment Identity and creates a new coordinator
instance linked to the predecessor. Workload launch and result-retrieval
methods may remain deployment-specific.

`status` is read-only, `cancel` is an idempotent explicit cancellation,
`resume` retains Execution Run ID and Deployment Identity, reconciles existing
work, and may submit Tasks that were never submitted, but it never retries a
failed Task. `restart` always returns a new Execution Run ID and Deployment
Identity and is the only retry boundary. Passing those fields does not grant
paid-work authorization; the CLI still performs its normal Modal
authentication.

Repeating `biomodals app run` or `biomodals workflow run` without predecessor
identity always creates a new root Execution Run. The kernel does not infer
identity from matching command text, file paths, Workload Run Keys, or
normalized arguments, and it maintains no command-fingerprint catalog. Both
launch commands accept `--restart-from <execution-run-id>` as a convenience
over `biomodals run restart`; it invokes the same successor operation rather
than mutating the predecessor.

A Successor Execution Run revalidates each expected Workload Publication.
Valid successes become cache-satisfied Tasks. Missing or conclusively invalid
publications are eligible for new Tasks only when predecessor ownership is
conclusively terminal; active or unknown ownership blocks replacement. Once
those Tasks succeed, previously untouched downstream Nodes become ready
normally.

Generic restart reuses the predecessor's stored immutable scientific plan.
Launch-time `--restart-from` normalizes the supplied workload inputs and
requires its Workload Plan Fingerprint to equal the predecessor's before
creating successor state. Changed file content, seeds, scientific parameters,
or declared result-affecting versions require a new root Run. Operational
concurrency, batching, resource settings, and the newly resolved Deployment
Identity may differ.

Deployment Identity is not scientific identity by itself. A new deployment's
workload adapter must accept the stored plan schema and preserve every
declared result-affecting tool, model, adapter, and schema version before the
successor may reuse publications. Otherwise restart is rejected and the user
must create a new root Run.

SQLite transactions and repository constraints are expected to turn an
interruption into a valid checkpoint, not broken partial rows. Recovery may
therefore encounter a valid `submitting` or `outcome_unknown` call and must
preserve its ownership. An unreadable database or invariant-invalid
predecessor fails closed: neither resume nor restart reconstructs ownership
from partial rows or scientific outputs alone. Publications remain reusable
after an operator separately resolves the risk of active provider work.

### Durable execution state

Every coordinator that promises restart or recovery requires an Execution
State Repository. Centralization applies to the state model, transition rules,
and persistence operations, not to one database process or file for all
Biomodals activity. Production CLI app runs are durable remote runs; only
tests and dry-run planning may use a transient in-memory repository.

Repository scope follows the coordinator boundary:

| Coordinator | Repository scope | Execution authority | Separate authority |
| --- | --- | --- | --- |
| API service | One long-lived `service.sqlite3` for every service-owned Job and Execution Run | Service-coordinated Nodes, Tasks, call IDs, and observed state | Users, Jobs, admission, runtime configuration, and result cache remain service-owned |
| Workflow orchestrator | The existing per-run Workflow Ledger | Workflow Nodes, fan-out Tasks, calls, and recovery | Workflow artifacts and Volume synchronization remain workflow-owned |
| Direct CLI app coordinator | One App Run Ledger in the app deployment's configured durable Volume | App Nodes, Tasks, batches, assignments, child calls, and recovery | Workload publications, scientific inputs, and outputs remain app-owned |
| Child App Call | No separate repository; use the parent Execution Run | Work attributed to the service, workflow, or Direct CLI App Run | Function implementation, resources, and scientific publication remain app-owned |

An ordinary API request therefore updates only the service database. It does
not create a database for the called app. If the service starts a remote
workflow orchestrator, the service repository tracks that child coordinator
call and the workflow's existing per-run ledger tracks the internal DAG. Those
repositories describe different scheduling levels; they do not duplicate the
same Tasks.

A Direct CLI App Run always creates its per-run ledger remotely and executes
through a Run-Scoped Coordinator Pool, even for one direct Provider Call. The
Local Entrypoint owns no SQLite file: it prepares local input, submits the run,
and optionally waits for or retrieves the result.

A workflow CLI run follows the same deployment rule and uses its remote
Workflow Orchestrator and per-run Workflow Ledger. Both CLI commands pin an
exact deployed version before admitting work. An explicit Development CLI Run
may execute current source through an ephemeral Modal app, but it cannot later
claim the durable resume semantics of a Deployed CLI Run.

Each app Deployment Coordinator Adapter binds its App Run Ledger to a reserved
`.biomodals/execution/runs/<execution-run-id>/ledger.sqlite3` path in that
app's configured durable Volume, normally its existing output Volume. The
workflow adapter keeps using the workflow orchestrator Volume and likewise
keys its physical run root by Execution Run ID rather than a user-supplied
name. The kernel receives a connection and explicit Volume synchronization
boundary from either host and never imports a global execution Volume.

The same app invoked from an API service or workflow is instead a Child App
Call. Its Tasks live in that parent coordinator's repository, so the child
does not create a duplicate ledger. A complex app such as AlphaFold3 exposes
its plan and workload hooks to the owning coordinator; its markers and
validated publications remain scientific authority in every call shape.

Provider workers do not write SQLite. A single coordinator applies
transitions and commits them using its transaction and Volume synchronization
boundary. Remote pull workers submit idempotent claim and completion commands
to that coordinator instead of mounting or opening the ledger.

The accepted implementation is one `SqliteExecutionRepository` for
single-writer durable coordinators. The host supplies the SQLite connection
and transaction; the repository does not select the database path, commit,
close, or synchronize a Modal Volume. This lets service admission and initial
execution state be committed atomically, while a workflow can commit
execution rows together with its Volume-backed ledger.

The first implementation has no persistence protocol. Tests exercise the
production repository through an in-memory SQLite connection. If a second real
backend later appears, its requirements provide evidence for extracting a
smaller, proven persistence interface instead of guessing one now.

### Coordinator interruption and checkpoints

An Execution Coordinator is a logical scheduling authority that may span
several Coordinator Attempts. Modal preemption, infrastructure loss, or a hard
container shutdown ends an Attempt; it does not cancel the Execution Run or
its attached child Provider Calls.

A graceful interruption handler performs best-effort draining:

1. stop admitting new Tasks and Dispatch Batches;
2. finish or roll back the current SQLite transaction;
3. close or checkpoint SQLite and cross the host's explicit durability
   boundary;
4. leave attached child Provider Calls running;
5. exit without projecting the Run as cancelled.

A replacement Attempt reloads the latest durable repository, reconstructs
active permits and batches, resolves attached calls by ID, and resumes
observation. Correctness must also survive a hard kill before the handler runs
or finishes. Lifecycle hooks and background Volume commits reduce lost
progress but are not correctness boundaries.

For a Volume-backed repository, a local SQLite commit is not sufficient when
ordering state against a provider side effect. Required preclaims and call
attachments must be made visible through an explicit, serialized Volume
checkpoint. The current workflow exit behavior that cancels active child calls
must be removed when it adopts this policy. Explicit user cancellation remains
separate and may cancel those calls.

Different Run-Scoped Coordinator Pools use distinct SQLite files. Their
deployment-specific Modal Volume v2 may accept concurrent commits to those
different paths, while the one-container pool cap prevents concurrent access
to one ledger file. A coordinator closes or checkpoints SQLite before a Volume
reload or commit and never places scientific outputs inside the reserved ledger
namespace.

Exactly one process may write one Volume-backed repository at a time. A remote
coordinator enforces this through a run-scoped provider pool:

1. the immutable Execution Run ID and pinned containing app or workflow
   deployment version identify a parameterized coordinator pool;
2. every unique parameter tuple has its own container pool, capped at one
   coordinator container;
3. concurrent coordinator-loop, claim, completion, and observation inputs
   route to that container;
4. method handlers submit commands to one in-process writer loop rather than
   opening transactions themselves;
5. the writer serializes SQLite transactions and explicit Volume checkpoints;
6. a replacement container reloads the last checkpoint after preemption;
7. different Execution Runs use independent pools and may proceed in parallel.

Stable request IDs make duplicate control inputs idempotent in SQLite. The
provider's per-parameter pool isolation, one-container cap, and replacement
behavior are correctness assumptions and require a manual Modal smoke test
before adoption. The API service keeps its existing process-level
single-writer exclusion. Direct CLI App Runs use the remote wrapper even when
they need no pull-worker RPC, so production app execution never creates a
local run database.

### Worker interruption and assignment recovery

Worker recovery is driven by provider-call state and publication validation,
not elapsed time:

- preemption or a rescheduled container retains the same Provider Call, Task,
  permit allocation, and Worker Assignment;
- a normal result may carry independent per-Task outcomes for a batch;
- a conclusive terminal call failure fails its unfinished Tasks and releases
  their permits;
- a successful call with a missing or invalid expected publication fails the
  affected Task;
- unknown call state preserves `state_unknown` and forbids replacement work.

Direct one-Task-per-call execution needs no separate remote assignment store.
For a pull Dispatch Batch, ready Task rows are the queue. A worker sends
`claim(worker_id, capacity, request_id)` to the coordinator. In one serialized
operation, the coordinator:

1. returns an existing result if `request_id` was already processed;
2. selects ready Tasks that fit the worker's capacity;
3. records their Worker Assignments and permit allocation;
4. commits SQLite and crosses the explicit Volume durability boundary;
5. only then returns the Task payloads.

A lost response can therefore be requested again without creating another
assignment. If the coordinator dies before the checkpoint, no payload has
been returned; if it dies afterward, the replacement coordinator recovers the
same assignments. A restarted provider input repeats its claim request and
retains its work. A conclusively failed owner call fails its unfinished Tasks;
no different Provider Call may claim them in the same Execution Run.

Workers publish their outputs before sending an idempotent completion report.
The coordinator records individual Task outcomes and releases permits in one
serialized transaction. A lost completion response is harmless, and
publication validation reconciles output that became durable before its
completion report.

Provider redelivery is not a second kernel submission. A claim request may
repeat automatically before or after a Worker Assignment is committed, and
Modal may restart or retry the same provider input without creating another
Provider Call record. Once a paid call is conclusively terminal and its Task
is failed, the kernel cannot submit that Task again in the same Execution Run.
`resume` only reconciles attached or unknown calls and schedules ready Tasks
that have never been submitted.

An explicit `restart` builds a Successor Execution Run under a newly resolved
Deployment Identity and Execution Run ID, records the predecessor Execution
Run ID, and reuses the Workload Run Key. It revalidates every expected
Workload Publication before creating Tasks. Satisfied work remains complete;
missing work becomes eligible only when the predecessor Provider Call or
Worker Assignment is conclusively terminal. Active or unknown predecessor
ownership blocks replacement work. The successor copies no mutable Task,
assignment, call, or permit state.

An exit hook may commit workload checkpoints and emit an advisory
`interrupted` event containing the call, slot, and Task IDs. The coordinator
does not treat receipt as failure, and correctness does not depend on receipt.
In particular, an exit hook must not remove a BoltzGen output claim or requeue
a Rosetta Task while Modal may restart the same input.

### Host-invariant execution state

Physical colocation does not make service metadata part of the execution
model. Dependencies point toward execution IDs only:

```text
Service Job ───────────────┐
                          ├──> Execution Run -> Node -> Task
Workflow Artifact Store ──┘                    ├-> Dispatch Batch
                                              └-> Provider Call

Task --optional direct ownership---------------> Provider Call
Task -> Worker Assignment ---------------------> Provider Call

Execution tables --X--> Service Job, user, HTTP, admin, or artifact tables
```

Execution tables may store only data needed to reconstruct and manage actual
work:

- stable run, node, task, batch, assignment, and call identifiers;
- an optional predecessor Execution Run ID for explicit restart lineage;
- the immutable Deployment Identity used to recover provider calls;
- the immutable Workload Plan Fingerprint and Task fingerprints;
- dependency edges and legal execution states;
- submission tokens, provider targets, call IDs, and observed outcomes;
- execution timestamps, Run `status_reason` and `status_message`, Task and Node
  errors, single-submission state, and resource permits;
- workload execution payloads required to reconstruct a Task.

They must not store:

- service user or Job ownership;
- display names, UI labels, or HTTP state;
- administrator configuration or the source of a resolved provider setting;
- result-download policy or service cache-management fields;
- workflow artifact schemas, selectors, or presentation metadata;
- arbitrary host `metadata_json` bags.

The service owns a one-way `execution_run_id` reference from a Service Job.
Workflow artifact rows may similarly refer to producing execution IDs.
Resolved provider bindings may be recorded because they determine actual
execution, but the kernel is invariant to whether those values came from an
administrator setting, CLI option, or workload default.

### Service Job projection

The service does not mirror compute state. Admission uses one host-owned
transaction to:

1. create an Execution Run and its initial plan through
   `SqliteExecutionRepository`; and
2. create a Service Job containing a distinct `job_id` and one-way
   `execution_run_id`.

The Service Job owns only service concerns such as:

- owner and authorization scope;
- request idempotency and submitted parameters;
- selected workload and user-facing display data;
- admission and configuration snapshots;
- input-upload references;
- result location, checksum, size, retention, and download metadata;
- service audit timestamps that are not compute transitions.

The Execution Run owns all actual work, including local result preparation,
provider submission, cancellation progress, unknown outcomes, and terminal
compute status. The service retains `JobState` as an HTTP/OpenAPI enum only and
derives it at read time from the Execution Run plus service result-delivery
facts. Workload presentation code maps stable Node and Task keys to labels and
timeline rows without writing those labels into execution tables.

The same projection is used for:

- Job detail and list endpoints;
- per-column Job filtering and sorting;
- active-job admission limits;
- administrator running-job counts;
- cancellation eligibility;
- stage timelines and Running Function values.

Do not add a trigger or materialized state column until query evidence shows
that the indexed join is insufficient. If a read projection is cached later,
it is disposable and never becomes a second authority.

### Resource ownership

The first kernel manages resources only inside one Execution Run owned by one
coordinator. Its SQLite state records:

- the resolved Run-Level Task Budget;
- permit cost for an admitted Task or Provider Call batch;
- active permit allocations tied to Tasks and Provider Calls;
- release or recovery of allocations when calls become terminal or unknown.

Permit allocation is atomic with the transition that admits work, so a
coordinator restart can reconstruct the active count from durable execution
rows. Batching policy decides whether permits account for a container, a Task,
or another workload-defined unit, but the meaning is fixed in the immutable
Task plan before submission.

The kernel does not own:

- per-user, per-tool, or service-wide active-Job admission limits;
- the administrator settings from which a service resolves a Run budget;
- Modal CPU, GPU, memory, timeout, accelerator, or deployment concurrency;
- limits shared across different coordinators or Execution Runs.

No shared-lease interface or Modal Dict implementation is added in this
refactor. If a future workflow truly needs a hard limit across multiple
coordinators, that concrete requirement must define the failure and recovery
semantics before introducing another storage seam. The existing ADR principle
still applies: a hard distributed limit cannot be implemented by pretending
separate in-process counters are global.

### Workflow Ledger decomposition

The existing physical workflow `ledger.sqlite3` file remains useful. The
existing `WorkflowLedger` class and its generic execution schema should not
remain as a parallel implementation.

The target split is:

| Current Workflow Ledger concern | Destination |
| --- | --- |
| Execution columns from `runs`, `nodes`, and `remote_calls` | Shared execution SQLite schema |
| `attempts`, `current_attempt_id`, attempt foreign keys, and attempt counters | Delete; move the one retained result or error to Task, Node, Provider Call, or artifact state |
| Run, Node, Task, Dispatch Batch, Worker Assignment, and Provider Call transitions | Shared execution repository implementation |
| `RunStatus`, `NodeStatus`, placement, and recovery policy | Shared execution models imported directly after cutover |
| `NodeExecutionPolicy`, `AttemptRecord`, and `NodeStatusRecord.attempts` | Delete; provider redelivery and Successor Execution Runs replace generic rerun policy |
| `artifacts`, `artifact_files`, `node_inputs`, and `node_outputs` tables | Workflow-specific run store |
| `WorkflowArtifact`, `ArtifactSelector`, and materialized `AppRunResult` handling | Workflow-specific artifact module |
| Run-root directories, node/task output paths, connection closure, and Volume synchronization | Workflow-specific run store |
| Finalizing execution state and artifacts together | One host-owned SQLite transaction spanning both implementations |

This leaves one SQLite file per workflow run, not an execution database beside
a workflow database. The file contains shared execution tables and
workflow-specific artifact tables on the same connection.

The kernel implementation and its tests are built alongside the current
workflow runtime, but the current `WorkflowLedger` is never adapted into a
compatibility facade and never dual-writes the new schema. One direct cutover
commit switches the workflow composition root to
`SqliteExecutionRepository` plus a narrow `WorkflowArtifactStore`, deletes the
old execution methods and attempt model, and rejects old unfinished ledgers.
The deletion test is that removing the old `WorkflowLedger` class must not
redistribute generic SQL or transition logic back into workflow callers.

The service follows the same pattern. The user-facing Service Job points to an
Execution Run. `job_operations`, persisted `JobOperationState`, and persisted
compute `JobState` are replaced by shared Execution Nodes, Tasks, and Provider
Calls. Service projections retain the existing HTTP state and timeline
vocabulary without preserving a second operation state machine.

### Paid-call lifecycle

The kernel uses exactly eight Provider Call statuses:

| Status | Terminal | Meaning |
| --- | --- | --- |
| `submitting` | No | The durable preclaim exists and spawn has not safely returned |
| `attached` | No | The provider call ID is durable but the call has not yet been observed |
| `running` | No | The provider reports that the call is active |
| `outcome_unknown` | No | Spawn may have occurred, but no provider call ID was durably attached |
| `state_unknown` | No | The call ID exists, but state or cancellation outcome is inconclusive |
| `succeeded` | Yes | The provider call returned successfully |
| `failed` | Yes | The provider conclusively reported failure |
| `cancelled` | Yes | The provider conclusively confirmed cancellation |

The primary lifecycle is:

```text
submitting
  -> attached | outcome_unknown | failed

attached
  -> running
  -> succeeded | failed | cancelled | state_unknown

running
  -> succeeded | failed | cancelled | state_unknown

outcome_unknown
  -> attached | running | succeeded | failed | cancelled

state_unknown
  -> running | succeeded | failed | cancelled
```

`outcome_unknown` means a spawn may have started but no call ID was durably
attached. `state_unknown` means an attached call exists but its current or
terminal provider state cannot be established. Both preserve Task ownership
and prohibit replacement work until explicit reconciliation establishes one
of the listed transitions.

There is no `planned` Provider Call: unsubmitted intent remains on the Task or
Dispatch Batch, and the durable preclaim creates a call directly in
`submitting`. There is no `expired` status either. An expired provider handle
with conclusive failure becomes `failed`; without a conclusive outcome it
becomes `state_unknown`, with the expiry retained as diagnostic reason.

The preclaim operation atomically creates the call, assigns its Tasks, and
returns whether this caller created the row. The creating caller must then
cross the host's durability boundary: a service transaction commits, while a
Volume-backed host checkpoints the committed SQLite file. Only after that
boundary succeeds does the creation result become a one-time in-process
authorization to invoke spawn. Duplicate request IDs, concurrent commands,
and later coordinators observe the existing call and perform no provider side
effect. The authorization is not persisted as a lease and cannot be recovered.

Provider resolution, version checks, normalized input preparation, and other
failure-prone work happen before preclaim. Once preclaim crosses the host
durability boundary:

- a returned provider call ID is attached and checkpointed;
- a conclusive provider rejection makes the call and unfinished Tasks
  `failed`;
- an ambiguous spawn exception makes the call `outcome_unknown`;
- recovery of an abandoned `submitting` row also makes it
  `outcome_unknown`, because the coordinator cannot prove spawn never began.

No path automatically invokes spawn again for that Provider Call. Failed Tasks
can run again only in a Successor Execution Run; unknown ownership must first
be resolved conclusively.

A provider adapter must expose only the operations the kernel needs: spawn,
resolve by call ID, observe or collect, and cancel. Modal-specific objects
must not leak into plans or persisted models.

### Cache and publication lifecycle

Every reuse decision returns exactly one observation:

- `available`: validated publication can satisfy the Task;
- `missing`: validation authoritatively established that no reusable
  publication exists;
- `unknown`: the checker failed, the storage was unavailable, or absence could
  not be established.

Only `missing` authorizes new work. After a call succeeds, the decoded result
and workload publication are committed before the Task and call are made
durably terminal. If a store cannot make those changes in one transaction, its
adapter must use a recoverable prepare/publish/finalize protocol.

### Failure modes

Nodes declare one of three workload-selected aggregation policies:

- `fail_fast`: stop admitting sibling Tasks after the first terminal failure;
- `collect_all`: allow all admitted Tasks to finish and report every outcome;
- `allow_partial`: publish an explicit partial result that downstream nodes
  must opt into.

These policies do not authorize another submission. A failed Task remains
failed for that Execution Run; retry requires an explicit Successor Execution
Run.

## Correctness work before extraction

Do not copy current runtime behavior into the kernel until these gaps have
characterization tests:

1. `RemoteCallManager.run_node()` spawns before recording a call ID. A crash in
   that interval can leave untracked paid work. Introduce a durable submitting
   claim before spawn and preserve an unknown outcome rather than rerunning.
2. The workflow runtime marks a call succeeded before the processed
   `AppRunResult`, artifacts, and node completion are durably finalized. A
   crash in that interval can cause another call. Finalize these under the
   existing Volume synchronization boundary.
3. `check_artifact_availability()` maps external checker exceptions to
   `missing`, contrary to the accepted tri-state ADR. Map them to `unknown`.
4. `CONTEXT.md` says `WorkflowLedger` records fan-out tasks, but its schema
   does not yet contain task rows. Keep the term marked planned until the
   durable-task phase implements it.
5. `WorkflowOrchestrator.exit()` currently cancels active remote calls during
   every container exit. Separate interruption from explicit cancellation and
   preserve attached calls for replacement-coordinator recovery.
6. Rosetta removes a Modal Queue item before any durable Worker Assignment, so
   a hard interruption can lose delivery. Replace that transport with an
   idempotent coordinator claim that commits and checkpoints the assignment
   before returning the Task payload.
7. BoltzGen removes its output lock from `@modal.exit`, which can expose the
   same output to another worker while Modal restarts the interrupted input.
   Exit must preserve call-bound ownership.

Fault-injection tests must stop immediately after preclaim, spawn, attachment,
collection, decode, publication, and final commit. Each restart assertion must
prove whether the old call is recovered, the run blocks as unknown, or a
Successor Execution Run may safely submit missing work.

## State-transition policy

This pre-release refactor does not carry old execution history into the new
model.

For the API service:

- preserve `users`, password and session data, `service_settings`, and
  `workload_settings`;
- recreate the Service Job table around its one-way `execution_run_id`;
- remove `job_operations` and all persisted compute-state columns;
- create the shared execution schema at the current kernel version;
- discard old Job history, result-cache rows, and unfinished execution state;
- leave remote Modal Volumes and workload publications untouched.

The transition is an explicit offline CLI operation, not an automatic
destructive startup migration. It stops if the source schema is unexpected,
performs the service-table preservation and execution-state replacement in a
transaction, and records the new schema version. No compatibility reader,
dual-write path, or automatic backup is required.

For workflows:

- write an execution-schema version into each new physical Workflow Ledger;
- reject an older ledger with a clear instruction to restart or force the run;
- do not migrate old Nodes, attempts, calls, or artifact rows;
- retain app-owned Volume outputs so workload validators can reuse valid
  scientific publications in the restarted run.

Local staged API result files left by discarded Job rows are service cache, not
kernel state. Clean them through the existing administrator cache-management
path rather than teaching the execution migration about service files.

## Migration plan

Each phase is reviewable and reversible as a Git commit. Internal predecessor
code may be replaced directly once scientific, cost-safety, and recovery tests
pass; do not maintain dual schemas or runtime compatibility switches.

### Phase 0 — accept boundaries and freeze behavior

Deliverables:

- accept or revise ADR 0006 and the proposed glossary;
- add pure characterization tests for GROMACS graph readiness, workflow
  recovery, PPIFlow candidate outcomes, and AlphaFold3 plan identities;
- add fault-injection fixtures using fake provider calls and in-memory SQLite;
- record scientific identities, publication markers, cost-safety behavior,
  CLI operations, and user-visible results as regression fixtures.

Exit gate:

- every subsequent phase can prove scientific and cost-safety invariants
  without Modal access.

Rollback:

- documentation and test-only changes can be reverted without state migration.

Phase 0 test inventory:

| Test location | Required case |
| --- | --- |
| `tests/workflow/test_artifacts.py` | An external checker exception returns `unknown`, never `missing` |
| `tests/service/test_submission.py` | Pre-preclaim failure leaves work unowned; one preclaim authorizes one spawn; definite post-preclaim rejection fails; unknown or abandoned submission blocks; an unattached returned call is cancelled and marked unknown |
| `tests/workflow/test_runtime.py` | A crash after durable claim but before attachment does not submit replacement work |
| `tests/workflow/test_runtime.py` | A recorded call ID is resolved and collected instead of resubmitted |
| `tests/workflow/test_runtime.py` | A crash after collection, decode, publication, or final commit never starts another provider call |
| `tests/workflow/test_runtime.py` | Graceful and hard coordinator interruption preserve attached calls and recover their permits |
| `tests/workflow/test_orchestrator.py` | Container exit drains and checkpoints without cancelling children; explicit cancellation still cancels |
| `tests/execution/test_remote_coordinator.py` | A detached loop reaches terminal without client polling; duplicate loop, claim, and completion inputs are idempotent; infrastructure replacement reloads checkpoints; uncaught coordinator errors stop without automatic retry or child cancellation; explicit resume reconciles; terminal status can reopen the ledger; different Execution Run IDs remain isolated |
| `tests/execution/test_dispatch.py` | Lost claim responses, claim replay, preemption with an active assignment, terminal-owner failure without same-run reassignment, and unknown-owner blocking |
| `tests/execution/test_single_submission.py` | Each Task gets at most one submission per Run; redelivery retains call identity; resume never retries failure; restart reuses valid publications and submits only conclusively unowned missing work |
| `tests/execution/test_deployment.py` | Explicit and history-resolved versions are pinned; an unavailable version fails with reason `deployment_unavailable`; restart creates a linked run and reuses publications |
| `tests/execution/test_run_status.py` | Exactly nine statuses and six reason codes exist; legal transitions, terminality, status-reason constraints, suspension/resume, unknown-state blocking, deployment failure reason, and service projections are deterministic |
| `tests/execution/test_node_status.py` | Exactly seven Node statuses exist; terminality, derived readiness, cache-success provenance, and non-duplication of Run control states are deterministic |
| `tests/execution/test_task_status.py` | Exactly six Task statuses exist; terminality, durable ownership, unknown-owner blocking, cache provenance, fail-fast skipping, and absence of partial Task state are deterministic |
| `tests/execution/test_relationships.py` | Every Task, Dispatch Batch, and Provider Call belongs to one Node; a call cannot own another Node's Task; a Task cannot acquire a second remote owner; zero-call cache and local completion remain valid |
| `tests/execution/test_provider_call_status.py` | Exactly eight Provider Call statuses exist; legal transitions, terminality, attachment identity, unknown-state ownership, and expiry projection are deterministic |
| `tests/execution/test_identity.py` | Execution UUIDs are opaque and unique; workload keys never select paths; successor lineage uses a new UUID |
| `tests/execution/test_cli_location.py` | Explicit deployment and run flags reach the correct coordinator; mismatched ledger fields fail; optional call IDs remain non-authoritative |
| `tests/execution/test_cli_recovery.py` | A repeated launch creates a root Run; resume never retries failures; generic restart and `--restart-from` create equivalent successors; valid publications are reused; unknown or invalid predecessor state fails closed |
| `tests/execution/test_restart_compatibility.py` | Result-affecting input, content, and declared version changes reject successor creation; operational policy and Deployment Identity changes remain compatible; generic and launch-time restart use the same fingerprint |
| `tests/workflow/test_ledger.py` | Execution result, artifacts, Task, Node, and Provider Call finalize atomically without attempt rows or paths |
| `tests/service/test_gromacs_plan.py` | The fixed GROMACS graph preserves its parallel readiness waves |
| `tests/workflow/ppiflow/test_coordinators.py` | Candidate outcomes preserve identity, order, partial failures, and configured concurrency |
| `tests/app/test_alphafold3_production_contracts.py` | Search, run, request, marker, and seed-batch identities remain unchanged |

Each fault test uses a deterministic injected failure point and asserts both
the durable rows and the number of fake paid calls. Merely asserting a final
status is insufficient.

### Phase 1 — repair independent safety semantics

Deliverables:

- map availability-check exceptions to `unknown`;
- stop cancelling attached child calls from the orchestrator exit hook and
  checkpoint best-effort instead;
- add characterization tests for the current preclaim, attachment,
  finalization, and interruption gaps without adding new legacy execution
  state.

Exit gate:

- cache-check failures cannot authorize work;
- orchestrator shutdown cannot cancel child calls without explicit user
  cancellation;
- fault tests describe the required kernel behavior before extraction.

Rollback:

- revert either focused safety commit; neither changes the ledger schema.

### Phase 2 — extract immutable plans and graph algorithms

Deliverables:

- add immutable `ExecutionPlan`, `NodePlan`, and dependency validation;
- add deterministic Workload Plan Fingerprints that separate result-affecting
  inputs and declared scientific versions from operational execution policy;
- extract deterministic readiness and terminal-reachability functions;
- build pure GROMACS and workflow adapters beside their current execution
  paths and prove graph equivalence without switching either composition root;
- keep dynamic work represented as a Node-owned Task factory, not mutable DAG
  vertices.

Exit gate:

- GROMACS selects the same parallel operations;
- workflows produce the same hashes, scheduled waves, and terminal pruning;
- no provider or database dependency exists in the plan module.

Rollback:

- revert the implementation commit; do not retain a runtime selection flag.

### Phase 3 — build durable single-submission state

Deliverables:

- add `SqliteExecutionRepository` tables and transitions for Execution Runs,
  Nodes, Tasks, and Provider Calls over a host-supplied SQLite connection;
- implement the nine-status Run transition table, `status_reason`, and
  `status_message`;
- implement the eight-status Provider Call lifecycle without `planned` or
  `expired` states;
- enforce Node-local dispatch, zero-to-many Tasks per Provider Call, and at
  most one durable remote owner path per Task without a generic many-to-many
  call association;
- persist immutable Task plans and enforce one submission claim per Task per
  Execution Run;
- implement preclaim, spawn, attachment, observation, collection,
  cancellation, and unknown-outcome recovery behind the provider port;
- reuse `ModalJobSubmitter`'s preclaim, attachment, and unknown-spawn
  classifications, but deliberately replace its retryable operation-release
  behavior with the one-spawn preclaim rule;
- expose a common read-only execution snapshot for logs and diagnostics.

Exit gate:

- every paid-call transition and crash boundary is exercised through fakes in
  CI;
- database constraints reject a second submission for the same Task;
- provider redelivery retains one Task and Provider Call identity;
- no production host has switched to the new repository yet.

Rollback:

- revert the kernel commits; no host schema or runtime path has changed.

### Phase 4 — add dispatch, budgets, and remote coordination

Deliverables:

- add Dispatch Batches, Worker Assignments, idempotent claim requests, and
  explicit Task-to-call or Task-to-assignment links;
- move bounded batching and permit accounting into execution internals;
- persist coordinator-scoped permit allocations and recovery without adding a
  distributed lease abstraction;
- add reusable sync and async coordinator loops;
- add the run-scoped Modal coordinator binding with one-container routing,
  serialized SQLite and Volume checkpoints, detached execution, lifecycle
  methods, and preemption recovery;
- keep provider workers behind idempotent claim and completion methods and
  prevent them from opening SQLite directly.

Exit gate:

- direct, batched, and pull-worker fake workloads obey the same
  single-submission rule;
- lost claim and completion responses replay idempotently;
- resource tests prove that one call batch consumes the intended permits;
- remote-coordinator assumptions pass a manual Modal smoke test before host
  adoption.

Rollback:

- revert the kernel dispatch and remote-binding commits; no host has cut over.

### Phase 5 — cut over fixed consumers and deployed CLI runs

Deliverables:

- migrate GROMACS service execution from `job_operations` to the shared
  repository while preserving its fixed parallel DAG, stage projection, logs,
  cancellation, and result archive;
- add the explicit offline service-state transition that preserves users and
  configuration while recreating Job and execution state;
- update OpenAPI and frontend-facing service projections with the backend
  cutover;
- derive existing service `queued` and `finalizing` labels from kernel
  `pending`, `running`, and the active Node rather than persisting them as Run
  statuses;
- switch the workflow composition root in one commit to
  `SqliteExecutionRepository` plus `WorkflowArtifactStore`;
- in that same workflow cutover, delete the old `WorkflowLedger` execution
  methods, attempt fields and tables, generic execution policy, and
  attempt-based paths without a facade or dual write;
- preserve workflow DAG hashes, artifact contracts, Volume synchronization,
  display, and scientific publication reuse;
- make `biomodals app run` and `biomodals workflow run` resolve an exact
  deployed coordinator version, start a remote ledger and detached coordinator
  loop, and create no local execution database;
- accept an explicit `--version` or resolve `modal app history --json` once,
  then persist and use only the exact Deployment Identity;
- add shared `biomodals run status`, `cancel`, `resume`, and `restart`
  lifecycle commands using explicit deployment and run flags;
- add `--restart-from <execution-run-id>` to app and workflow launch commands
  as a thin convenience over the generic restart operation, without implicit
  command matching;
- keep local input staging, result retrieval, dry-run, help, and explicit
  source-backed development mode in thin CLI clients.

Exit gate:

- GROMACS API behavior, OpenAPI, timelines, log IDs, cancellation, and archives
  are unchanged except for the intentional execution schema replacement;
- the workflow runtime contains no attempt state or compatibility facade and
  passes DAG, recovery, artifact, and publication-equivalence tests;
- a second CLI process can address the same version-pinned run, while
  development mode clearly lacks cross-invocation recovery;
- a repeated launch creates a new root Run, while `--restart-from` and generic
  restart create the same linked successor behavior;
- successor creation rejects a changed Workload Plan Fingerprint while
  allowing operational policy and Deployment Identity changes;
- restart creates a Successor Execution Run and cannot replace active or
  unknown predecessor work;
- manual Modal tests validate deployment lookup, run-scoped routing,
  preemption recovery, and terminal ledger reopening.

Rollback:

- revert service, workflow, and CLI cutover commits independently;
- recreate incompatible unfinished Job or workflow execution state while
  preserving scientific publications and app-owned outputs.

### Phase 6 — adopt runtime-discovered PPIFlow Tasks

Deliverables:

- translate each fixed PPIFlow stage into an Execution Node whose Task factory
  discovers stable candidate Tasks at runtime;
- retain candidate IDs, manifests, stage-specific inputs, joins, attrition,
  and result ordering as workload-owned contracts;
- use Task and batch state rather than bare `bounded_map` orchestration for
  durable candidate scheduling;
- represent per-candidate outcomes and configured aggregation policy without
  inferring Node success from a partial batch;
- apply the existing run-level concurrency configuration through kernel
  permits;
- reuse validated candidate publications after interruption or in a Successor
  Execution Run.

Exit gate:

- candidate identities and manifests remain byte- or semantic-equivalent;
- interrupted stages do not repeat uncertain calls;
- partial candidate failures retain the same successful publications and
  deterministic joins;
- PPIFlow becomes the first production proof of runtime Task discovery without
  a mutable DAG.

Rollback:

- revert the PPIFlow adapter commit; validated candidate publications remain
  reusable and no scientific format changes.

### Phase 7 — adapt AlphaFold3 without changing scientific authority

Model the existing pipeline rather than replacing it:

```text
stage request input
  -> raw database-search Tasks
  -> combined-MSA publication
  -> template-search Tasks
  -> enriched and staged inference input
  -> independently claimed seed Tasks
  -> inference summary
  -> request view and retrieval archive
```

Deliverables:

- translate pure search and inference plans into Nodes and Tasks;
- wrap existing marker validators as availability adapters;
- wrap generation claims as writer-coordination adapters;
- claim each missing seed Task independently before batching compatible seeds
  into one GPU call;
- persist the one-call-to-many-seed mapping and per-seed outcomes;
- make a Successor Execution Run the only retry path while retaining current
  scientific request and workload run identities;
- route the Local Entrypoint through its deployment-local remote coordinator;
- let service and workflow hosts execute the same plan through their parent
  coordinator without creating a nested ledger;
- preserve current CLI inputs, outputs, and direct Child App Call behavior.

Exit gate:

- AlphaFold run IDs, request IDs, search identities, marker payloads, Volume
  paths, seed reuse, ranking order, and retrieval archives remain unchanged;
- an overlapping seed request performs only its missing seed work;
- partial search and seed failures preserve the same reusable publications;
- no automatic paid retry or nested execution ledger is introduced.

Rollback:

- revert the AlphaFold3 adapters; marker and publication formats require no
  reverse migration.

### Phase 8 — replace app-local schedulers and remove duplication

Adopt two concrete dispatch adapters only after the kernel and dynamic Task
model are proven:

1. BoltzGen exercises bounded direct fan-out, where each workload run key is
   one Task and one GPU Provider Call.
2. Rosetta exercises the SQLite-backed pull worker pool, where several
   Provider Calls claim Task microbatches from one Dispatch Batch.

Deliverables:

- adapt BoltzGen to reusable bounded direct fan-out and Task state;
- retain its output lock only if needed as a cross-coordinator publication
  claim, never as the scheduler queue;
- adapt Rosetta to ready Task rows, idempotent claims, durable Worker
  Assignments, and call-bound work recovery;
- remove Modal Queue, Modal Dict, output-file scheduling, and generic
  worker-pool lifecycle from execution coordination;
- preserve workload commands, Task payloads, scientific result validation,
  deterministic output ordering, and CLI behavior;
- remove GROMACS-local readiness duplication, workflow-specific paid-call
  transitions, PPIFlow bare concurrency scheduling, AlphaFold3 generic bounded
  outcome loops, BoltzGen incomplete-run fan-out, Rosetta queue orchestration,
  and repeated claim mechanics only after their replacements pass equivalence
  tests;
- publish the final supported execution inspection surface and delete stale
  adapters, migration notes, and dead scheduler helpers.

Exit gate:

- interrupted BoltzGen runs reuse validated publications without duplicating an
  uncertain GPU call;
- Rosetta workers balance Tasks dynamically and reconcile each Task after
  partial worker failure;
- worker exit callbacks can be dropped without changing recovery;
- each execution concern has one implementation or a documented
  workload-specific reason to differ;
- no compatibility facade, migration switch, dead adapter, Modal queue, or
  duplicate generic scheduler remains.

Rollback:

- revert each workload-adoption commit independently; validated publications
  remain reusable and require no reverse migration.

## Suggested incremental commits

Use small commits in dependency order:

1. `docs: plan unified task scheduler`
2. `workflow: fix availability uncertainty`
3. `workflow: preserve calls on exit`
4. `execution: add plans and graph`
5. `execution: add durable task state`
6. `execution: add call lifecycle`
7. `execution: add dispatch coordinators`
8. `service: transition execution state`
9. `service: adopt execution kernel`
10. `cli: target deployed coordinators`
11. `workflow: cut over execution kernel`
12. `ppiflow: adopt durable task fanout`
13. `alphafold3: adopt execution adapters`
14. `boltzgen: adopt direct task fanout`
15. `rosetta: adopt sqlite work pool`
16. `execution: remove duplicate schedulers`

Split a commit further whenever its predecessor and replacement cannot be
reviewed side by side. Never combine an AlphaFold3 scientific contract change
with scheduler extraction.

### First two implementation commits

The first two implementation commits have fixed scope:

#### `workflow: fix availability uncertainty`

Files:

- `src/biomodals/workflow/core/artifact_availability.py`
- `tests/workflow/test_artifacts.py`

Change:

- map external checker exceptions to `ArtifactAvailabilityStatus.UNKNOWN`;
- put the exception diagnostic in `unknown_reason`;
- leave `errors` empty because absence was not established;
- add the focused regression test and change no scheduling code.

Verification:

```text
uv run pytest tests/workflow/test_artifacts.py
prek run --files \
  src/biomodals/workflow/core/artifact_availability.py \
  tests/workflow/test_artifacts.py
```

Rollback: revert the commit; it has no schema or durable-state effect.

#### `workflow: preserve calls on exit`

Files:

- `src/biomodals/workflow/core/orchestrator.py`
- `tests/workflow/test_orchestrator.py`

Change:

- remove exit-time child cancellation from the Modal lifecycle hook;
- retain runtime closure and the best-effort Volume commit;
- leave explicit user cancellation behavior unchanged;
- prove repeated exit-hook calls never cancel attached children.

Verification:

```text
uv run pytest tests/workflow/test_orchestrator.py
prek run --files \
  src/biomodals/workflow/core/orchestrator.py \
  tests/workflow/test_orchestrator.py
```

Rollback: revert the commit; it has no schema effect.

## Verification matrix

| Layer | Required verification |
| --- | --- |
| Pure model | Graph cycles, readiness, terminal closure, stable fingerprints, task discovery determinism |
| Call lifecycle | Fault injection around every transition, attach validation, recovery, expiry, cancellation, unknown outcomes |
| Cache | Available/missing/unknown, checker exceptions, marker validation, cache hit starts no call |
| Batching | Call-to-many mapping, per-Task result decode, partial and failed batches, deterministic ordering |
| Dispatch | Direct fan-out, many-call pull pools, idempotent claim replay, call-bound Worker Assignments, partial outcomes |
| Interruption | Graceful drain, hard kill, child-call preservation, replacement recovery, explicit cancellation |
| Resources | Node parallelism independent from Task permits, batched permit accounting, no permit leak on failure |
| Service | API/OpenAPI unchanged unless intentionally versioned; admission, timeline, logs, cancel, cache staging, ZIP contents |
| Workflow | DAG hashes, scheduler waves, terminal pruning, artifact selection/materialization, coordinator resume, and successor restart behavior |
| PPIFlow | Candidate identity, manifests, attrition, joins, partial outcomes, and successor publication reuse |
| AlphaFold3 | Search/run/request identities, claims, publications, seed batching/reuse, summaries, archive hashes |
| CLI | App and workflow discovery/help, version resolution and overrides, deployed versus development launch, representative dry tests |

CI uses an in-memory SQLite repository plus fake provider and workload-storage
implementations. Remote Modal validation remains a manual, explicitly
authorized smoke test after local and CI gates pass.

## Risks and controls

| Risk | Control |
| --- | --- |
| A universal abstraction hides scientific differences | Workload-owned hooks and provider adapters; migrate AF3 last |
| Extraction duplicates rather than replaces code | Each phase names deletion candidates and has a final deletion gate |
| Async and sync consumers distort the API | Share pure transitions; keep thin separate host loops |
| A batch obscures individual outcomes | Persist Task identities, per-Task outcomes, and explicit call links |
| A claim response is lost after assignment | Commit and Volume-checkpoint the assignment before responding; replay by stable request ID |
| An exit callback races a restarted worker | Treat exit events as advisory and retain call-bound Worker Assignments |
| Recovery silently spends on a second call | Enforce one Task submission per Run in SQLite; require a Successor Execution Run for failed work |
| “Exactly once” hides provider re-execution | Promise single scheduler submission only; require idempotent work or authoritative publication validation across redelivery |
| Cache checker outage triggers expensive recomputation | Tri-state availability; only `missing` authorizes work |
| Crash after paid spawn duplicates work | Preclaim, attach protocol, explicit outcome unknown, no blind retry |
| Preemption is mistaken for cancellation | Preserve child calls, checkpoint best-effort, and recover by call ID |
| Two coordinator containers open one Volume ledger | Route by Execution Run ID and pinned deployment version, cap the pool at one container, serialize writes, and smoke-test provider behavior |
| One coordinator must understand every workload | Ship a thin binding with each app or workflow deployment; keep shared mechanics in the kernel |
| Latest deployment changes between CLI calls | Resolve history once, persist the exact Deployment Identity, and use only versioned handles |
| Version-pinned lookup is unsupported or expired | Preflight workspace support and exact availability; fail closed with the recorded identity |
| A newer deployment mutates an old run | Make Deployment Identity immutable; create a linked successor after publication revalidation |
| One Volume couples unrelated app ledgers | Store app ledgers in deployment-specific Volumes and reserve a kernel-owned path namespace |
| A workload name collides with execution state | Generate an opaque Execution Run ID and keep workload keys only in immutable plan input |
| CLI identity flags are treated as authority | Verify Deployment Identity and Execution Run ID against the ledger before observing or mutating state |
| Resource limits are mistaken for Modal decorators | Separate operational requirements from run-level permit accounting |
| One ledger becomes a cross-context bottleneck | Embed the same execution tables into coordinator-owned databases |
| Refactor changes scientific or user-visible behavior accidentally | Scientific, cost-safety, CLI-operation, and result regression tests |

## Explicit non-goals

- a universal API Job base class;
- one database or ledger for all consumers;
- a universal scientific cache or marker schema;
- a mutable runtime DAG;
- a YAML workflow language;
- global plugin registration or autodiscovery;
- automatic retries of paid provider calls;
- a cross-coordinator or cross-run global resource scheduler;
- remote workers opening or mutating the coordinator's SQLite file;
- a generic provider-native message-queue abstraction;
- automatic expiry or background garbage collection of execution ledgers;
- scheduler-driven mutation of Modal function decorators;
- moving scientific parsers into generic execution code;
- rewriting AlphaFold3, GROMACS, or PPIFlow public interfaces as part of the
  extraction.

## Decision gates for the grill

Resolve these one at a time and update ADR 0006 and `CONTEXT.md` immediately
after each decision:

1. **Authority boundary — accepted 2026-07-29**: the kernel owns execution
   mechanics and its durable state contract, while host stores and scientific
   publications retain their established authority.
2. **Repository scope and Workflow Ledger decomposition — accepted
   2026-07-29**: repositories follow durable coordinator boundaries. The
   physical workflow ledger remains, but its generic tables and transitions
   move to the shared execution repository. Workflow code retains a narrow
   Workflow Artifact Store and Volume lifecycle.
3. **Repository implementation — accepted 2026-07-29**: the kernel provides
   one concrete `SqliteExecutionRepository` over a host-supplied connection
   and transaction. Tests use in-memory SQLite; a generic persistence protocol
   waits for a second real backend.
4. **Internal surface — accepted 2026-07-29**:
   `biomodals.execution` has a small interface for depth and testability but no
   compatibility promise for Python imports, unfinished database schemas, or
   in-progress run formats.
5. **Service Job projection — accepted 2026-07-29**: a Service Job stores a
   one-way Execution Run ID and service-owned metadata but no duplicate compute
   state. `JobState`, timelines, filters, limits, and counts derive from
   execution rows.
6. **Resource scope — accepted 2026-07-29**: the kernel persists permit
   accounting only within one Execution Run and coordinator. Service admission
   and Modal resources remain with their current owners; no distributed lease
   interface is added without a concrete cross-coordinator requirement.
7. **State transition — accepted 2026-07-29**: preserve service-owned users,
   authentication, and administrator configuration; recreate Service Job and
   execution state without old Job history; reject and restart old workflow
   ledgers; preserve remote scientific publications and caches.
8. **Adoption order — accepted 2026-07-29**: GROMACS and the basic workflow
   runtime prove fixed scheduling first, PPIFlow is the first
   runtime-discovered Task consumer, and AlphaFold3 adopts the proven kernel
   afterward.
9. **App-internal fan-out — accepted 2026-07-29**: after the Task lifecycle is
   proven, BoltzGen adopts reusable direct fan-out and Rosetta adopts a
   SQLite-backed pull work-pool adapter. Ready Task rows and Worker
   Assignments are the durable queue; workers access them only through
   idempotent coordinator methods.
10. **Coordinator interruption — accepted 2026-07-29**: preemption suspends
    scheduling without cancelling child calls. A Coordinator Attempt drains
    and checkpoints best-effort, correctness survives hard interruption, and
    a replacement recovers attached calls and permits from durable state.
11. **Single-writer topology — accepted 2026-07-29**: a Volume-backed remote
    coordinator is parameterized by Execution Run ID and pinned deployment
    version. Its provider pool is capped at one container, and concurrent
    method inputs submit commands to one in-process SQLite writer. Different
    Runs use independent pools.
12. **Worker interruption — accepted and revised 2026-07-29**: preemption
    retains the Task, Provider Call, and committed Worker Assignment. Claim and
    completion methods are idempotent, and assignment is checkpointed before
    payload delivery. A terminal owner fails unfinished Tasks; no successor
    assignment is created in the same Execution Run.
13. **Single submission and retry boundary — accepted 2026-07-29**: the
    kernel stores no Task Attempt identity. Each Task receives at most one
    Provider Call submission or Worker Assignment in an Execution Run.
    Provider redelivery may re-execute the same call, so exactly-once execution
    is not promised. `resume` never retries failed Tasks. Explicit `restart`
    creates a Successor Execution Run, revalidates publications, and submits
    only missing work whose predecessor ownership is conclusively terminal.
14. **Task queue storage — accepted 2026-07-29**: SQLite is the only durable
    queue and assignment store. The design adds neither Modal Dict nor Modal
    Queue; remote workers claim bounded microbatches through the run-scoped
    coordinator.
15. **Coordinator placement — accepted 2026-07-29**: every Direct CLI App Run
    and workflow CLI run uses a remote run-scoped coordinator and remote
    per-run repository. API service calls use `service.sqlite3`. Child App
    Calls use their parent Run and never create a redundant nested ledger.
    Local Entrypoints are thin clients and create no local execution database.
16. **Remote coordinator deployment — accepted 2026-07-29**: each app and
    workflow deployment includes a thin Deployment Coordinator Adapter over
    the shared kernel. The containing deployment version pins coordinator and
    workload code together. There is no universal coordinator deployment,
    workload registry, or deployment-global Volume in the kernel.
17. **CLI deployment lifetime — accepted 2026-07-29**: both
    `biomodals app run` and `biomodals workflow run` target exact named
    deployment versions by default, keeping their coordinator pools
    addressable across CLI processes. Source-backed ephemeral execution is an
    explicit Development CLI Run with no cross-invocation resume guarantee.
    Local dry-run planning needs no deployment.
18. **Deployment version selection — accepted 2026-07-29**: `--version`
    explicitly selects a version. When omitted, the CLI resolves
    `modal app history --json` once and pins the current deployed version.
    Environment, deployment name, and numeric version are preflighted and
    persisted before admission; later lookups never float to latest.
19. **Expired deployment recovery — accepted 2026-07-29**: an Execution Run
    never changes Deployment Identity. Existing calls remain observable by ID.
    An incomplete run with an unavailable version becomes `failed` with reason
    `deployment_unavailable`. Explicit restart creates a linked Successor
    Execution Run on a new version, revalidates publications, and schedules
    only missing Tasks whose predecessor ownership is conclusively terminal.
20. **Remote ledger storage — accepted 2026-07-29**: each Direct CLI App Run
    stores an App Run Ledger at
    `.biomodals/execution/runs/<execution-run-id>/ledger.sqlite3` in its
    deployment's configured durable Volume. Workflows retain their
    orchestrator Volume and the API retains `service.sqlite3`. The kernel
    receives these host bindings and defines no shared cross-app execution
    Volume.
21. **Execution Run identity — accepted 2026-07-29**: the admitting host
    generates an opaque UUID before repository creation or work admission.
    It keys execution rows, coordinator routing, lineage, and ledger paths.
    User and scientific names remain Workload Run Keys in immutable plan input.
    A successor receives a new UUID while reusing appropriate workload keys
    and validated publications.
22. **CLI run location — accepted 2026-07-29**: launch commands print
    Deployment Identity, Execution Run ID, and optional root coordinator
    FunctionCall ID. Shared `biomodals run` lifecycle commands verify the
    deployment and run fields against the ledger. No local registry or global
    run index is added.
23. **CLI run arguments — accepted 2026-07-29**: lifecycle commands take
    `--environment`, `--deployment-name`, `--deployment-version`, and
    `--execution-run-id` explicitly on every invocation. An optional
    `--coordinator-call-id` is a replaceable observation hint. There is no
    encoded run-reference format or parsing abstraction.
24. **Remote ledger retention — accepted 2026-07-29**: terminal Direct CLI
    App Run and workflow ledgers are retained without a TTL or background
    garbage collector. A future explicit host or CLI cleanup operation must
    reject non-terminal Runs and may remove only execution state, never
    Workload Publications or scientific outputs. Service database retention
    remains service-owned.
25. **Active remote coordinator — accepted 2026-07-29**: each remote
    top-level CLI app or workflow run submits one detached coordinator loop
    that advances it until terminal without client polling. Concurrent control
    and worker inputs share its serialized writer. Preemption starts a
    replacement Coordinator Attempt from the ledger; after terminal completion
    the loop returns, and later status calls may reopen the retained ledger in
    a fresh container. The loop is not a CLI subcommand or public domain type.
26. **Coordinator application errors — accepted 2026-07-29**: provider
    redelivery may replace an infrastructure-interrupted Coordinator Attempt.
    An uncaught coordinator application exception is not automatically
    retried: admission stops, attached calls remain running, a diagnostic is
    recorded when possible, and the Run becomes `suspended` until explicit
    resume reconciles durable state and returns it to `running`.
27. **Workflow attempt removal — accepted 2026-07-29**: the workflow
    migration deletes `attempts`, `current_attempt_id`, attempt foreign keys and
    counters, `AttemptRecord`, generic `NodeExecutionPolicy`, attempt status
    output, and the `attempts/<attempt-id>/` path layer. Task, Node, Provider
    Call, and artifact records retain the one result, error, timing, and output
    state needed by the single-submission model.
28. **Direct workflow cutover — accepted 2026-07-29**: build and test the
    kernel beside the unchanged workflow execution implementation, then switch
    the workflow composition root in one cutover commit. Do not add a
    `WorkflowLedger` compatibility facade, dual schema, dual writes, or
    attempt-preserving adapter. Delete the replaced execution implementation
    and reject old unfinished ledgers at cutover.
29. **Execution Run statuses — accepted 2026-07-29**: the kernel uses
    `pending`, `running`, `cancel_requested`, `suspended`, `state_unknown`,
    `succeeded`, `partial`, `failed`, and `cancelled`. The first five are
    nonterminal and the final four terminal. Deployment unavailability is
    `failed` with reason `deployment_unavailable`; provider
    `outcome_unknown` projects to `state_unknown`. Preemption and finalization
    do not create Run statuses, and host Job labels remain derived projections.
30. **Execution Run status reasons — accepted 2026-07-29**: a Run stores one
    nullable stable `status_reason` code plus one nullable human-readable
    `status_message`. Transitions replace or clear them atomically with
    `status`, and control flow never parses the message. Task- and Node-specific
    failures stay canonical on those records; there are no status-specific
    reason columns.
31. **Initial Run reason vocabulary — accepted 2026-07-29**: the closed
    kernel enum contains `coordinator_error`,
    `submission_outcome_unknown`, `provider_outcome_unknown`,
    `cancellation_outcome_unknown`, `required_work_failed`, and
    `deployment_unavailable`. The first applies only to `suspended`, the next
    three only to `state_unknown`, the final two only to `failed`, and every
    other Run status requires a null reason. The repository rejects invalid
    combinations.
32. **Execution Node statuses — accepted 2026-07-29**: Nodes use `pending`,
    `running`, `succeeded`, `partial`, `failed`, `cancelled`, and `skipped`;
    only the first two are nonterminal. Readiness is derived, cache reuse is
    Task provenance on a successful Node, and Run-level cancellation request,
    suspension, and unknown state are not duplicated onto Nodes. `skipped`
    means an upstream terminal outcome made a planned Node unreachable.
33. **Task statuses — accepted 2026-07-29**: Tasks use `pending`, `running`,
    `succeeded`, `failed`, `cancelled`, and `skipped`; only the first two are
    nonterminal. Durable local or provider ownership moves a Task to
    `running`, which is retained while the owner is uncertain. Cache reuse is
    success provenance, partiality belongs to Node aggregation, provider
    submission phases stay on Provider Calls, and `skipped` is reserved for
    sibling Tasks not admitted after `fail_fast`.
34. **Node, Task, and Provider Call relationships — accepted 2026-07-29**:
    Nodes are fixed semantic stages, Tasks are independently scheduled and
    validated items, and Provider Calls are concrete remote worker
    invocations. Every Task, Dispatch Batch, and Provider Call belongs to one
    Node. A call may own zero or many Tasks from that Node, while a Task has at
    most one durable remote owner path per Run. There is no cross-Node call or
    generic many-to-many Task-to-call relation. Cache and local execution need
    no call; provider redelivery retains call identity; coordinator hosting is
    a separate Coordinator Attempt.
35. **Provider Call statuses — accepted 2026-07-29**: calls use
    `submitting`, `attached`, `running`, `outcome_unknown`, `state_unknown`,
    `succeeded`, `failed`, and `cancelled`. The first five are nonterminal and
    preserve ownership; the final three are terminal. The preclaim creates a
    `submitting` call directly. Unsubmitted intent is not a `planned` call, and
    handle expiry projects to conclusive `failed` or unresolved
    `state_unknown` rather than an `expired` status.
36. **Explicit CLI recovery — accepted 2026-07-29**: a repeated app or
    workflow launch without predecessor identity creates a new root Run and
    never infers identity from command text, paths, arguments, or Workload Run
    Keys. `resume` retains the Run and never retries failed Tasks. Generic
    restart or launch-time `--restart-from` creates a linked successor, reuses
    valid publications, and schedules conclusively unowned missing work before
    advancing downstream Nodes. Active, unknown, unreadable, or
    invariant-invalid predecessor ownership fails closed; there is no implicit
    run catalog.
37. **Submission preclaim boundary — accepted 2026-07-29**: the atomic
    preclaim creates a `submitting` Provider Call, assigns its Tasks, and
    authorizes only the caller that created the row and crossed the host
    durability boundary to invoke spawn. Duplicate requests perform no side
    effect. Recovery of `submitting` becomes `outcome_unknown` and never
    spawns again. Conclusive rejection fails the call and Tasks; ambiguous
    failure preserves unknown ownership. Resolution and input preparation
    precede preclaim, and only a Successor Execution Run can retry failed
    Tasks.
38. **Successor scientific compatibility — accepted 2026-07-29**: every Run
    stores a Workload Plan Fingerprint over normalized result-affecting inputs,
    content digests, and declared scientific tool, model, adapter, and schema
    versions. Generic restart reuses that plan; `--restart-from` must match its
    fingerprint. Scientific changes require a new root Run. Operational
    concurrency, batching, resources, and Deployment Identity may change, but
    a new adapter may reuse publications only if it accepts the stored plan and
    preserves all result-affecting declarations.

## Definition of ready for implementation

Implementation begins only when:

- ADR 0006 is accepted;
- all decision gates are resolved;
- the proposed execution terms are reconciled into `CONTEXT.md`;
- Phase 0 scientific, cost-safety, and user-visible regression fixtures plus
  fault-injection points are enumerated as test cases;
- the first two implementation commits have exact file and rollback scopes.

Implementation remains paused until the remaining decision gates are resolved.
