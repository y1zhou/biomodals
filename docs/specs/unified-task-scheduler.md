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
- Scientific and cache identity excludes operational concurrency, placement,
  and resource allocation.
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
  unavailable, an explicit restart creates a linked successor run.
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
        ├── Node
        │     └── Task
        └── Dispatch Batch
              ├── contains one or more Tasks
              ├── records call-bound Worker Assignments
              └── served by one or more Provider Calls
```

The relationship between Task and Provider Call is not one-to-one:

- a cache hit completes a Task without a call;
- GROMACS usually has one Task per Modal call;
- one AlphaFold3 inference worker call may serve several seed Tasks;
- several Rosetta worker calls may steal Tasks from one Dispatch Batch;
- a Task receives at most one Provider Call or Worker Assignment in its
  Execution Run;
- a failed Task can be represented again only in a Successor Execution Run.

### Proposed terms

**Execution Run** is one invocation of an immutable execution plan. An API Job
may own one run, while a CLI workflow or app invocation can create a run
without an API Job.

**Execution Run ID** is a kernel-generated UUID that keys execution state,
coordinator routing, lineage, and ledger location.

**Workload Run Key** is an optional workload-owned name or scientific key in
the immutable plan. It may be reused by successor runs and publications but is
never an execution primary key or ledger path.

**Execution Node** is a fixed semantic DAG step. It replaces neither
`WorkflowNode` nor user-facing service stages immediately; adapters map those
concepts to it during migration.

**Task** is the smallest independently identified unit whose cache and outcome
can be reasoned about. Tasks may be discovered only when their containing Node
starts.

**Single-Submission Rule** means the kernel schedules each Task once and
creates at most one Provider Call submission or Worker Assignment for it in
one Execution Run. Provider redelivery can re-execute that same call, so this
is not an exactly-once execution claim.

**Provider Call** is one detached Modal function call, including its durable
call ID and observed lifecycle. A call can cover a batch of Tasks.

**Dispatch Batch** is a durable grouping of Tasks offered to one provider call
or a shared worker pool. Exact Task-to-call attribution is recorded only when
it is observed.

**Worker Assignment** is a durable, call-bound SQLite record electing the
worker allowed to execute one Task from a shared pull work pool. It is
checkpointed before the Task payload is returned.

**Task Claim Request** is an idempotent request for a bounded set of ready
Tasks. Repeating its stable request ID returns the same Worker Assignments.

**Deployment Identity** is the Modal Environment, deployed app or workflow
name, and exact numeric deployment version fixed before run admission.

**Deployment-Blocked Run** is a terminal, incomplete run whose Deployment
Identity is unavailable. It remains inspectable and admits no new work.

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
when possible. The adapter does not automatically retry that exception.
Explicit `resume` reloads the durable ledger, reconciles attached calls, and
then continues the same Execution Run.

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
Workflow Artifact Store ──┘                         └-> Provider Call

Execution tables --X--> Service Job, user, HTTP, admin, or artifact tables
```

Execution tables may store only data needed to reconstruct and manage actual
work:

- stable run, node, task, batch, assignment, and call identifiers;
- an optional predecessor Execution Run ID for explicit restart lineage;
- the immutable Deployment Identity used to recover provider calls;
- immutable plan and task fingerprints;
- dependency edges and legal execution states;
- submission tokens, provider targets, call IDs, and observed outcomes;
- execution timestamps, errors, single-submission state, and resource permits;
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
| `RunStatus`, `NodeStatus`, placement, and recovery policy | Shared execution models, with temporary workflow re-exports if needed |
| `NodeExecutionPolicy`, `AttemptRecord`, and `NodeStatusRecord.attempts` | Delete; provider redelivery and Successor Execution Runs replace generic rerun policy |
| `artifacts`, `artifact_files`, `node_inputs`, and `node_outputs` tables | Workflow-specific run store |
| `WorkflowArtifact`, `ArtifactSelector`, and materialized `AppRunResult` handling | Workflow-specific artifact module |
| Run-root directories, node/task output paths, connection closure, and Volume synchronization | Workflow-specific run store |
| Finalizing execution state and artifacts together | One host-owned SQLite transaction spanning both implementations |

This leaves one SQLite file per workflow run, not an execution database beside
a workflow database. The file contains shared execution tables and
workflow-specific artifact tables on the same connection.

During migration, `WorkflowLedger` can be a short-lived facade between
incremental commits so the runtime does not change in one large step. It must
not become a permanent pass-through interface duplicating every execution
repository method. Once callers use the kernel, either:

- rename the remaining workflow-specific implementation to
  `WorkflowRunStore`; or
- delete the facade and let the workflow runtime compose the shared execution
  repository with a narrow artifact store directly.

The preferred end state is the second option if composition remains readable.
The deletion test is that removing `WorkflowLedger` must not redistribute
generic SQL or transition logic back into workflow callers.

The service follows the same pattern. The user-facing Service Job points to an
Execution Run. `job_operations`, persisted `JobOperationState`, and persisted
compute `JobState` are replaced by shared Execution Nodes, Tasks, and Provider
Calls. Service projections retain the existing HTTP state and timeline
vocabulary without preserving a second operation state machine.

### Paid-call lifecycle

The durable lifecycle must distinguish:

```text
planned
  -> submitting
  -> attached
  -> running
  -> succeeded | failed | cancelled | expired

submitting -> outcome_unknown
attached/running -> state_unknown
```

`outcome_unknown` means a spawn may have started but no call ID was durably
attached. `state_unknown` means an attached call exists but its current or
terminal provider state cannot be established. Neither state automatically
returns to `planned`.

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
| `tests/service/test_submission.py` | Definite rejection releases work; unknown spawn blocks; an unattached returned call is cancelled and marked unknown |
| `tests/workflow/test_runtime.py` | A crash after durable claim but before attachment does not submit replacement work |
| `tests/workflow/test_runtime.py` | A recorded call ID is resolved and collected instead of resubmitted |
| `tests/workflow/test_runtime.py` | A crash after collection, decode, publication, or final commit never starts another provider call |
| `tests/workflow/test_runtime.py` | Graceful and hard coordinator interruption preserve attached calls and recover their permits |
| `tests/workflow/test_orchestrator.py` | Container exit drains and checkpoints without cancelling children; explicit cancellation still cancels |
| `tests/execution/test_remote_coordinator.py` | A detached loop reaches terminal without client polling; duplicate loop, claim, and completion inputs are idempotent; infrastructure replacement reloads checkpoints; uncaught coordinator errors stop without automatic retry or child cancellation; explicit resume reconciles; terminal status can reopen the ledger; different Execution Run IDs remain isolated |
| `tests/execution/test_dispatch.py` | Lost claim responses, claim replay, preemption with an active assignment, terminal-owner failure without same-run reassignment, and unknown-owner blocking |
| `tests/execution/test_single_submission.py` | Each Task gets at most one submission per Run; redelivery retains call identity; resume never retries failure; restart reuses valid publications and submits only conclusively unowned missing work |
| `tests/execution/test_deployment.py` | Explicit and history-resolved versions are pinned; unavailable versions block; restart creates a linked run and reuses publications |
| `tests/execution/test_identity.py` | Execution UUIDs are opaque and unique; workload keys never select paths; successor lineage uses a new UUID |
| `tests/execution/test_cli_location.py` | Explicit deployment and run flags reach the correct coordinator; mismatched ledger fields fail; optional call IDs remain non-authoritative |
| `tests/workflow/test_ledger.py` | Execution result, artifacts, Task, Node, and Provider Call finalize atomically without attempt rows or paths |
| `tests/service/test_gromacs_plan.py` | The fixed GROMACS graph preserves its parallel readiness waves |
| `tests/workflow/ppiflow/test_coordinators.py` | Candidate outcomes preserve identity, order, partial failures, and configured concurrency |
| `tests/app/test_alphafold3_production_contracts.py` | Search, run, request, marker, and seed-batch identities remain unchanged |

Each fault test uses a deterministic injected failure point and asserts both
the durable rows and the number of fake paid calls. Merely asserting a final
status is insufficient.

### Phase 1 — repair recovery semantics in place

Deliverables:

- map availability-check exceptions to `unknown`;
- preclaim workflow remote submission and preserve submission outcome unknown;
- finalize a successful workflow call, processed result, artifacts, Task, and
  Node under one recoverable synchronization protocol;
- stop cancelling attached child calls from the orchestrator exit hook and
  checkpoint best-effort instead;
- add explicit tests that no restart automatically duplicates uncertain work.

Exit gate:

- existing workflow APIs pass, and every injected crash has a deterministic,
  cost-safe recovery outcome.

Rollback:

- revert the implementation commit. Unfinished ledgers created with the
  reverted schema may be recreated; they need not remain readable.

### Phase 2 — extract immutable plans and graph algorithms

Deliverables:

- add immutable `ExecutionPlan`, `NodePlan`, and dependency validation;
- extract deterministic readiness and terminal-reachability functions;
- adapt the pure GROMACS operation plan and workflow builder to the same graph
  representation, replacing internal types directly where that simplifies the
  interface;
- keep dynamic work represented as a Node-owned Task factory, not mutable DAG
  vertices.

Exit gate:

- GROMACS selects the same parallel operations;
- workflows produce the same hashes, scheduled waves, and terminal pruning;
- no provider or database dependency exists in the plan module.

Rollback:

- revert the implementation commit; do not retain a runtime selection flag.

### Phase 3 — extract the provider-call state machine

Deliverables:

- move the proven preclaim/attach/recover transitions into
  `SqliteExecutionRepository` and behind the provider port;
- use `ModalJobSubmitter` as the behavioral baseline for uncertain spawn;
- add thin async service and sync workflow coordinator loops;
- replace service `job_operations` and workflow `remote_calls` with shared
  execution tables as each host migrates;
- add the explicit offline service-state transition command;
- expose a common read-only execution snapshot for logs and diagnostics.

Exit gate:

- GROMACS API timelines, log call IDs, cancellation, and result archives are
  unchanged;
- workflow call recovery and cancellation are unchanged or safer;
- all provider behavior is exercised through fakes in CI.

Rollback:

- revert the host migration commit. Do not retain a runtime migration switch
  after a host uses the shared repository.

### Phase 4 — add durable Tasks, batches, and budgets

Deliverables:

- persist immutable Task plans before submission;
- add explicit Task-to-call and Task-to-assignment links with database
  constraints enforcing at most one submission per Task per Execution Run;
- move bounded batching and permit accounting into execution internals;
- make PPIFlow the first runtime-discovered Task consumer;
- represent partial candidate outcomes without making a Node successful by
  implication;
- persist coordinator-scoped permit allocations and recovery without adding a
  distributed lease abstraction.

Exit gate:

- interrupted PPIFlow stages reuse validated candidate publications and do not
  repeat uncertain calls;
- stable candidate IDs and manifests are byte- or semantic-equivalent;
- resource tests prove that one call batch consumes the intended permits.

Rollback:

- revert the implementation commit and recreate incompatible unfinished
  workflow runs. Keep validated app-owned publications for reuse.

### Phase 5 — adapt AlphaFold3 without changing scientific authority

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
- route the Local Entrypoint through a remote run-scoped coordinator;
- let service and workflow hosts execute the same plan through their parent
  coordinator without creating a nested ledger;
- preserve current CLI inputs, outputs, and direct Child App Call behavior.

Exit gate:

- AlphaFold run IDs, request IDs, search identities, marker payloads, Volume
  paths, seed reuse, ranking order, and retrieval archives remain unchanged;
- an overlapping seed request performs only its missing seed work;
- partial search and seed failures preserve the same reusable publications;
- no automatic paid retry is introduced.

Rollback:

- adapters can return control to the existing pure pipelines because no marker
  or publication format changes in this phase.

### Phase 6 — replace generic App-Local Scheduler mechanics

Adopt two concrete dispatch adapters only after durable Tasks and batches are
proven:

1. BoltzGen exercises bounded direct fan-out, where each workload run key is
   one Task and one GPU Provider Call.
2. Rosetta exercises a SQLite-backed pull worker pool, where several Provider
   Calls claim Task microbatches from one Dispatch Batch through the
   coordinator.

Deliverables:

- add a reusable direct-fan-out adapter that admits ready Tasks as permits
  become available;
- add a reusable run-scoped remote coordinator and pull-work adapter for
  idempotent claims, worker-call attachment, and outcome reconciliation;
- keep only workload hooks, Modal decorators, and Volume bindings in each
  deployment's thin Coordinator Adapter;
- bind each Direct CLI App Run to a distinct App Run Ledger in the app's
  configured durable Volume;
- add durable Worker Assignments so lost claim responses are harmless and
  preempted provider inputs recover their current Tasks;
- keep SQLite single-writer: workers return outcome records or publish
  outputs, and only the coordinator commits execution transitions;
- remove BoltzGen's generic `bounded_map` orchestration and use Task state to
  prevent duplicate work within one coordinator;
- retain or replace BoltzGen's output lock only as a cross-coordinator
  publication claim;
- remove Rosetta's generic queue and worker-pool lifecycle from workload code
  and use ready Task rows as the remote work-stealing queue;
- preserve deterministic result ordering and existing CLI behavior.

Exit gate:

- interrupted BoltzGen runs reuse validated completed runs and do not duplicate
  an uncertain GPU call;
- Rosetta workers balance Tasks dynamically, and every Task is reconciled
  independently after partial worker failure;
- worker exit callbacks can be dropped without changing any recovery result;
- a coordinator restart reconstructs permits, batches, and attached worker
  calls from the SQLite ledger;
- workload modules contain scientific execution and validation rather than
  generic concurrent scheduling loops.

Rollback:

- revert each workload-adoption commit independently; validated outputs remain
  reusable and no publication format changes are required.

### Phase 7 — make WorkflowRuntime a host and remove duplication

Deliverables:

- delegate graph traversal, Task scheduling, call lifecycle, availability
  policy, and budgets from `WorkflowRuntime` to the execution kernel;
- retain workflow-specific artifact materialization, Volume synchronization,
  display, run layout, and Workflow Artifact Store;
- migrate remaining durable fan-out consumers;
- remove the temporary `WorkflowLedger` migration facade after callers
  compose the shared execution repository and Workflow Artifact Store;
- delete the legacy `attempts` table, `current_attempt_id`, attempt foreign
  keys, attempt counters, generic `NodeExecutionPolicy`, attempt status fields,
  and `attempts/<attempt-id>/` path segment;
- key node and Task staging paths directly by their stable identities;
- remove replaced coordinator loops and stale planned documentation;
- publish the final supported execution inspection surface.

Likely deletion candidates, only after equivalence:

- GROMACS-local readiness and all-completed algorithms;
- workflow-specific paid-call transition logic;
- AlphaFold3 `_bounded_remote_outcomes` and claimed seed-batch loops;
- PPIFlow durable candidate scheduling through bare `bounded_map`;
- BoltzGen incomplete-run fan-out and same-coordinator output locking;
- Rosetta queue population, worker-pool sizing, and cleanup orchestration;
- repeated generation-claim mechanics that have converged on the common port.

Exit gate:

- each concern has one implementation or a documented workload-specific
  reason to differ;
- the physical `ledger.sqlite3` contains shared execution tables and
  workflow-specific artifact tables without a second execution state machine;
- no `WorkflowLedger` migration facade remains;
- no migration switch or dead adapter remains.

Rollback:

- phase commits remain independently revertible; scientific publications
  require no reverse migration. Old local databases and unfinished workflow
  runs may require explicit recreation.

### Phase 8 — switch the CLI launch contract

Deliverables:

- make `biomodals app run` and `biomodals workflow run` resolve the
  deployment-local coordinator at an exact app or workflow version;
- accept an explicit `--version` override or parse `modal app history --json`
  once to select the current deployed version;
- preflight and persist the resulting Environment, deployment name, and
  numeric version before admitting any Task;
- use only exact versioned Function and Cls lookups after resolution;
- retain workload-specific argument parsing and local input staging in thin
  Local Entrypoints without creating local execution state;
- submit one detached coordinator loop per remote top-level run so progress
  does not depend on the CLI process remaining connected;
- add an explicit development mode for source-backed ephemeral execution and
  label its lack of cross-invocation resume;
- keep help, shell, and workflow dry-run behavior local and free of paid calls;
- print the Execution Run ID, Workload Run Key when present, Deployment
  Identity, and coordinator FunctionCall ID needed for inspection and recovery;
- add shared `biomodals run status`, `cancel`, `resume`, and `restart`
  commands backed by the standard deployment-local `ExecutionCoordinator`;
- require `--environment`, `--deployment-name`, `--deployment-version`, and
  `--execution-run-id` on lifecycle commands, with
  `--coordinator-call-id` as an optional observation hint;
- do not add an encoded reference format, reference parser, or local registry;
- update CLI command builders, help, README examples, and characterization
  tests together.

Exit gate:

- default app and workflow runs never place their coordinator in an ephemeral
  deployment;
- a second CLI process can address the same run-scoped coordinator using the
  recorded Deployment Identity and Execution Run ID;
- app and workflow fields exercise the same generic lifecycle commands;
- mismatched deployment or run fields fail before state mutation;
- a rolling deployment after admission cannot change any Provider Call target;
- unavailable or unretained versions fail before new paid work rather than
  falling back to a floating latest handle;
- attached calls from a now-unavailable deployment remain observable by
  FunctionCall ID and their publications are still validated;
- an explicit restart creates a linked Successor Execution Run and schedules
  only Tasks whose publications remain missing and whose predecessor ownership
  is conclusively terminal;
- development mode remains useful for source iteration but cannot be mistaken
  for a resumable deployed run;
- dry-run and help start no remote execution.

Rollback:

- restore the old ephemeral command builders only before users depend on the
  new durable CLI run contract. Remote ledgers and publications need no
  conversion.

## Suggested incremental commits

Use small commits in dependency order:

1. `docs: plan unified task scheduler`
2. `workflow: fix availability uncertainty`
3. `workflow: harden call recovery`
4. `execution: add plans and graph`
5. `service: adopt execution graph`
6. `workflow: adopt execution graph`
7. `execution: add call lifecycle`
8. `service: adopt call lifecycle`
9. `workflow: adopt call lifecycle`
10. `execution: add durable task state`
11. `ppiflow: adopt durable task fanout`
12. `alphafold3: adopt execution adapters`
13. `execution: add dispatch coordinators`
14. `boltzgen: adopt direct task fanout`
15. `rosetta: adopt sqlite work pool`
16. `cli: target deployed coordinators`
17. `execution: remove duplicate schedulers`

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

#### `workflow: harden remote call recovery`

Files:

- `src/biomodals/workflow/core/ledger.py`
- `src/biomodals/workflow/core/_runtime/remote_calls.py`
- `src/biomodals/workflow/core/_runtime/node_runner.py`
- `src/biomodals/workflow/core/orchestrator.py`
- `tests/workflow/test_ledger.py`
- `tests/workflow/test_runtime.py`
- `tests/workflow/test_orchestrator.py`

Change:

- record a durable submission identity before invoking the provider;
- attach the returned call ID to that identity;
- leave an unattached or interrupted submission explicitly outcome-unknown;
- finalize the processed result, artifacts, Task, Node, and Provider Call in
  one host transaction and Volume synchronization boundary;
- replace exit-time child cancellation with best-effort drain and checkpoint;
- remove the earlier transition that marks a call succeeded before its result
  is recoverable;
- add the fault-injection cases from the Phase 0 inventory.

Verification:

```text
uv run pytest tests/workflow/test_ledger.py tests/workflow/test_runtime.py
prek run --files \
  src/biomodals/workflow/core/ledger.py \
  src/biomodals/workflow/core/_runtime/remote_calls.py \
  src/biomodals/workflow/core/_runtime/node_runner.py \
  src/biomodals/workflow/core/orchestrator.py \
  tests/workflow/test_ledger.py \
  tests/workflow/test_runtime.py \
  tests/workflow/test_orchestrator.py
```

Rollback: revert the commit and recreate any unfinished workflow ledger
written by it. No compatibility reader is required.

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
   move to the shared execution repository. After migration, the
   `WorkflowLedger` facade is removed and workflow code retains a narrow
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
    An incomplete run with an unavailable version becomes Deployment-Blocked.
    Explicit restart creates a linked Successor Execution Run on a new version,
    revalidates publications, and schedules only missing Tasks whose
    predecessor ownership is conclusively terminal.
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
    recorded when possible, and the Run stays incomplete until explicit
    resume reconciles durable state.
27. **Workflow attempt removal — accepted 2026-07-29**: the workflow
    migration deletes `attempts`, `current_attempt_id`, attempt foreign keys and
    counters, `AttemptRecord`, generic `NodeExecutionPolicy`, attempt status
    output, and the `attempts/<attempt-id>/` path layer. Task, Node, Provider
    Call, and artifact records retain the one result, error, timing, and output
    state needed by the single-submission model.

## Definition of ready for implementation

Implementation begins only when:

- ADR 0006 is accepted;
- all decision gates are resolved;
- the proposed execution terms are reconciled into `CONTEXT.md`;
- Phase 0 scientific, cost-safety, and user-visible regression fixtures plus
  fault-injection points are enumerated as test cases;
- the first two implementation commits have exact file and rollback scopes.

Implementation remains paused until the remaining decision gates are resolved.
