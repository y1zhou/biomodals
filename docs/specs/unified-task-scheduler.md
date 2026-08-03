<!-- markdownlint-disable MD013 -->

# Unified task scheduler

Status: accepted and implemented; manual Modal smoke validation pending.

Implemented:

- the shared model, SQLite repository, deterministic admission, fixed-batch
  and pull-worker dispatch, Modal call boundary, cancellation, restart, and
  coordinator loops;
- the GROMACS service cutover and explicit pre-release service-state
  transition;
- the workflow ledger decomposition, remote workflow coordinator, deployed
  workflow CLI lifecycle, and successor recovery;
- PPIFlow runtime Task discovery for ReFold, Partial, and LigandMPNN, plus
  directly tracked initial-design, FlowPacker, and DockQ calls;
- AlphaFold3 search and inference adoption;
- BoltzGen direct Task fan-out;
- Rosetta pull-worker adoption and removal of its Modal Queue;
- remote per-run coordination and generic lifecycle commands for direct
  `biomodals app run` and `biomodals workflow run`;
- fail-closed production app launch when an entrypoint has not yet adopted a
  Deployment Coordinator Adapter, with source-backed execution available only
  through explicit `--development`.

Still pending:

- explicitly authorized manual Modal smoke tests. CI and local verification do
  not make paid provider calls.

This specification records the execution and recovery contract shared by the API
service, reusable workflow runtime, PPIFlow fan-out, and AlphaFold3 search and
inference pipelines. It defines a narrow `biomodals.execution` kernel rather
than an all-purpose `TaskManager`.

The target is one place to reason about:

- fixed DAG validation and readiness;
- runtime-discovered tasks inside fixed semantic nodes;
- single-submission Task state and its relationship to Provider Calls;
- safe submission, attachment, polling, cancellation, and recovery;
- recorded cache observations and the decision to reuse or compute;
- durable Result Envelopes between Modal completion and Task completion;
- batching and Run-level total/GPU Provider Call limits;
- reusable direct-fan-out and SQLite-backed pull worker-pool dispatch.

It is not a framework for workload handlers, scientific input/output parsing,
cache validation, or publication. Existing app and workflow code performs
those operations and reports their results to the kernel.

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
5. A successful call releases its slots only after a recoverable Result
   Envelope is durable; unfinished Tasks remain active until their scientific
   publications validate.
6. Parallel tasks run when their dependencies and Provider Call slots allow
   it.
7. One provider call may represent a batch of independently identified tasks.
8. Several provider calls may claim work from one SQLite-backed Dispatch Batch
   without opening or writing the repository themselves.
9. Each scheduling cycle greedily fills every feasible Provider Call slot in
   deterministic DAG-priority order.
10. Interrupted coordinator-local work is re-entered only after its
    publication is authoritatively missing; conclusive failure never replays.
11. Service, workflow, and app CLI entrypoints remain functionally testable;
   internal imports and unfinished schemas carry no compatibility promise.

## Pre-refactor execution authorities

This historical inventory records the duplicated authorities that motivated the
refactor. It is not a description of the current implementation; the shared
execution mechanics now live in `biomodals.execution`, while each row's
scientific and publication boundary remains workload-owned.

| Area | Pre-refactor authority | Reusable strength | Boundary preserved |
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
- A Task receives at most one scheduler admission in an Execution Run and at
  most one Provider Call submission or Worker Assignment. Retrying failed paid
  or local work requires a Successor Execution Run. Interrupted
  Coordinator-Local Tasks may re-enter the same operation only after publication
  observation, and active or unknown predecessor ownership blocks replacement.
- Only the repository call that creates a durable `submitting` Provider Call
  may invoke spawn. Duplicate or recovered callers never resubmit it.
- Scientific and cache identity excludes operational concurrency, placement,
  and resource allocation.
- A successor requires the same Workload Plan Fingerprint over normalized
  result-affecting inputs and declared scientific versions. Changed science
  requires a new root Run.
- A workflow declares the versions of its workflow-local scientific logic and
  every app or model that can affect its publications. These versions enter
  the Workload Plan Fingerprint even when a particular Node class has no
  node-local version hook.
- Workflow Node parallelism and Run-level Provider Call limits are different
  controls.
- AlphaFold3 raw searches, assemblies, templates, and seeds retain their
  current identities and Volume layouts.
- GROMACS continues to call the deployed app's established functions. The
  service does not rewrite the app or replace its CLI entrypoint.
- A GROMACS run directory has an immutable plan-identity marker. Reusing the
  human run name is allowed only for the same Workload Plan Fingerprint; a
  changed input or simulation setting requires another run name. Preparation
  is reusable only when the input PDB and every NVT, NPT, and production file
  required by downstream Nodes validates against its publication marker.
- Before writing an incomplete GROMACS directory, the adapter atomically
  claims its human run name in a workload-scoped Modal Dict. The Volume marker
  records the elected Execution Run but never acts as a file lock. Matching
  complete terminal publications remain read-only cache hits; only the same
  Run or an explicit Successor may repair incomplete output.
- The GROMACS coordinator records only the top-level deployed Function as the
  Task-owning Provider Call. Existing private nested calls to
  `find_traj_last_time_ns` and `postprocess_traj` remain an explicit legacy
  exception: they create no Tasks or ledger rows and are outside Run-scoped
  call-limit accounting. Do not use this exception as a template for new apps.
- PPIFlow keeps a fixed stage DAG while candidate work fans out inside a stage.
- Provider workers never write the coordinator's SQLite repository.
- Ready Task rows and Worker Assignments are the durable pull-work queue.
- A remote SQLite coordinator is routed by Execution Run and pinned deployment
  version to a provider pool capped at one container.
- The coordinator binding ships with each app or workflow deployment; there
  is no universal execution-coordinator deployment or workload registry.
- `biomodals app run` and `biomodals workflow run` target exact deployed
  versions directly from their local thin clients by default; they create no
  ephemeral launcher App. Source-backed ephemeral execution is explicit
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
  it without client polling until it becomes terminal or requires explicit
  recovery in `suspended` or `state_unknown`, then returns so its container may
  scale to zero.
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

## Domain model

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

### Terms

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
user-facing service stages; adapters map those concepts at their boundaries.

**Task** is the smallest independently identified unit whose cache and outcome
can be reasoned about. Every Task belongs to exactly one Node, and Tasks may be
discovered only when their containing Node starts. Each has a stable
Node-local key and deterministic fingerprint.

**Coordinator-Local Task** is a Task executed by caller-owned code inside the
exclusive coordinator process. It uses no Provider Call or call slot and may
re-enter the same idempotent operation after coordinator interruption only
when its publication is authoritatively missing.

**Task Fingerprint** is computed by the kernel once at discovery:

```text
sha256(canonical_json({
  "plan": workload_plan_fingerprint,
  "node": node_key,
  "task": task_key,
  "science": scientific_payload
}))
```

The workload supplies a JSON-compatible normalized `scientific_payload`; the
kernel uses compact sorted-key JSON, rejects non-finite numbers, and persists
the digest. File inputs are already represented by content digests.
Operational execution payloads are separate and excluded.

**Single-Submission Rule** means the kernel admits each Task once and creates
at most one Provider Call submission or Worker Assignment for it in one
Execution Run. Provider redelivery can re-execute that same call, and
coordinator recovery can re-enter one interrupted Coordinator-Local Task, so
this is not an exactly-once execution claim.

**Provider Call** is one concrete remote worker invocation, including its
durable provider call ID and observed lifecycle. It belongs to one Node and
can cover zero or many Tasks from that Node. A Modal Function Call is its
current provider implementation.

**Result Envelope** is a small durable JSON-compatible operational record
captured before a successfully returned Provider Call becomes `succeeded`. It
maps that call to Task-specific durable result references or conclusive
diagnostics needed to resume decoding and publication. It contains no large
scientific payload and is not a Workload Publication.

**Dispatch Batch** is a durable grouping of Tasks from one Node offered to one
Provider Call or a shared worker pool. Fixed-batch dispatch binds every Task
to its call at preclaim; pull-worker dispatch records Task-to-call ownership
only through Worker Assignments.

**Fixed-Batch Dispatch** groups compatible ready Tasks from one Node into one
Provider Call and persists the complete mapping at preclaim. The mapping is
immutable after spawn authorization.

**Pull-Worker Dispatch** admits worker Provider Calls before Task ownership is
known. Workers later claim bounded Task microbatches through the coordinator,
which checkpoints Worker Assignments before returning payloads.

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
| `running` | No | The DAG is advancing or waiting on dependencies, call slots, publications, or attached calls |
| `cancel_requested` | No | Explicit cancellation is durable and attached work is being reconciled |
| `suspended` | No | A coordinator error or unknown result validation stopped admission until explicit resume |
| `state_unknown` | No | Provider submission, call state, or cancellation outcome cannot be established; replacement is forbidden |
| `succeeded` | Yes | Every terminal Node has a complete validated scientific result |
| `partial` | Yes | The terminal result boundary is usable but explicitly incomplete |
| `failed` | Yes | A required terminal result cannot be produced or the Run cannot continue |
| `cancelled` | Yes | Cancellation completed conclusively |

Two nullable fields refine the current status:

- `status_reason` is a stable, machine-readable kernel code;
- `status_message` is a human-readable diagnostic and is never control-flow
  input.

A repository transition atomically replaces or clears both fields with
`status`. The first version has a closed, kernel-owned `RunStatusReason`:

| Status | Required `status_reason` |
| --- | --- |
| `suspended` | `coordinator_error` or `result_validation_unknown` |
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
| `cancelled` | Yes | Explicit cancellation or result pruning stopped active Node work |
| `skipped` | Yes | The Node became unreachable or was unnecessary before work started |

Readiness is derived from dependencies and their accepted outcomes; it is not
a persisted `ready` status. A Node satisfied entirely by reusable
publications becomes `succeeded`, while its Tasks retain cache provenance.
Run-level `cancel_requested`, `suspended`, and `state_unknown` are not
duplicated onto Nodes. A workload excludes an optional branch from its
immutable plan rather than creating a Node that is already `skipped`.

Each immutable dependency edge has `accept_partial: bool = False`:

- `succeeded` always satisfies the edge;
- `partial` satisfies only an edge that explicitly opts in;
- `failed`, `cancelled`, and `skipped` never satisfy an edge.

A Node becomes ready only when every dependency is satisfied. If an
unacceptable upstream outcome is terminal, the dependent Node becomes
`skipped`; that skip may propagate through its descendants. Explicit Run
cancellation takes precedence and marks unfinished Nodes `cancelled` instead
of applying dependency-skip propagation. The edge-level boolean lets one Node
accept a partial candidate set from one dependency while requiring complete
output from another, without a generic accepted-status policy.

Terminal Execution Nodes—the DAG leaves with no downstream dependency—form
the scientific result boundary. Scheduling is result-driven:

1. caller-owned workload code validates every terminal Node publication and
   records its observation before preparing dependency inputs or Tasks;
2. treat an `available` complete publication as successful without scheduling
   its ancestors;
3. for each incomplete terminal, walk backward through only its ancestor
   closure, stopping whenever another Node result is `available`;
4. schedule the remaining required Nodes forward in dependency order.

The recorded observation uses the same `available`, `missing`, or `unknown`
vocabulary as Task publication validation. `missing` expands the backward
closure. `unknown` blocks new work; it never becomes a cache miss. A workload
without an aggregate reusable publication deliberately reports `missing`. A
partial publication is not an available complete Node result, but its
successful Task publications remain eligible for granular reuse after the
Node enters the repair closure.

An `unknown` Node or Task result observation leaves that record nonterminal
and moves the Run to `suspended` with
`status_reason=result_validation_unknown`. New admission stops, attached
Provider Calls retain ownership and continue, and no automatic validator retry
loop runs. Explicit `resume` rechecks the publication and continues the same
Run after a conclusive observation. `state_unknown` is not used because no
provider ownership is ambiguous.

The Run succeeds when every terminal Node succeeds. A failed, cancelled,
partial, or skipped upstream Node does not override complete terminal results.
This is intentionally not an aggregation across every planned Node: the graph
describes ways to obtain scientific results, while validated terminal
publications determine whether those results exist. It preserves fast cached
return and the current workflow runtime's terminal-pruning semantics.

After provider ownership and control states are conclusive, terminal Nodes
aggregate strictly for that Execution Run:

| Terminal result boundary | Run outcome |
| --- | --- |
| Every terminal Node is `succeeded` | `succeeded` |
| Every terminal Node is `succeeded` or `partial`, and at least one is `partial` | `partial` |
| At least one terminal Node is `failed` or `skipped`, and none is `cancelled` | `failed` |
| At least one terminal Node is `cancelled` | `cancelled` |

An upstream status never changes this table. If a subset of independent
outputs is scientifically usable, the workload represents that fact with a
terminal aggregation Node whose Node policy can produce `partial`; the kernel
does not guess that an arbitrary subset is usable.

Result pruning preserves ownership safety:

- a pending unnecessary ancestor becomes `skipped` with
  `status_reason=result_already_satisfied`;
- an already-terminal ancestor keeps its historical status;
- a running unnecessary ancestor admits no more work while the coordinator
  cancels and reconciles its attached Provider Calls;
- if cancellation wins, that Node becomes `cancelled` with
  `status_reason=result_already_satisfied`; if work reaches another conclusive
  terminal outcome first, that observed outcome is retained;
- the Run remains `running` during cleanup and moves to `state_unknown` if a
  cancellation outcome cannot be established.

`result_already_satisfied` is the only initial Node `status_reason` and is
valid only with `skipped` or `cancelled`. Every other Node status requires a
null reason; workload failures remain in the Node error record. A successful
result boundary may determine Run success only after every unnecessary remote
owner is conclusively terminal. The normal cache-hit path remains immediate
because no ancestor work is admitted before terminal validation.

### Task statuses

The kernel uses exactly six Task statuses:

| Status | Terminal | Meaning |
| --- | --- | --- |
| `pending` | No | Discovered but not yet cache-satisfied or assigned |
| `running` | No | Durably owned by local execution, a Provider Call, or a Worker Assignment |
| `succeeded` | Yes | The required Workload Publication was validated |
| `failed` | Yes | Execution or publication validation conclusively failed |
| `cancelled` | Yes | Explicit cancellation or result pruning stopped owned Task work |
| `skipped` | Yes | Admission stopped before the Task acquired an execution owner |

Provider submission and attachment phases belong to the Provider Call rather
than the Task. If the owner's outcome becomes unknown, the Task remains
`running`, the owner retains it, and the Run becomes `state_unknown`; the Task
does not become eligible for replacement work. Task success records whether it
came from cache validation or execution as provenance, not as a `cached`
status. Individual Tasks are never `partial`; the Node aggregation policy
derives partiality.

A `running` Coordinator-Local Task has no independently surviving remote
owner. If its coordinator disappears, the exclusive replacement observes its
publication before deciding whether the same operation may be re-entered.
Missing output authorizes recovery of that same Task, not a transition back
to `pending`, a new attempt, or a second scheduler admission.

Result pruning applies the same ownership boundary at Task granularity:

- do not create rows for Tasks that were never discovered;
- mark discovered, pending, unowned Tasks `skipped` with
  `status_reason=result_already_satisfied`;
- keep owned Tasks `running` until their Provider Call or Worker Assignment is
  conclusive;
- mark an owned Task `cancelled` with the same reason only when cancellation
  wins; if execution or publication validation completes first, retain that
  observed terminal outcome;
- never rewrite an already-terminal Task;
- preserve `running` ownership and set the Run to `state_unknown` when
  ownership or cancellation remains unknown.

`result_already_satisfied` is the only initial Task `status_reason` and is
valid only with `skipped` or `cancelled`. Every other Task status requires a
null reason; Task failure diagnostics remain in its error record.

### Task discovery transaction and durability

Caller-owned workload code constructs the complete finite
`Sequence[TaskPlan]` for a Node and gives it to the kernel. The kernel validates
unique stable Node-local keys, assigns each Task its zero-based encounter
ordinal from that Sequence, computes each Task Fingerprint once, then
atomically inserts every Task and marks the Node `discovery_complete`. Resume
loads persisted ordinals and fingerprints and does not ask the caller to
reconstruct them during status polling or Modal-call observation.

The discovery transaction is committed to coordinator-local SQLite before
admission continues. A Volume-backed coordinator does not issue a standalone
remote Volume commit for discovery alone. The first provider preclaim,
Worker Assignment response, or Coordinator-Local Task ownership barrier
checkpoints the discovery transaction together with the ownership transition
before paid work or another container can act on it. A crash before that
boundary may cause the caller to reconstruct the whole set, but no paid work
has been authorized. A crash after the boundary reloads the persisted Tasks
and never reconstructs them again for that Node in the same Run. Empty sets
follow `allow_empty_result`.

The first kernel version has no streaming, incremental, paginated, or
worker-side discovery. This deliberately keeps SQLite's ready queue complete
and prevents workers from observing a partially discovered Node. Encounter
order is an operational admission tie-break only. Scientific ordering, when
relevant, belongs in stable Task payloads and therefore in the Task
Fingerprint rather than in the encounter ordinal.

`TaskPlan` keeps its normalized scientific payload separate from the
operational execution payload used to prepare provider arguments. The latter
may contain staging paths, batching choices, concurrency, and resource
requirements without changing scientific cache identity. A Successor
Execution Run reuses a Task publication only when Node key, Task key,
kernel-computed fingerprint, and workload validation all match. The first
version uses one fixed standard-library SHA-256/canonical-JSON implementation;
it has no codec or hashing plugin layer and never reads large files while
fingerprinting Tasks.

### Admission-set and synchronization boundaries

A scheduling cycle treats all selected Provider Call candidates as one
admission set:

1. Resolve each distinct Provider Binding once, outside the SQLite writer.
2. Create all authorized preclaims in one transaction and issue one explicit
   Volume checkpoint.
3. Submit calls outside the writer.
4. Attach all returned handles or classify submission errors in one transaction
   and issue one explicit Volume checkpoint.
5. Observe calls outside the writer, then apply one reconciliation transaction
   and at most one terminal-result checkpoint for that observation set.

An interruption after step 2 cannot authorize the same paid call again.
Unattached `submitting` records become `outcome_unknown` during recovery.
If cancellation becomes durable during step 3, every newly attached handle is
cancelled immediately after step 4 without placing the Modal RPC under the
writer. Before each later spawn in the admission set, the spawn-owning process
rechecks cancellation under the writer and conclusively cancels any preclaim
whose provider side effect has not begun. This prevents a cancellation that
arrives during one slow spawn from launching the rest of the selected set.
Ordinary planning, cache observation, policy persistence, and unchanged or
running provider polls use only their local SQLite transactions. They do not
issue a remote Volume commit.

The run-scoped drive lock excludes a second scheduling loop. Status and
pull-worker callbacks do not acquire it; cancellation acquires it only when
taking over driving after its request is durable. The SQLite writer protects
short state snapshots and transitions plus explicit transition-and-Volume
barriers; Modal resolve, spawn, observe, cancel, and result encoding never run
while that writer is held. SQLite is closed only for an explicit Volume commit
or reload. Initial Volume refresh belongs to the coordinator host, and reloads
thereafter occur only when another container may have published data.
Workload callbacks snapshot immutable ledger records under the writer, perform
filesystem or provider work after releasing it, and reacquire the current
repository for mutations. A repository reference must not cross a Volume
barrier because that barrier closes and reopens SQLite. Cache and model Volumes
are refreshed only for successful Nodes whose functions write those Volumes.

## Responsibility boundary

| Concern | Execution kernel owns | Workload or host owns |
| --- | --- | --- |
| DAG | Validation, topological readiness, terminal reachability, and deterministic admission rank | Ordered Nodes, dependencies, semantic labels |
| Task planning | Immutable task records, encounter ordinals, canonical fingerprint calculation, dependency links | Constructing the complete ordered Task sequence, normalized scientific payload, and content digests |
| Cache | Recording `available` / `missing` / `unknown` observations and applying scheduling policy | Running validation logic and inspecting markers, manifests, and content |
| Inputs | Persisting already-normalized Task plans and fingerprints | Parsing, validation, staging, and Modal function arguments |
| Calls | Modal submit, attach, resolve, poll, cancel, recover state machine, and durable Result Envelope boundary | Function selection and normalizing its small returned value |
| Local execution | Durable ownership and publication-first recovery transitions | Invoking idempotent local code, Task-specific staging, and atomic publication |
| Dispatch | Durable fixed batches, direct fan-out, pull claims, call tracking, stable image cohorts, and outcome routing | Modal binding, Runtime Image Key, Task payloads, compatibility keys, and per-Task decoding |
| Outputs | Persisting Result Envelopes and reported Task outcomes in legal order | Envelope decoding, schemas, scientific validation, paths, and publication |
| Batching | Stable grouping by compatibility and encounter order, immutable call mapping, and outcome distribution | Positive maximum Tasks per call and whether batching changes scientific identity |
| Resources | Run-scoped total and GPU Provider Call admission counts | Service admission, Modal decorators, deployment limits, cross-coordinator policy |
| Persistence | State schema, legal transitions, and atomic repository operations | Repository location, transaction integration, Volume synchronization |
| Presentation | Bounded lifecycle overviews, full diagnostic snapshots, and stable events for adapters | HTTP Jobs, CLI output, timelines, logs, admin policy |

The kernel is a caller-driven library rather than an inversion-of-control
framework. Workload code constructs plans and Tasks, records cache
observations, consumes Result Envelopes, publishes outputs, and reports Task
outcomes through ordinary runtime operations. The kernel never loads a
per-Node handler, calls a workload protocol, or parses PDB, A3M, Parquet,
archives, or other scientific formats.

## Minimal kernel shape

Add the package incrementally. Do not scaffold empty modules in advance.

```text
src/biomodals/execution/
  __init__.py             # deliberately small supported surface
  model.py                # immutable plan and state value objects
  sqlite.py               # schema and transitions on a host connection
  scheduler.py            # graph readiness, batching, and call limits
  modal.py                # Modal call lifecycle and remote coordination
  runtime.py              # caller-driven composition facade
```

The initial internal interface should be no larger than:

- `ExecutionPlan`
- `NodePlan`
- `TaskPlan`
- `TaskResult`
- `ExecutionRuntime`

Each composition root supplies its SQLite connection and transaction,
already-constructed plans and Task inputs, and Modal bindings explicitly. It
uses the runtime to advance scheduling, reads durable Result Envelopes,
performs workload-specific publication, and records outcomes. Do not add a
workload-handler hierarchy, provider plugin layer, callback registry, global
registry, plugin discovery, YAML workflows, or import-time Modal app or Volume
bindings. Each app and workflow declares only its thin decorated coordinator
wrapper over `execution.modal`.

`ExecutionRuntime.advance_once` owns the common reconciliation order. Its
smaller operations own the terminal-first publication walk, ready-Node Task
discovery, completed-call decoding boundary, result-driven pruning, Node/Run
aggregation, and fixed-call candidate selection. Workload adapters pass
direct validator and constructor functions; they must not reimplement the
cycle or introduce a handler registry.

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
and advances the DAG until the Execution Run becomes terminal or enters
`suspended` or `state_unknown`, where explicit operator action is required. The
launching CLI may observe it but need not stay connected or poll to make
progress. Lifecycle methods and pull-worker claims or completions may enter the
same Run-Scoped Coordinator Pool concurrently; they all use its serialized
writer. After preemption, a replacement Coordinator Attempt reloads the ledger
and continues reconciliation. When automatic driving stops, the input logs its
status and durable Provider Call slot occupancy, then returns. Coordinator
classes use Modal's two-second minimum idle scale-down window, so a later
`status` call may briefly start a fresh container to read the retained ledger.
“Coordinator loop” is an internal activity, not a public `drive` CLI command or
a new kernel domain type.

Provider redelivery of an interrupted coordinator input is recovery, not an
application-level retry. An uncaught exception from coordinator code stops
admission, leaves attached Provider Calls running, and records its diagnostic
when possible. It sets the Run to `suspended`; the adapter does not
automatically retry that exception. Explicit `resume` reloads the durable
ledger, reconciles attached calls, and then returns the same Execution Run to
`running`.

A workload or Provider Call failure is different: it terminally fails the
affected Task, and the Node aggregation policy derives the Node outcome. The
Run outcome is derived from the terminal scientific result boundary rather
than every intermediate Node. The coordinator reports that outcome and
returns when the required result boundary is terminal. It does not convert a
known Task failure into a resumable coordinator error.

Every app and workflow deployment exports its thin class under the standard
name `ExecutionCoordinator`. Its version-pinned, run-parameterized instance
provides the common `status`, `cancel`, and `resume` lifecycle methods.
`restart` resolves a new Deployment Identity and creates a new coordinator
instance linked to the predecessor. Workload launch and result-retrieval
methods may remain deployment-specific.

Before submitting an app coordinator, the client stages both its bounded
immutable workload request and a bounded immutable launch identity declaring
either a root Run or the exact predecessor Execution Run ID. This lets an
immediate `cancel` create the durable Run with correct lineage even when it
wins the race with `run` or `restart`; launch identity is never inferred from
workload fields, output claims, or matching command text.

Generic restart uses two coordinator inputs. `prepare_restart` synchronously
validates the predecessor, applies operational overrides, and checkpoints the
immutable Successor request and launch identity. Only after that succeeds does
the CLI spawn `drive_prepared`. This is a submission boundary, not a new Run
status: preparation never admits Provider Calls. Direct apps stage immutable
request and launch files. Workflows may also persist the pending Successor
ledger and reusable publication rows so immediate cancellation can act on
durable state. A workflow entrypoint's `--restart-from` option uses the same
two-input boundary through `prepare_restart_from` and `drive_prepared`; the
combined workflow coordinator method is intentionally not exposed.

Whenever a direct-app coordinator opens a staged request, it compares the
plan's declared scientific versions with the versions loaded by the target
deployment. Conditional versions, such as an optional engine or reference
dataset, are compared only when the plan uses them. A mismatch prevents root
or Successor execution before claims or ledger state are written. Pre-staged
request and launch files may remain as inert submission inputs.

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
Its repair closure starts at predecessor terminal Nodes in `partial`,
`failed`, `skipped`, or `cancelled`, plus any `succeeded` terminal whose
publication no longer validates. It walks backward through their ancestor
closure and stops at complete reusable publications. Valid Task successes
inside the closure become cache-satisfied Tasks; partial publications are not
mistaken for complete terminal results. Missing or conclusively invalid
publications are eligible for new Tasks only when predecessor ownership is
conclusively terminal; active or unknown ownership blocks replacement. Once
those Tasks succeed, previously untouched downstream Nodes become ready
normally. The predecessor remains terminal, and the successor receives its
own independently aggregated outcome.

Generic restart reuses the predecessor's stored immutable scientific plan.
Launch-time `--restart-from` normalizes the supplied workload inputs and
requires its Workload Plan Fingerprint to equal the predecessor's before
creating successor state. Changed file content, seeds, scientific parameters,
or declared result-affecting versions require a new root Run. Operational
concurrency, batching, resource settings, and the newly resolved Deployment
Identity may differ.

Deployment Identity is not scientific identity by itself. A new deployment's
caller-owned workload code must accept the stored plan schema and preserve
every declared result-affecting tool, model, adapter, and schema version before
the successor may reuse publications. Otherwise restart is rejected and the
user must create a new root Run.

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
and optionally waits for or retrieves the result. Normal CLI execution invokes
that client in the CLI process and hydrates the pinned deployed coordinator
directly rather than starting a source-backed ephemeral launcher. The client
also resolves lazy named objects in the coordinator's explicit Modal
Environment so input staging and result retrieval cannot drift to the local
profile default.

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
ordinary plan-construction and result-processing functions to the owning
coordinator; its markers and validated publications remain scientific
authority in every call shape.

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

A replacement Attempt reloads the latest durable repository, derives active
total and GPU call counts from nonterminal Provider Calls, reconstructs
batches, resolves attached calls by ID, and resumes observation. Correctness
must also survive a hard kill before the handler runs or finishes. Lifecycle
hooks and background Volume commits reduce lost progress but are not
correctness boundaries.

For a Volume-backed repository, a local SQLite commit is not sufficient when
ordering state against a provider side effect. Required preclaims and call
attachments must be made visible through an explicit, serialized Volume
checkpoint. The current workflow exit behavior that cancels active child calls
must be removed when it adopts this policy. Explicit user cancellation remains
separate and may cancel those calls.

A local SQLite commit is otherwise sufficient within one live coordinator
container. Planning, cache observation, dispatch-policy persistence, and
unchanged provider polling must not explicitly commit or reload the Modal
Volume on each pass. Modal's periodic and final snapshots provide ordinary
progress persistence. Explicit Volume commits are reserved for state ordered
before external side effects or cross-container responses, plus terminal and
error handoff. Explicit reloads are reserved for publications made by another
container; the workload adapter must invalidate planning observations after a
reload.

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
  active call slot, and Worker Assignment;
- a normal result may carry independent per-Task outcomes for a batch;
- a conclusive terminal call failure fails its unfinished Tasks and releases
  its total and optional GPU call slots;
- a successful call releases its slots after its Result Envelope is durable;
  unfinished Tasks remain `running` through decode and publication;
- a missing or invalid expected publication then fails only the affected Task;
- unknown call state preserves `state_unknown` and forbids replacement work.

Direct one-Task-per-call execution needs no separate remote assignment store.
For a pull Dispatch Batch, ready Task rows are the queue. A worker sends
`claim(worker_id, capacity, request_id)` to the coordinator. In one serialized
operation, the coordinator:

1. returns an existing result if `request_id` was already processed;
2. selects ready Tasks that fit the worker's capacity;
3. records their Worker Assignments against the already-admitted worker call;
4. commits SQLite and crosses the explicit Volume durability boundary;
5. only then returns the Task payloads.

A lost response can therefore be requested again without creating another
assignment. If the coordinator dies before the checkpoint, no payload has
been returned; if it dies afterward, the replacement coordinator recovers the
same assignments. A restarted provider input repeats its claim request and
retains its work. A conclusively failed owner call fails its unfinished Tasks;
no different Provider Call may claim them in the same Execution Run.

Workers publish their outputs before sending an idempotent completion report.
The coordinator records individual Task outcomes in one serialized
transaction. An early validated completion report may finish a Task while its
worker call is still active. The call retains its single active slot until its
terminal Result Envelope is durable, then releases that slot independently of
any unfinished Task publication. A lost completion response is harmless, and
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
assignment, or call state.

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
- Node discovery-complete checkpoints and immutable Task plans;
- submission tokens, provider targets, call IDs, and observed outcomes;
- Node result observation timestamps and cache-versus-current-Run completion
  provenance, without workload publication contents;
- execution timestamps, Run `status_reason` and `status_message`, Task and Node
  errors, single-submission state, Run call limits, and each call's GPU
  classification;
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

Routine lifecycle and service reads use a bounded Execution Overview: the Run,
its Nodes, one representative Provider Call per Node, and aggregate active-call
counts. The representative is the newest active call when one exists, otherwise
the latest terminal call. The view deliberately excludes Task payloads and
historical call ownership.
The complete Execution Snapshot remains available for explicit diagnostics and
tests, but is not a list- or status-endpoint projection.

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

### Dispatch modes and deterministic batching

The first kernel supports exactly two remote dispatch modes. Both keep Tasks
as the independently cacheable, validated, and reported unit while Provider
Calls remain the unit counted by remote-call limits.

#### Fixed-batch dispatch

The workload supplies operational dispatch descriptors for ready Tasks:

- the resolved provider binding and whether it uses a GPU;
- an optional opaque Runtime Image Key;
- a stable compatibility key;
- a positive `max_tasks_per_call`;
- provider argument construction for an ordered Task collection;
- result decoding into independent per-Task outcomes.

The kernel first removes Tasks whose publications validate as `available`.
Within one Node and provider binding, it groups the remaining ready Tasks by
compatibility key, preserves `task_ordinal` within each group, and chunks each
group up to `max_tasks_per_call`. A batch containing fewer Tasks is valid only
as the remaining tail. The batch candidate's admission tie-break is the first
constituent Task ordinal. Batches never span Nodes or provider bindings.

Batch formation is a pure, side-effect-free operation. The serialized
preclaim then atomically:

1. rechecks Task readiness and ownership;
2. checks the total and optional GPU call ceilings;
3. creates the Dispatch Batch and `submitting` Provider Call;
4. assigns every constituent Task directly to that call; and
5. returns spawn authorization only after the host durability boundary.

After preclaim, the Task-to-call mapping is immutable. Recovery observes or
collects the same call and never repacks its Tasks. Its Result Envelope
preserves the returned per-Task references or diagnostics before the call
becomes `succeeded`. A successful call may still contain failed or invalid
Task results, which per-Task decoding and publication validation record
independently. A failed call preserves any already-validated Task successes
and fails only its unfinished Tasks.

#### Pull-worker dispatch

For work-stealing Nodes, the Dispatch Batch represents the Node's durable
ready Task pool. DAG-priority admission creates worker Provider Calls without
preassigning Tasks. Each call consumes one total and optional GPU slot.

A running worker sends an idempotent claim request with its bounded capacity.
The coordinator selects ready Tasks in `task_ordinal` order, writes
call-bound Worker Assignments, commits and checkpoints them, and only then
returns payloads. The worker may repeat the same request after a lost response
and may claim another microbatch after reporting completion. No Task moves to
a different call after assignment in the same Execution Run.

An existing request ID always replays its durable assignments. A new request
receives work only while its Run and Node are both `running` and its Provider
Call is `submitting`, `attached`, or `running`. Suspended, unknown, cancelling,
pruned, or terminal work returns an empty claim. Before selecting Tasks, the
repository applies the Node aggregation policy so the first `fail_fast`
failure skips unowned siblings without waiting for another coordinator poll.

The workload declares one positive `claim_capacity`, the maximum Tasks a
worker can own concurrently. Pool size is derived rather than separately
configured:

```text
desired_workers =
    ceil(nonterminal_tasks_in_node / claim_capacity)

new_worker_candidates =
    max(0, desired_workers - nonterminal_worker_calls_in_node)
```

The scheduler then applies DAG priority and the remaining total and GPU call
slots. Pending Tasks and assigned-but-unfinished Tasks count toward the
numerator. Cache-satisfied, succeeded, failed, cancelled, and skipped Tasks do
not. `submitting`, `attached`, `running`, `outcome_unknown`, and
`state_unknown` worker calls count against the desired pool and the Run's call
limits.

A worker may claim repeatedly, which provides work stealing when Task
durations differ. The coordinator does not cancel already-admitted workers
when the desired count falls. They finish owned work and exit once no unowned
Task remains. Because claims are serialized, a worker can lose a race, receive
no Tasks, and return successfully. The kernel adds no separate
`max_worker_calls`, utilization feedback loop, lease, or idle timeout; Modal
decorator limits still govern actual provider containers.

#### Policy persistence and scientific identity

One Run persists its dispatch mode, compatibility descriptors, provider
bindings, GPU declarations, Runtime Image Keys, and maximum batch or claim
sizes as operational policy. Resume reloads that policy. A Successor Execution
Run may choose different operational batching or worker counts while reusing
validated scientific publications.

These values are excluded from Workload Plan and Task Fingerprints unless they
change scientific meaning. Any result-affecting ordering or batching parameter
must instead appear in the normalized scientific payload. The kernel adds no
dynamic bin-packing, duration estimation, byte-size optimization, cross-Node
batching, or workload-owned durable queue.

### Resource ownership

The first kernel has exactly two remote-admission limits inside one Execution
Run:

- `max_active_provider_calls`: a positive ceiling on all nonterminal Provider
  Calls;
- `max_active_gpu_provider_calls`: a nonnegative ceiling on the subset bound
  to functions with a GPU allocation, no greater than the total ceiling.

The resolved provider binding declares `uses_gpu: bool`, which is persisted on
the Provider Call and excluded from scientific fingerprints. The kernel does
not inspect Modal decorators. Under the serialized writer, submission preclaim
atomically counts nonterminal calls, checks the total and optional GPU ceiling,
and creates the `submitting` call only when both slots are available.

Every nonterminal Provider Call consumes exactly one total slot. A GPU call
also consumes exactly one GPU slot. `submitting`, `outcome_unknown`, and
`state_unknown` retain those slots because a container may exist. A terminal
call releases its slots automatically by leaving the derived active count. A
successful return becomes terminal only after its Result Envelope crosses the
host durability boundary; unfinished Task publication does not retain the
call slot. There is no allocation table or variable permit cost.

A call that owns a batch of Tasks still consumes one slot, and each pull
worker call consumes one slot regardless of how many Tasks it claims. Local
coordinator work consumes none. These limits bound in-flight remote calls and
conservatively approximate container fan-out. Modal may pack concurrent calls
into fewer containers, and its decorators remain authoritative for CPU, RAM,
accelerator type, GPU device count, timeout, and deployment container limits.

The kernel does not own:

- per-user, per-tool, or service-wide active-Job admission limits;
- the administrator settings from which a service resolves Run call limits;
- Modal CPU, GPU, memory, timeout, accelerator, or deployment concurrency;
- limits shared across different coordinators or Execution Runs.

No shared-lease interface or Modal Dict implementation is added in this
refactor. If a future workflow truly needs a hard limit across multiple
coordinators, that concrete requirement must define the failure and recovery
semantics before introducing another storage seam. The existing ADR principle
still applies: a hard distributed limit cannot be implemented by pretending
separate in-process counters are global.

### DAG-priority admission

The coordinator fills all currently feasible Provider Call slots in one
scheduling cycle. It does not admit at most one call per Node and then require
another pass. This follows Snakemake's useful scheduling shape—greedily select
a feasible set from ready work—while keeping Biomodals' ranking deliberately
small and specific to its result-driven DAG.

After result observation determines the current required DAG or Successor
Repair Closure, the plan layer annotates each required Node with:

- `depth`: the longest dependency path from a required source Node to this
  Node, where a larger value means farther downstream and closer to a
  scientific terminal result;
- `unblocking_span`: the number of distinct required, unfinished descendant
  Nodes reachable from this Node;
- `node_ordinal`: the zero-based position at which the Node occurs in the
  ordered `ExecutionPlan`.

The first two values are recalculated only when the required closure or a Node
terminal outcome changes, not during provider polling. `node_ordinal` is
immutable. Each discovered Task similarly stores its position in the returned
`TaskPlan` Sequence as `task_ordinal`. A direct-call candidate uses its Task
ordinal; a batch uses the ordinal of its first constituent Task. Pull-worker
call candidates enumerate the Node's ready Tasks in the same order.

The scheduler processes candidates in this order:

```text
greater depth
  -> greater unblocking_span
    -> GPU candidates, then CPU candidates
      -> stable Runtime Image Key cohorts
        -> node_ordinal, then task_ordinal
```

Depth and unblocking span form the graph rank. Resource and image preferences
apply only among candidates with the same graph rank: lower-ranked GPU work
never displaces graph-critical CPU work.

Within an equal graph-rank band, all GPU candidates are considered before CPU
candidates. Within each resource class, the scheduler stably groups candidates
by Runtime Image Key. Image cohorts are ordered by the earliest encountered
candidate they contain, and candidates retain encounter order within the
cohort. For example:

```text
encounter: A(image-x), B(image-y), C(image-x), D(image-y)
cohorted:  A(image-x), C(image-x), B(image-y), D(image-y)
```

The resolved provider binding supplies the opaque Runtime Image Key. An absent
key is treated as unique to that binding. The key and `uses_gpu` are persisted
operational metadata and are excluded from Workload Plan and Task
Fingerprints.

The coordinator walks the resulting sequence once, preclaiming candidates
that fit the remaining total and GPU ceilings until no total slot or feasible
ready work remains. With five total slots, two feasible GPU candidates, and
any number of equal-graph-rank CPU candidates, it admits both GPU candidates
and then three CPU candidates. A GPU candidate skipped because the GPU ceiling
is full does not prevent a CPU candidate from using a total slot. The GPU
ceiling is an upper bound, not a quota.

Admission operates on Provider Call candidates rather than raw Tasks. Two
compatible GPU Tasks may already form one fixed-batch candidate and therefore
consume one GPU and one total slot.

Encounter ordinals are persisted so recovery does not depend on SQLite row
order, Python set iteration, completion timing, or random identifiers. Runtime
image cohorting never uses currently active calls as a signal, never holds a
slot open, and never changes provider autoscaling settings. Modal documents
[Image layer caching](https://modal.com/docs/guide/custom-container)
separately from [Function and Class container
lifecycle](https://modal.com/docs/guide/lifecycle-functions), so the kernel
does not promise warm-container reuse across different Functions sharing an
Image. If execution order changes scientific meaning, the workload must encode
that order in its normalized scientific payload instead.

The ready set is finite because Node Task discovery is checkpointed as one
complete finite Sequence. Consequently, this policy needs no round-robin
cursor, aging, priority weights, per-Node quota, preemption, or scheduler
plugin. A high-ranked Node may fill every available call slot; lower-ranked
finite work becomes eligible as those calls finish.

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
| `state_unknown` | No | The call ID exists, but state, terminal result recovery, or cancellation is inconclusive |
| `succeeded` | Yes | The provider call returned and its Result Envelope is durably recoverable |
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
terminal provider state, returned result recovery, or cancellation outcome
cannot be established. Both preserve Task ownership and prohibit replacement
work until explicit reconciliation establishes one of the listed transitions.

Explicit cancellation intent takes precedence over every later provider
uncertainty. While a cancelling Run is represented as `state_unknown`, it keeps
`status_reason=cancellation_outcome_unknown`; another call cannot overwrite
that reason. When the final unknown owner becomes conclusive, reconciliation
returns to `cancel_requested`, never `running`, so no new work is admitted.

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

The first runtime targets Modal directly. `modal.py` isolates only the
operations needed for spawn, call-ID resolution, observation or collection,
and cancellation so tests can use fakes. This is an internal test seam, not a
public provider abstraction or plugin system. Modal-specific objects do not
leak into plans or persisted models.

### Result Envelope and split completion

Provider Call completion and scientific Task completion are deliberately
separate durability boundaries. When a Modal call returns successfully,
caller-owned coordinator code normalizes its small returned value into a
JSON-compatible Result Envelope and records it with the kernel. The envelope
records call identity plus the durable per-Task result references or
conclusive diagnostics needed by workload code. Large trajectories, models,
archives, and other scientific files remain in workload-owned durable
storage; the envelope may refer to them but never embeds them and is excluded
from scientific fingerprints.

Workflow adapters also keep serialized provider returns out of SQLite. They
write the return once to a checksum-addressed file in the workflow Run's
durable output directory and store only its relative path, SHA-256 digest, and
byte count in the Result Envelope. Recovery verifies that reference before
decoding the return. This keeps InlineBytes and other potentially large app
results from inflating the coordinator ledger while preserving the split
completion recovery boundary.

The repository stores the envelope and the call's transition to `succeeded`
atomically, then the coordinator crosses the host durability boundary. Only
then does the call leave the derived total and optional GPU active counts. No
`finalizing` status or separate slot row is needed. Any owned Task not already
finished by an idempotent worker completion report remains `running` while the
caller loads the envelope, decodes it, stages and publishes output, validates
the publication, and reports the outcome:

- a valid decoded outcome whose publication validates as `available` succeeds
  that Task;
- a conclusive per-Task error, malformed outcome, invalid publication, or
  authoritative post-publication `missing` fails only that Task;
- `unknown` publication validation leaves the Task `running` and suspends the
  Run with `result_validation_unknown`;
- one call may therefore be `succeeded` while its Tasks independently become
  succeeded, failed, or temporarily unresolved.

Once explicit cancellation is durable, an unfinished Task whose direct call
or pull worker is already terminal becomes `cancelled` instead of entering
publication validation. The same rule applies when the owner becomes terminal
after the cancellation request. A Task whose publication was already
validated remains terminal and is never rewritten.

A provider return that is conclusively malformed may be represented by a
durable diagnostic envelope. The call still succeeded as a provider
invocation, while decoding fails its affected Tasks. If the coordinator cannot
obtain, reconstruct, or make any envelope durable, it must not invent call
success: the call remains or becomes `state_unknown`, retains its slots, and
requires explicit reconciliation.

The recovery path follows the last durable boundary:

1. before envelope durability, resolve and collect the same attached provider
   call ID without spawning replacement work;
2. after envelope durability but before Task publication, resume decode and
   publication from the stored envelope without contacting Modal for another
   execution;
3. after publication but before Task commit, observe the publication and
   commit the recovered Task outcome.

### Cache and publication lifecycle

Node and Task reuse decisions return exactly one observation:

- `available`: a validated complete publication can satisfy the Node or Task;
- `missing`: validation authoritatively established that no reusable
  publication exists;
- `unknown`: the checker failed, the storage was unavailable, or absence could
  not be established.

Only `missing` authorizes new work. Caller-owned code validates and records a
Node result observation before dependency input or Task preparation; that
observation may prune the Node's ancestor closure. Nodes without a standalone
complete publication report `missing`; partial aggregate output does not
satisfy the probe. The repository records when the observation occurred and
whether Node completion was cache-validated or produced in the current Run,
without copying workload manifests into generic execution state. `unknown`
leaves the observed Node or Task nonterminal and suspends the Run with
`result_validation_unknown`; explicit resume lets the caller validate and
record another observation.

Result Envelope durability does not make scientific output reusable. The
workload publication and validator remain authoritative, and the Task becomes
terminal only under the split-completion protocol above.

### Coordinator-local execution lifecycle

A caller may mark a Task for coordinator-local execution instead of remote
dispatch. This is appropriate for deterministic operations such as result
assembly or archive preparation that need durable timeline and dependency
state but no separate Provider Call.

The normal Task publication observation runs first:

- `available` succeeds the Task without executing local code;
- `unknown` suspends the Run with `result_validation_unknown`;
- only `missing` permits local ownership and execution.

For `missing`, the repository durably marks the Task `running` with
coordinator-local ownership before returning permission for the caller to run
the operation. It consumes no total or GPU call slot. Caller code writes only
to Task-specific staging until its result is complete, publishes atomically or
idempotently, validates it, and reports the observation. The kernel marks the
Task `succeeded` only after validation returns `available`. A reported
`unknown` suspends the Run. Conclusive local failure, invalid output, or
authoritative post-publication `missing` terminally fails the Task.

A hard interruption can leave the Task `running` after its process and local
stack disappear. Under the exclusive coordinator topology, no old local
operation continues concurrently when the replacement starts. Recovery does
not create a Task Attempt or move the Task back to `pending`; caller-owned code
observes the publication and records the result:

- `available` commits success without re-execution;
- `missing` permits the caller to re-enter the same operation for the same
  Task;
- `unknown` suspends the Run and admits no work.

Repeated infrastructure interruptions may re-enter the operation until the
publication validates. This is crash recovery, not failure retry. A caught or
otherwise conclusive local failure remains `failed`, and `resume` cannot
re-enter it. Explicit cancellation also prevents re-entry and follows the
normal Task cancellation path.

Caller-owned coordinator-local code must therefore tolerate re-entry through
idempotence or Task-keyed staging and atomic publication. Code with an
uncontrolled non-idempotent external side effect is not eligible for
coordinator-local execution; it must use an external idempotency key or a
tracked Provider Call. The kernel stores no local attempt counter, local call
record, or additional status. Coordinator-local versus provider execution is
persisted operational placement for one Run and is excluded from scientific
fingerprints; a Successor Run may change placement only while preserving the
same publication contract.

Dynamic Task discovery never treats an empty collection as vacuous success.
Every `NodePlan` carries `allow_empty_result: bool = False`, which is included
in the Workload Plan Fingerprint:

- zero Tasks with the default `False` fails the Node with a workload
  diagnostic;
- zero Tasks with `True` requires caller-owned code to publish and validate an
  empty result and creates no synthetic Task;
- only an explicit empty Node publication that validates as `available` makes
  the Node `succeeded`;
- `unknown` suspends the Run under the existing result-validation policy, and
  a missing or invalid publication after finalization fails the Node.

This rule runs before Task aggregation and is independent of `fail_fast`,
`collect_all`, and `allow_partial`.

### Failure modes

Nodes declare one of three workload-selected aggregation policies:

- `fail_fast`: the first failed Task stops new admission; unowned pending
  siblings become `skipped`, already-owned Tasks continue without
  cancellation, and the Node becomes `failed` after every owner is
  conclusive;
- `collect_all`: admit every Task subject to Provider Call limits; all successes
  produce `succeeded`, while any failure produces `failed`;
- `allow_partial`: admit every Task; all successes produce `succeeded`, some
  successes and some failures produce `partial`, and no successes produce
  `failed`.

Cache-validated Tasks count as successes. A partial Node publication may be
consumed downstream only through an `accept_partial` dependency edge.
Explicit Run cancellation and result pruning take precedence over aggregation.
These policies neither cancel already-owned work nor authorize another
submission. A failed Task remains failed for that Execution Run; retry
requires an explicit Successor Execution Run.

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

## Verification matrix

| Layer | Required verification |
| --- | --- |
| Pure model | Graph cycles, readiness, terminal closure, stable fingerprints, task discovery determinism |
| Call lifecycle | Fault injection around every transition, attach validation, Result Envelope durability and replay, recovery, expiry, cancellation, unknown outcomes |
| Result publication | Call success releases slots before unfinished Task decode/publish/validate; malformed per-Task outcomes fail independently; publication recovery never respawns |
| Cache | Available/missing/unknown, checker exceptions, marker validation, cache hit starts no call |
| Batching | Compatibility grouping, maximum-size chunking, immutable call-to-many mapping, per-Task result decode, partial and failed batches, deterministic ordering |
| Dispatch | Fixed preclaim assignment, direct fan-out, many-call pull pools, idempotent claim replay, call-bound Worker Assignments, partial outcomes |
| Local execution | Durable local ownership, no call slot, staged publication, repeated crash recovery, unknown suspension, no replay after conclusive failure |
| Interruption | Graceful drain, hard kill, child-call preservation, replacement recovery, explicit cancellation |
| Resources | Node parallelism independent from total/GPU call slots, one slot per active call, conservative unknown-state retention |
| Scheduling | Graph rank before resource class; GPU before CPU within a rank; stable image cohorts before encounter order; no active-image or slot-reservation heuristic |
| Service | API/OpenAPI unchanged unless intentionally versioned; admission, timeline, logs, cancel, cache staging, ZIP contents |
| Workflow | DAG hashes, scheduler waves, terminal pruning, artifact selection/materialization, coordinator resume, and successor restart behavior |
| PPIFlow | Candidate identity, manifests, attrition, joins, partial outcomes, and successor publication reuse |
| AlphaFold3 | Search/run/request identities, claims, publications, seed batching/reuse, summaries, archive hashes |
| CLI | App and workflow discovery/help, version resolution and overrides, deployed versus development launch, representative dry tests |

CI uses an in-memory SQLite repository, a fake internal Modal call driver, and
temporary workload storage. Remote Modal validation remains a manual,
explicitly authorized smoke test after local and CI gates pass.

## Risks and controls

| Risk | Control |
| --- | --- |
| A universal abstraction hides scientific differences | Keep workload code caller-owned, add no handler framework, and migrate AF3 last |
| Extraction duplicates rather than replaces code | Delete replaced implementations in the same change and test only the shared path |
| Async and sync consumers distort the API | Share pure transitions; keep thin separate host loops |
| A batch obscures individual outcomes | Persist Task identities, per-Task outcomes, and explicit call links |
| Call slots remain occupied by local result processing | Persist a small recoverable Result Envelope, terminally succeed the call, and publish each Task independently |
| A claim response is lost after assignment | Commit and Volume-checkpoint the assignment before responding; replay by stable request ID |
| An exit callback races a restarted worker | Treat exit events as advisory and retain call-bound Worker Assignments |
| Recovery silently spends on a second call | Enforce one Task submission per Run in SQLite; require a Successor Execution Run for failed work |
| “Exactly once” hides provider re-execution | Promise single scheduler submission only; require idempotent work or authoritative publication validation across redelivery |
| Cache checker outage triggers expensive recomputation | Tri-state availability; only `missing` authorizes work; `unknown` suspends until explicit resume |
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
| Call limits are mistaken for Modal resource decorators | Document call-count ceilings as conservative fan-out controls; decorators remain authoritative for actual resources and container packing |
| One ledger becomes a cross-context bottleneck | Embed the same execution tables into coordinator-owned databases |
| Refactor changes scientific or user-visible behavior accidentally | Scientific, cost-safety, CLI-operation, and result regression tests |

## Explicit non-goals

- a universal API Job base class;
- one database or ledger for all consumers;
- a universal scientific cache or marker schema;
- a workload-handler framework or generic scientific input/output lifecycle;
- a public multi-provider abstraction before a second provider exists;
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
