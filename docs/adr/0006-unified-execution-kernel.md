# Centralize execution mechanics without centralizing workload state

Status: accepted.

Biomodals should introduce a provider-aware execution kernel under
`biomodals.execution` for DAG traversal, task readiness, paid-call attachment
and recovery, cache-observation policy, batching, and Provider Call limits. The
kernel should embed its execution tables into host-owned databases without
replacing `ServiceStore` domain state, workflow artifact records, or
AlphaFold3's claims and publications. A service Job remains the user-facing
admission and result envelope; workflows retain the physical per-run ledger;
workload publications remain authoritative for scientific cache completion.

The authority boundary was accepted on 2026-07-29. “Without centralizing
workload state” does not mean that execution state may remain ephemeral. The
kernel governs the durable state model and atomic transition contract for
runs, nodes, tasks, dispatch batches, worker assignments, and provider calls.
Separate repository instances implement that contract so an API service, a
per-run workflow, and an app coordinator do not depend on one shared database.

Repository scope follows the coordinator boundary, not each API request,
application call, or Modal function. The API service uses one long-lived
database for all service-owned execution runs. A workflow keeps its existing
per-run ledger because its remote workflow orchestrator is a separate durable
coordinator. Every direct CLI app invocation creates a remote run-scoped
coordinator and remote per-run repository. App functions invoked by a service
or workflow remain child calls in the parent's Execution Run and do not create
another repository.

The coordinator-placement policy was accepted on 2026-07-29. A Local
Entrypoint is a thin client: it validates and stages local input, submits a
Direct CLI App Run, and observes or retrieves its result. It never creates a
local SQLite ledger. API service calls remain coordinated through the service
database, and workflow CLI calls remain coordinated through the remote
workflow ledger. Generic scheduling inside a child app moves to the parent
coordinator instead of introducing a nested execution database.

The deployment-binding policy was accepted on 2026-07-29. There is no
universal execution-coordinator Modal app. Each app and workflow deployment
includes a thin Deployment Coordinator Adapter that binds the shared kernel
to its workload hooks, Volumes, and provider configuration. The containing
deployment version pins the coordinator code and workload adapter together.
The shared kernel contains reusable Modal coordination mechanics but declares
no app object, workload registry, or deployment-global Volume.

The CLI deployment-lifetime policy was accepted on 2026-07-29. Both
`biomodals app run` and `biomodals workflow run` submit to an exact named
deployment version by default. This keeps the same parameterized coordinator
pool addressable after the launching process exits. Source-backed
`modal run` execution remains available only as an explicit Development CLI
Run and carries no cross-invocation resume guarantee. Local dry-run planning
does not require a deployment.

The deployment-resolution policy was accepted on 2026-07-29. A CLI
`--version` value is an explicit override. Otherwise the CLI queries
`modal app history --json` once, selects the current deployed version, and
forms an exact Deployment Identity from the Modal Environment, deployment
name, and numeric version. It preflights and persists that identity before
admitting Tasks or starting Provider Calls. Every lookup then supplies the
exact version; a missing or unretained version fails closed rather than
falling back to the latest deployment. Modal support for version-pinned
lookups is therefore a deployment prerequisite.

The expired-deployment policy was accepted on 2026-07-29. An Execution Run
never changes its Deployment Identity in place. The kernel continues to
observe already attached calls by FunctionCall ID and validates any resulting
Workload Publications. If the pinned coordinator version is unavailable while
the run remains incomplete, the run becomes `failed` with
`status_reason=deployment_unavailable` and admits no new work. An explicit
restart creates a Successor Execution Run with a newly resolved Deployment
Identity and predecessor link. It revalidates publications and schedules only
missing Tasks; active or unknown predecessor ownership blocks replacement
work, and the predecessor repository remains read-only and inspectable.

The remote-ledger storage policy was accepted on 2026-07-29. A Direct CLI App
Run stores its App Run Ledger beneath the reserved
`.biomodals/execution/runs/` namespace in the app deployment's configured
durable Volume, normally its existing output Volume. A workflow retains its
physical ledger in the workflow orchestrator Volume, and the API service
retains execution tables in `service.sqlite3`. The host supplies the Volume,
mountpoint, and repository path; the kernel declares no global execution
Volume or workload output location.

The remote-ledger retention policy was accepted on 2026-07-29. The initial
kernel neither expires terminal ledgers nor runs background garbage
collection. Direct CLI app and workflow ledgers remain available for status,
diagnosis, and restart until an operator explicitly removes them. Any future
cleanup operation belongs to the host or CLI, must reject non-terminal Runs,
and removes only execution state—not Workload Publications, scientific
outputs, or service-owned records. The API service retains its own database
according to service policy.

The execution-identity policy was accepted on 2026-07-29. The admitting host
uses the kernel's generator to create an opaque UUID Execution Run ID before
repository creation or work admission. That UUID keys execution rows,
coordinator routing, predecessor lineage, and remote ledger paths.
User-provided run names and scientific IDs remain optional Workload Run Keys
inside the immutable workload plan; they never select a ledger or coordinator
pool. A Successor Execution Run receives a new UUID while retaining the same
Workload Run Key and publication identity where scientifically appropriate.

The CLI location policy was accepted on 2026-07-29 and simplified the same
day. App and workflow launch commands print the Deployment Identity, Execution
Run ID, and optional root coordinator FunctionCall ID. Each Deployment
Coordinator Adapter exposes the same lifecycle surface, so `biomodals run
status`, `cancel`, `resume`, and `restart` accept those values as explicit CLI
flags without app-specific command implementations. There is no encoded run
reference, parser, local registry, or global remote run index. The adapter
verifies the Deployment Identity and Execution Run ID against the App Run
Ledger or Workflow Ledger before acting; the optional FunctionCall ID is only
an observation hint. `resume` retains the Execution Run ID and Deployment
Identity and never retries failed Tasks, while `restart` always creates a
successor and prints its new fields.

The explicit CLI recovery policy was accepted on 2026-07-29. Repeating
`biomodals app run` or `biomodals workflow run` without predecessor identity
creates a new root Execution Run; command text, file paths, and Workload Run
Keys are not implicit execution identity. Both launch commands accept
`--restart-from <execution-run-id>` as a convenience that delegates to the
same Successor Execution Run operation as `biomodals run restart`.
`resume` reconciles the same Run and never retries failed Tasks. Restart
revalidates Workload Publications, reuses valid successes, schedules only
missing or invalid work whose predecessor ownership is conclusively terminal,
and then advances untouched downstream Nodes. Active, unknown, unreadable, or
invariant-invalid predecessor ownership fails closed. An interrupted SQLite
transaction must recover as a valid durable state such as `submitting` or
`outcome_unknown`; physical database corruption is not reconstructed from
partial rows. The design adds no command-fingerprint catalog or implicit
latest-run lookup.

The successor-compatibility policy was accepted on 2026-07-29. Each Run stores
a Workload Plan Fingerprint over normalized result-affecting inputs and
declared scientific tool, model, adapter, and schema versions. Generic restart
reuses the predecessor's immutable scientific plan. Launch-time
`--restart-from` constructs and compares the candidate fingerprint before
creating state; a mismatch requires a new root Run. File identity uses content
digests rather than paths. Operational concurrency, batching, resource
allocation, and Deployment Identity may change without changing the
fingerprint. Deployment Identity is not itself scientific cache identity, but
a new deployment may reuse publications only if its workload adapter accepts
the stored plan and all result-affecting version declarations still match.

Keeping a workflow's physical `ledger.sqlite3` file does not preserve a
separate workflow implementation of generic execution state. The shared
execution repository should own run, node, task, dispatch-batch,
worker-assignment, and provider-call tables and transitions inside that file.
It deliberately has no attempt table or attempt foreign key. Workflow code
should retain only its artifact records, run-directory lifecycle, and Modal
Volume synchronization.

The repository scope and ledger decomposition were accepted on 2026-07-29.
The workflow runtime composes the shared execution repository and a narrow
Workflow Artifact Store over the same connection. The physical
`ledger.sqlite3` file remains.

The workflow cutover policy was accepted on 2026-07-29. Kernel modules and
tests are built beside the unchanged workflow execution implementation. There
is no `WorkflowLedger` compatibility facade, dual schema, dual write, or
attempt-preserving adapter. Once the kernel replacement is complete, one
cutover commit switches the workflow composition root, deletes the old
execution methods and attempt model, and rejects old unfinished ledgers.
Rollback reverts that commit and recreates unfinished runs; scientific
publications remain reusable.

The repository implementation was accepted on 2026-07-29. The first kernel
uses one concrete `SqliteExecutionRepository` over a host-supplied SQLite
connection and transaction. It owns the execution schema and transitions but
does not choose a file path, commit, close, or synchronize a Volume. Tests use
the same implementation with an in-memory SQLite connection. A generic
persistence protocol is deferred until a second real storage backend exists.

The surface policy was accepted on 2026-07-29. `biomodals.execution` is an
internal shared module with no compatibility promise for Python imports,
unfinished database schemas, or in-progress run formats. Its interface remains
small for depth, locality, and testability rather than external stability.

The service-invariance rule was accepted on 2026-07-29. Execution tables store
only facts required to schedule, recover, and account for actual work. They
contain no user, API Job, display, authentication, administrator, HTTP, or
workflow-artifact metadata and have no foreign keys back to host tables. A
Service Job or Workflow Artifact Store may refer to execution IDs in the
opposite direction. Physical colocation in one SQLite file does not weaken
this dependency rule.

The service projection rule was accepted on 2026-07-29. A Service Job stores
service-owned identity, admission, request, and result-delivery data plus a
one-way Execution Run ID. It does not persist a second compute state or
operation ledger. `JobState`, timelines, active-job counts, and administrative
running-job counts are service views derived from execution rows and
service-owned result data.

The Run-status policy was accepted on 2026-07-29. The kernel has exactly nine
Execution Run statuses: `pending`, `running`, `cancel_requested`, `suspended`,
`state_unknown`, `succeeded`, `partial`, `failed`, and `cancelled`. The first
five are nonterminal; the final four are terminal. An unexpected coordinator
application error sets `suspended`; provider submission, state, or
cancellation uncertainty sets `state_unknown`. Deployment unavailability is
`failed` with `status_reason=deployment_unavailable`, not a separate status.
Provider-call `outcome_unknown` remains lower-level detail. Provider
preemption does not change Run status, finalization remains an ordinary
running Node, and the kernel adds no `queued`, `finalizing`, generic `blocked`,
`interrupted`, or `retrying` state. Service Job projections may retain
user-facing labels without persisting another compute state machine.

The Run-reason representation was accepted on 2026-07-29. Every Execution Run
has one nullable, stable machine-readable `status_reason` and one nullable
human-readable `status_message`. Repository transitions replace or clear both
alongside `status`; control flow never parses `status_message`. Workload errors
remain canonical on their Task or Node records, while the Run fields summarize
only why the overall lifecycle changed. The kernel does not add separate
failure, suspension, or unknown-reason columns.

The initial Run-reason vocabulary was accepted on 2026-07-29 and extended by
the result-validation policy below. A `suspended` Run requires
`coordinator_error` or `result_validation_unknown`. A `state_unknown` Run
requires one of `submission_outcome_unknown`, `provider_outcome_unknown`, or
`cancellation_outcome_unknown`. A `failed` Run requires either
`required_work_failed` or `deployment_unavailable`. Every other Run status
requires a null `status_reason`. The repository rejects all mismatched and
unknown codes.

The Execution Node status policy was accepted on 2026-07-29. Nodes have
exactly seven statuses: `pending`, `running`, `succeeded`, `partial`, `failed`,
`cancelled`, and `skipped`. Only the first two are nonterminal. `ready` is a
derived dependency predicate, not persisted state. A fully cache-satisfied
Node is `succeeded`, with cache reuse retained as Task provenance. Run-level
`cancel_requested`, `suspended`, and `state_unknown` are not duplicated on
Nodes. `skipped` means that an upstream terminal outcome made a planned Node
unreachable.

The partial-dependency policy was accepted on 2026-07-29. Every immutable Node
dependency edge has `accept_partial: bool = False`. A `succeeded` upstream Node
always satisfies the edge; `partial` satisfies only an opted-in edge; and
`failed`, `cancelled`, or `skipped` never satisfies it. All dependencies must
be satisfied before a Node is ready. Once an unacceptable upstream outcome is
terminal, the dependent Node becomes `skipped`. Explicit Run cancellation
takes precedence and marks unfinished Nodes `cancelled` rather than obscuring
the cancellation through skip propagation.

The Node aggregation policy was accepted on 2026-07-29. Every Node declares
exactly one of `fail_fast`, `collect_all`, or `allow_partial`. On the first
Task failure, `fail_fast` stops admitting pending siblings, marks unowned
siblings `skipped`, lets already-owned work finish without cancellation, and
then fails the Node. `collect_all` admits every Task subject to Provider Call
limits and fails the Node if any Task fails. `allow_partial` also admits every
Task: all successes produce `succeeded`, a mixture of successes and failures
produces `partial`, and no successes produces `failed`. Cache-validated Tasks
count as successes. Explicit Run cancellation and result pruning take
precedence over aggregation. None of these policies retries a Task or releases
uncertain ownership.

The empty-result policy was accepted on 2026-07-29. `NodePlan` has
`allow_empty_result: bool = False`, and that result-affecting declaration is
included in the Workload Plan Fingerprint. Discovering zero Tasks fails a Node
whose flag is false. When it is true, the kernel invokes the workload
finalizer with the empty result set; the workload must publish and validate an
explicit complete empty Node result. `available` succeeds the Node, `unknown`
suspends under the result-validation policy, and a missing or invalid
post-finalization publication fails the Node. Task aggregation policy does not
infer an outcome for an empty set, and no synthetic Task row is created.

The Task-discovery checkpoint policy was accepted on 2026-07-30. A workload's
read-only `discover_tasks(node, inputs)` hook returns the complete finite set
of `TaskPlan` values for that Node, each with a unique stable Node-local key
and deterministic fingerprint. The repository validates and inserts all
Tasks and marks the Node `discovery_complete` in one transaction. The host
crosses its durability boundary before any Task may acquire a Provider Call,
Worker Assignment, or local owner. A crash before that boundary exposes no
paid work and recovery rediscovers the complete set; a crash after it reloads
the persisted set and never invokes discovery again for that Node in the same
Run. Empty discovery follows `allow_empty_result`. The first kernel version
does not support streaming, incremental, or worker-side Task discovery, so a
worker can never observe a half-populated SQLite queue.

The Task-fingerprint policy was accepted on 2026-07-30 with an explicit
simplicity and performance constraint. `TaskPlan` separates a JSON-compatible
workload-normalized scientific payload from its operational execution
payload. The kernel computes SHA-256 once during discovery over compact,
sorted-key canonical JSON containing the Workload Plan Fingerprint, Node key,
Task key, and scientific payload; non-finite JSON numbers are rejected.
Workloads supply content digests rather than file paths or file bytes. The
persisted fingerprint is loaded on resume and never recomputed during polling.
Provider kwargs, staging paths, batching, concurrency, resources, and call
identity are excluded. A successor may reuse a publication only when Node
key, Task key, computed fingerprint, and workload validation all match. The
kernel has no pluggable hash registry, custom codec framework, or repeated
large-file hashing.

The result-boundary policy was accepted on 2026-07-29. Terminal Execution
Nodes—the DAG leaves with no downstream dependency—collectively define the
scientific result boundary. The scheduler validates their workload
publications first and walks backward only through the ancestor closure of
terminal Nodes whose results are incomplete. A reusable terminal publication
therefore completes that Node without scheduling already-unnecessary
ancestors. If every terminal Node succeeds, the Execution Run succeeds even
when an upstream Node previously failed, was cancelled, or was skipped.
Intermediate lifecycle history remains available for diagnosis but is not an
all-Node vote on the scientific outcome. This generalizes the workflow
runtime's existing terminal-pruning behavior to every kernel consumer.

The Node-result observation policy was accepted on 2026-07-29. Every workload
adapter exposes a lightweight `observe_node_result(node)` hook that can run
before dependency inputs or Tasks are prepared and returns the shared
`available`, `missing`, or `unknown` vocabulary. `available` means the
complete workload publication validated and may mark the Node `succeeded`;
`missing` authorizes backward expansion into its dependencies; and `unknown`
blocks new work. A workload with no aggregate reusable publication
deliberately returns `missing`. Partial publications are not complete Node
hits, although their successful Task publications may be reused later inside
the repair closure. The kernel records the observation time and whether Node
completion was cache-validated or produced in this Run; scientific evidence,
markers, manifests, and validation logic remain workload-owned. This hook is
part of the workload port and does not add a fake terminal Task or another
top-level kernel abstraction.

The unknown-result policy was accepted on 2026-07-29. An `unknown` Node or
Task result observation leaves that record nonterminal, stops new admission,
and suspends the Run with
`status_reason=result_validation_unknown`. Attached Provider Calls continue
under their existing ownership; they are not cancelled. The coordinator does
not poll or retry the failed validator automatically. Explicit `resume`
repeats the observation and continues the same Run if it becomes conclusive.
`state_unknown` remains reserved for ambiguous provider submission, call, or
cancellation ownership, where replacement work could duplicate paid compute.

The terminal-aggregation and repair policy was accepted on 2026-07-29. For one
Execution Run, all-successful terminal Nodes produce `succeeded`; a boundary
containing only `succeeded` and `partial` produces `partial` when at least one
is partial; any `failed` or `skipped` terminal produces `failed`; and any
`cancelled` terminal produces `cancelled`. Upstream outcomes do not enter this
aggregation. A restart never reopens that result. It creates a Successor
Execution Run whose repair closure starts from every predecessor terminal
that was `partial`, `failed`, `skipped`, or `cancelled`, and from any
previously successful terminal whose publication no longer validates. The
successor walks backward to complete reusable publications, reuses successful
Task publications inside the closure, and submits only conclusively unowned
missing work. Unknown publication or predecessor ownership blocks new work
rather than guessing. The successor is aggregated independently and may reach
the same terminal outcome again.

The result-pruning cleanup policy was accepted on 2026-07-29. A pending
ancestor made unnecessary by a complete result becomes `skipped` with
`status_reason=result_already_satisfied`; a previously terminal ancestor keeps
its historical outcome. The coordinator stops admission for a running
unnecessary ancestor, cancels and reconciles its attached Provider Calls, and
marks the Node `cancelled` with the same reason when cancellation wins.
Provider work that reaches another conclusive terminal outcome first retains
that observed outcome. The Run remains `running` during this internal cleanup;
an inconclusive cancellation moves it to `state_unknown`. Only after every
unnecessary remote owner is conclusive may the coordinator exit with the Run
outcome derived from its terminal scientific results. This prevents cached
return from abandoning paid work without adding another Node status.

The Task status policy was accepted on 2026-07-29. Tasks have exactly six
statuses: `pending`, `running`, `succeeded`, `failed`, `cancelled`, and
`skipped`. Only the first two are nonterminal. Durable local execution,
Provider Call ownership, or Worker Assignment moves a Task to `running`.
Provider uncertainty remains on that owner and projects the Run to
`state_unknown`; it does not release the Task from `running`. `partial` is a
Node aggregation result, provider submission phases remain Provider Call
state, and cache reuse is success provenance rather than a Task status.
`skipped` represents unowned Tasks not admitted after a `fail_fast` failure or
after result pruning; the latter carries
`status_reason=result_already_satisfied`.

The Task-level result-pruning policy was accepted on 2026-07-29. A pruned Node
creates no records for Tasks it never discovered. Discovered pending Tasks
without an owner become `skipped` with
`status_reason=result_already_satisfied`. An owned Task remains `running`
until its Provider Call or Worker Assignment reaches a conclusive outcome.
Conclusive cancellation makes the Task `cancelled` with the same reason;
success, failure, or a validated publication that wins the race is retained
instead. Existing terminal Task outcomes never change. Unknown ownership or
cancellation keeps the Task `running` and the Run `state_unknown`. These rules
make pruning close every durable work record without inventing completion or
releasing uncertain ownership.

The Node, Task, and Provider Call relationship policy was accepted on
2026-07-29. A Node is a fixed semantic DAG stage, a Task is one independently
scheduled and validated item in that stage, and a Provider Call is one
concrete remote worker invocation. Every Task, Dispatch Batch, and Provider
Call belongs to exactly one Node. A Provider Call may own zero or many Tasks,
but only from that Node; a Task has at most one durable remote owner path in
one Execution Run. Fixed dispatch may link Tasks directly to the call, while a
pull worker uses one unique Worker Assignment per Task. The kernel has no
generic many-to-many Task-to-call association. Cache or local execution can
complete Tasks without a Provider Call, provider redelivery retains the same
call identity, and restart creates new Tasks in a Successor Execution Run.
The remote invocation hosting an Execution Coordinator is a Coordinator
Attempt, not a Task-owning Provider Call.

The Provider Call status policy was accepted on 2026-07-29. Calls have exactly
eight statuses: `submitting`, `attached`, `running`, `outcome_unknown`,
`state_unknown`, `succeeded`, `failed`, and `cancelled`. The first five are
nonterminal and preserve Task ownership; the final three are terminal. The
submission preclaim creates a call directly in `submitting`, so unsubmitted
intent remains on the Task or Dispatch Batch rather than a `planned` call.
`outcome_unknown` means spawn may have occurred without a durably attached
provider ID, while `state_unknown` means the ID exists but observation or
cancellation is inconclusive. An expired provider handle is an observation,
not a status: conclusive failure becomes `failed`, while an unresolved outcome
becomes `state_unknown`.

The submission-preclaim boundary was accepted on 2026-07-29. The atomic
repository preclaim creates the `submitting` Provider Call, assigns its Tasks,
and tells exactly one in-process caller that it created the row. That caller
must commit or checkpoint through the host's durability boundary before it may
invoke provider spawn. A duplicate request observes the existing call and
performs no side effect. Recovery of an abandoned `submitting` row cannot
prove whether spawn began, so it moves the call to `outcome_unknown` and never
invokes spawn again. A conclusive provider rejection moves the call and its
unfinished Tasks to `failed`; an ambiguous exception preserves unknown
ownership. Provider resolution and input preparation happen before preclaim.
Retrying failed Tasks requires a Successor Execution Run.

The resource scope was accepted on 2026-07-29. The first kernel persists and
enforces Provider Call admission limits within one Execution Run and
coordinator. Service-wide admission limits remain service-owned, and Modal
CPU, GPU, memory, timeout, and deployment limits remain workload-owned.
Cross-coordinator and cross-run global enforcement, including a shared-lease
interface, is deferred until a concrete requirement exists.

The concrete call-limit policy was accepted on 2026-07-30. Each Run stores
`max_active_provider_calls` and `max_active_gpu_provider_calls`, with the GPU
limit nonnegative and no greater than the positive total limit. A workload's
resolved provider binding declares whether its target function has a GPU
allocation; the kernel persists that boolean on the Provider Call but does not
inspect or reproduce Modal decorators. Every nonterminal call consumes one
total slot, and a GPU call also consumes one GPU slot. The serialized preclaim
checks both counts atomically before creating `submitting`; terminal status
releases the slot by removing the call from the derived active count.
`outcome_unknown` and `state_unknown` retain their slots conservatively. A
single call containing many Tasks and each pull-worker call consume one slot;
local work consumes none. The kernel stores no variable permit cost, named
resource pool, or allocation table. These operational limits are excluded
from scientific fingerprints and conservatively bound in-flight remote calls,
not actual Modal container packing or CPU, RAM, accelerator type, or GPU
device count.

The admission-order policy was accepted on 2026-07-30. The coordinator uses a
Snakemake-inspired greedy selection: every scheduling cycle fills as many
currently feasible total and GPU Provider Call slots as ready work permits,
without an artificial one-call-per-Node pass. Ready call candidates are
ordered lexicographically by greater Node depth in the required DAG closure,
then by the greater number of required unfinished descendant Nodes reachable
from that Node, then by stable encounter order. `ExecutionPlan` records each
Node's sequence ordinal and Task discovery records each Task's sequence
ordinal; batches retain the first constituent Task's ordinal. These ordinals
are operational, excluded from scientific fingerprints, and reused after
coordinator recovery. If a higher-ranked GPU candidate cannot fit the
remaining GPU slots, selection continues to feasible CPU candidates rather
than leaving total slots idle. The kernel adds no fairness cursor, priority
weights, preemption, or scheduler plugin surface.

The dispatch-batching policy was accepted on 2026-07-30. The kernel implements
exactly two remote dispatch mechanics. In fixed-batch dispatch, the workload
declares each Task's provider binding, GPU use, compatibility key, positive
maximum Tasks per call, argument construction, and per-Task result decoding.
The kernel removes cache-satisfied Tasks, groups compatible ready Tasks in
encounter order up to that maximum, and ranks each resulting call candidate by
its first Task ordinal. The serialized preclaim atomically creates the
Dispatch Batch and Provider Call and assigns every constituent Task before
spawn authorization. That mapping never changes afterward. In pull-worker
dispatch, the kernel admits Node-ranked worker Provider Calls without
preassigning Tasks; each worker later obtains bounded microbatches through
idempotent, checkpointed Worker Assignments. The Run persists its operational
dispatch policy, so resume reproduces it while a Successor Run may choose
different batching. Dispatch metadata stays outside scientific fingerprints
unless it changes scientific meaning. There is no dynamic batching optimizer,
cross-Node batch, or workload-owned durable scheduling state.

The pull-worker sizing policy was accepted on 2026-07-30. A workload declares
one positive `claim_capacity`, meaning the maximum number of Tasks a worker
may own concurrently. For an eligible pull-worker Node, the kernel derives
`desired_workers = ceil(nonterminal_tasks / claim_capacity)` and creates at
most `max(0, desired_workers - nonterminal_worker_calls)` new call candidates
before applying DAG priority and the Run's remaining total and GPU call slots.
Pending and assigned-but-unfinished Tasks count; cache-satisfied and terminal
Tasks do not. Unknown worker calls remain nonterminal and count
conservatively. Workers may claim repeatedly to rebalance uneven durations.
The coordinator does not cancel excess workers when the target shrinks; they
finish owned work and exit when no unowned Task remains. A claim race may
produce a successful zero-Task call. The kernel adds no per-Node worker limit,
adaptive throughput controller, lease, or idle timeout.

The state-transition policy was accepted on 2026-07-29. The service preserves
users, authentication data, and administrator configuration while recreating
the Job and execution schema without old Job history. Existing workflow
ledgers are incompatible and must restart. Remote scientific publications,
markers, and caches remain reusable and are never deleted by this transition.

The adoption order was accepted on 2026-07-29. GROMACS and the basic workflow
runtime establish fixed graph and one-Task-per-call behavior, PPIFlow is the
first runtime-discovered fan-out consumer, and AlphaFold3 adopts the proven
kernel afterward.

The app-fan-out scope was accepted on 2026-07-29. The kernel also replaces
generic App-Local Scheduler mechanics after its Task lifecycle is proven.
BoltzGen supplies the direct one-Task-per-call case and Rosetta supplies the
SQLite-backed pull work-pool case. Ready Task rows are the durable queue;
provider workers claim bounded microbatches through coordinator methods and
never open the coordinator's SQLite repository.

The interruption policy was accepted on 2026-07-29. A Modal preemption ends
one Coordinator Attempt but does not cancel its Execution Runs or attached
child Provider Calls. The Attempt stops admitting work and checkpoints
best-effort during graceful shutdown; correctness also survives a hard kill
from the last durable checkpoint. A replacement Attempt reloads execution
state, reconstructs active call-slot counts, and resolves attached calls by
ID. Only an explicit user cancellation authorizes cancelling child calls.

The single-writer topology was accepted on 2026-07-29. A Volume-backed remote
coordinator runs in a parameterized, run-scoped provider pool identified by
the Execution Run and the pinned containing app or workflow deployment
version. That pool is capped at one coordinator container. Concurrent run,
claim, completion, and observation requests enter that container, but one
in-process writer loop serializes every SQLite transition and Volume
checkpoint. Different Run IDs have independent pools and may execute
concurrently. The provider routing and single-container assumptions require a
manual Modal smoke test before remote adoption.

The active-run lifecycle was accepted on 2026-07-29. Each remote top-level CLI
app or workflow run submits one detached coordinator-loop input to its
Run-Scoped Coordinator Pool. This is an internal activity, not a CLI
subcommand or public kernel type. It reconciles durable state, dispatches ready
work, observes attached calls, and advances the DAG until the Run becomes
terminal; progress never depends on the launching CLI remaining connected or
polling. Concurrent lifecycle and worker inputs enter the same pool and use
the serialized writer. A replacement Coordinator Attempt reloads the ledger
after preemption. Once terminal, the loop returns and the container may scale
to zero; a later status request may start a fresh container and read the
retained ledger.

The coordinator-error policy was accepted on 2026-07-29. Provider
redelivery after infrastructure interruption may transparently create a
replacement Coordinator Attempt, but the coordinator adapter configures no
automatic retry loop for an uncaught application exception. Such an exception
stops admission, preserves attached Provider Calls, and records a diagnostic
when possible. The Execution Run becomes `suspended` and requires an explicit
`resume` command to reconcile durable state and continue. Resume may schedule
Tasks that were never submitted, but it does not retry a failed Task.

The worker-interruption policy was accepted on 2026-07-29. Worker preemption
does not fail a Task or release its Worker Assignment because Modal restarts
the same provider input. Pull workers use stable, idempotent claim request IDs.
The coordinator transactionally records an assignment, crosses the Volume
durability boundary, and only then returns its Task payload; a repeated request
returns the same assignment. A conclusively failed owner call fails its
unfinished Tasks and releases its active call slot by becoming terminal, but
no other call may claim those Tasks in the same Execution Run. Lifecycle exit
hooks may checkpoint work or emit diagnostics, but they never authorize
reassignment.

The single-submission policy superseded the earlier attempt-based retry policy
on 2026-07-29. The kernel stores no Task Attempt identity or counter. Within
one Execution Run, each Task is scheduled once and receives at most one
Provider Call submission or Worker Assignment. Modal may redeliver and
re-execute the same provider input, so the kernel does not claim exactly-once
execution. A conclusive provider or workload failure terminates the Task; the
Node aggregation policy then determines the Node outcome, while terminal
scientific results determine the Run outcome. `resume` reconciles interrupted
coordination and may submit Tasks that were never submitted, but it never
resubmits failed Tasks. Retrying failed work requires an explicit `restart`,
which creates a Successor Execution Run, revalidates Workload Publications,
and schedules only missing Tasks whose predecessor ownership is conclusively
terminal. Active or unknown predecessor calls block replacement work.

## Considered options

- Expanding `WorkflowRuntime` into the universal scheduler would reuse its DAG
  and ledger implementation, but would force asynchronous API jobs and
  multi-writer AlphaFold3 publications into workflow-specific persistence and
  artifact contracts.
- Expanding the API service coordinator would preserve its stronger paid-call
  attachment protocol, but would couple CLI workflows and apps to service
  authentication, global SQLite state, and HTTP-facing Job concepts.
- Introducing one universal Job class and database would make inspection look
  uniform while conflating API admission, workflow recovery, provider calls,
  and scientific cache authority.
- Letting remote workers open SQLite on a shared Volume would remove a control
  hop, but Modal Volumes do not provide safe distributed file locking or
  coherent concurrent same-file writes.
- Mirroring assignments into Modal Dict and transporting Tasks through Modal
  Queue would add a second state model with expiry and delivery-recovery
  semantics even though SQLite already contains the durable ready set.
- One universal coordinator deployment would avoid small per-deployment
  wrappers, but would require a workload registry, access to every workload's
  storage, and independent compatibility between coordinator and workload
  deployment versions.
- Keeping ephemeral `modal run` as the default CLI launcher would preserve the
  current source-first behavior, but a later process could recover a
  FunctionCall by ID without being able to address the same parameterized
  coordinator deployment reliably.
- Requiring users to enter a numeric deployment version on every run would be
  deterministic but unnecessarily cumbersome. Resolving history once and
  immediately switching to exact versioned lookups provides the same pin.
- Calling an unversioned handle throughout a run would be simpler, but Modal
  may route later calls to a newer deployment during a rolling update.
- Migrating an incomplete run to a newer deployment in place would preserve
  its Execution Run ID, but would mix plan, adapter, schema, and coordinator
  versions inside one execution authority.
- One execution-state Volume shared by every deployment would simplify global
  discovery, but would broaden storage access and couple unrelated apps to one
  Volume lifecycle. Existing deployment-specific Volumes already provide the
  required durable boundary.
- Reusing a user or scientific run name as the execution primary key would
  make CLI lookup familiar, but would couple untrusted paths, publication
  reuse, restart lineage, and one scheduler invocation to the same string.
- A local run registry would shorten later commands, but would make recovery
  machine-specific and introduce another mutable state store. Separate app and
  workflow lifecycle commands would duplicate an identical kernel surface.
- Retaining Task Attempt rows would support retries inside one Execution Run,
  but would duplicate run lineage and complicate cost safety. A failed Run and
  an explicit Successor Execution Run provide one retry boundary.
- A small execution kernel with one embeddable SQLite implementation and
  explicit provider and workload adapters reuses the common algorithms while
  allowing each host to preserve its transaction and durability model.

## Consequences

The common hierarchy is an execution run containing fixed semantic nodes,
runtime-discovered tasks, dispatch batches, worker assignments, and provider
calls. One provider call may serve several tasks, and a cache hit may complete
a task without a provider call. Workloads continue to define scientific
identity, cache validation, input and output contracts, function arguments,
resource requirements, and publication rules. The kernel determines when
those hooks run and how their observations affect scheduling.

Execution identity is deliberately operational. A Service Job, workflow
request, GROMACS run name, or AlphaFold run identity may refer to it, but none
is interchangeable with it. This lets a new execution reuse valid scientific
outputs without reopening or overwriting its predecessor ledger.

An execution repository is authoritative for scheduling facts such as the
immutable plan, readiness, single-submission claims, attached call IDs,
observed provider state, and timestamps. It records that a workload publication
was validated, but the publication's marker, manifest, or workload-specific
validator remains authoritative for whether scientific output is reusable.

Concurrent Task execution does not imply concurrent SQLite writers. One
coordinator durably admits work and records returned events while direct
calls, batched calls, or SQLite-backed pull worker pools run concurrently. A
Dispatch Batch relates Tasks to one or more Provider Calls without introducing
another queue. Workload code retains task identity, batch compatibility,
execution, and publication validation; shared adapters own reusable fan-out
and worker-pool mechanics.

An Execution Coordinator is logical and may span several Coordinator Attempts.
Graceful lifecycle hooks improve checkpoint freshness but are not part of the
correctness proof. Any transition that must precede an external side effect is
committed through the host's durability boundary before that side effect, and
an attached call ID is checkpointed promptly after submission. Preemption is
therefore recovery, not implicit cancellation.

A remote run-scoped coordinator relies on provider routing, a one-container
pool cap, and deployment-version pinning to ensure that only one process opens
its Volume-backed SQLite file. Concurrent method inputs enqueue commands to
one in-process writer rather than executing database transactions directly.
Host-exclusive coordinators such as the single-process API service keep their
existing process ownership.

Different remote runs write distinct ledger files. Modal Volume v2 supports
concurrent writes to distinct files, while the run-scoped one-container rule
protects each individual SQLite file. The coordinator closes or checkpoints
SQLite before reloading or committing its host Volume; scientific outputs and
the reserved ledger namespace never share files.

SQLite-backed work stealing uses the same call-bound principle at Task scope.
Ready Tasks and committed Worker Assignments form the durable work pool.
Workers request capacity through idempotent coordinator calls; the assignment
and Volume checkpoint precede the response. Completion reports are likewise
idempotent. Publication validation and terminal provider state recover lost
responses or interrupted delivery without a timeout-based steal, Modal Dict,
or Modal Queue.

This separates recovery from repeated spending. Provider-native redelivery
preserves one Provider Call and Task identity. A terminal failure cannot create
another submission in the same Execution Run; retrying requires a Successor
Execution Run after publication validation and conclusive predecessor
termination.

This is an incremental extraction, not a rewrite. Internal types, tables, and
imports may change directly while each consumer adopts the kernel. Scientific
identities, publications, cost-safety rules, and documented user behavior
remain regression constraints unless a separate decision deliberately changes
them. Duplicated orchestration code is removed after its replacement passes
characterization and recovery tests.

The deployment-local wrappers are intentional composition roots, not duplicate
schedulers. They contain only Modal decorators and bindings that must remain
with the workload deployment; graph traversal, state transitions, task claims,
recovery, and resource accounting remain in `biomodals.execution`.
