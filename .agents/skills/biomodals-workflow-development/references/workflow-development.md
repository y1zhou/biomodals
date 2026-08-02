# Biomodals Workflow Development

Use this guide when creating or changing files under
`src/biomodals/workflow/` or shared workflow contracts under
`src/biomodals/schema/`.

Use `src/biomodals/workflow/shortmd_workflow.py` as the primary end-to-end
workflow example. Use
`src/biomodals/workflow/rfd_ligandmpnn_workflow.py` as the reference for
workflows that select files from one app's volume-backed output and fan those
files out into another app's workflow-compatible function. Do not use
`src/biomodals/workflow/ppiflow_workflow.py` as a generic starter template, but
use it as the reference for candidate-manifest joins, retained-candidate
filtering, candidate-wide remote stage coordinators, and PPIFlow-specific stage
wiring.

Treat [ADR 0006](../../../../docs/adr/0006-unified-execution-kernel.md) and the
[scheduler specification](../../../../docs/specs/unified-task-scheduler.md) as
authoritative when changing execution statuses, ownership, restart, durability,
or coordinator behavior. Keep this guide focused on workflow composition.

## Contents

- [Vocabulary](#vocabulary)
- [ShortMD Reference Pattern](#shortmd-reference-pattern)
- [RFD LigandMPNN Reference Pattern](#rfd-ligandmpnn-reference-pattern)
- [Schema Boundaries](#schema-boundaries)
- [Node Execution Policy](#node-execution-policy)
- [Artifact Availability And Recovery](#artifact-availability-and-recovery)
- [Execution Boundaries](#execution-boundaries)
- [Ledger Layout](#ledger-layout)
- [Modal Preemption](#modal-preemption)
- [Fan-Out](#fan-out)
- [Orchestrator Submission](#orchestrator-submission)
- [Runtime Diagnostics](#runtime-diagnostics)
- [CLI Namespace](#cli-namespace)
- [Workflow App Composition](#workflow-app-composition)
- [App Interfaces](#app-interfaces)
- [Volumes And Artifacts](#volumes-and-artifacts)
- [DAG Construction](#dag-construction)
- [Testing](#testing)

## Vocabulary

- **App**: a deployed Modal app that owns tool runtime and app functions.
- **App Function**: a callable remote Modal function exposed by an app.
- **Local Entrypoint**: a CLI-only `@app.local_entrypoint`.
- **Workflow-Compatible App Function**: a remote app function returning
  `AppRunResult`.
- **Workflow Node**: one semantic DAG vertex.
- **App-Backed Node**: a workflow node that calls app functions.
- **Workflow-Native Node**: a workflow node implemented in workflow code.
- **Task**: the smallest independently identified, cacheable, and verifiable
  work item inside one Node.
- **Provider Call**: one concrete Modal function invocation owned by one Node
  and zero or more Tasks.
- **Execution Run**: one immutable workflow plan invocation, keyed by an opaque
  UUID.
- **Successor Execution Run**: a new compatible Run that may reuse validated
  predecessor publications and schedule conclusively missing work.
- **Workflow Runtime**: the workflow-owned adapter that validates
  publications, discovers Tasks, and calls `biomodals.execution`.
- **Execution Coordinator**: the run-scoped Modal class that owns the only
  SQLite writer and drives one workflow Run.
- **Workflow Artifact**: durable data passed between workflow nodes.
- **Artifact Selector**: a named reference to upstream artifacts.

Use the canonical execution terms in `CONTEXT.md`. Avoid `app node`, `runner
node`, `engine`, `workflow entrypoint`, and `attempt`; they are ambiguous or
obsolete in this codebase.

## ShortMD Reference Pattern

ShortMD is the current reference for executable workflow apps. Its data flow is:

1. The local entrypoint discovers local `.pdb` files, sanitizes the workflow
   `run_id`, reads PDB bytes, builds a static `Workflow`, and submits that
   object to the included `ExecutionCoordinator`.
2. The workflow app composes the shared orchestrator and the GROMACS app with
   `modal.App(...).include(orchestrator.app)` plus
   `include_dependency_apps(app, CONF.depends_on_apps)`.
3. Each remote Node returns a `RemoteNodeCall` with the exact included
   function name. The kernel resolves it against the pinned ShortMD deployment;
   explicit development runs provide a temporary name-to-handle map.
4. `ShortMDPrepNode` prepares one input PDB once through the GROMACS app.
5. `ShortMDCloneNode` clones prepared production inputs into per-replicate
   directories. This file management is workflow-native because the standalone
   GROMACS app does not need it.
6. `ShortMDReplicateNode` runs each production replicate through the GROMACS app
   and collects trajectory stats.
7. `ShortMDSummaryNode` emits a Markdown report from completed production
   artifacts.

Follow this structure for new app-composed workflows: stage local inputs before
DAG construction, build a static fan-out DAG, keep app-specific runtime work in
included app functions, keep workflow-only adapters in the workflow module, and
return durable artifacts as `AppRunResult` outputs.

## RFD LigandMPNN Reference Pattern

`rfd_ligandmpnn_workflow.py` is the current reference for workflows that chain
workflow-compatible app functions across an app-owned output volume. Its data
flow is:

1. The local entrypoint reads one local PDB, sanitizes `run_id`, builds a static
   fan-out DAG, and submits it to the included `ExecutionCoordinator`.
2. Each `RFdiffusionTrajectoryNode` calls the RFdiffusion app's
   workflow-compatible remote function and receives a durable `VolumePath`
   directory plus log artifact metadata.
3. A workflow-native remote selector reads RFdiffusion PDB/TRB pairs from the
   RFdiffusion output volume, using RFdiffusion metadata to derive the residues
   downstream LigandMPNN should redesign.
4. Each `LigandMPNNDesignNode` calls the LigandMPNN app's workflow-compatible
   remote function with PDB bytes and MPNN CLI args, receiving a small inline
   zstd archive that the workflow runtime materializes.
5. The summary node reports all LigandMPNN archive artifacts.

Use this pattern when the source app owns durable outputs but downstream nodes
need selected small files or derived arguments. Keep selector/adaptation logic in
the workflow module unless it is also useful to the standalone app.

## Schema Boundaries

Shared contracts live in `biomodals.schema`.

Schema modules must not import `modal`, `biomodals.app`, or
`biomodals.workflow`. They should contain Pydantic models and primitive fields
only. The shared `AppConfig` Pydantic schema lives in `biomodals.schema.app`.
Modal-specific helpers that construct volumes, images, or apps must stay in
`biomodals.app` or `biomodals.helper`, with compatibility imports allowed during
the transition from `biomodals.app.config`.

Workflow-compatible app functions return `AppRunResult`. The workflow runtime
materializes each `AppOutput` into one or more `WorkflowArtifact` manifests.
Inline byte outputs are for UTF-8 text bytes or small zstd archives with
`media_type="application/zstd"`. `InlineBytes` should rely on Pydantic's
`ser_json_bytes` and `val_json_bytes` configuration for JSON byte encoding and
decoding; keep text-vs-archive policy in the workflow runtime materialization
layer rather than adding manual byte decoding validators to the shared schema.
Inline byte outputs are materialized into the workflow run volume when the
runtime records workflow artifacts. Other binary outputs, large archives, and
non-text bytes must be written to deterministic volume paths and returned as
`VolumePath` storage.

`AppRunResult.logs` are durable workflow artifacts too. For a single-Task Node,
the runtime materializes inline logs below
`nodes/<node-id>/result/logs/`. Runtime-discovered Tasks use
`nodes/<node-id>/tasks/<task-key>/result/logs/`. Artifact manifests retain the
exact storage paths.

Volume path outputs may either be referenced in place or copied into the
workflow run volume when the source volume is mounted locally. Reference mode is
the default because many app outputs are already durable in their owning app
volume. Copy mode is for workflows that need a self-contained run directory.

When staging selected files from upstream workflow artifacts into app input
directories, never reuse the full pipeline/provenance-derived selected name as
the filesystem basename. Those names can accumulate node ids, artifact ids,
archive paths, and run names and exceed per-component filename limits. Use a
short deterministic basename from a candidate id, sanitized stem, or content
hash, and keep provenance in manifests or metadata instead.

The first workflow runtime is Python-first. Pass a `Workflow` object across the
orchestrator boundary; serialized workflow dictionaries are intentionally
deferred until the node and app-function contracts stabilize.

## Node Execution Policy

Workflow run completion is terminal-node driven. If all terminal DAG nodes have
durable completion and no missing recorded outputs, the run succeeds without
rechecking or scheduling intermediate nodes. If only some terminal nodes are
incomplete, schedule only those terminals and their ancestor closure.

Within the required closure, the runtime records tri-state publication
observations before authorizing work. `available` completes or reuses the Task,
`missing` permits execution, and `unknown` suspends the Run without spending.

A Task has no retry policy or attempt counter. It receives at most one
scheduler submission or remote owner in an Execution Run. `resume` continues a
suspended Run or explicitly reconciles `state_unknown`, but never retries
conclusive failure. Retrying missing or failed work requires an explicit
Successor Execution Run with the same Workload Plan Fingerprint.

Coordinator-local Tasks may re-enter the same idempotent operation after a
coordinator interruption only when their publication is authoritatively
missing. Long-running work must use deterministic run, node, and Task
identifiers and store checkpoints in durable volumes rather than
container-local scratch paths.

`AppRunStatus.PARTIAL` is meaningful only when a Node's declared aggregation
policy permits it. A single ordinary remote or local Task must return
`SUCCEEDED` to publish success.

A workflow-specific `force` flag may deliberately replace workload-owned
scientific outputs, as ShortMD does through tracked cleanup Nodes. It does not
reset or mutate kernel execution state.

## Artifact Availability And Recovery

The runtime verifies workflow-volume artifact availability before reusing a
recorded publication. A conclusively missing publication may authorize work in
the current incomplete Run or its explicit Successor; an unavailable checker
returns `unknown` and authorizes no work.

An app-owned Volume cannot be inferred available when the workflow runtime has
not mounted it. Without a checker its observation is `unknown`, which
authorizes no work. Workflows that publish or reuse app-owned outputs should
enable `strict_external_artifact_checks` and name one workflow-local checker
function that mounts and inspects the required volumes. Keep checks run-level
and derived from recorded `WorkflowArtifact` locations; do not add per-Node
user settings or tool-specific logic to workflow core.

ShortMD, RFdiffusion-to-LigandMPNN, and PPIFlow install their checker
unconditionally because they publish app-owned Volume paths. New workflows
with external outputs must do the same; do not expose a flag that makes normal
publication validation inconclusive.

The helpers in `workflow.core.artifact_availability` are pure Python so workflow
modules can call them from a lightweight Modal function that mounts the
app-owned volumes needed for the run. Use the typed availability contract to
distinguish `available`, `missing`, and `unknown` app-owned volume state; only
missing artifacts may authorize execution.

## Execution Boundaries

Use `WorkflowNativeNode` for lightweight coordinator-local logic such as
filtering, ranking, reporting, and small manifest transforms. Its `run()`
method executes through the kernel's Coordinator-Local Task boundary and
consumes no Provider Call slot.

`max_parallel_nodes` limits how many workflow Nodes may be `running` at once.
It is independent from the Run's total and GPU Provider Call limits: one
running Node may fan out to several calls, while a local Node consumes no call
slot.

Use `RemoteWorkflowNode` or its semantic alias `AppBackedNode` for one tracked
remote call. Implement `prepare_remote(context)` to return a `RemoteNodeCall`;
this prepares arguments and an exact function name but never submits work.
Implement `process_remote_result(result, metadata)` when the provider result
needs adaptation before publication.

Use `RemoteTaskWorkflowNode` when one semantic Node discovers a finite set of
independently identified Tasks at runtime. Implement
`discover_remote_tasks()`, `prepare_remote_task()` or
`prepare_remote_task_batch()`, result decoding, publication observation when
needed, and `finalize_remote_tasks()`.

Use `RemotePullTaskWorkflowNode` only for large variable-duration Task sets that
benefit from lock-free work stealing. Implement `prepare_pull_worker()` with a
bounded claim capacity. Ready Tasks and durable Worker Assignments in SQLite
are the queue; workers claim and complete them through idempotent coordinator
methods and never open the database.

The runtime preclaims every Provider Call before invoking Modal, attaches the
returned call ID durably, and recovers that exact call. Node implementations
must not call `.spawn()`, `.remote()`, or `.get()` themselves.

An app function invoked by a workflow is a Provider Call in the workflow's
Execution Run. It must not launch the app's top-level coordinator or create a
nested app-run ledger.

Do not add a generic remote-node wrapper that accepts arbitrary workflow nodes.
Workflow-native file-management adapters and app-backed nodes that combine
multiple non-`AppRunResult` app calls should expose their own workflow-local
Modal functions or prepare the primary app call and adapt its raw result with
`process_remote_result(...)`. Unit tests use fake Modal drivers and must not
call live Modal APIs.

## Ledger Layout

Each workflow Execution Run uses one opaque UUID and one coordinator-scoped
SQLite repository:

```text
<workflow-volume>/
  .biomodals/execution/runs/<execution-run-id>/
    ledger.sqlite3
    workflow-plan.pkl
  workflow-runs/<execution-run-id>/
    nodes/
      <node-id>/
        result/
        cache/
        tasks/<task-key>/
          result/
          cache/
    artifacts/
      <artifact-id>.json
```

`WorkflowRunStore` owns the connection and transaction boundary. It embeds the
shared `SqliteExecutionRepository` tables and the narrow
`WorkflowArtifactStore` tables in the same database so publication and
execution transitions can commit together. There is no separate
`WorkflowLedger`, attempt table, or attempt-directory layer.

The Execution Coordinator is the only SQLite writer. Provider workers write
scientific outputs or Result Envelopes and communicate claims and completions
through coordinator methods; they never open `ledger.sqlite3`.

The shared repository stores execution records:

```text
execution_runs
execution_nodes
execution_tasks
execution_dispatch_batches
execution_provider_calls
execution_provider_call_tasks
execution_worker_assignments
```

Workflow-owned tables store artifacts, input/output links, and materialized
Node and Task `AppRunResult` records:

```text
workflow_artifacts
workflow_artifact_files
workflow_node_inputs
workflow_node_outputs
workflow_node_results
workflow_task_outputs
workflow_task_results
```

Keep large payloads in files and only durable `VolumePath` references in
SQLite. `InlineBytes` are materialized before the result is recorded. Commit
coordinator-local SQLite changes locally. Close SQLite and synchronize its
Volume only at a cross-container visibility or ownership boundary; reload
before reading another container's committed files. Do not commit the Volume
after every same-container state or file mutation.

The persisted `workflow-plan.pkl` is trusted internal state tied to the exact
deployment. A reopened Run must match its Workload Plan Fingerprint, Workload
Run Key, and Deployment Identity. Old pre-kernel workflow ledgers are rejected;
there is no compatibility facade or migration.

## Modal Preemption

All Modal functions are subject to preemption. Provider redelivery may execute
the same Provider Call and Task identity again, so remote functions must be
idempotent against deterministic inputs and publication paths.

Coordinator interruption is different from Task failure. The exit hook closes
and checkpoints state best-effort without cancelling children. A replacement
coordinator reloads SQLite, reattaches recorded call IDs, and derives active
call counts from durable records. An uncaught coordinator application error
suspends the Run until explicit `resume`.

Worker exit callbacks are advisory. They may checkpoint workload-owned data or
record diagnostics, but they do not fail, reassign, or retry a Task. Pull-worker
claims and completions use stable request IDs so a lost response can be replayed
without creating a new assignment.

Remote workflow code should:

- split large work into independently identified Tasks where scientifically
  meaningful;
- use deterministic output paths from Run, Node, and Task identities;
- validate publications rather than treating a claim or marker alone as
  completion;
- leave enough output and logs to reconcile the retained Provider Call;
- never submit replacement work when ownership or availability is `unknown`.

## Fan-Out

Use static DAG fan-out when the input cardinality is known during DAG
construction, as ShortMD does for per-PDB preparation and per-replicate
production runs. Use fixed DAG nodes with runtime task fan-out when a semantic
stage owns a candidate set, as PPIFlow does for candidate-wide stage
coordinators.

Use barriered fan-out first: a node starts only after all declared upstream
dependencies are complete. Streaming between nodes is deferred. Independent
ready nodes may run in parallel when all dependencies for each node are
satisfied.

## Orchestrator Submission

The reusable workflow orchestrator lives under `biomodals.workflow.core` and is
not a user-facing workflow script. Workflow scripts should import the module and
compose its app into their own Modal app:

```python
from biomodals.workflow.core import orchestrator

app = modal.App(...).include(orchestrator.app)
```

All remote orchestration functions should live as methods on
`ExecutionCoordinator`. Workflow apps obtain the class handle with
`orchestrator.execution_coordinator_handle(...)` and call
`orchestrator.submit_workflow_run(...)`. That helper submits `run` for a root
Run, or synchronously calls `prepare_restart_from` before spawning
`drive_prepared` for a Successor. The reusable orchestrator must not discover
workflow modules, perform floating deployed-app lookups, or own
workflow-specific input staging. Domain-specific staging, DAG construction,
and development function handles belong in top-level workflow scripts.

Pass workflow Node parallelism as `max_parallel_nodes` and remote fan-out
ceilings as `max_active_provider_calls` and
`max_active_gpu_provider_calls`. Do not collapse these into one runtime field;
a user-facing workflow flag may deliberately set both to the same value.

## Runtime Diagnostics

Use `ExecutionSnapshot` and the durable execution, Provider Call, and workflow
artifact rows for diagnostics. Do not expose private scheduler, repository, or
Volume-sync collaborators as routine workflow authoring APIs.

Keep the public coordinator surface minimal: `run`, `status`, `cancel`,
`resume`, `prepare_restart`, `prepare_restart_from`, `drive_prepared`, and the
pull-worker claim/completion callbacks. The coordinator does not expose generic
per-Node execution methods; runtime-managed Nodes only prepare work and the
kernel owns submission.

The reusable orchestrator module should not expose a local entrypoint for generic
workflow submission. Each user-facing workflow script owns its own local
entrypoint, stages its own inputs, builds its `Workflow` object, and submits that
object to the included `ExecutionCoordinator`.

The coordinator API accepts `Workflow` objects only. Workflow scripts build the
DAG locally and submit that object to `ExecutionCoordinator.run`. The
coordinator should not accept serialized workflow dictionaries or workflow
factory import strings as its primary run contract. Workflow Node classes must
therefore be importable in remote containers by canonical package-qualified
module names.

## CLI Namespace

Use `biomodals app ...` for app commands and `biomodals workflow ...` for
workflow commands. App and workflow discovery should live behind catalog helper
APIs; `cli.py` should not import app or workflow home constants directly.

Workflows should be launched through the `biomodals workflow run` CLI rather
than by running workflow Python files directly. The run command is responsible
for importing workflow modules through the catalog/package path so workflow node
classes serialize with stable canonical module names before being submitted to
the included `ExecutionCoordinator`. Coordinator-aware workflows resolve and
pin an exact deployed version by default. Their user-facing flags mirror
`biomodals app run`, including environment, deployment name/version, detach,
timeout, `--restart-from`, and pass-through workflow flags after `--`.

Launches print Deployment Identity, Execution Run ID, and Coordinator
FunctionCall ID. Use `biomodals run status|cancel|resume|restart` with the
explicit deployment and Run fields from any later CLI process. Repeating a
launch creates a new root Run unless `--restart-from` is supplied.

The run command also exposes `--dry-run`, which forwards `--dry-run` to the
selected workflow local entrypoint. User-facing workflow local entrypoints should
accept `dry_run: bool = False`; when set, they should build and validate the DAG,
call `print_workflow_dag(workflow.validate())`, and return before constructing
or submitting the orchestrator. DAG graph output should stay compact and print
Node IDs, execution boundary, workflow Node class qualnames, and dependencies
without module-qualified class names.
The command may accept workflow paths only when they resolve to package-qualified
modules under the Biomodals workflow package. Reject ad hoc workflow files that
cannot be imported by a stable package module path.
Use Modal's module mode for workflow runs, for example
`python -m modal run -m biomodals.workflow.shortmd_workflow::submit_shortmd_workflow`,
so local and remote containers agree on workflow node class module names.
Source-backed execution is explicit development mode and provides no
cross-process recovery.

## Workflow App Composition

Workflow scripts should compose every Modal app they need at import time. Define
dependency app names once on `AppConfig.depends_on_apps`, mirror that list into
`CONF.tags["depends_on"]` for Modal UI visibility, and call
`include_dependency_apps(app, CONF.depends_on_apps)` after including the shared
orchestrator app. Modal tag values cannot contain commas, so use a Modal-valid
delimiter such as `"-".join(DEPENDENCY_APPS)`.

```python
DEPENDENCY_APPS = ("gromacs",)
CONF = AppConfig(
    name="ShortMDWorkflow",
    depends_on_apps=DEPENDENCY_APPS,
    tags={"depends_on": "-".join(DEPENDENCY_APPS)},
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)
```

`depends_on_apps` is a composition declaration, not a deployment command. Do not
auto-deploy dependency apps from workflow submission paths. Including dependency
apps makes their functions and classes part of the containing workflow deployment
while letting Modal reuse normal image caching behavior.

Import dependency app modules directly for app metadata, volume objects, volume
names, and mountpoints. Do not duplicate volume names or mount paths. A
`RemoteNodeCall` must still declare the exact deployed function name; reuse an
app-exported name constant when one exists.

## App Interfaces

Local entrypoints stay CLI-only. They parse local paths, submit remote work,
download or report outputs, print user messages, and return `None`.

Workflow reuse happens through workflow-compatible remote app functions. These
functions may reuse behavior from local entrypoints or existing remote
functions, but they return `AppRunResult` and avoid local filesystem UX.
When developing a new app that may be used by a workflow, ask whether it needs a
workflow-compatible app function. If yes, coordinate with the app-development
skill and use `rfdiffusion_app.py` as the reference for durable `VolumePath`
outputs and `ligandmpnn_app.py` as the reference for small inline zstd archive
outputs.

For new Biomodals workflows that depend on other Biomodals apps, include those
apps in the workflow deployment and return exact function names from
`RemoteNodeCall`. The execution kernel resolves each name against the Run's
pinned containing deployment. Do not call `modal.Function.from_name(...)` or
carry hydrated Modal handles inside workflow Nodes. Explicit source-backed
development may pass a temporary function-name-to-handle map to the coordinator
without making those handles part of the DAG or Workload Plan Fingerprint.

Prefer `AppBackedNode` for nodes whose primary job is to invoke app functions.
Workflow definitions should reuse existing app functions whenever possible. Add
`WorkflowNativeNode` implementations only when the source app lacks a needed
function or when workflow-specific adapters are required to transform artifacts
between apps. Use native nodes for lightweight transforms, selectors, summaries,
and file-management glue that is not part of the source app's standalone
contract.

If a workflow-native adapter needs a remote Modal boundary, define a top-level
`@app.function` in the workflow module and use its exact function name in the
Node's `RemoteNodeCall`. Do not try to make ordinary node methods remote Modal
methods; node methods are plain Python methods unless the node itself is a Modal
`@app.cls`, which is not the generic workflow-node model.

Keep workflow-specific file cloning, cleanup, and adapter logic in workflow
scripts, not in app modules, when the standalone app does not require that
behavior. Conversely, if a function is useful to the app outside workflows, add
it to the app and preserve the app's existing standalone local entrypoints.

Group repeated app arguments in a compact workflow settings dataclass when that
keeps node constructors readable. Avoid extracting trivial two- or three-line
helpers that are used once or twice; inline those operations with a comment if
the intent is not obvious.

## Volumes And Artifacts

Workflows that import multiple apps should treat each app's volume metadata as
owned by that app. Import volume handles, volume names, and mountpoints from the
source app module rather than hardcoding them in the workflow.

When an app function returns an absolute path under its mounted volume, convert
that path to workflow storage with
`biomodals.helper.app_run.volume_path_from_mount_path(...)`. The helper takes
`str` inputs and returns a single validated `VolumePath`; do not construct a
`VolumePath` only to extract `.path` and wrap it again.

Workflow-native remote functions must `reload()` before reading data committed
by another container. Explicitly `commit()` writes, copies, or deletions before
another container consumes or acts on them; code in the same container sees its
own writes without a commit. Validate artifact storage paths with `VolumePath`
before joining them to mounted paths.

When a caller waits for a remote function that committed created, copied, or
deleted files in a mounted Volume, reload that same Volume before reading,
selecting, materializing, or validating those paths in the caller.

## DAG Construction

Build workflow DAGs locally from already-staged primitive data or Pydantic
models. Discover local inputs before DAG construction, sanitize user-derived
identifiers with `sanitize_filename`, and reject duplicate sanitized names. Use
stable Node IDs derived from sanitized names and deterministic indices so
fingerprints, successor recovery, and ledger debugging stay predictable.

Use static fan-out when the input cardinality is known at submission time. For
example, create one prep node per input, one clone node per replicate, one
production node per clone, and a final summary node that depends on all
production outputs. Keep per-run namespace prefixes explicit when the same input
filenames may appear across workflow runs.

Summary/report nodes should usually be `WorkflowNativeNode` instances when they
only aggregate manifests or emit text reports. Return reports as UTF-8
`InlineBytes`; return small zstd archives as `InlineBytes` with
`media_type="application/zstd"`; return other binary files, directories, and
large archives as durable `VolumePath` outputs.

When adding a workflow-compatible app function, keep existing local entrypoint
behavior unchanged and add a focused pytest contract test that does not call
Modal live APIs.

## Testing

Keep tests under top-level `tests/`.

Use pytest for non-Modal tests. Tests must not call `.remote()`, `.spawn()`,
`modal.Function.from_name(...)`, real `modal.Queue`, real `modal.Volume`, or
deployed Modal apps. Mock Modal boundaries with fake objects and deterministic
`AppRunResult` or `WorkflowArtifact` payloads.

For included-app workflows, tests should assert that the workflow app declares
the expected `depends_on_apps`, composes dependency apps through
`include_dependency_apps`, and imports app-owned volume metadata instead of
hardcoding it. Patch `modal.Function.from_name` to fail in tests that exercise
new included-app nodes so accidental deployed-app lookup regressions are caught.

Use fake Modal drivers and deterministic function-name-to-handle maps at the
coordinator boundary. The production Node contract remains primitive and names
the exact function; it does not carry Modal objects.
