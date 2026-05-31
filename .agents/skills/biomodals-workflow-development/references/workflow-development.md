# Biomodals Workflow Development

Use this guide when creating or changing files under
`src/biomodals/workflow/` or shared workflow contracts under
`src/biomodals/schema/`.

Use `src/biomodals/workflow/shortmd_workflow.py` as the primary end-to-end
workflow example. Ignore `src/biomodals/workflow/ppiflow_workflow.py` as a
reference pattern for now; it is expected to be refactored.

## Vocabulary

- **App**: a deployed Modal app that owns tool runtime and app functions.
- **App Function**: a callable remote Modal function exposed by an app.
- **Local Entrypoint**: a CLI-only `@app.local_entrypoint`.
- **Workflow-Compatible App Function**: a remote app function returning
  `AppRunResult`.
- **Workflow Node**: one semantic DAG vertex.
- **App-Backed Node**: a workflow node that calls app functions.
- **Workflow-Native Node**: a workflow node implemented in workflow code.
- **Workflow Runtime**: validates and schedules workflow nodes.
- **Workflow Orchestrator**: the Modal-hosted coordinator class hosting the runtime for one workflow run.
- **Workflow Artifact**: durable data passed between workflow nodes.
- **Artifact Selector**: a named reference to upstream artifacts.

Avoid the terms `app node`, `runner node`, `engine`, and `workflow
entrypoint`; they are ambiguous in this codebase.

## ShortMD Reference Pattern

ShortMD is the current reference for executable workflow apps. Its data flow is:

1. The local entrypoint discovers local `.pdb` files, sanitizes the workflow
   `run_id`, reads PDB bytes, builds a static `Workflow`, and submits that
   object to the included `WorkflowOrchestrator`.
2. The workflow app composes the shared orchestrator and the GROMACS app with
   `modal.App(...).include(orchestrator.app)` plus
   `include_dependency_apps(app, CONF.depends_on_apps)`.
3. `ShortMDModalNamespace` carries the hydrated GROMACS functions and
   workflow-native remote functions across the orchestrator boundary.
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
Inline byte outputs are for UTF-8 text bytes only. They must be written into the
workflow run volume before they cross a node boundary. Binary outputs, archives,
and other non-text bytes must be written to deterministic volume paths and
returned as `VolumePath` storage.

`AppRunResult.logs` are durable workflow artifacts too. The runtime writes log
outputs under `nodes/<node-id>/attempts/<attempt-id>/logs/` and records
artifact manifests for them so failed or partial attempts retain diagnostic
state.

Volume path outputs may either be referenced in place or copied into the
workflow run volume when the source volume is mounted locally. Reference mode is
the default because many app outputs are already durable in their owning app
volume. Copy mode is for workflows that need a self-contained run directory.

The first workflow runtime is Python-first. Pass a `Workflow` object across the
orchestrator boundary; serialized workflow dictionaries are intentionally
deferred until the node and app-function contracts stabilize.

## Node Execution Policy

Every workflow node checks durable SQLite run state before execution and skips work
when completed artifact manifests already exist.

Incomplete nodes use one of two policies:

- `RERUN`: discard incomplete attempt state and recompute.
- `RESUME`: use a durable node cache to resume or skip completed subwork.

Long-running nodes must be idempotent against deterministic run, node, input,
and attempt identifiers. Store resumable state in volumes, not container-local
scratch paths.

`AppRunStatus.PARTIAL` is terminal but not successful in the first runtime. The
runtime records the node as failed, records the run as failed, preserves logs,
and does not unblock downstream nodes.

Forced workflow runs replace the existing run directory before creating a fresh
ledger. Use force only when discarding previous artifacts, node caches, and
attempt records is intentional.

## Node Placement

Use `ORCHESTRATOR` placement for lightweight workflow-native logic such as
filtering, ranking, reporting, and small manifest transforms.

Use `REMOTE` placement for long-running work, app-backed work, and work that
benefits from failure isolation.

The runtime routes `REMOTE` nodes through an injected remote-node runner when
one is available. The Modal orchestrator supplies a thin remote runner that
executes one node in a separate Modal function and commits workflow volume
writes after node code returns. Unit tests use fake runners and must not call
live Modal APIs.

## Ledger Layout

The first durable run layout is:

```text
<workflow-volume>/<workflow-name>/<run-id>/
  ledger.sqlite3
  inputs/
  nodes/
    <node-id>/
      attempts/
        <attempt-id>/
          logs/
          raw_outputs/
          materialized_outputs/
      cache/
  artifacts/
    <artifact-id>/
  final/
```

The workflow ledger is one SQLite database per run. The orchestrator is the only
ledger writer. Remote nodes write deterministic output files and logs, then the
orchestrator reloads the volume, reconciles those files, and updates the ledger.

Ledger updates mutate SQLite rows directly. Do not preserve obsolete
Pydantic-status update patterns such as `model_copy(update=...)` for ledger
state.

After orchestrator ledger writes inside Modal containers, call `commit()`.
Before reading data written by another container, call `reload()`. Resuming a
run with a different DAG hash fails unless the run is forced, because stale node
state cannot safely be reused across workflow definition changes.

Record a Modal `FunctionCall.object_id` in `remote_calls` immediately after
submitting remote node work. On orchestrator startup or restart, reattach with
`modal.FunctionCall.from_id(call_id)` and poll before launching replacement
work. Reconcile existing pending, succeeded, failed, or expired calls and their
deterministic output files before applying `RERUN` or `RESUME`. Do not blindly
resubmit work while an older call may still be writing the same node outputs.

Use these tables for the first ledger schema:

```text
runs(run_id, workflow_name, dag_hash, status, created_at, updated_at, metadata_json)
nodes(node_id, status, execution_policy, placement, current_attempt_id, error, started_at, completed_at, updated_at)
attempts(attempt_id, node_id, status, started_at, completed_at, app_result_json, error, metadata_json)
remote_calls(call_id, node_id, attempt_id, function_name, call_kind, status, submitted_at, completed_at, error, metadata_json)
artifacts(artifact_id, producing_node_id, kind, volume_name, storage_path, source_app_output_name, created_at, metadata_json)
artifact_files(artifact_id, path, role, media_type, size_bytes, metadata_json)
node_inputs(node_id, input_name, artifact_id)
node_outputs(node_id, artifact_id)
```

Keep large payloads in files and store paths in SQLite. Store non-Pydantic
metadata JSON text with `orjson`. Store Pydantic payload snapshots with
`model_dump_json()` and load them with `model_validate_json(...)`. A human
should be able to debug a run with `sqlite3` by
checking `runs.status`, stalled rows in `nodes`, outstanding `remote_calls`,
and artifact paths in `artifacts` plus `artifact_files`.

## Modal Preemption

All Modal functions are subject to preemption. Treat remote functions as
restartable with the same inputs.

Remote workflow code should:

- split long work into smaller retryable tasks;
- expose enough attempt status, artifacts, and logs for the orchestrator to
  record ledger state before and after work;
- write cache checkpoints for `RESUME` nodes;
- use deterministic output paths from run and node identifiers;
- leave enough artifacts and logs to reconcile after restart.

## Fan-Out

The first workflow runtime supports static DAG fan-out. Build one node per known
unit of work during DAG construction, as ShortMD does for per-PDB preparation
and per-replicate production runs.

Use barriered fan-out first: a node starts only after all declared upstream
dependencies are complete. Streaming between nodes is deferred.

Independent ready nodes may run in parallel when all dependencies for each node
are satisfied.

## Orchestrator Submission

The reusable workflow orchestrator lives under `biomodals.workflow.core` and is
not a user-facing workflow script. Workflow scripts should import the module and
compose its app into their own Modal app:

```python
from biomodals.workflow.core import orchestrator

app = modal.App(...).include(orchestrator.app)
```

All remote orchestration functions should live as methods on
`WorkflowOrchestrator`. Workflow apps may use the included
`WorkflowOrchestrator` methods for run submission, but the reusable orchestrator
must not perform deployed app lookups, import workflow app functions by name, or
handle hydration details for workflow-specific apps. Domain-specific input
staging and DAG construction belong in top-level workflow scripts.

Keep the public orchestrator method surface minimal. The intended remote methods
are `WorkflowOrchestrator.run(...)` for a whole workflow run and
`WorkflowOrchestrator.run_node(...)` for isolated remote node execution. Do not
add convenience wrappers or alternate submission APIs unless they cover a large
missing capability or a clear ergonomics gap.

The reusable orchestrator module should not expose a local entrypoint for generic
workflow submission. Each user-facing workflow script owns its own local
entrypoint, stages its own inputs, builds its `Workflow` object, and submits that
object to the included `WorkflowOrchestrator`.

The orchestrator API accepts `Workflow` objects only. Workflow scripts build the
DAG locally and submit that object to the included `WorkflowOrchestrator`
method. The orchestrator should not accept serialized workflow dictionaries or
workflow factory import strings as its primary run contract. Workflow node
classes must therefore be importable in remote containers by their canonical
package-qualified module names.

## CLI Namespace

Use `biomodals app ...` for app commands and `biomodals workflow ...` for
workflow commands. App and workflow discovery should live behind catalog helper
APIs; `cli.py` should not import app or workflow home constants directly.

The workflow namespace should expose `list` and `help` first. Other workflow
commands can exist as placeholders until the runtime execution interface is
stable. Existing top-level app commands may remain as deprecated aliases for one
transition period, but documentation and smoke tests should prefer the
namespaced commands.

Workflows should be launched through the `biomodals workflow run` CLI rather
than by running workflow Python files directly. The run command is responsible
for importing workflow modules through the catalog/package path so workflow node
classes serialize with stable canonical module names before being submitted to
the included `WorkflowOrchestrator`. Its user-facing flags should mirror
`biomodals app run`, including Modal mode, detach, timeout, and pass-through
workflow flags after `--`.
The command may accept workflow paths only when they resolve to package-qualified
modules under the Biomodals workflow package. Reject ad hoc workflow files that
cannot be imported by a stable package module path.
Use Modal's module mode for workflow runs, for example
`python -m modal run -m biomodals.workflow.shortmd_workflow::submit_shortmd_workflow`,
so local and remote containers agree on workflow node class module names.

## Workflow App Composition

Workflow scripts should compose every Modal app they need at import time. Define
dependency app names once on `AppConfig.depends_on_apps`, mirror that list into
`CONF.tags["depends_on"]` for Modal UI visibility, and call
`include_dependency_apps(app, CONF.depends_on_apps)` after including the shared
orchestrator app.

```python
DEPENDENCY_APPS = ("gromacs",)
CONF = AppConfig(
    name="ShortMDWorkflow",
    depends_on_apps=DEPENDENCY_APPS,
    tags={"depends_on": ",".join(DEPENDENCY_APPS)},
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)
```

`depends_on_apps` is a composition declaration, not a deployment command. Do not
auto-deploy dependency apps from workflow submission paths. Including dependency
apps gives the workflow access to hydrated Modal functions and classes while
letting Modal reuse normal image caching behavior.

Import dependency app modules directly for app metadata, Modal function handles,
volume objects, volume names, and mountpoints. Do not duplicate volume names,
mount paths, or app function names as workflow-local string constants when the
source app exports them.

## App Interfaces

Local entrypoints stay CLI-only. They parse local paths, submit remote work,
download or report outputs, print user messages, and return `None`.

Workflow reuse happens through workflow-compatible remote app functions. These
functions may reuse behavior from local entrypoints or existing remote
functions, but they return `AppRunResult` and avoid local filesystem UX.

For new Biomodals workflows that depend on other Biomodals apps, prefer
included-app Modal handles over deployed-app lookup strings. Avoid
`modal.Function.from_name(...)` in workflow definitions when the dependency app
can be included; remote orchestrator containers can otherwise re-import the
workflow module and see unhydrated function globals. Use deployed-app lookup only
for legacy workflows or external apps that cannot be composed into the workflow
app, and document that reason near the node.

When nodes need included Modal functions or classes, group those hydrated
objects in a small workflow-local dataclass named `*ModalNamespace`. Type the
fields as `modal.Function` or `modal.Cls`; avoid overly generic callable
protocols. Store the namespace on nodes as runtime-only state:

```python
@dataclass(frozen=True)
class ShortMDModalNamespace:
    prepare_gpu: modal.Function
    production_gpu: modal.Function


@dataclass
class ShortMDPrepNode(AppBackedNode):
    modal_namespace: ShortMDModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
```

The namespace is allowed to cross the orchestrator boundary because it contains
Modal objects from apps included into the workflow app. Excluding it from the
DAG hash keeps retry and resume behavior tied to semantic workflow inputs rather
than runtime hydration objects.

Prefer `AppBackedNode` for nodes whose primary job is to invoke app functions.
Workflow definitions should reuse existing app functions whenever possible. Add
`WorkflowNativeNode` implementations only when the source app lacks a needed
function or when workflow-specific adapters are required to transform artifacts
between apps. Use native nodes for lightweight transforms, selectors, summaries,
and file-management glue that is not part of the source app's standalone
contract.

If a workflow-native adapter needs a remote Modal boundary, define a top-level
`@app.function` in the workflow module and put that hydrated function in the
workflow's `*ModalNamespace`. Do not try to make ordinary node methods remote
Modal methods; node methods are plain Python methods unless the node itself is a
Modal `@app.cls`, which is not the generic workflow-node model.

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
`biomodals.helper.volume_run.volume_path_from_mount_path(...)`. The helper takes
`str` inputs and returns a single validated `VolumePath`; do not construct a
`VolumePath` only to extract `.path` and wrap it again.

Workflow-native remote functions that mutate mounted volumes must call
`reload()` before reading data written by other containers and `commit()` after
writing, copying, or deleting files. Validate artifact storage paths with
`VolumePath` before joining them to mounted paths.

## DAG Construction

Build workflow DAGs locally from already-staged primitive data or Pydantic
models. Discover local inputs before DAG construction, sanitize user-derived
identifiers with `sanitize_filename`, and reject duplicate sanitized names. Use
stable node ids derived from sanitized names and deterministic indices so
resume, force, and ledger debugging stay predictable.

Use static fan-out when the input cardinality is known at submission time. For
example, create one prep node per input, one clone node per replicate, one
production node per clone, and a final summary node that depends on all
production outputs. Keep per-run namespace prefixes explicit when the same input
filenames may appear across workflow runs.

Summary/report nodes should usually be `WorkflowNativeNode` instances with
`ORCHESTRATOR` placement when they only aggregate manifests or emit text
reports. Return reports as UTF-8 `InlineBytes`; return binary files,
directories, and archives as durable `VolumePath` outputs.

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

Use fake Modal namespace objects at node boundaries. Cast those fakes to
`modal.Function` or `modal.Cls` in tests when needed to satisfy static typing;
the production node contract should remain Modal-object based.
