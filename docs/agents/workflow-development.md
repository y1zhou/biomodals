# Biomodals Workflow Development

Use this guide when creating or changing files under
`src/biomodals/workflow/` or shared workflow contracts under
`src/biomodals/schema/`.

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
- **Worker Pool**: fixed-size remote workers for one node's fan-out tasks.

Avoid the terms `app node`, `runner node`, `engine`, and `workflow
entrypoint`; they are ambiguous in this codebase.

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
ledger writer. Remote nodes and workers write deterministic output files and
logs, then the orchestrator reloads the volume, reconciles those files, and
updates the ledger.

Ledger updates mutate SQLite rows directly. Do not preserve obsolete
Pydantic-status update patterns such as `model_copy(update=...)` for ledger
state.

After orchestrator ledger writes inside Modal containers, call `commit()`.
Before reading data written by another container, call `reload()`. Resuming a
run with a different DAG hash fails unless the run is forced, because stale node
state cannot safely be reused across workflow definition changes.

Record a Modal `FunctionCall.object_id` in `remote_calls` immediately after
submitting remote node or worker work. On orchestrator startup or restart,
reattach with `modal.FunctionCall.from_id(call_id)` and poll before launching
replacement work. Reconcile existing pending, succeeded, failed, or expired
calls and their deterministic output files before applying `RERUN` or `RESUME`.
Do not blindly resubmit work while an older call may still be writing the same
node outputs.

Use these tables for the first ledger schema:

```text
runs(run_id, workflow_name, dag_hash, status, created_at, updated_at, metadata_json)
nodes(node_id, status, execution_policy, placement, current_attempt_id, error, started_at, completed_at, updated_at)
attempts(attempt_id, node_id, status, started_at, completed_at, app_result_json, error, metadata_json)
remote_calls(call_id, node_id, attempt_id, function_name, call_kind, status, submitted_at, completed_at, error, metadata_json)
node_tasks(task_id, node_id, attempt_id, status, input_artifact_id, output_artifact_id, remote_call_id, claimed_by, started_at, completed_at, error, metadata_json)
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
fan-out progress in `node_tasks`, and artifact paths in `artifacts` plus
`artifact_files`.

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

## Fan-Out And Worker Pools

The first workflow runtime supports static DAGs with dynamic task fan-out.
The DAG shape is fixed, but a node may derive a runtime task list from upstream
artifacts.

Use barriered fan-out first: a node starts only after all declared upstream
dependencies are complete. Streaming between nodes is deferred.

A fan-out node may spawn a fixed-size worker pool. Workers process that node's
task queue until empty, then expose deterministic outputs that let the
orchestrator write one completion status.

Independent ready nodes may run in parallel when all dependencies for each node
are satisfied.

Keep worker-pool naming, task ids, queue enqueueing, and result aggregation in
pure helpers. Keep Modal queue creation, worker spawning, and
`FunctionCall.gather` calls in thin integration helpers so unit tests can use
fake queues and fake function calls.

## Orchestrator Submission

The reusable workflow orchestrator lives under `biomodals.workflow.core` and is
not a user-facing workflow script. Define it as a Modal class so lifecycle hooks
can reload and commit durable state around run execution and cleanup. Its local
helper accepts a no-argument Python factory in `module:function` form and
submits the returned `Workflow` object to the remote orchestrator.
Domain-specific input staging belongs in top-level workflow scripts.

## CLI Namespace

Use `biomodals app ...` for app commands and `biomodals workflow ...` for
workflow commands. App and workflow discovery should live behind catalog helper
APIs; `cli.py` should not import app or workflow home constants directly.

The workflow namespace should expose `list` and `help` first. Other workflow
commands can exist as placeholders until the runtime execution interface is
stable. Existing top-level app commands may remain as deprecated aliases for one
transition period, but documentation and smoke tests should prefer the
namespaced commands.

## App Interfaces

Local entrypoints stay CLI-only. They parse local paths, submit remote work,
download or report outputs, print user messages, and return `None`.

Workflow reuse happens through workflow-compatible remote app functions. These
functions may reuse behavior from local entrypoints or existing remote
functions, but they return `AppRunResult` and avoid local filesystem UX.

App-backed workflow nodes either define `app_name` and `function_name` so the
runtime can lazily import `modal.Function.from_name(...)`, or override
`load_app_function()`. They should override `build_app_function_kwargs()` to
translate `NodeRunContext.inputs` into the app function's primitive or Pydantic
arguments.

`load_app_function()` and any stored app function reference should be typed as
`modal.Function`. An app-backed node is not expected to call a regular Python
`Callable`; unit tests may use fakes at the Modal boundary, but production node
contracts should stay Modal-function based.

When adding a workflow-compatible app function, keep existing local entrypoint
behavior unchanged and add a focused pytest contract test that does not call
Modal live APIs.

## Testing

Keep tests under top-level `tests/`.

Use pytest for non-Modal tests. Tests must not call `.remote()`, `.spawn()`,
`modal.Function.from_name(...)`, real `modal.Queue`, real `modal.Volume`, or
deployed Modal apps. Mock Modal boundaries with fake objects and deterministic
`AppRunResult` or `WorkflowArtifact` payloads.
