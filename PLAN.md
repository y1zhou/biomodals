<!-- markdownlint-disable MD013 -->

# Modal Workflow Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Biomodals workflow interface that composes Modal app functions into a DAG, passes durable artifacts between nodes, survives Modal preemption, and preserves logs and local outputs.

**Architecture:** Add shared Pydantic schemas under `biomodals.schema`, then build a Python-first workflow builder and runtime under `biomodals.workflow`. Workflows run through a Modal-hosted workflow orchestrator class that schedules workflow nodes, records durable state in a workflow volume, materializes app outputs into workflow artifacts, and uses Modal lifecycle hooks plus durable run state to resume or reconcile interrupted nodes according to explicit policies.

**Tech Stack:** Python 3.12+ for the package, Python 3.13 for the dependency-light workflow orchestrator runtime, Modal, Pydantic v2, Polars for tabular parsing/writing, pytest for focused non-Modal tests, shared Biomodals app config schema and Modal helper APIs.

______________________________________________________________________

## Context

The current example, `src/biomodals/workflow/ppiflow_workflow.py`, proves the target behavior but is not a reusable runtime. It hardcodes the PPIFlow stage order, imports app functions by name in-line, mixes app-specific output handling with orchestration, manages Modal queues directly, and returns a final `.tar.zst` archive after copying outputs into a workflow volume.

The refactor should keep that working behavior as the reference scenario, but extract reusable contracts and scheduling behavior so future workflows can compose any deployed Biomodals app functions.

Execution note: in this feature branch, `ppiflow_workflow.py` was already the
working behavioral reference before the reusable runtime extraction. The runtime
work keeps that branch version intact and does not replace it with a separate
definition-only scaffold.

## Vocabulary

Use the terms defined in `CONTEXT.md`.

- **App**: a deployed Modal app that owns runtime images, volumes, and exported app functions.
- **App Function**: a callable remote Modal function exposed by an app.
- **Local Entrypoint**: a CLI-only `@app.local_entrypoint`; workflows must not call it.
- **Workflow-Compatible App Function**: a remote `@app.function` with standardized workflow input/output schemas.
- **Workflow Definition**: the Python-declared DAG.
- **Workflow Node**: one semantic DAG vertex.
- **App-Backed Node**: a workflow node implemented by calling one or more app functions.
- **Workflow-Native Node**: a workflow node implemented directly in workflow code.
- **Workflow Runtime**: the reusable library that validates and schedules workflow nodes.
- **Workflow Orchestrator**: the Modal-hosted coordinator class that hosts the workflow runtime for one run.
- **Workflow Artifact**: a durable record of data passed between nodes.
- **Artifact Selector**: a named input reference that selects upstream artifacts.
- **Worker Pool**: a fixed-size set of remote workers used by one node for dynamic task fan-out.

Avoid the overloaded terms "app node", "runner node", "engine", and "workflow entrypoint".

## Documentation Checked

- Modal preemption: <https://modal.com/docs/guide/preemption>
- Modal retries: <https://modal.com/docs/guide/retries>
- Modal volumes: <https://modal.com/docs/guide/volumes>
- Modal queues: <https://modal.com/docs/guide/queues>
- Modal apps and local entrypoints: <https://modal.com/docs/guide/apps>
- Snakemake rules and DAG concepts: <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html>
- Snakemake basics: <https://snakemake.readthedocs.io/en/stable/tutorial/basics.html>
- Snakemake logs and benchmarks: <https://snakemake.readthedocs.io/en/stable/tutorial/additional_features.html>
- pytest examples and fixtures: <https://docs.pytest.org/>

## Architecture Decisions

01. Workflows are Python-first. YAML or JSON workflow serialization is not required for the first version.
02. Shared cross-module contracts live in `biomodals.schema`; the pure Pydantic `AppConfig` schema belongs in `biomodals.schema.app`, while Modal-specific helpers stay outside `biomodals.schema`.
03. App local entrypoints stay CLI-only. Workflow reuse happens through remote `@app.function`s.
04. Workflow-compatible app functions return `AppRunResult`.
05. The workflow runtime converts `AppRunResult.outputs` into durable `WorkflowArtifact` manifests.
06. `InlineBytes` app outputs are UTF-8 text only and are always materialized into the workflow run volume before crossing a node boundary.
07. The first runtime uses static DAGs with dynamic task fan-out. Runtime task counts may vary, but node and edge types do not appear dynamically.
08. Node dependencies are inferred from named `ArtifactSelector`s; `ControlEdge`s are available for ordering without data passage.
09. Every node checks the durable SQLite ledger before execution and skips completed work.
10. Incomplete nodes use one of two execution policies:
    - `RERUN`: discard incomplete attempt state and recompute.
    - `RESUME`: use a durable node cache to resume or safely skip completed subwork.
11. Nodes declare placement:
    - `ORCHESTRATOR`: lightweight workflow-native logic runs inline in the orchestrator.
    - `REMOTE`: long-running or failure-isolated work runs in a separate remote Modal function.
12. Long-running nodes must be idempotent against deterministic run, node, input, and attempt identifiers, and must store durable cache state in volumes.
13. Barriered fan-out is the first supported fan-out model. A node starts after its declared upstream dependencies complete, then may spawn a fixed-size worker pool to process selected inputs.
14. Independent ready nodes may run in parallel when all dependencies for each node are satisfied.
15. Streaming between nodes is outside the first implementation.
16. The workflow ledger is one SQLite database per run. The **Workflow Orchestrator** is the only writer; remote nodes and workers write deterministic output files and logs, then the orchestrator reloads the volume and records ledger rows.
17. Generic JSON serialization uses `orjson`; Pydantic model JSON serialization uses `model_dump_json()`, and Pydantic model JSON parsing uses `model_validate_json(...)`.
18. The CLI is namespaced: `biomodals app ...` is the canonical app interface, `biomodals workflow ...` is the canonical workflow interface, and old top-level app commands remain temporary deprecated aliases for one transition period.

## Target File Structure

Create or modify these files.

- Create `src/biomodals/schema/__init__.py`: public exports for shared schemas.
- Create `src/biomodals/schema/storage.py`: `InlineBytes`, `VolumePath`, and storage discriminators.
- Create `src/biomodals/schema/app.py`: pure Pydantic `AppConfig`, `AppOutput`, `AppRunResult`, and app status enums.
- Modify `src/biomodals/app/config.py`: keep compatibility imports and Modal-specific helpers such as output volume construction outside the schema package.
- Create `src/biomodals/schema/workflow.py`: `WorkflowArtifact`, `ArtifactFile`, node/run status schemas, execution policy, placement, and selector models.
- Create `src/biomodals/workflow/core/artifacts.py`: materialize app outputs into workflow volume paths and create artifact manifests.
- Create `src/biomodals/workflow/core/ledger.py`: read/write run, node, attempt, remote-call, task, and artifact rows in the run SQLite ledger.
- Create `src/biomodals/workflow/core/nodes.py`: base workflow node protocol plus app-backed and workflow-native node helpers.
- Create `src/biomodals/workflow/core/builder.py`: Python workflow definition API and DAG validation.
- Create `src/biomodals/workflow/core/runtime.py`: scheduler, skip-if-complete logic, parallel ready-node execution, and node placement dispatch.
- Create `src/biomodals/workflow/core/orchestrator.py`: workflow orchestrator class, Modal boundary, lifecycle hooks, and submission helpers.
- Create `src/biomodals/workflow/core/workers.py`: reusable worker-pool queue helpers for fan-out nodes.
- Modify `src/biomodals/app/catalog.py`: expose separate app and workflow catalog helpers so `cli.py` does not import `APP_HOME` or `WORKFLOW_HOME`.
- Modify `src/biomodals/cli.py`: add `app` and `workflow` subcommands and keep top-level aliases as deprecated compatibility wrappers.
- Modify `src/biomodals/workflow/__init__.py`: export the builder and core runtime types.
- Keep `src/biomodals/workflow/ppiflow_workflow.py` intact during the first runtime extraction; use it as a behavioral reference.
- Modify `pyproject.toml`: add `pytest` to the `dev` dependency group.
- Create `tests/schema/test_workflow_schemas.py`: schema validation tests.
- Create `tests/workflow/test_artifacts.py`: local artifact materialization tests using temporary directories.
- Create `tests/workflow/test_builder.py`: DAG and selector validation tests.
- Create `tests/workflow/test_ledger.py`: durable ledger read/write tests.
- Create `tests/workflow/test_runtime.py`: scheduler behavior tests with fake nodes.
- Create `tests/workflow/test_orchestrator.py`: mocked orchestrator boundary tests that do not call Modal.
- Create `tests/workflow/test_workers.py`: worker-pool helper tests using fake queues.
- Create `tests/app/test_flowpacker_workflow_contract.py`: focused app contract test with mocked app dependencies.

## Testing Policy

Keep tests in the top-level `tests/` directory. Use pytest with plain `assert`, `tmp_path`, and `monkeypatch` for focused tests around schema validation, DAG construction, ledger row writes, artifact materialization, scheduler decisions, and worker-pool naming.

Do not execute live Modal work in tests. Tests must not call `.remote()`, `.spawn()`, `modal.Function.from_name(...)`, real `modal.Queue`, real `modal.Volume`, or deployed Modal apps. When integration-style coverage is useful, mock the Modal boundary with fake objects that assert expected inputs and return deterministic `AppRunResult` or `WorkflowArtifact` payloads.

## Shared Schema Sketch

These names and fields are the first target contract.

```python
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AppRunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


class ArtifactKind(StrEnum):
    STRUCTURES = "structures"
    SCORES = "scores"
    REPORT = "report"
    ARCHIVE = "archive"
    DIRECTORY = "directory"
    TABLE = "table"
    LOGS = "logs"


class StorageKind(StrEnum):
    INLINE_BYTES = "inline_bytes"
    VOLUME_PATH = "volume_path"


class InlineBytes(BaseModel):
    kind: Literal[StorageKind.INLINE_BYTES] = StorageKind.INLINE_BYTES
    data: bytes
    filename: str
    media_type: str | None = None


class VolumePath(BaseModel):
    kind: Literal[StorageKind.VOLUME_PATH] = StorageKind.VOLUME_PATH
    volume_name: str
    path: str
    media_type: str | None = None


class AppOutput(BaseModel):
    name: str
    kind: ArtifactKind
    storage: InlineBytes | VolumePath = Field(discriminator="kind")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppRunResult(BaseModel):
    status: AppRunStatus
    outputs: list[AppOutput] = Field(default_factory=list)
    metrics: dict[str, str | int | float | bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    logs: list[AppOutput] = Field(default_factory=list)
```

`WorkflowArtifact` should reference durable workflow state after materialization:

```python
class ArtifactFile(BaseModel):
    path: str
    role: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowArtifact(BaseModel):
    artifact_id: str
    producing_node_id: str
    kind: ArtifactKind
    storage: VolumePath
    files: list[ArtifactFile] = Field(default_factory=list)
    source_app_output_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

`biomodals.schema` must not import `modal`, `biomodals.app`, or `biomodals.workflow`. Move the pure Pydantic `AppConfig` fields and validators into `biomodals.schema.app`, but keep helpers that construct `modal.Volume`, Modal images, or Modal apps in `biomodals.app` or `biomodals.helper`.

`InlineBytes.data` is for UTF-8 text bytes only. Do not configure Pydantic byte serialization as base64 for this schema. Binary outputs, archives, and other non-text bytes must be written to deterministic volume paths and represented by `VolumePath` storage instead.

## Initial Ledger Layout

The first durable workflow run layout should be:

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

`ledger.sqlite3` records workflow run state, node status, attempts, submitted Modal function calls, fan-out tasks, artifact manifests, and selected node inputs/outputs. Keep large payloads in files, not database rows. Store non-Pydantic metadata JSON text with `orjson`; store Pydantic payload snapshots with `model_dump_json()` and load them with `model_validate_json(...)`.

SQLite writes must go through `WorkflowLedger`; do not let remote nodes or workers open the database for mutation. After orchestrator ledger writes in Modal containers, call `commit()` on the mounted volume. Before reconciling files written by another function/container, call `reload()`. Remote nodes and workers should write deterministic files under their attempt/task directories and return or expose enough information for the orchestrator to update the ledger.

When the orchestrator submits remote node or worker work, it records the Modal `FunctionCall.object_id` in `remote_calls` immediately, before waiting for the result. During startup or restart recovery, the orchestrator reconciles every non-terminal `remote_calls` row by reattaching with `modal.FunctionCall.from_id(call_id)` and polling with `get(timeout=0)` or a short timeout. It classifies calls as pending, succeeded, failed, or expired, reloads the workflow volume, reconciles deterministic output files, and only then applies `RERUN` or `RESUME`. The scheduler must not blindly resubmit work for a node while a previously submitted call may still be writing that node's outputs.

Human-debuggable first table schema:

```text
runs(
  run_id text primary key,
  workflow_name text not null,
  dag_hash text,
  status text not null,
  created_at text not null,
  updated_at text not null,
  metadata_json text not null default '{}'
)

nodes(
  node_id text primary key,
  status text not null,
  execution_policy text not null,
  placement text not null,
  current_attempt_id text,
  error text,
  started_at text,
  completed_at text,
  updated_at text not null
)

attempts(
  attempt_id text primary key,
  node_id text not null references nodes(node_id),
  status text not null,
  started_at text not null,
  completed_at text,
  app_result_json text,
  error text,
  metadata_json text not null default '{}'
)

remote_calls(
  call_id text primary key,
  node_id text not null references nodes(node_id),
  attempt_id text references attempts(attempt_id),
  function_name text not null,
  call_kind text not null,
  status text not null,
  submitted_at text not null,
  completed_at text,
  error text,
  metadata_json text not null default '{}'
)

node_tasks(
  task_id text primary key,
  node_id text not null references nodes(node_id),
  attempt_id text references attempts(attempt_id),
  status text not null,
  input_artifact_id text,
  output_artifact_id text,
  remote_call_id text references remote_calls(call_id),
  claimed_by text,
  started_at text,
  completed_at text,
  error text,
  metadata_json text not null default '{}'
)

artifacts(
  artifact_id text primary key,
  producing_node_id text not null references nodes(node_id),
  kind text not null,
  volume_name text not null,
  storage_path text not null,
  source_app_output_name text,
  created_at text not null,
  metadata_json text not null default '{}'
)

artifact_files(
  artifact_id text not null references artifacts(artifact_id),
  path text not null,
  role text,
  media_type text,
  size_bytes integer,
  metadata_json text not null default '{}',
  primary key (artifact_id, path)
)

node_inputs(
  node_id text not null references nodes(node_id),
  input_name text not null,
  artifact_id text not null references artifacts(artifact_id),
  primary key (node_id, input_name, artifact_id)
)

node_outputs(
  node_id text not null references nodes(node_id),
  artifact_id text not null references artifacts(artifact_id),
  primary key (node_id, artifact_id)
)
```

Debugging expectation: a human should be able to inspect `ledger.sqlite3` with `sqlite3`, check `runs.status`, see stalled nodes in `nodes`, correlate `attempts` and `remote_calls` by `attempt_id`, inspect fan-out work in `node_tasks`, and map output artifacts to volume paths through `artifacts` and `artifact_files`.

## Python Builder Shape

Target authoring API:

```python
from biomodals.workflow import Workflow
from biomodals.workflow.ppiflow_v2 import PPIFlowWorkflowNode

workflow = Workflow("ppiflow")

designs = workflow.add_node(
    PPIFlowWorkflowNode("PPIFlowStep", {"stage": 1}),
    id="designs",
)

packed = workflow.add_node(
    PPIFlowWorkflowNode("MPNNStep_stage1", {"model_type": "soluble_mpnn"}),
    id="packed",
    inputs={
        "structures": designs.outputs(
            kind="structures",
            pattern="**/*.pdb",
        ),
    },
)

scores = workflow.add_node(
    PPIFlowWorkflowNode("AF3scoreStep_stage1", {"max_batches": 10}),
    id="scores",
    inputs={
        "structures": packed.outputs(
            kind="structures",
            pattern="**/*.pdb",
        ),
    },
)

filtered = workflow.add_node(
    FilterStructuresNode(filters={"iptm": "> 0.7"}),
    id="filtered",
    inputs={
        "structures": packed.outputs(kind="structures", pattern="**/*.pdb"),
        "scores": scores.outputs(kind="scores", pattern="**/*.csv"),
    },
)
```

Builder rules:

- Node ids are unique and sanitized.
- Dependencies are inferred from `ArtifactSelector`s returned by `NodeHandle.outputs(...)` and used in node inputs.
- `depends_on=[ranked]` adds control edges without artifact passage.
- Artifact selectors can filter by producing node, artifact kind, file role, glob pattern, and metadata.
- The DAG must be acyclic.
- Selector resolution happens at runtime after upstream artifacts exist.

## Task Plan

The first implementation slice is now in code. Finished tasks are summarized here
instead of restating implementation details that can be read from the files and
tests.

### Completed Implementation Summary

- **Shared schemas:** `biomodals.schema` now exports storage, app, and workflow
  Pydantic contracts. `biomodals.app.config.AppConfig` remains a compatibility
  import plus Modal-specific helpers outside the schema package. Schema tests
  cover round trips, schema import boundaries, and inline-byte constraints.
- **Artifact materialization:** `workflow.core.artifacts` materializes
  `AppRunResult` outputs and logs into workflow artifacts, validates UTF-8
  `InlineBytes`, supports volume-path reference/copy modes, and writes artifact
  manifests.
- **Workflow ledger:** `workflow.core.ledger.WorkflowLedger` owns the per-run
  SQLite layout, run/node/attempt/remote-call/task/artifact tables, deterministic
  run paths, and raw-SQL-debuggable state transitions.
- **Builder and node contracts:** `workflow.core.builder` provides the
  Python-first `Workflow` API, data/control dependency inference, cycle checks,
  and node handles that return `ArtifactSelector`s directly. `workflow.core.nodes`
  provides node context plus native and app-backed node contracts.
- **Runtime scheduler:** `workflow.core.runtime.WorkflowRuntime` validates DAGs,
  skips completed nodes, schedules ready nodes in waves, dispatches remote
  placement through an injected runner, records Modal call ids with an explicit
  remote function name before waiting, materializes outputs, and handles
  `RERUN`/`RESUME` state.
- **Modal orchestrator:** `workflow.core.orchestrator` defines the Modal-hosted
  `WorkflowOrchestrator`, local submission helper, runtime wiring, volume
  reload/commit hooks, documented `module:function` workflow factory loading,
  and mocked boundary tests.
- **Worker helpers:** `workflow.core.workers` covers worker-pool naming,
  deterministic task ids, queue enqueueing, spawn/gather adapters, and result
  summaries.
- **First workflow-compatible app function:** `flowpacker_app.py` now exposes
  `run_flowpacker_workflow(...) -> AppRunResult` while preserving the existing
  local entrypoint behavior.
- **Workflow discovery and docs:** `biomodals app ...` / `biomodals workflow ...`
  discovery is wired, workflow development guidance lives in
  `docs/agents/workflow-development.md`, and app-development guidance points to
  the repo-local skill.
- **Definition-only PPIFlow DAG:** `ppiflow_v2.py` now uses the data-driven
  `PPIFlowWorkflowNode` for app-backed steps and keeps semantic identity in
  `step_name` until individual steps own distinct behavior.

### Current Verification Status

- `rtk uv run pytest -q -W error::DeprecationWarning` passes: 96 passed.
- `rtk uv run ty check <changed-python-files>` passes for the workflow files and
  tests touched by these fixes.
- `rtk prek run --files <changed files>` passes.

### Further Tasks

- [ ] **Make the PPIFlow v2 definition executable deliberately.** Wire real
  workflow-compatible app functions into app-backed nodes, implement
  `build_app_function_kwargs(...)`, and replace placeholder native filter/report
  nodes with implementations that emit durable `WorkflowArtifact`s.
- [ ] **Add workflow-compatible scoring outputs.** Decide whether DockQ and
  AF3Score should expose `AppRunResult` workflow functions directly or be called
  through adapter nodes, then add focused contract tests that avoid live Modal
  calls.
- [ ] **Strengthen runtime recovery tests.** Add coverage for resumed remote
  calls that write output files before the orchestrator finalizes them, failed
  remote calls with preserved logs, and copied empty `VolumePath` directories.
- [ ] **Keep CLI namespace transition explicit.** Before making workflow runs
  user-facing, add or document `biomodals workflow run` semantics separately
  from `biomodals app run`, and decide when deprecated top-level app aliases can
  be removed.

## Deferred Decisions

- External YAML or JSON workflow authoring.
- Arbitrary runtime DAG mutation.
- Streaming artifacts from one node to another before the upstream node completes.
- Cross-run content-addressed caching.
- A global registry of workflow-compatible app functions.
- A stable public plugin API for third-party apps.

## Rollback Notes

If the runtime extraction stalls, shared schemas and docs can remain because they do not alter deployed app behavior. If workflow-compatible app functions cause issues, keep existing local entrypoints and existing remote app functions unchanged and revert only the new workflow-specific functions.
