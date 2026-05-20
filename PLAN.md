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

workflow = Workflow("ppiflow")

designs = workflow.add_node(
    PPIFlowDesignNode(stage=1),
    id="designs",
)

packed = workflow.add_node(
    LigandMPNNNode(model_type="soluble_mpnn"),
    id="packed",
    inputs={
        "structures": designs.outputs(
            kind="structures",
            pattern="**/*.pdb",
        ),
    },
)

scores = workflow.add_node(
    AF3ScoreNode(max_batches=10),
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
- Dependencies are inferred from `NodeOutputRef`s used in node inputs.
- `depends_on=[ranked]` adds control edges without artifact passage.
- Artifact selectors can filter by producing node, artifact kind, file role, glob pattern, and metadata.
- The DAG must be acyclic.
- Selector resolution happens at runtime after upstream artifacts exist.

## Task Plan

### Task 1: Add Shared Schema Package

**Files:**

- Create: `src/biomodals/schema/__init__.py`

- Create: `src/biomodals/schema/storage.py`

- Create: `src/biomodals/schema/app.py`

- Modify: `src/biomodals/app/config.py`

- Create: `src/biomodals/schema/workflow.py`

- Modify: `pyproject.toml`

- Create: `tests/schema/test_workflow_schemas.py`

- [x] **Step 1: Write schema tests**

Create tests covering:

```python
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    StorageKind,
    VolumePath,
    WorkflowArtifact,
)


def test_inline_bytes_round_trip():
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="summary",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=b"summary text",
                    filename="summary.txt",
                    media_type="text/plain",
                ),
            )
        ],
    )
    dumped = result.model_dump()
    loaded = AppRunResult.model_validate(dumped)
    assert loaded.outputs[0].storage.kind == StorageKind.INLINE_BYTES
    assert loaded.outputs[0].storage.filename == "summary.txt"


def test_workflow_artifact_is_volume_backed():
    artifact = WorkflowArtifact(
        artifact_id="art-packed",
        producing_node_id="packed",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(
            volume_name="Workflow-outputs",
            path="ppiflow/run-1/artifacts/art-packed",
        )
    )
    assert artifact.storage.path == "ppiflow/run-1/artifacts/art-packed"
```

- [x] **Step 2: Run tests to confirm failure**

Run:

```bash
rtk uv run pytest tests/schema/test_workflow_schemas.py -q
```

Expected: failure because `biomodals.schema` does not exist.

- [x] **Step 3: Implement schemas**

Implement the schema sketch above. Move pure Pydantic `AppConfig` fields and validators into `schema/app.py`, export it from `biomodals.schema`, and keep `biomodals.app.config.AppConfig` as a compatibility import. Move Modal helpers such as `get_out_volume()` out of the schema model. Include `NodeExecutionPolicy`, `NodePlacement`, `NodeStatus`, `RunStatus`, `ArtifactFile`, `ArtifactSelector`, and `ControlEdge` in `workflow.py`. Add `pytest` to the `dev` dependency group in `pyproject.toml`.

- [x] **Step 4: Run schema tests**

Run:

```bash
rtk uv run pytest tests/schema/test_workflow_schemas.py -q
```

Expected: all tests pass.

### Task 2: Add Artifact Materialization

**Files:**

- Create: `src/biomodals/workflow/core/artifacts.py`

- Create: `tests/workflow/test_artifacts.py`

- [x] **Step 1: Write materialization tests**

Cover these cases:

- `InlineBytes(filename="out.txt")` accepts UTF-8 text bytes and writes `raw_outputs/out.txt`.

- Binary archives are rejected as `InlineBytes`; archive outputs use `VolumePath` storage and record archive metadata on the `AppOutput` or resulting `WorkflowArtifact`.

- `VolumePath` app outputs are copied or referenced according to an explicit materialization mode.

- Materialization returns `WorkflowArtifact` objects with volume-backed storage.

- [x] **Step 2: Implement materialization helpers**

Add these functions:

```python
def materialize_app_run_result(
    *,
    result: AppRunResult,
    workflow_volume_name: str,
    attempt_dir: Path,
    artifact_dir: Path,
    producing_node_id: str,
) -> list[WorkflowArtifact]:
    """Write app outputs into the workflow volume and return artifact manifests."""
    raise NotImplementedError("Task 2 implements artifact materialization")
```

Use existing archive helper behavior from `src/biomodals/app/helper/shell.py` and extraction behavior from `ppiflow_workflow.py` as references.

- [x] **Step 3: Run artifact tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_artifacts.py -q
```

Expected: all tests pass.

### Task 3: Add Durable Ledger Utilities

**Files:**

- Create: `src/biomodals/workflow/core/ledger.py`

- Create: `tests/workflow/test_ledger.py`

- [x] **Step 1: Write ledger tests**

Cover:

- Creating a run ledger creates `ledger.sqlite3` and inserts `runs`.

- Marking a node running upserts `nodes` and inserts `attempts`.

- Recording a remote submission inserts `remote_calls` before waiting for the Modal function call.

- Restart recovery reattaches to existing `remote_calls` with `modal.FunctionCall.from_id(...)` before scheduling replacement work.

- Recording artifacts inserts `artifacts`, `artifact_files`, and `node_outputs`.

- A completed node is detected from `nodes.status`, `node_outputs`, and matching artifact/file rows.

- [x] **Step 2: Implement ledger**

Implement a `WorkflowLedger` class that accepts a mounted volume root path, creates or opens `<workflow-name>/<run-id>/ledger.sqlite3`, and never imports app modules. The orchestrator owns write access. Tests should assert the documented tables exist and that rows are readable with raw SQL for manual debugging. Ledger updates mutate SQLite rows directly; do not preserve obsolete Pydantic status-copy patterns such as `model_copy(update=...)` for ledger state.

Required methods:

```python
create_run(run: WorkflowRun) -> WorkflowRun
load_run(workflow_name: str, run_id: str) -> WorkflowRun
mark_node_pending(node_id: str) -> NodeStatusRecord
mark_node_running(node_id: str, attempt_id: str) -> NodeStatusRecord
mark_node_succeeded(node_id: str, artifact_ids: list[str]) -> NodeStatusRecord
mark_node_failed(node_id: str, error: str) -> NodeStatusRecord
record_attempt_started(node_id: str, attempt_id: str) -> AttemptRecord
record_remote_call(call_id: str, node_id: str, attempt_id: str, function_name: str, call_kind: str) -> None
record_task(task_id: str, node_id: str, attempt_id: str, status: str, ...) -> None
record_app_result(node_id: str, attempt_id: str, result: AppRunResult) -> Path
record_artifacts(artifacts: list[WorkflowArtifact]) -> None
node_is_complete(node_id: str) -> bool
```

- [x] **Step 3: Run ledger tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_ledger.py -q
```

Expected: all tests pass.

### Task 4: Add Workflow Builder and DAG Validation

**Files:**

- Create: `src/biomodals/workflow/core/builder.py`

- Create: `src/biomodals/workflow/core/nodes.py`

- Modify: `src/biomodals/workflow/__init__.py`

- Create: `tests/workflow/test_builder.py`

- [x] **Step 1: Write builder tests**

Cover:

- Adding two nodes with a selector creates one data dependency.

- `depends_on` creates a control edge.

- Duplicate node ids raise `ValueError`.

- Cycles raise `ValueError`.

- Two nodes depending on the same upstream node are both ready after the upstream succeeds.

- [x] **Step 2: Implement node contracts**

Define:

```python
class WorkflowNode(Protocol):
    id: str
    execution_policy: NodeExecutionPolicy
    placement: NodePlacement

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute node implementation and return app-compatible outputs."""
        raise NotImplementedError
```

Define `AppBackedNode` and `WorkflowNativeNode` base classes for common behavior, but keep app-specific logic out of the runtime.

`AppBackedNode.load_app_function()` returns `modal.Function`. `AppBackedNode.app_function` and any node function resolver should be typed as `modal.Function`, not `Any` or a generic `Callable`; app-backed nodes do not invoke ordinary Python callables except in unit-test fakes that model the Modal boundary.

- [x] **Step 3: Implement builder**

Expose:

```python
Workflow(name: str)
Workflow.add_node(node, *, id: str, inputs=None, depends_on=None)
NodeHandle.outputs(kind: ArtifactKind, pattern: str | None = None, role: str | None = None)
Workflow.validate()
Workflow.ready_nodes(completed_node_ids: set[str])
```

- [x] **Step 4: Run builder tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_builder.py -q
```

Expected: all tests pass.

### Task 5: Add Runtime Scheduler

**Files:**

- Create: `src/biomodals/workflow/core/runtime.py`

- Create: `tests/workflow/test_runtime.py`

- [x] **Step 1: Write runtime tests with fake nodes**

Cover:

- Completed nodes are skipped.

- Independent ready nodes run in parallel or are submitted in the same scheduler wave.

- A failed required node prevents downstream nodes from running.

- `RERUN` reruns incomplete nodes.

- `RESUME` calls node code with durable cache paths available in the context.

- [x] **Step 2: Implement runtime**

Implement scheduler waves:

```text
load ledger
find completed nodes
find ready nodes
run ready nodes by placement
materialize outputs
record artifacts
repeat until all nodes complete or a failure stops the run
```

For local unit tests, use fake node implementations and a temporary directory as the ledger root.

- [x] **Step 3: Run runtime tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_runtime.py -q
```

Expected: all tests pass.

### Task 6: Add Modal Orchestrator

**Files:**

- Create: `src/biomodals/workflow/core/orchestrator.py`

- Modify: `src/biomodals/workflow/__init__.py`

- Create: `tests/workflow/test_orchestrator.py`

- [x] **Step 1: Create workflow AppConfig**

Use the app-development conventions where they apply to Modal code:

```python
CONF = AppConfig(
    tags={"group": "workflow"},
    name="WorkflowOrchestrator",
    package_name="biomodals-workflow-orchestrator",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)
```

- [x] **Step 2: Add remote orchestrator class**

Add:

```python
@app.cls(
    cpu=(1.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
class WorkflowOrchestrator:
    @modal.enter()
    def enter(self) -> None:
        OUT_VOLUME.reload()

    @modal.method()
    def run(
        self,
        workflow_name: str,
        run_id: str,
        workflow_definition: Workflow,
        force: bool = False,
    ) -> AppRunResult:
        runtime = WorkflowRuntime.from_definition(
            workflow_name=workflow_name,
            workflow_definition=workflow_definition,
            volume_root=Path(CONF.output_volume_mountpoint),
            workflow_volume=OUT_VOLUME,
        )
        return runtime.run(run_id=run_id, force=force)

    @modal.exit()
    def exit(self) -> None:
        OUT_VOLUME.commit()
```

The orchestrator class should live in `orchestrator.py`; do not create a separate `orchestrator_app.py`. It should instantiate `WorkflowRuntime`, run it against the mounted output volume, commit volume writes after state changes, and return an `AppRunResult` or final workflow artifact manifest list.

- [x] **Step 3: Add local submission helper**

Keep this helper CLI-facing and minimal: stage local inputs, submit the remote orchestrator, and optionally download final outputs. It should not become a second scheduler. Expose it under the `biomodals workflow ...` namespace when it becomes user-facing.

- [x] **Step 4: Write mocked orchestrator boundary tests**

Use pytest `monkeypatch` to replace `WorkflowRuntime.from_definition` with a fake runtime that records `workflow_name`, `workflow_definition`, `volume_root`, `run_id`, and `force`. Call the class method body directly and assert it returns the fake `AppRunResult`. Do not call `.remote()`, `.spawn()`, or any live Modal API.

- [x] **Step 5: Run orchestrator tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_orchestrator.py -q
```

Expected: mocked orchestrator tests pass without contacting Modal.

- [x] **Step 6: Run discovery smoke tests**

Run:

```bash
rtk uv run biomodals app list
rtk uv run biomodals workflow list
rtk uv run biomodals workflow help ppiflow
```

Expected: app discovery works, workflow discovery exposes executable workflow scripts, and runtime-only orchestrator internals stay under `biomodals.workflow.core` and are not listed as user workflows. Temporary deprecated aliases (`biomodals list`, `biomodals help`, and `biomodals run`) may remain for app commands during one transition period.

### Task 7: Add Worker Pool Helpers

**Files:**

- Create: `src/biomodals/workflow/core/workers.py`

- Create: `tests/workflow/test_workers.py`

- [x] **Step 1: Write worker-pool tests**

Cover queue naming, deterministic task ids, worker count calculation, and completion status aggregation using fake queue objects.

- [x] **Step 2: Implement worker-pool helper API**

Expose:

```python
def build_worker_pool_name(workflow_name: str, run_id: str, node_id: str) -> str:
    parts = [workflow_name, run_id, node_id, "workers"]
    return "-".join(sanitize_filename(part) for part in parts)

def bounded_worker_count(max_workers: int, task_count: int) -> int:
    if task_count < 1:
        return 0
    return max(1, min(max_workers, task_count))
```

Keep the Modal `Queue` and `FunctionCall.gather` calls in thin integration functions so pure unit tests can exercise most behavior.

- [x] **Step 3: Run worker tests**

Run:

```bash
rtk uv run pytest tests/workflow/test_workers.py -q
```

Expected: all tests pass.

### Task 8: Add First Workflow-Compatible App Function

**Files:**

- Modify the first narrow app: `src/biomodals/app/fold/flowpacker_app.py`

- Create `tests/app/test_flowpacker_workflow_contract.py`

- Update `examples/app/` only if user-facing invocation changes

- [x] **Step 1: Read app development standards**

Read:

```text
.agents/skills/biomodals-app-development/SKILL.md
.agents/skills/biomodals-app-development/references/app-development.md
docs/agents/app-development.md
```

- [x] **Step 2: Add a workflow-compatible remote app function**

The function should:

- Accept workflow-friendly primitive/Pydantic inputs.

- Call the same core behavior used by the existing local entrypoint or existing remote run function.

- Return `AppRunResult`.

- Preserve the existing local entrypoint behavior and return value.

- Inline trivial result construction into `run_flowpacker_workflow` instead of adding a helper that only wraps return values.

- [x] **Step 3: Add a focused app contract test**

Use pytest to call the workflow contract helper with fake inputs and monkeypatched app-function dependencies. Assert the returned `AppRunResult` contains expected `AppOutput` names, kinds, and storage metadata. Do not call `.remote()`, `.spawn()`, `modal.Function.from_name(...)`, or deployed Modal apps.

- [x] **Step 4: Run the focused app test**

Run:

```bash
rtk uv run pytest tests/app/test_flowpacker_workflow_contract.py -q
```

Expected: the app contract test passes without contacting Modal.

- [x] **Step 5: Smoke test app discovery**

Run:

```bash
rtk uv run biomodals app list
rtk uv run biomodals app help flowpacker
rtk prek run --files src/biomodals/app/fold/flowpacker_app.py tests/app/test_flowpacker_workflow_contract.py src/biomodals/schema/app.py src/biomodals/schema/storage.py src/biomodals/schema/workflow.py
```

Expected: discovery and pre-commit checks pass.

### Task 9: Documentation and Verification

**Files:**

- Modify: `docs/agents/app-development.md` if workflow-compatible app function standards become app-development standards.

- Create: `docs/agents/workflow-development.md`

- Modify: `AGENTS.md` only if repo-wide agent instructions need a pointer to workflow-development docs.

- [x] **Step 1: Document workflow-development rules**

Include:

- Workflow vocabulary.

- Schema boundaries.

- Node execution policies.

- Node placement.

- Ledger layout.

- Modal preemption requirements.

- Worker pool and barriered fan-out behavior.

- App local entrypoint versus workflow-compatible app function split.

- CLI namespace split between `biomodals app ...` and `biomodals workflow ...`.

- [x] **Step 2: Run verification**

Run:

```bash
rtk uv run pytest tests -q
rtk prek run --files PLAN.md CONTEXT.md src/biomodals/schema/__init__.py src/biomodals/schema/storage.py src/biomodals/schema/app.py src/biomodals/schema/workflow.py src/biomodals/workflow/__init__.py
rtk uv run biomodals app list
rtk uv run biomodals workflow list
```

Expected: unit tests pass, pre-commit passes for changed files, and CLI app/workflow discovery still works.

## Deferred Decisions

- External YAML or JSON workflow authoring.
- Arbitrary runtime DAG mutation.
- Streaming artifacts from one node to another before the upstream node completes.
- Cross-run content-addressed caching.
- A global registry of workflow-compatible app functions.
- A stable public plugin API for third-party apps.

## Rollback Notes

If the runtime extraction stalls, shared schemas and docs can remain because they do not alter deployed app behavior. If workflow-compatible app functions cause issues, keep existing local entrypoints and existing remote app functions unchanged and revert only the new workflow-specific functions.
