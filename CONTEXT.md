<!-- markdownlint-disable MD013 -->

# Biomodals

Biomodals runs bioinformatics tools as Modal apps and composes them into reusable computational workflows.

## Language

**Workflow Artifact**:
A durable record of data produced or consumed by a workflow step, including its data category, storage location, and metadata needed by downstream steps.
_Avoid_: raw app output, untyped file path, loose tarball

**Workflow Node**:
A semantic step in a workflow DAG that consumes workflow artifacts and produces workflow artifacts.
_Avoid_: Modal function, app function

**App**:
A deployed Modal app that owns tool runtime, images, volumes, and exported app functions.
_Avoid_: workflow node, app node

**App Function**:
A callable Modal remote function exposed by a Biomodals app or another Modal app and invoked by a workflow node.
_Avoid_: workflow node

**Local Entrypoint**:
A CLI-facing Modal entrypoint that parses local user inputs, submits app functions, downloads or reports outputs, and returns no workflow contract.
_Avoid_: workflow entrypoint

**Workflow-Compatible App Function**:
An app function with standardized workflow input and output schemas suitable for app-backed workflow nodes.
_Avoid_: local entrypoint, submit function

**Shared Schema**:
A stable Pydantic contract used across Biomodals packages without depending on app or workflow implementation modules.
_Avoid_: app config, internal model

**App-Backed Node**:
A workflow node implemented by calling one or more app functions.
_Avoid_: app node, runner node

**Workflow-Native Node**:
A workflow node implemented directly in workflow code for orchestration, transformation, selection, ranking, packaging, or reporting.
_Avoid_: runtime node, orchestrator node

**Workflow Builder**:
A Python interface for declaring workflow nodes, dependencies, artifact selectors, and execution settings before a workflow run.
_Avoid_: workflow YAML, scheduler config

**Artifact Selector**:
A named input reference that selects upstream workflow artifacts by kind, file role, path pattern, metadata, or producing node.
_Avoid_: raw input path, wildcard-only dependency

**Control Edge**:
A dependency between workflow nodes that enforces execution order without passing workflow artifacts.
_Avoid_: dummy artifact

**Dynamic Task Fan-Out**:
A workflow node execution pattern where the DAG node is fixed but the number of per-input tasks is determined from upstream artifacts at runtime.
_Avoid_: dynamic DAG

**Worker Pool**:
A fixed-size set of remote workers spawned by one workflow node to process that node's runtime task queue.
_Avoid_: server pool, runner server

**Workflow Runtime**:
The reusable library that validates a workflow DAG, schedules workflow nodes, tracks durable run state, and materializes workflow artifacts.
_Avoid_: engine

**Workflow Orchestrator**:
A remote Modal function that hosts the workflow runtime for one workflow run.
_Avoid_: workflow node, runner

**Node Placement**:
The execution location for a workflow node, either inline in the workflow orchestrator or in a separate remote Modal function.
_Avoid_: runner location, execution site

**Node Execution Policy**:
The restart and recovery contract for an incomplete workflow node when Modal interrupts or retries the node.
_Avoid_: runner tag, retry hint

**Durable Node Cache**:
Volume-backed intermediate state that lets a long-running workflow node resume or safely skip completed work after restart.
_Avoid_: temporary scratch, local cache

## Relationships

- A **Workflow Builder** defines a workflow DAG in Python code.
- A **Workflow Builder** connects nodes through named **Artifact Selectors** and optional **Control Edges**.
- A **Workflow Node** declares a **Node Execution Policy**.
- A **Workflow Node** declares **Node Placement**.
- A **Workflow Node** may use **Dynamic Task Fan-Out** without changing the workflow DAG shape.
- A **Workflow Node** may use a **Worker Pool** to process dynamically fanned-out tasks.
- A **Workflow Runtime** schedules **Workflow Nodes** and does not contain tool-specific biological logic.
- A **Workflow Runtime** may run independent ready nodes in parallel when all of each node's dependencies are satisfied.
- A **Workflow Orchestrator** runs the **Workflow Runtime** remotely on Modal.
- A workflow step produces zero or more **Workflow Artifacts**.
- A workflow step consumes zero or more **Workflow Artifacts** from upstream steps.
- A **Workflow Artifact** may reference inline bytes or files stored in a remote Modal volume.
- Byte-backed **Workflow Artifacts** are normalized into volume-backed artifacts after step completion when the data needs to be preserved, inspected, resumed, or consumed by downstream steps.
- A **Workflow Node** may invoke one or more **App Functions** to fulfill one semantic step.
- An **App** may expose many **App Functions**.
- A **Local Entrypoint** remains CLI-only and should not be called by the workflow orchestrator.
- A **Workflow-Compatible App Function** may reuse behavior from a **Local Entrypoint**, but exposes a remote app function contract for workflows.
- A **Shared Schema** may be imported by app and workflow modules, but it must not import app or workflow modules.
- App-specific configuration models remain with their app until they become stable cross-module contracts.
- An **App-Backed Node** calls one or more **App Functions** and processes their outputs into workflow artifacts.
- A **Workflow-Native Node** performs lightweight workflow logic without calling a bioinformatics app.
- A long-running **Workflow Node** must use a **Durable Node Cache** so interruption and restart do not corrupt outputs or repeat unsafe work.
- A short-running **Workflow Node** may choose a rerun-on-restart policy when recomputation is cheaper than durable checkpointing.
- A lightweight **Workflow-Native Node** may run inline in the **Workflow Orchestrator** when remote execution overhead is not justified.
- A long-running or failure-isolated **Workflow Node** should run as a separate remote Modal function.
- Every **Workflow Node** checks durable run state before execution and skips work when completed artifacts already exist.
- A **Workflow** may compose **App Functions** from any Modal app when those functions can be described by node input and output contracts.

## Example dialogue

> **Dev:** "Should the LigandMPNN app pass its tarball directly to FlowPacker?"
> **Domain expert:** "No — the workflow should record a **Workflow Artifact** for the LigandMPNN result, materialize it into the run volume, and let FlowPacker consume the artifact's selected structure files."
>
> **Dev:** "Is AF3Score four workflow steps because it has lock, prepare, run, and postprocess functions?"
> **Domain expert:** "No — those are **App Functions** inside one **Workflow Node** when the workflow cares about one scoring step."
>
> **Dev:** "Should users write workflow YAML first?"
> **Domain expert:** "No — complex workflows should start with the **Workflow Builder** so node contracts and artifact selectors stay explicit in Python."
>
> **Dev:** "Can every interrupted node just run again?"
> **Domain expert:** "Only short-running nodes should default to rerun. Long-running nodes need a **Durable Node Cache** and a **Node Execution Policy** that makes restart behavior explicit."
>
> **Dev:** "Does the workflow orchestrator spawn app-backed nodes as runners?"
> **Domain expert:** "No — the **Workflow Orchestrator** runs the **Workflow Runtime**, the runtime schedules **Workflow Nodes**, and an **App-Backed Node** calls app functions as its implementation."
>
> **Dev:** "Should every node be its own Modal function?"
> **Domain expert:** "No — **Node Placement** determines whether the node runs inline for lightweight workflow logic or remotely for long-running and failure-isolated work."
>
> **Dev:** "Can the workflow call an app's local entrypoint?"
> **Domain expert:** "No — **Local Entrypoints** stay CLI-only. Workflows call **Workflow-Compatible App Functions** that may be derived from the same behavior."
>
> **Dev:** "How does one node consume only PDB files from an upstream design step?"
> **Domain expert:** "Use an **Artifact Selector** that names the upstream node, selects structure artifacts, and filters files by role or pattern."
>
> **Dev:** "Does one PPIFlow output structure create a new node in the DAG?"
> **Domain expert:** "No — the downstream **Workflow Node** uses **Dynamic Task Fan-Out** to create per-structure tasks while the DAG shape stays fixed."
>
> **Dev:** "If two scoring nodes depend on the same design node, should they wait for each other?"
> **Domain expert:** "No — once their shared upstream dependency is complete, the **Workflow Runtime** can schedule both nodes in parallel."

## Flagged ambiguities

- "artifact" can mean either inline bytes or remote files. Resolved: a **Workflow Artifact** may hold either storage form, but the **Workflow Runtime** should normalize byte outputs into volume-backed state when the output crosses a workflow step boundary.
- "step" can mean either a semantic workflow operation or one callable remote function. Resolved: use **Workflow Node** for the semantic DAG unit and **App Function** for a Modal remote callable.
- "app node" can mean either a Modal deployment unit or a DAG vertex backed by that app. Resolved: use **App** for the deployment unit and **App-Backed Node** for the DAG vertex.
- "workflow entrypoint" can be confused with Modal's local entrypoint. Resolved: use **Workflow-Compatible App Function** for reusable remote app functions and **Local Entrypoint** for CLI wrappers.
- "dynamic workflow" can mean changing the DAG at runtime or changing only the task count. Resolved: first-version workflows use static DAGs with **Dynamic Task Fan-Out** only.
