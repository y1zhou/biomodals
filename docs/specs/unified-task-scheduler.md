<!-- markdownlint-disable MD013 -->

# Unified task scheduler refactor plan

Status: proposed; implementation must not begin until the decision gates are
accepted.

This plan consolidates the execution and recovery findings from the API
service, reusable workflow runtime, PPIFlow fan-out, and AlphaFold3 search and
inference pipelines. It proposes a narrow `biomodals.execution` kernel rather
than a new all-purpose `TaskManager`.

The target is one place to reason about:

- fixed DAG construction and readiness;
- runtime-discovered tasks inside fixed semantic nodes;
- task attempts and their relationship to provider calls;
- safe submission, attachment, polling, cancellation, and recovery;
- cache observations and the decision to reuse or compute;
- input preparation and output collection boundaries;
- batching and run-level resource budgets.

It is not one place to encode every workload's scientific semantics or persist
every kind of state.

## Success definition

The refactor is complete when GROMACS service jobs, reusable workflows,
PPIFlow candidate fan-out, and AlphaFold3 search and inference all use the same
execution-state vocabulary and scheduling primitives without changing their
public behavior, scientific identities, durable layouts, or retry authority.

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
8. Service, workflow, and app CLI entrypoints retain their existing contracts.

## Current execution authorities

| Area | Current authority | Reusable strength | Boundary to preserve |
| --- | --- | --- | --- |
| API service | `ServiceStore`, `ModalJobSubmitter`, GROMACS coordinator and plan | API admission, per-Job locking, preclaim-before-spawn, call attachment, cancellation, user-visible operations | User/auth/config state and global service transactions stay service-owned |
| Workflow runtime | `Workflow`, `WorkflowRuntime`, `WorkflowLedger`, `RemoteCallManager` | Static DAG validation, node attempts, artifacts, per-run recovery, terminal pruning | Per-run ledger and workflow artifact contracts stay workflow-owned |
| PPIFlow | Fixed workflow nodes plus candidate manifests and bounded coordinator loops | Runtime candidate identity, partial outcomes, stage-specific fan-out | Scientific candidate schemas and joins stay PPIFlow-owned |
| AlphaFold3 | Pure search/inference plans, Modal adapters, generation claims, Volume markers | Fine-grained cache identity, multi-writer claims, per-seed reuse, batched inference, publication validation | Markers and validated publications remain scientific completion authority |
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
- Retrying a paid operation requires a later explicit invocation unless the
  provider proves that the original operation did not start.
- Scientific and cache identity excludes operational concurrency, placement,
  and resource allocation.
- Workflow node parallelism and child-call task budgets are different limits.
- AlphaFold3 raw searches, assemblies, templates, and seeds retain their
  current identities and Volume layouts.
- GROMACS continues to call the deployed app's established functions. The
  service does not rewrite the app or replace its CLI entrypoint.
- PPIFlow keeps a fixed stage DAG while candidate work fans out inside a stage.
- No paid Modal calls run in CI.

## Proposed domain model

```text
Service Job (optional API envelope)
  └── Execution Run
        ├── Node
        │     └── Task
        │           └── Task Attempt
        └── Provider Call
              └── serves one or more Task Attempts
```

The relationship between Task Attempt and Provider Call is not one-to-one:

- a cache hit completes a Task Attempt without a call;
- GROMACS usually has one Task Attempt per Modal call;
- one AlphaFold3 inference worker call may serve several seed Task Attempts;
- a Task may have several explicitly authorized Task Attempts over its
  lifetime.

### Proposed terms

**Execution Run** is one invocation of an immutable execution plan. An API Job
may own one run, while a CLI workflow or app invocation can create a run
without an API Job.

**Execution Node** is a fixed semantic DAG step. It replaces neither
`WorkflowNode` nor user-facing service stages immediately; adapters map those
concepts to it during migration.

**Task** is the smallest independently identified unit whose cache and outcome
can be reasoned about. Tasks may be discovered only when their containing Node
starts.

**Task Attempt** is one explicitly authorized effort to complete a Task. It
owns the decision to reuse, submit, recover, or report an unknown outcome.

**Provider Call** is one detached Modal function call, including its durable
call ID and observed lifecycle. A call can cover a batch of Task Attempts.

**Publication** is workload-owned durable evidence that a Task's scientific
output is complete. The kernel records the observation but does not prescribe
its file or marker format.

These terms should be added to `CONTEXT.md` only after the architectural
boundary is accepted. Existing terms should then be reconciled rather than
duplicated.

## Responsibility boundary

| Concern | Execution kernel owns | Workload or host owns |
| --- | --- | --- |
| DAG | Validation, topological readiness, terminal reachability | Nodes, dependencies, semantic labels |
| Task planning | Immutable task records, fingerprints, dependency links | Task discovery and scientific identity payload |
| Cache | `available` / `missing` / `unknown` vocabulary and scheduling policy | Validation logic, markers, manifests, content checks |
| Inputs | Calling preparation hooks and recording normalized fingerprints | Parsing, validation, staging, provider kwargs |
| Calls | Claim, submit, attach, resolve, poll, cancel, recover state machine | Function selection and provider adapter binding |
| Outputs | Calling decode/validate/publish hooks and committing outcome ordering | Schemas, scientific validation, paths, publication |
| Batching | Mapping Tasks to call batches and distributing outcomes | Batch compatibility and workload-specific limits |
| Resources | Run budget, permit accounting, later shared-lease protocol | Modal decorators and actual CPU/GPU/memory selection |
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
  ports.py                # persistence, provider, and workload protocols
  runtime.py              # composition facade after primitives stabilize
  _internal/
    scheduler.py          # ready-node/task selection
    submission.py         # paid-call lifecycle
    batching.py           # task-to-call grouping
    resources.py          # budgets and permit accounting
```

The first supported imports should be no larger than:

- `ExecutionPlan`
- `NodePlan`
- `TaskPlan`
- `TaskResult`
- `ExecutionRuntime`

Provider, store, and workload implementations are supplied explicitly by each
composition root. Do not add global registries, plugin discovery, YAML
workflows, or import-time Modal bindings.

### Driver model

The pure transition and readiness functions are shared. The API service keeps
an async driver around them; the workflow and current app entrypoints use a
sync driver. Maintaining two small drivers is preferable to infecting every
consumer with an async abstraction or running nested event loops.

### Durable execution state

Every coordinator that promises restart or recovery requires an Execution
State Repository. Centralization applies to the state model, transition rules,
and persistence operations, not to one database process or file for all
Biomodals activity. A deliberately non-resumable one-shot entrypoint may use a
transient repository, but it must not present transient state as durable.

Repository scope follows the coordinator boundary:

| Coordinator | Repository scope | Execution authority | Separate authority |
| --- | --- | --- | --- |
| API service | One long-lived `service.sqlite3` for every service-owned Job and Execution Run | Job-owned Nodes, Tasks, Task Attempts, call IDs, and observed state | Users, admission, runtime configuration, and result cache remain service-owned |
| Workflow orchestrator | The existing per-run Workflow Ledger | Workflow Nodes, fan-out Tasks, Task Attempts, calls, and recovery | Workflow artifacts and Volume synchronization remain workflow-owned |
| Nested app coordinator | None for a simple app call; a durable repository only when the app independently schedules recoverable child work | Nested Tasks, Task Attempts, batches, and child calls | AlphaFold3 claims and validated publications remain scientific authority |

An ordinary API request therefore updates only the service database. It does
not create a database for the called app. If the service starts a remote
workflow orchestrator, the service repository tracks that child coordinator
call and the workflow's existing per-run ledger tracks the internal DAG. Those
repositories describe different scheduling levels; they do not duplicate the
same Tasks.

Likewise, a simple direct app invocation does not create a SQLite file merely
because it uses a kernel plan or provider adapter. A complex app such as
AlphaFold3 needs another durable repository only if its app-owned coordinator
promises restartable tracking of its internal search and seed Tasks. If the
service directly owns those Tasks instead, they live only in the service
repository.

Provider workers do not write SQLite. A single coordinator applies
transitions and commits them using its transaction and Volume synchronization
boundary. Modal Dict remains appropriate for distributed writer claims; it is
not a replacement for the coordinator's attempt and call ledger.

The recommended implementation is a reusable SQLite repository for
single-writer durable coordinators. It should be embeddable into a host-owned
transaction. That lets service admission and initial execution state be
committed atomically, while a workflow can commit execution rows together with
its Volume-backed ledger. If a host cannot share a transaction, its adapter
must implement the same prepare/publish/finalize recovery protocol.

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
and workload publication are committed before the Task Attempt and call are
made durably terminal. If a store cannot make those changes in one
transaction, its adapter must use a recoverable prepare/publish/finalize
protocol.

### Failure modes

Nodes declare one of three workload-selected aggregation policies:

- `fail_fast`: stop admitting sibling Tasks after the first terminal failure;
- `collect_all`: allow all admitted Tasks to finish and report every outcome;
- `allow_partial`: publish an explicit partial result that downstream nodes
  must opt into.

These policies do not authorize retries. Retry authority remains a separate,
explicit Task Attempt decision.

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

Fault-injection tests must stop immediately after preclaim, spawn, attachment,
collection, decode, publication, and final commit. Each restart assertion must
prove whether the old call is recovered, the run blocks as unknown, or a new
call is safe.

## Migration plan

Each phase is reviewable and reversible. A phase deletes no predecessor until
the new path has equivalent contract and recovery coverage.

### Phase 0 — accept boundaries and freeze behavior

Deliverables:

- accept or revise ADR 0006 and the proposed glossary;
- add pure characterization tests for GROMACS graph readiness, workflow
  recovery, PPIFlow candidate outcomes, and AlphaFold3 plan identities;
- add fault-injection fixtures using fake provider calls and stores;
- record current public APIs, CLI entrypoints, run layouts, archive layouts,
  marker formats, and cache fingerprints as compatibility fixtures.

Exit gate:

- every subsequent phase can prove semantic equivalence without Modal access.

Rollback:

- documentation and test-only changes can be reverted without state migration.

### Phase 1 — repair recovery semantics in place

Deliverables:

- map availability-check exceptions to `unknown`;
- preclaim workflow remote submission and preserve submission outcome unknown;
- finalize a successful workflow call, processed result, artifacts, Task
  Attempt, and Node under one recoverable synchronization protocol;
- add explicit tests that no restart automatically duplicates uncertain work.

Exit gate:

- existing workflow APIs pass, and every injected crash has a deterministic,
  cost-safe recovery outcome.

Rollback:

- no schema replacement; keep additive state fields readable by the old path
  until the phase is accepted.

### Phase 2 — extract immutable plans and graph algorithms

Deliverables:

- add immutable `ExecutionPlan`, `NodePlan`, and dependency validation;
- extract deterministic readiness and terminal-reachability functions;
- adapt the pure GROMACS operation plan and workflow builder to the same graph
  representation while retaining their public types;
- keep dynamic work represented as a Node-owned Task factory, not mutable DAG
  vertices.

Exit gate:

- GROMACS selects the same parallel operations;
- workflows produce the same hashes, scheduled waves, and terminal pruning;
- no provider or database dependency exists in the plan module.

Rollback:

- adapters can switch back to the existing readiness functions without
  changing persisted state.

### Phase 3 — extract the provider-call state machine

Deliverables:

- move the proven preclaim/attach/recover transitions behind persistence and
  provider ports;
- use `ModalJobSubmitter` as the behavioral baseline for uncertain spawn;
- add thin async service and sync workflow drivers;
- adapt service `job_operations` and workflow `remote_calls` without initially
  replacing either schema;
- expose a common read-only execution snapshot for logs and diagnostics.

Exit gate:

- GROMACS API timelines, log call IDs, cancellation, and result archives are
  unchanged;
- workflow call recovery and cancellation are unchanged or safer;
- all provider behavior is exercised through fakes in CI.

Rollback:

- keep old coordinators behind a temporary composition switch for one phase;
  remove the switch when both adapters pass compatibility tests.

### Phase 4 — add durable Tasks, batches, and budgets

Deliverables:

- persist immutable Task plans before submission;
- add Task Attempt rows and an explicit many-to-many Task-Attempt-to-call
  link;
- move bounded batching and permit accounting into execution internals;
- make PPIFlow the first runtime-discovered Task consumer;
- represent partial candidate outcomes without making a Node successful by
  implication;
- add a shared-lease port, but defer distributed hard-limit implementation
  unless required by the decision gate.

Exit gate:

- interrupted PPIFlow stages reuse validated candidate publications and do not
  repeat uncertain calls;
- stable candidate IDs and manifests are byte- or semantic-equivalent;
- resource tests prove that one call batch consumes the intended permits.

Rollback:

- additive workflow-ledger tables can be ignored by the predecessor runtime;
  old PPIFlow runs retain the restart policy already documented in the ADR.

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
- retain explicit-invocation retry authority and current request/run identity;
- preserve the current local entrypoint and direct app-call behavior.

Exit gate:

- run IDs, request IDs, search identities, marker payloads, Volume paths, seed
  reuse, ranking order, and retrieval archives remain unchanged;
- an overlapping seed request performs only its missing seed work;
- partial search and seed failures preserve the same reusable publications;
- no automatic paid retry is introduced.

Rollback:

- adapters can return control to the existing pure pipelines because no marker
  or publication format changes in this phase.

### Phase 6 — make WorkflowRuntime a host and remove duplication

Deliverables:

- delegate graph traversal, Task scheduling, call lifecycle, availability
  policy, and budgets from `WorkflowRuntime` to the execution kernel;
- retain workflow-specific artifact materialization, Volume synchronization,
  display, and ledger implementation;
- migrate remaining durable fan-out consumers;
- remove replaced coordinator loops and stale planned documentation;
- publish the final supported execution inspection surface.

Likely deletion candidates, only after equivalence:

- GROMACS-local readiness and all-completed algorithms;
- workflow-specific paid-call transition logic;
- AlphaFold3 `_bounded_remote_outcomes` and claimed seed-batch loops;
- PPIFlow durable candidate scheduling through bare `bounded_map`;
- repeated generation-claim mechanics that have converged on the common port.

Exit gate:

- each concern has one implementation or a documented workload-specific
  reason to differ;
- `WorkflowLedger` documentation matches its real task schema;
- no compatibility switch or dead adapter remains.

Rollback:

- phase commits remain independently revertible; scientific publications and
  public contracts require no reverse migration.

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
10. `execution: add durable task attempts`
11. `ppiflow: adopt durable task fanout`
12. `alphafold3: adopt execution adapters`
13. `execution: remove duplicate schedulers`

Split a commit further whenever its predecessor and replacement cannot be
reviewed side by side. Never combine an AlphaFold3 scientific contract change
with scheduler extraction.

## Verification matrix

| Layer | Required verification |
| --- | --- |
| Pure model | Graph cycles, readiness, terminal closure, stable fingerprints, task discovery determinism |
| Call lifecycle | Fault injection around every transition, attach validation, recovery, expiry, cancellation, unknown outcomes |
| Cache | Available/missing/unknown, checker exceptions, marker validation, cache hit starts no call |
| Batching | Call-to-many mapping, per-Task result decode, partial and failed batches, deterministic ordering |
| Resources | Node parallelism independent from Task permits, batched permit accounting, no permit leak on failure |
| Service | API/OpenAPI unchanged unless intentionally versioned; admission, timeline, logs, cancel, cache staging, ZIP contents |
| Workflow | DAG hashes, scheduler waves, terminal pruning, artifact selection/materialization, resume and force behavior |
| PPIFlow | Candidate identity, manifests, attrition, joins, partial outcomes, stage restart |
| AlphaFold3 | Search/run/request identities, claims, publications, seed batching/reuse, summaries, archive hashes |
| CLI | Existing app and workflow discovery/help plus representative local-entrypoint dry tests |

CI uses fake provider and storage adapters. Remote Modal validation remains a
manual, explicitly authorized smoke test after local and CI gates pass.

## Risks and controls

| Risk | Control |
| --- | --- |
| A universal abstraction hides scientific differences | Workload-owned hooks and persistence adapters; migrate AF3 last |
| Extraction duplicates rather than replaces code | Each phase names deletion candidates and has a final deletion gate |
| Async and sync consumers distort the API | Share pure transitions; keep thin separate drivers |
| A batch obscures individual outcomes | Persist Task and Task Attempt identities plus explicit call links |
| Cache checker outage triggers expensive recomputation | Tri-state availability; only `missing` authorizes work |
| Crash after paid spawn duplicates work | Preclaim, attach protocol, explicit outcome unknown, no blind retry |
| Resource limits are mistaken for Modal decorators | Separate operational requirements from run-level permit accounting |
| One ledger becomes a cross-context bottleneck | Keep existing stores and define atomic persistence ports |
| Migration changes established app behavior | Compatibility fixtures and adapter-first rollout |

## Explicit non-goals

- a universal API Job base class;
- one database or ledger for all consumers;
- a universal scientific cache or marker schema;
- a mutable runtime DAG;
- a YAML workflow language;
- global plugin registration or autodiscovery;
- automatic retries of paid provider calls;
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
2. **Repository scope**: should repositories follow durable coordinator
   boundaries: one database for the API service, the existing per-run database
   for each workflow orchestrator, and no app-owned database unless an app
   independently schedules recoverable nested Tasks? Recommendation: yes.
3. **Repository implementation**: should the kernel provide one reusable,
   embeddable SQLite execution repository for those durable coordinators?
   Recommendation: yes.
4. **Surface stability**: should `biomodals.execution` be a supported Python
   package with a deliberately small public surface, while adapters remain
   internal until two consumers use them? Recommendation: yes.
5. **Distributed resource limits**: should the first extraction define a lease
   port but defer a Modal Dict-backed global lease implementation?
   Recommendation: yes, until a concrete cross-container limit requires it.
6. **Migration scope**: should service database changes be additive and
   migratable while in-progress workflow runs keep their documented restart
   policy? Recommendation: yes.
7. **First dynamic consumer**: should PPIFlow adopt durable Tasks before
   AlphaFold3? Recommendation: yes; it exercises fan-out with lower scientific
   and cache risk.

## Definition of ready for implementation

Implementation begins only when:

- ADR 0006 is accepted;
- the seven decision gates are resolved;
- the proposed execution terms are reconciled into `CONTEXT.md`;
- Phase 0 compatibility fixtures and fault-injection points are enumerated as
  test cases;
- the first two implementation commits have exact file and rollback scopes.
