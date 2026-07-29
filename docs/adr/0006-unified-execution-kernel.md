# Centralize execution mechanics without centralizing workload state

Status: proposed.

Biomodals should introduce a provider-aware execution kernel under
`biomodals.execution` for DAG traversal, task readiness, paid-call attachment
and recovery, cache-observation policy, batching, and resource budgets. The
kernel should embed its execution tables into host-owned databases without
replacing `ServiceStore` domain state, workflow artifact records, or
AlphaFold3's claims and publications. A service Job remains the user-facing
admission and result envelope; workflows retain the physical per-run ledger;
workload publications remain authoritative for scientific cache completion.

The authority boundary was accepted on 2026-07-29. “Without centralizing
workload state” does not mean that execution state may remain ephemeral. The
kernel governs the durable state model and atomic transition contract for
runs, nodes, tasks, attempts, and provider calls. Separate repository
instances implement that contract so an API service, a per-run workflow, and
an app coordinator do not depend on one shared database.

Repository scope follows the coordinator boundary, not each API request,
application call, or Modal function. The API service uses one long-lived
database for all service-owned execution runs. A workflow keeps its existing
per-run ledger because its remote workflow orchestrator is a separate durable
coordinator. An app needs another repository only if it independently owns
nested, recoverable scheduling; simple app calls do not create databases.

Keeping a workflow's physical `ledger.sqlite3` file does not preserve a
separate workflow implementation of generic execution state. The shared
execution repository should own run, node, task, attempt, and provider-call
tables and transitions inside that file. Workflow code should retain only its
artifact records, run-directory lifecycle, and Modal Volume synchronization.
The current `WorkflowLedger` may serve as a short-lived migration facade
between incremental commits, then be removed.

The repository scope and ledger decomposition were accepted on 2026-07-29.
The accepted end state removes the migration facade: the workflow runtime
composes the shared execution repository and a narrow Workflow Artifact Store
over the same connection. The physical `ledger.sqlite3` file remains.

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
- A small execution kernel with one embeddable SQLite implementation and
  explicit provider and workload adapters reuses the common algorithms while
  allowing each host to preserve its transaction and durability model.

## Consequences

The common hierarchy is an execution run containing fixed semantic nodes,
runtime-discovered tasks, task attempts, and provider calls. One provider call
may serve several tasks, and a cache hit may complete a task without a provider
call. Workloads continue to define scientific identity, cache validation,
input and output contracts, function arguments, resource requirements, and
publication rules. The kernel determines when those hooks run and how their
observations affect scheduling.

An execution repository is authoritative for scheduling facts such as the
immutable plan, readiness, attempts, submission tokens, attached call IDs,
observed provider state, and timestamps. It records that a workload
publication was validated, but the publication's marker, manifest, or
workload-specific validator remains authoritative for whether scientific
output is reusable.

This is an incremental extraction, not a rewrite. Internal types, tables, and
imports may change directly while each consumer adopts the kernel. Scientific
identities, publications, cost-safety rules, and documented user behavior
remain regression constraints unless a separate decision deliberately changes
them. Duplicated orchestration code is removed after its replacement passes
characterization and recovery tests.
