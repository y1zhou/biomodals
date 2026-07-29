# Centralize execution mechanics without centralizing workload state

Status: proposed.

Biomodals should introduce a provider-aware execution kernel under
`biomodals.execution` for DAG traversal, task readiness, paid-call attachment
and recovery, cache-observation policy, batching, and resource budgets. The
kernel should use persistence and workload adapters instead of replacing
`ServiceStore`, `WorkflowLedger`, or AlphaFold3's claim and publication
records. A service Job remains the user-facing admission and result envelope;
workflows retain per-run ledgers; workload publications remain authoritative
for scientific cache completion.

The authority boundary was accepted on 2026-07-29. “Without centralizing
workload state” does not mean that execution state may remain ephemeral. The
kernel governs the durable state model and atomic transition contract for
runs, nodes, tasks, attempts, and provider calls. Separate repository
instances implement that contract so an API service, a per-run workflow, and
an app coordinator do not depend on one shared database. The exact reusable
repository implementation remains a pending design decision.

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
The current `WorkflowLedger` may serve as a compatibility facade during
migration, then shrink into a workflow run store or be removed.

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
- A small execution kernel with explicit adapters reuses the common algorithms
  while allowing each existing store to preserve its transaction and
  durability model.

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

This is an incremental extraction, not a rewrite. Existing contracts must
remain usable while each consumer adopts the kernel, and duplicated
orchestration code is removed only after its replacement passes
characterization and recovery tests.
