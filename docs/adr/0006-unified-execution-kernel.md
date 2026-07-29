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

This is an incremental extraction, not a rewrite. Existing contracts must
remain usable while each consumer adopts the kernel, and duplicated
orchestration code is removed only after its replacement passes
characterization and recovery tests.
