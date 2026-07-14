# Implement planned workflow runtime capabilities through existing boundaries

The `[planned]` glossary terms in `CONTEXT.md` are workflow runtime capabilities,
not new workflow authoring concepts. Implement them through the existing runtime
boundaries: fixed workflow DAGs, `WorkflowLedger`, `RemoteCallManager`, and
`NodeRunContext.cache_dir`.

## Dynamic task fan-out

Keep the DAG static. Runtime task fan-out belongs inside one workflow node when
the semantic stage owns a runtime-discovered candidate or input set; static DAG
fan-out remains the default when cardinality is known during DAG construction.
Add durable per-task ledger rows only when a node needs retry or skip state, with
deterministic task ids derived from the node id, upstream artifact identity, and
normalized candidate or input id. PPIFlow candidate-wide stage coordinators are
the first implementation target.

## Stale node attempts

Reconcile stale attempts before submitting replacement work. Orchestrator-placed
attempts can follow their declared `RERUN` or `RESUME` policy because no
independent work survives the orchestrator. Remote attempts must first reattach
by the recorded Modal function-call id, process any recovered result through the
node contract, and finalize under the existing volume-sync lock; if no
recoverable call id exists, block instead of duplicating remote work.

## Durable node cache

Keep durable node cache as the node-owned directory already exposed through
`NodeRunContext.cache_dir`, under `nodes/<node-id>/cache`. The runtime creates
and preserves it for `RESUME` nodes, clears it on `force` or `RERUN` reset, and
leaves checkpoint file formats node-local until at least two workflows need the
same cache contract.

## Rejected options

Do not add mutable dynamic DAG construction, a shared cache schema, or blind
remote-call replacement in the first implementation. Those choices would make
recovery harder to reason about before the runtime has multiple consumers that
need the extra surface area.
