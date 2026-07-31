# Use kernel Tasks for OligoFormer off-target scoring

Status: accepted and implemented.

## Decision

OligoFormer direct runs use the shared execution kernel and a deployment-local
remote coordinator. The coordinator owns the run-scoped SQLite ledger,
constructs the scientific DAG, records Task and Provider Call state, and admits
deployed Modal functions under the run's container limits. The local entrypoint
only stages bounded inputs, starts or restarts the coordinator, waits for the
terminal snapshot, and downloads the published archive.

The off-target DAG uses these durable scientific Task boundaries:

- one RNAplfold reference-shard Task for each deterministic full-human UTR
  shard;
- one PITA candidate Task for each selected siRNA; and
- one TargetScan Task for each candidate-batch by reference-shard tile.

PITA and TargetScan Tasks are independent and may run concurrently when their
dependencies are ready. The kernel applies the configured run-wide Provider
Call limit and divides `off_target_process_slots` between their Nodes. RNAplfold
uses its own configured Node ceiling. The efficacy Node is the only GPU
Provider Call.

The provider functions retain the workload-specific inner algorithms. A PITA
candidate container prepares its target-discovery shards, processes them with a
bounded local pool, consolidates their rows, scores the row shards with another
bounded local pool, and publishes one candidate result. A TargetScan tile
container prepares its reusable reference/candidate inputs, scores context
shards with a bounded local pool, and publishes one tile result. These local
pools are implementation details, not another durable scheduler.

There is no OligoFormer-owned Modal Queue, nested scientific `.remote()`,
`.map()`, or `.spawn()` call, branch coordinator, or same-Run outer retry loop.
A conclusive Provider Call failure fails its Tasks. Recovery that retries work
uses a Successor Execution Run; valid scientific publications are discovered
from their existing markers and reused.

The rejected alternatives are:

- a dynamic Modal Queue of fine-grained row jobs, because it duplicates the
  kernel's durable queue and call ownership;
- nested PITA and TargetScan branch functions, because their child calls are
  not owned directly by the execution ledger; and
- one Provider Call per internal row shard, because the extra Modal startup and
  ledger cost is not justified by the current workloads.

`top_n=-1` still means score every efficacy candidate. The evidence-planning
Provider Call constructs the finite candidate and tile manifests after efficacy
is available. Those Tasks are persisted before admission, so the configured
limits bound active containers even when the complete Task set is large.

## Publication and cache authority

The execution ledger records orchestration facts, not scientific truth.
OligoFormer's existing files and validated markers remain authoritative for
cache reuse and Task completion:

- RNA-FM weights and converted full-human references remain immutable assets in
  the shared model Volume;
- derived RNAplfold files remain in the OligoFormer output Volume's
  `reference-cache/` tree;
- efficacy, off-target evidence, final tables, and the downloadable archive
  remain in the output Volume; and
- compact result envelopes carry plans, paths, counts, or publication metadata,
  never large result files.

The converted UTR and ORF content digests are part of scientific identity.
Declared source URLs alone are not sufficient identity for persisted results.
An all-human plan pins the reference identity, and providers reject a changed
model/reference publication rather than attaching new-reference outputs to the
old identity.

The RNAplfold cache is split into deterministic shards. Each shard marker binds
its reference identity and input digest to the expected output names, sizes, and
publication-time SHA-256 values. A top marker binds the exact set of shard
markers. Routine readiness checks manifests, small inputs, existence, and
recorded sizes without re-hashing the full human cache; the recorded output
hashes remain available for explicit deep validation. Missing shards alone are
admitted on a Successor Run.

Cross-Run publication exclusion remains workload-owned. The coordinator claims
the content-keyed reference cache and per-stem evidence publication before it
submits writers. The existing generation guard still protects mutable model,
efficacy, and final-table cache publications. These claims and guards do not
become execution state or generic kernel cache behavior.

PITA builds its extended UTR and UTR STAB reference data once per evidence stem.
That data remains scoped to the evidence publication; moving it into a global
reference cache would require a separate immutable publication and retention
policy and is not part of this refactor.

Only after every required PITA candidate and TargetScan tile succeeds does the
evidence publisher merge `pita.tab` and `targetscan.tab` and write
`off_target.done`. The marker records schema, row count, size, digest, and
scientific identity. Final-table construction consumes only this published
compact evidence. It may then remove bulky per-tile intermediates while
preserving logs, efficacy outputs, reusable references, compact evidence, final
tables, and their manifests.

Off-target scoring fails closed. Missing PITA or TargetScan Tasks cannot produce
a successful final table, because incomplete evidence can make a candidate look
safer than it is.

## Scientific semantics retained

Final PITA and TargetScan consolidation uses Polars lazy scans and streaming CSV
sinks. Numeric sort keys are explicit while upstream-shaped columns are
retained. Filtering groups merged evidence to one summary row per candidate, so
`top_n=-1` does not reload the complete human evidence table as Python objects.

PITA and TargetScan evidence are interpreted independently. No hit from one
tool means that tool found no evidence for the evaluated siRNA; it does not mean
the candidate was skipped. `off_target_filter == -5` remains reserved for
candidates outside the evaluated `top_n` set. Final `filter` values count enabled
filter failures, so sentinel values cannot cancel functionality or toxicity
failures.

For positive `top_n`, Biomodals ranks by efficacy, keeps each selected row's
actual siRNA sequence, and names the off-target record by the candidate's
original row identity. The off-target cache salt changes if this binding
semantics changes.

The TargetScan reducer keeps the existing Biomodals correction to the pinned
upstream shell pipeline. Upstream projects each row to four fields and then
tests missing field 28 as zero, discarding qualifying `7mer-1a`, `7mer-m8`, and
`8mer-1a` sites. Biomodals tests the projected score field and preserves sites
that meet the configured thresholds. Focused tests cover every site type and
threshold boundary; this correction remains part of the off-target cache
identity.

Efficacy and final-table markers remain table manifests, not existence flags.
They record schema, row count, size, SHA-256, output identities, and the relevant
semantics salt. Efficacy identity contains GPU inputs and semantics; evidence
identity adds selected candidates and reference content; final-table identity
adds filtering settings. Threshold variants can therefore reuse valid efficacy
and compact evidence without conflating scientific outputs.
