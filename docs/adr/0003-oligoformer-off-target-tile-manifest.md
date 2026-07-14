# Use bounded deterministic tiles for OligoFormer off-target scoring

Status: accepted, implemented for the current OligoFormer app.

OligoFormer off-target scoring represents parallel work as deterministic tiles,
with bounded queues only where measured shard skew requires work stealing.
TargetScan tiles are candidate-batch by reference-shard preparation tasks that
expand into bounded context-score row tiles. Candidate batches default to 20
siRNAs and are independently configurable from reference shards, so no tile
receives the whole selected set merely because `top_n=-1` was requested. PITA
prepares reference-only UTR stabilization once per evidence stem, streams it
into bounded shards, and combines those shards with per-siRNA MIR stabilization
before row scoring. This preserves the finite shape of a run after efficacy
prediction, lets the app count and bound fanout before submitting work, and
makes retries, cache keys, result comparison, and packaging inspectable.

The rejected first implementation is a dynamic queue of single-siRNA jobs over
reference chunks. That shape is viable but too fine-grained for Modal startup
overhead, repeats tool setup that upstream TargetScan and PITA can share across
multi-siRNA inputs, and makes `top_n=-1` harder to reason about before work is
submitted. The accepted compromise is to keep deterministic manifests for
candidate and reference tiling, then use bounded per-stage queues or fanout only
inside stages with measured imbalance.

When `top_n=-1` is used with off-target scoring, the app still means "score all
efficacy candidates". The app lazily emits candidate-by-reference waves under a
run-wide process-slot budget instead of materializing or submitting the complete
product. The default and maximum budget is 64 slots, split between TargetScan
and PITA, so all-candidate scoring does not become unbounded Modal fanout.

Transcript-aligned TargetScan reference FASTAs are persisted before candidate
waves. The first wave for each reference shard publishes digest-validated
reference-only UTR, ORF, length, and branch-length-bin inputs; later candidate
waves reuse them. PITA likewise builds its extended UTR and UTR STAB data once
per content-keyed evidence stem; it reads the complete STAB output as a stream
while publishing deterministic UTR shards rather than materializing all lines
in Python. The PITA cache is deliberately scoped to the evidence stem for now,
so final cleanup and cache ownership remain atomic; moving it to a global
reference-digest cache would first require an immutable publication and
retention policy independent of run cleanup. This keeps reference-only work out
of the candidate fanout without introducing a second mutable shared cache.

The converted full-human UTR and ORF sources remain immutable assets in the
shared model volume. The derived RNAplfold files are written under the
OligoFormer output volume's `reference-cache/` tree instead of the shared model
volume. The output volume uses Modal Volume v2, which is the required storage
backend for the stage's default fanout of up to 32 concurrent distinct-file
writer nodes. The reference-cache tree also stores the converted UTR and ORF
content digests used by evidence cache keys; declared source URLs alone are not
sufficient identity for persisted scientific results.

The RNAplfold cache is partitioned into deterministic reference shards. Each
shard marker binds the reference identity and shard input digest to every
expected output name, nonzero size, and publication-time SHA-256; a top marker
binds the exact set of shard markers. Routine readiness re-hashes the small
inputs and manifests and checks every output's recorded size, avoiding a full
human-cache reread. A damaged shard repairs only missing or invalid outputs, and
setup overrides cannot exceed 32 nodes or 8 local workers.

An all-human run plan pins that content identity. Before constructing evidence,
post-processing reacquires the stable global reference-state generation,
reloads both volumes, revalidates the pinned identity, and holds the generation
through evidence commit. A concurrent forced reference refresh therefore waits
or makes the prepared run fail closed and require re-planning; it cannot relabel
evidence built from another reference version.

The global guard remains necessary while force-refresh replaces fixed model,
converted-reference, and RNAplfold paths that evidence workers read. Releasing
it after validating only a current pointer would allow those fixed paths to
change during the off-target DAG. Its exclusive generation mode queues an
unrelated all-human run behind the current reader instead of coalescing it as an
identical cache request. The guard can shrink to pointer publication only after
assets and derived references move to immutable digest-versioned directories
and run plans pin those directories directly.

Stages whose row counts are known only after discovery use two independent
expansion hierarchies. TargetScan emits bounded candidate-batch by
reference-shard waves, then context-score row shards. PITA prepares one shared
reference STAB plan, expands that into per-siRNA target-discovery shards, then
emits potential-target energy row shards. Both shapes keep initial fanout
bounded while allowing heavy work to split by observed row count.

The remote `run_oligoformer_postprocess` function owns this off-target tile
scheduler. The local entrypoint remains a thin app-run submitter, while the
manifest, reference prep, child tile expansion, reducers, and final tables live
with the output-volume cache on Modal. This avoids making the user's local
process responsible for thousands of scheduling decisions.

Tile workers write off-target evidence tables and diagnostics into the output
volume and return only lightweight paths, row counts, or status objects.
Returning large result bytes through Modal RPC would make all-candidate runs
more fragile and less inspectable than volume-backed evidence files that
reducers can merge and failed tiles can preserve.

Off-target scoring should fail closed: all required TargetScan and PITA tiles
must succeed before the final ranked table is emitted. A partial off-target run
can make candidates appear safer than they are because missing reference shards
remove evidence, so failed tile diagnostics should be preserved without treating
the incomplete ranking as a valid result.

Different threshold variants may request the same evidence concurrently. A
per-evidence, per-stem distributed build generation uses Modal Dict's atomic
`skip_if_exists` insertion to elect one writer. The writer publishes
`off_target.done` only after both merged evidence tables pass schema validation;
the marker records their row counts, sizes, and SHA-256 digests. It commits that
evidence before recording its generation complete. Waiters reload and validate
the tables. A missing or corrupt completed publication advances exactly one
repair generation, while failed and stale generations advance through
append-only status records instead of deleting a possibly replaced lock.

The tiled implementation is intended to preserve upstream-equivalent scores
except for the explicit TargetScan reducer correction below. It does not add
approximate prefilters or heuristic pruning. Earlier pinned-upstream exercises
compared raw and final artifacts for maintained fixtures; focused site-type and
boundary tests establish the corrected 7mer and 8mer behavior. A future
independent scientific comparison must apply the same projected-score
correction to its upstream oracle before claiming equivalence.

TargetScan and PITA run as independent Modal branch functions because their
inputs and outputs do not depend on each other. `run_oligoformer_postprocess`
starts PITA once, executes bounded TargetScan candidate waves while PITA runs,
then gathers both branches before merging the raw `pita.tab` and
`targetscan.tab` evidence tables. On failure, branch cancellation is best-effort
so cleanup errors do not hide the original failure.

TargetScan context-score shards are submitted through a per-run Modal Queue with
a bounded number of active Modal worker nodes. Each worker node runs local
threads that pull one shard at a time until the queue is empty. This avoids
static batch tail latency where one container receives multiple expensive shards
while other containers finish early. Local workers are capped at 32, and the
actual worker-node count is bounded by the queue size and the TargetScan share
of the run-wide slot budget, so small queues do not spawn idle containers and
environment overrides cannot exceed the run bound.

Each TargetScan context worker commits the output volume once when the worker
finishes or fails, rather than after every shard. When a worker call fails, the
parent reloads the output volume, rebuilds a fresh queue from completion-marker
gaps, and retries the missing deterministic shards up to the configured attempt
limit. A dequeued shard is therefore not lost when its worker terminates.

TargetScan per-reference-batch reduction is also a bounded parallel stage.
After all context-score shards finish, each reference batch is reduced by a
remote finalizer that warms the context output directory with
`warmup_directory()` before the Polars scan, writes that batch's `targetscan.tab`,
and commits the output volume. Merge fanout is capped by both its stage setting
and the assigned process slots, removing the serial reduction tail without
unbounded concurrent reads from the Modal output volume.

Final PITA and TargetScan evidence consolidation uses Polars lazy scans and
streaming CSV sinks. Numeric sort keys are cast explicitly while the published
upstream-shaped columns are retained, avoiding a Python object per evidence row
and keeping tabular parsing inside the repository's standard dataframe engine.
Filtering also lazily groups those merged tables to one summary row per
candidate and collects only the summaries, so `top_n=-1` does not immediately
reload all-human evidence into Python memory.

PITA and TargetScan evidence are interpreted independently. A missing PITA hit
or missing TargetScan hit for an evaluated siRNA means that tool found no
off-target evidence for that siRNA; it does not mean the candidate was skipped.
The `off_target_filter == -5` sentinel is reserved for candidates outside the
`top_n` evaluated set. Final `filter` values count enabled filter failures
instead of summing raw signed filter codes, so sentinel values cannot cancel
functionality or toxicity failures.

For positive `top_n` values, Biomodals scores the semantically ranked
candidates: it ranks by efficacy, keeps each selected row's actual siRNA
sequence, and names the off-target record by the candidate's original row
identity. This intentionally does not preserve upstream-compatible ambiguity
where ranked candidate names could be paired with the first N original
sequences. The off-target cache salt changes when this binding semantics
changes.

TargetScan context-score reducers are implemented with Polars instead of the
upstream shell pipeline. The reducer reads the upstream positional fields used
by TargetScan, applies the site-type thresholds before aggregation, and writes
the same no-header evidence table shape consumed by the final off-target merge.
The off-target cache salt changes when these reducer semantics change.

This reducer intentionally patches a bug in OligoFormer's pinned upstream
`scripts/targetscan.sh`. Upstream projects each context-score row to four fields
and then tests field 28 in that projected row. The missing field is treated as
zero, so all thresholded `7mer-1a`, `7mer-m8`, and `8mer-1a` sites are discarded
and only unthresholded `6mer` sites survive. Biomodals tests the projected score
field instead, preserving qualifying 7mer and 8mer evidence. For this decision,
"upstream equivalence" means equivalence to the pinned upstream pipeline with
this field-selection correction applied. Tests must cover each site type and
the exact threshold boundaries, and this policy remains part of the off-target
cache identity.

Efficacy and final-table completion markers are table manifests, not existence
flags: they record exact schemas, row counts, sizes, and SHA-256 digests. Final
markers also carry a post-processing key and semantics salt separate from the
compute cache key and off-target evidence salt. Toxicity mode and filter
thresholds select distinct final output directories below the same compute run
root. This lets the app recompute final tables after threshold, sentinel
handling, final filter aggregation, corruption, or output-column semantics
changes while reusing valid efficacy and merged off-target evidence.

TargetScan branch-reduction optimization keeps the reducer semantics unchanged:
only the placement changes from serial in the parent branch node to bounded
parallel remote finalizers. The one-time direct-upstream sandbox verifier used
during broader equivalence work has served its purpose and is not retained as a
permanent maintenance tool. Focused regression tests now protect the corrected
site-type thresholds, candidate identities, raw PITA and TargetScan evidence,
and final filter semantics. Any future scientific reducer change requires a new
independent pinned-upstream comparison rather than treating the app as its own
oracle.

Only after both off-target branches have produced and committed their complete
merged evidence, and the final outputs and their manifest are committed, may
workers clean up bulky generated shard inputs and transient intermediates under
the run's `prepare/off_target/<stem>/` tree. Cleanup first revalidates the compact
evidence and preserves final output tables, completion manifests, logs, efficacy
outputs, model caches, reusable reference caches, and compact merged evidence
(`pita.tab`, `targetscan.tab`, and `off_target.done`). The efficacy key contains
only GPU inputs and semantics; the evidence key adds `top_n`, candidate binding,
and reference content identity; and the final-table key adds filtering
thresholds. Retaining merged evidence therefore supports threshold variants
without accumulating large shard trees, while custom-reference and `top_n`
variants still reuse GPU efficacy.
