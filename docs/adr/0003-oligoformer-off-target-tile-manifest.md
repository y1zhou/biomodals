# Use bounded deterministic tiles for OligoFormer off-target scoring

Status: accepted, implemented for the current OligoFormer app.

OligoFormer off-target scoring represents parallel work as deterministic tiles
rather than a dynamic worker queue. TargetScan tiles are candidate-batch by
reference-shard preparation tasks that expand into bounded context-score row
tiles. PITA tiles are per-siRNA preparation tasks that expand into bounded UTR
and row-scoring shards. This preserves the finite shape of a run after efficacy
prediction, lets the app count and bound fanout before submitting work, and
makes retries, cache keys, result comparison, and packaging inspectable.

The rejected first implementation is a dynamic queue of single-siRNA jobs over
reference chunks. That shape is viable but too fine-grained for Modal startup
overhead, repeats tool setup that upstream TargetScan and PITA can share across
multi-siRNA inputs, and makes `top_n=-1` harder to reason about before work is
submitted. A queue can be introduced later if real reference shards show enough
runtime skew to require work stealing.

When `top_n=-1` is used with off-target scoring, the app still means "score all
efficacy candidates". The app first materializes the bounded tile manifests and
applies the run-level task budget before submitting child work, so
all-candidate scoring does not become unbounded Modal fanout.

Reference-shard preparation is a separate cached stage before candidate-scoring
tiles run. Each transcript-aligned reference shard produces tool-specific
prepared artifacts for TargetScan and RNAplfold once, then candidate-batch tiles
consume those artifacts. This keeps reference-only work out of the
candidate-batch fanout and makes reruns reuse the expensive parts that do not
depend on the siRNA candidate set.

The manifest expands in two levels for stages whose row counts are only known
after upstream discovery. The first level contains candidate-batch by
reference-shard tiles known before submission; those tiles may then emit
deterministic TargetScan context-score row tiles or PITA potential-target energy
row tiles. This keeps initial fanout bounded while allowing skewed reference
shards to split heavy downstream work by actual row count.

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

The tiled implementation preserves exact upstream-equivalent scores before
adding approximate prefilters or heuristic pruning. Validation should compare
the merged TargetScan evidence, merged PITA evidence, and final ranked table
against the whole-input pipeline for the same candidates, references, and
thresholds.

TargetScan and PITA run as independent Modal branch functions because their
inputs and outputs do not depend on each other. `run_oligoformer_postprocess`
spawns both branches and gathers them before merging the raw `pita.tab` and
`targetscan.tab` evidence tables. On failure, branch cancellation is best-effort
so cleanup errors do not hide the original failure.

TargetScan context-score shards are submitted in worker-sized batches with a
bounded number of active Modal nodes. This avoids the earlier shape where one
100-plus-shard local batch could contain multiple expensive stragglers and pin
the whole run even though most shards had finished. The current default remains
32 local workers per context batch and up to 32 active context nodes.

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

Final-table completion markers carry a post-processing semantics salt separate
from the run cache key and off-target evidence salt. This lets the app recompute
final tables after sentinel handling, final filter aggregation, or output-column
semantics change while still reusing the same run root and expensive prepared
intermediates.

TargetScan branch-reduction optimization is deferred until upstream-equivalence
tests exist for the raw PITA evidence, raw TargetScan evidence, and final ranked
tables. Context scoring is bounded, but final per-batch reduction remains a
tail-latency risk; correctness takes priority because the recent reducer and
candidate-identity fixes changed evidence semantics.

Upstream-equivalence tests should compare canonicalized table output instead of
raw bytes. The reference remains the upstream OligoFormer, PITA, and TargetScan
programs run directly, but comparison should normalize row ordering, headers,
and numeric formatting before checking raw PITA evidence, raw TargetScan
evidence, and final ranked tables. This keeps the tests strict about scientific
results without making harmless formatting differences block reducer or
packaging cleanup.

The upstream-equivalence suite should run as a separate Modal sandbox
verification command, not as a normal fast `pytest` integration test. Local
tests should cover canonicalization helpers and small synthetic reducer
behavior; the sandbox verifier should invoke upstream OligoFormer, PITA, and
TargetScan directly and compare their canonicalized artifacts with Biomodals
artifacts from the same inputs. This keeps local tests fast while preventing the
app implementation from becoming its own oracle.

The first verifier input should be the existing OligoFormer example FASTA,
`examples/data/sirna_target.fa`, because it is small, already maintained as a
user-facing smoke input, and fast enough to debug canonicalization failures. The
real CFB mRNA should be a later scale and performance verification once the
example-input oracle is stable.

The first verifier should include off-target scoring, not only efficacy and
toxicity. It should default to `top_n=-1` so it compares upstream-compatible
all-candidate scoring rather than Biomodals' intentional positive-`top_n`
candidate-binding fix. It should use small checked-in UTR and ORF fixture
references instead of `--all-human`, so the verifier exercises TargetScan,
PITA, and final off-target merging without pulling the full human reference set
into the fast debug loop.

Those UTR and ORF fixture references should be biologically plausible
mini-transcripts designed to produce at least one expected positive and one
expected negative off-target case. Arbitrary tiny references would only prove
that the command path runs; they would not protect TargetScan site-type
thresholds, PITA evidence merging, or final off-target filter semantics.

The verifier should live under `scripts/`, initially as
`scripts/verify_oligoformer_upstream_equivalence.py`. It is an engineering
verification tool rather than a user-facing app example because it may launch
Modal sandboxes, write canonicalized artifacts, and compare direct upstream
outputs against Biomodals outputs.

The verifier may call the Biomodals OligoFormer app to generate the candidate
artifacts being checked, but the oracle side must come from direct upstream
OligoFormer, PITA, and TargetScan commands in Modal sandboxes. The app must not
be used as a proxy for running arbitrary upstream scripts or as its own oracle.

Verifier artifacts should be written to a persistent Modal output-volume path,
with the local script printing those paths and downloading only compact
canonical comparison summaries by default. Failed equivalence runs need
inspectable upstream logs, raw evidence, Biomodals artifacts, and canonicalized
tables without flooding local stdout or making large off-target outputs depend
on local temporary directories.

Canonicalized equivalence should require exact equality for candidate
identities, sequences, filter flags, categorical columns, and table membership.
Floating-point score columns may use tight, explicit per-column tolerances so
formatting differences and tiny arithmetic roundoff do not fail otherwise
equivalent runs. Any tolerance failure in a score that changes a pass/fail flag
is still a failed equivalence check.

The final-table `efficacy` tolerance is wider than the PITA and TargetScan
score tolerances because the verifier compares two separate upstream GPU
inference executions. Candidate identities, table membership, filter flags, raw
PITA evidence, and raw TargetScan evidence remain exact or tightly bounded.

The verifier canonicalizes upstream's legacy aggregate `filter` column to
Biomodals' count-of-failed-enabled-filters semantics before comparison. The
underlying individual filter columns remain exact comparisons, so this
normalization does not hide functionality, off-target, or toxicity flag
differences.

When raw PITA or TargetScan scores differ only within their explicit tolerance
and all candidate identities, table membership, and pass/fail filters match, the
verifier should pass. The summary should still report max absolute deltas for
floating-point columns so small drift remains visible during future reducer or
runtime changes.

The verifier should reuse persistent app and upstream artifacts by default and
provide an explicit `--force` option for recomputation. Cached reruns make
canonicalization and comparison debugging fast, while `--force` remains
available when runtime setup, fixture references, or direct upstream invocation
details change.

After final outputs are generated, workers should clean up bulky generated
off-target shard inputs and transient intermediates under the run's
`prepare/off_target/<stem>/` tree. The cleanup must preserve final output
tables, raw merged evidence such as `pita.tab` and `targetscan.tab`, completion
markers, logs, efficacy outputs, model caches, and reusable reference caches.
This keeps the output volume from accumulating large per-run shard files that
are not needed once the merged evidence and final tables exist.
