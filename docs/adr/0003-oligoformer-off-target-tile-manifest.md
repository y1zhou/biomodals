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
