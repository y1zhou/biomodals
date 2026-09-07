# Antibody humanization workflow

Status: implemented; one-pair live Modal smoke test passed on 2026-09-07.

The subsequent [service integration spec](humanization-service.md) extends the
original CLI-only delivery scope below with website submission and result
browsing; the scientific contracts remain unchanged.

## Agreed scope

- Integrate Sapiens, Humatch, p-AbNatiV2, and HuDiff-Ab into one end-to-end
  workflow that produces humanization candidates for experimental
  characterization.
- Run the methods independently from the same parental sequences, with
  parallel execution.
- Accept complete, explicitly paired VH-VL variable domains only. Full-length
  chains, unpaired chains, and VHHs are outside this release.
- Exclude CUMAb from this release. Retain its
  [research and implementation findings](../research/humanization/cumab.md)
  for future work.
- Submit through `biomodals workflow run` in this release. A future service
  will expose job submission to the frontend; no HTTP endpoint is required now.
- Use existing scientific defaults with optional per-method overrides, including
  HuDiff candidate count and seed. Do not require equal yields across generators.
- Accept CSV input with `id,vh,vl`, explicit pairing, unique IDs, and early
  validation. Defer FASTA input and inferred pairing conventions.

## Existing contracts

The [domain glossary](../../CONTEXT.md) defines a Humanization Candidate Set
as method-attributed results from the same parental VH-VL pair. The four apps
currently accept explicit complete VH-VL pairs. Their scores, mutation
coordinates, and scientific success or yield semantics differ.

The [output identifier decision](../adr/0011-humanization-output-identifiers.md)
preserves parental IDs in tables and seed derivation while normalizing FASTA
whitespace. The workflow must account for this existing mapping.

## Verified scoring capabilities

- [Sapiens](../research/humanization/sapiens.md): independent per-chain,
  per-residue amino-acid probabilities. A sequence summary requires an explicit
  aggregation definition; the existing unmasked probabilities are not masked
  pseudo-likelihood.
- [Humatch](../research/humanization/humatch.md): independent heavy/light family
  and pairing classifier distributions. Target-family comparisons require a
  named, consistent reference; selecting a new target for every candidate changes
  the comparison.
- [p-AbNatiV2](../research/humanization/pabnativ2.md): independent paired-sequence
  scoring, including joint, chain, and regional nativeness, pairing scores,
  percentiles, and residue profiles. Scoring does not require structure prediction
  or RASA calculation.
- [HuDiff-Ab](../research/humanization/hudiff-ab.md): candidate generation, with
  no established native standalone humanness scorer. Its published evaluation
  uses external metrics, which must not be attributed as native HuDiff scores.

These capabilities exist upstream or in wrapper internals; workflow-facing
scoring-only operations still need integration. Each evaluator has its own
alignment/validation contract, so a union table must retain explicit failures
rather than silently dropping candidates that cannot be scored.

## Agreed union evaluation

Generate independently, deduplicate exact VH-VL pairs within each parent, retain
all generating-method provenance, and include the unchanged parent as a baseline.
Evaluate this union without further sequence mutation using Sapiens, Humatch,
and p-AbNatiV2 in parallel. Export one row per parent/candidate with separate
method-specific metrics and evaluation errors, plus detailed residue artifacts.
Do not blend unlike scores into a single ranking by default.

Candidate identity is the complete VH-VL sequence pair within its parental
context; do not recombine chains. Preserve separate parental memberships when
identical sequences arise from different parents. Deliver the complete scored
table, including mutation counts and parent-relative score changes, rather than
an automatically selected experimental shortlist.

## Agreed partial-result policy

Continue independent work when a generator fails or an evaluator cannot score
a candidate. Preserve successful candidates and scores, record explicit errors
for missing evaluations, and mark the overall result incomplete. Successful
generation with zero candidates is distinct from execution failure.

Reject malformed CSV, missing chains, invalid amino-acid characters, and duplicate
IDs before remote work. Distinguish these request errors from valid pairs that
are incompatible with a particular method: record those failures and continue
the other methods.

## Agreed CDR preservation policy

Keep native generation behavior and independently check every candidate for
parental CDR preservation using IMGT boundaries. Retain and score candidates
with CDR changes, flagging those changes explicitly. This is an annotation
policy, not a guarantee that every candidate preserves IMGT CDRs. Do not equate
numbering schemes with CDR boundary definitions.

Verified generator contracts:

- Sapiens supports Kabat, Chothia, IMGT, and North CDR definitions, independently
  of its numbering scheme; its default is Kabat and it restores parental CDRs.
- Humatch protects fixed IMGT CDR regions by default and accepts extra protected
  IMGT positions.
- p-AbNatiV2 protects native CDR regions in AHo coordinates by default and accepts
  extra protected AHo positions.
- HuDiff-Ab uses the released Kabat-no-Vernier protection mask on an IMGT grid,
  with no exposed alternate CDR definition or extra-position controls.

A shared post-generation CDR check is possible without changing native
generation. Enforcing an arbitrary common definition during all four generators
is not supported by their existing controls. Do not silently repair candidate
sequences or exclude them for failing the shared preservation check.

If IMGT annotation cannot be established confidently, retain the candidate with
`cdr_preservation=unknown`, record the explanation, and mark the assessment
incomplete. An unavailable check is never evidence of preservation.

## Evaluation summaries

- Sapiens: separately for VH and VL, average the probability assigned to each
  actual amino acid in the supplied sequence. Retain the detailed residue
  distributions. This is an unmasked model-fit summary, not an experimental
  success probability.
- Humatch (agreed): retain all family classifier outputs and pairing scores.
  Select each parent's reference families once (automatic or explicit), then
  report every descendant's probability and germline likeness against those
  same references. Also report each candidate's best-family label and score for
  each chain, without replacing the fixed comparison target. Changing the target
  during generation can change mutations; changing the evaluation reference
  changes the reported comparison, not the candidate sequence.
- p-AbNatiV2: retain native pair, chain, region, pairing, and percentile summaries,
  alongside residue profiles. Percentiles refer to the bundled reference data,
  not the generated candidate union.
- Parent-relative changes compare the same metric, evaluator configuration,
  and reference between a candidate and its parent. Missing evaluations produce
  missing changes, not zero changes.

These summary definitions are agreed. The main table must be easy to sort and
filter for manual selection for further characterization: one row per
parent/candidate, separate scalar numeric score and delta columns, explicit
error and CDR-preservation columns, mutation counts, sequences, baseline flag,
and generating-method provenance. Missing scores remain null, never zero or
error strings in numeric columns. No composite fitness score is imposed.

Keep full family distributions, residue-level outputs, and detailed mutations
in separate tables keyed by parent/candidate identity. Native regional scores
and percentiles remain available in companion evaluation tables; the main
selection table emphasizes chain/pair summaries and their parental changes.

## Agreed deliverables and recovery

- A readable candidate-score CSV.
- VH/VL sequences in the main CSV; no duplicate FASTA or selection Parquet.
- Detailed Parquet score tables under `scores/` and `imgt_mutations.parquet`.
- A manifest recording parameters, model versions, provenance, and failures.
- Reuse the existing execution system's recovery mechanism to reuse matching
  successful work and retry missing or failed work. Changed inputs or scientific
  settings must not reuse incompatible results.

Existing recovery distinguishes `resume` (continue suspended/state-unknown work,
without retrying failed tasks) from `restart` (an explicit successor run).
Conclusive failures require the successor-run path; do not add automatic retry
or replacement-call loops.

## Verified run controls

- HuDiff defaults to 10 sampling attempts per parent, seed 42. The exposed
  `candidate_count` is an attempt budget, not a guaranteed unique yield; current
  bounds are 1–25 attempts per parent and 10,000 per standalone request.
- Sapiens, Humatch, and p-AbNatiV2 each return one final pair per parent. With
  default HuDiff attempts, the union has at most 13 generated pairs plus its
  parental baseline, before deduplication and failed/invalid attempts.
- Existing CSV parsers cap requests at 1,000 parents and 3 MiB. Method-specific
  length and alignment limits differ and must not become an implicit requirement
  that every accepted parent be processable by every generator.
- The CLI exposes shared `--max-containers` and `--max-gpu-containers` ceilings;
  GPU calls count toward the total. These are concurrency limits, not monetary
  or total-work budgets.

## Agreed workflow controls

Reuse global `--max-containers` and `--max-gpu-containers`; do not introduce
separate workflow or per-method concurrency flags. Expose native per-method
sampling constraints and scientific controls as tunable workflow arguments,
retaining native defaults and bounds. Clearly label attempts versus unique yield.

## Implementation task list

The user approved this implementation sequence. The implementation and local
verification evidence are recorded below; live cloud validation is separate.

1. Define typed workflow inputs, candidate identity, summary columns, detailed
   evaluation records, and explicit failure/CDR statuses. Document column meanings
   and units; keep numeric columns nullable and sortable.
2. Add workflow CLI discovery, paired CSV validation, and method-specific
   arguments. Reuse global resource controls; distinguish malformed requests
   from per-parent/per-method incompatibility.
3. Make existing app execution components composable in one workflow graph.
   Namespace method node IDs and result descriptors (existing collectors share
   a run-scoped `result.json` path). Include the required app dependencies and
   coordinator-side numbering tools without changing standalone scientific behavior.
4. Connect the four independent generators under the shared execution budget.
   Preserve per-parent successes, zero-yield records, attempts, seeds, and errors
   when other work fails; do not add automatic scientific resampling.
5. Build the exact-pair union per parent, retaining all provenance and the
   unchanged parent baseline. Add the common IMGT CDR check and mutation
   annotations, including an explicit unknown state.
6. Add scoring-only app operations for Sapiens, Humatch, and p-AbNatiV2 using
   existing pinned models. Verify unchanged sequences and native score semantics;
   preserve standalone humanization entry points.
7. Fan out union scoring and join by explicit parent/candidate identity. Compute
   fixed-parent-family and best-family Humatch summaries, Sapiens chain means,
   native p-AbNatiV2 summaries, and comparable parental deltas. Account for every
   candidate/evaluator result, including alignment failures and omissions.
8. Export the sortable selection CSV and detailed Parquet tables,
   and provenance/failure manifest. Keep raw method score-region definitions
   separate from the shared IMGT preservation annotations.
9. Integrate and test existing resume/restart semantics, successful-work reuse,
   scientific-configuration invalidation, and partial-result aggregation.
10. Add tests throughout the preceding slices, then run end-to-end local tests,
    standalone-app regressions, type/lint checks, and CLI discovery/help/dry-run
    smoke checks. Cover identifier collisions, duplicate/baseline candidates,
    score parity, nulls, CDR unknowns, zero yield, failures, and recovery. Provide
    a small example input and usage documentation. Request separate approval
    before live cloud inference; local doubles do not establish live score parity.

Out of scope: HTTP service/frontend, CUMAb, automatic top-N truncation, composite
fitness scores, cross-parent chain recombination, and unapproved live inference.

## Panel ordering (2026-09-07 amendment)

The main CSV adds nullable integer `quality_tier` and `panel_order`
after the parent and candidate IDs. There is no `selection_reason` column.
Every candidate and all original scores are retained. Within each parent, the
baseline appears first, followed by ranked candidates in panel order, followed
by unranked candidates in stable candidate-ID order. Select ranks 1 through N;
the parental control does not consume one of those N slots.

Ranking v1 is an explicit provisional selection heuristic, not experimental
validation or a calibrated confidence score:

- Require complete candidate and parental evaluations, preserved IMGT CDRs,
  complete framework mutation evidence, and positive parental pairing scores.
- More than 10% relative loss from the parent in either Humatch or p-AbNatiV2
  pairing score puts the candidate outside the default ranking pool. Both new
  fields remain null. This review guardrail is not a validated biological cutoff;
  affected rows remain available for manual exploratory selection. It does not
  alter generation settings, sequences, score values, or execution status.
- Compute Pareto layers within each parent's eligible candidates: maximize
  p-AbNatiV2 paired nativeness and both evaluators' pairing scores; minimize total
  VH/VL changes. Layer 1 is nondominated, then repeat on the remaining candidates.
  Compare scores rounded to six decimals to suppress numerical jitter, not to
  claim biological significance. Raw scores retain their original precision.
- Seed panel order by lowest tier, then descending paired nativeness, p-AbNatiV2
  pairing, Humatch pairing, fewer changes, and finally candidate ID. For subsequent
  picks, consider the best remaining tier and the immediately following tier;
  maximize minimum framework distance to already selected candidates. Break ties
  with the seed ordering. Thus a tier-2 alternative can precede a tier-1 near-copy,
  but diversity cannot immediately promote an arbitrarily worse tier.
- Framework distance counts differing IMGT positions across VH and VL using
  recorded mutations, including insertion codes and deletions. Two different
  replacements at one position count as one difference. Generator labels do not
  affect order. No new numbering/model call is needed.

Chain-level nativeness, Sapiens summaries, and Humatch family scores remain
visible for review but are not additional correlated votes in this first policy.
All ranks are per-run/per-parent and can change when the candidate pool changes.
The manifest records the full ranking policy and its version; that version also
participates in the workflow fingerprint. This amendment supersedes the original
no-ranking decision while retaining the full, non-truncated candidate table.

Ranking uses Polars filtering, grouped mutation counts, pairwise dominance
joins, and position-overlap aggregation on narrow frames. Pairwise comparisons
are bounded within each parent; sequences and unrelated score columns are not
copied into Python row dictionaries. Only Pareto-layer peeling and the inherently
sequential greedy panel picks use small control loops. Mutation JSON is parsed
once into typed Polars frames and reused for ranking and Parquet export.

## Implementation evidence

Final local verification: 1,491 tests passed across the full repository suite.
All changed production modules pass scoped `ty`; all changed files pass `prek`.
App discovery/help and workflow discovery/help/dry-run smoke checks pass.

- Added pure Pydantic pair, provenance, candidate, evaluation, and annotation
  contracts under `biomodals.workflow.humanization.contracts`.
- Added bounded CSV parsing reusing the existing format validator, deterministic
  per-parent exact-pair union, baseline retention, and sortable nullable selection
  tables with separate score/delta columns. Best-family scores have no delta
  because their reference can change; fixed-family deltas reject family mismatch.
- Added independent IMGT mutation/CDR annotation with explicit unknown results;
  substitutions, insertions, and deletions count as changed numbered positions.
  Generator sequences are neither repaired nor filtered.
- Added `sapiens_score`, a scoring-only app operation returning unchanged input,
  chain-mean summaries, detailed residue scores, and pinned scientific metadata.
  Shared chain validation remains in use by standalone humanization.
- Added `humatch_score` with fixed parental references and separate best-family
  summaries/full distributions, and `pabnativ2_score` with native paired scoring,
  mapped input IDs, omitted-input checks, and no structure prediction. All three
  operations share a bounded-table archive layout and digest manifest.
- Added the kernel-owned generation, union, and terminal evaluation graph. It
  uses direct app operations rather than nested coordinators or standalone
  collectors, so their node/result-path collisions do not enter this workflow.
  Generation and evaluation expose partial-result aggregation; a terminal
  generation-completeness check prevents earlier failures being hidden by a
  successful summary. Local baseline publications preserve useful output even
  when every model call fails.
- Real-kernel tests with fake providers verify complete and all-failed
  generation/evaluation, native-result decoding, multi-candidate deduplication,
  joined detail tables, manifest digests, and sortable terminal CSV output under
  shared total/GPU call limits.
- Modal composition and the `biomodals workflow run humanization` entrypoint
  are now wired, with explicit native method flags and global concurrency flags.
  The CLI dry-run succeeds on `examples/data/sapiens_pairs.csv`; discovery,
  composition, and no-launch validation are covered by local tests.
- Terminal output now includes a self-contained `humanization_results` directory:
  selection CSV, compact candidate provenance in the manifest, common IMGT
  mutation Parquet, joined scoring detail Parquets under `scores/`, retained
  native publications, and a digest manifest with explicit software/model
  identities. These identities also participate in the scientific fingerprint.
- Real-coordinator successor tests verify that failed scorer tasks alone are
  rerun, successful generator/evaluator publications are reused, and changed
  scientific settings are rejected before preparing an incompatible successor.
  Generation/evaluation retain predecessor reuse; the union collector is
  refreshed when its upstream closure needs work.
- Workflow-enabled images now explicitly include app source as well as workflow
  source, so the shared coordinator can import app-composed graph classes. Image
  source contracts and the annotation image's Python 3.12 parsing floor are
  tested locally.
- Local tests use fake numbering and model predictions. They do not establish
  scientific equivalence or experimental suitability.

### Live smoke test

Deployment `main/HumanizationWorkflow/v1`, run
`a6038d07-7ef4-4945-bec6-340f0809bfa8`, completed successfully in approximately
nine minutes. Input was `examples/data/sapiens_pairs.csv` (one pembrolizumab
VH–VL pair), with two HuDiff sampling attempts, seed 42, and global limits of
four total/two GPU provider calls. All 24 tasks succeeded.

The exact-pair union contained the parent and three modified candidates:
Sapiens, p-AbNatiV2, and HuDiff each contributed one; Humatch returned the parent.
Every union member was scored by all three evaluators and retained its IMGT
CDRs. All 69 manifested files passed SHA-256 verification, including the
four-row selection table, eight-record paired FASTA, and native detail tables.
The method scores disagreed on some candidates; successful execution is not a
claim of improved binding, developability, or experimental performance.

The run exposed a reporting defect: the coordinator returned status without
terminal artifact references, so the CLI omitted result locations. The shared
runtime now returns persisted terminal outputs on run and resume, covered by
regression tests. The workflow composition root and workflow-owned contracts
now live in `biomodals.workflow.humanization`; evaluator images explicitly
include its focused scoring-archive module. These follow-up changes require a
new deployment and were not part of the v1 live test.

## Usage

Run the checked-in no-spend example with `bash examples/workflow/humanization.sh`.
To inspect all native scientific controls:

```bash
uv run biomodals workflow help humanization
```

After the existing model-staging prerequisites are satisfied and the containing
workflow is deployed, submit through the normal CLI:

```bash
uv run biomodals workflow run --max-containers 4 --max-gpu-containers 2 \
  humanization -- --input-csv examples/data/sapiens_pairs.csv \
  --hudiff-ab-candidate-count 10 --hudiff-ab-seed 42
```

The CLI reports the execution/deployment identity and result locations. The
`humanization_results` directory is self-contained. Open `selection.csv` for
sorting; this is the only main selection table and includes both sequences.
Detailed score Parquets live under `scores/`. Compact candidate origins
(source IDs, attempts, seeds) live in `manifest.json`, without duplicated
sequences. No separate selection Parquet, provenance JSON, or FASTA is emitted.
Parental rows have null `generating_methods`, even when a generator returned
the unchanged parent; that event is retained only in manifest provenance.
The three evaluator `*_error` columns replace redundant `*_status` columns:
null means the evaluation succeeded; missing evaluations have an explicit error.
`evaluation_complete` still indicates whether all required evidence is present.
Humatch target-family output labels are `hv1`–`hv7` for VH, and `kv1`–`kv7`
(kappa) or `lv1`–`lv10` (lambda) for VL. The CLI also accepts `auto`, which
selects the parent's highest-probability human family and holds that reference
fixed for all candidates from that parent; the output records the resolved
family rather than `auto`.
`imgt_mutations.parquet` omits constant numbering/CDR-definition columns;
the manifest records the common IMGT definition. Unique generation publications
remain under `native/`. Scorer detail tables are consolidated under `scores/`;
their original manifests and publication metadata are retained in the main
manifest's `scoring_publications`. Scorer input pairs and all summary fields
are represented in `selection.csv`, so their CSV copies and enclosing archives
are not exported. Unknown extra scorer files are retained under `native/`.
Normalized adapter JSON and temporary detail shards are not exported. Execution
checkpoints remain intact outside the result bundle. Missing or failed model evaluations remain
explicit, and a changed CDR is not an automatic exclusion.

The waited CLI prints completion status, the result directory's volume/path,
and the selection-table path, rather than every intermediate task output.
The finalizer publishes only the self-contained directory, without additional
selection CSV or error JSON copies.

Apps expose their parameter validators for reuse by standalone calls and other
workflows. The workflow fingerprint includes p-AbNatiV2's full scientific
runtime identity (wrapper, compatibility patches, and PSSMs) and HuDiff's patch
digest. Corrected fingerprints require a new root run rather than restart from
an older incompatible plan; historical results and model caches are untouched.
The app-owned p-AbNatiV2 protocol is recorded once under `protocols` in the
workflow manifest: four-structure source behavior, PSSM cutoff, objective
weights, and pairing-score interpretation. This does not change inference.

Review decision: defer Humatch's repeated parental-scoring optimization because
it is not the bottleneck in the current concurrent workflow. Revisit it if
profiling shows meaningful runtime or cost impact.

Offline export verification against the saved 21-candidate run checked all 63
scorer archives: input pairs, every summary field, detailed table values, and
original scorer manifests were preserved, as were the four unique generation
publications. The self-contained bundle shrank from 273 files to 11. This
checks export equivalence, not a new cloud inference run.
