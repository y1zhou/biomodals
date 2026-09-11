# Antibody humanization workflow

Status: implemented. This is the scientific contract for CLI and website runs.
User instructions are in [Humanizing antibodies](../humanization.md); HTTP and
editor behavior belong to the [service spec](humanization-service.md).

## Scope

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
- Submit through `biomodals workflow run` or the website's shared Job lifecycle.
- Use existing scientific defaults with optional per-method overrides, including
  HuDiff candidate count and seed. Do not require equal yields across generators.
- Accept CSV input with `id,vh,vl`, explicit pairing, unique IDs, and early
  validation. Defer FASTA input and inferred pairing conventions.

## Existing contracts

The [domain glossary](../../CONTEXT.md) defines a Humanization Candidate Set
as method-attributed results from the same parental VH-VL pair. The four apps
currently accept explicit complete VH-VL pairs. Their scores, mutation
coordinates, and scientific success or yield semantics differ.

The [output identifier decision](../adr/0010-humanization-publication-contract.md#identifiers)
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

The workflow calls the scoring-only `sapiens_score`, `humatch_score`, and
`pabnativ2_score` operations. Each evaluator has its own
alignment/validation contract, so a union table must retain explicit failures
rather than silently dropping candidates that cannot be scored.

## Union evaluation

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

## Partial-result policy

Continue independent work when a generator fails or an evaluator cannot score
a candidate. Preserve successful candidates and scores, record explicit errors
for missing evaluations, and mark the overall result incomplete. Successful
generation with zero candidates is distinct from execution failure.

Reject malformed CSV, missing chains, invalid amino-acid characters, and duplicate
IDs before remote work. Distinguish these request errors from valid pairs that
are incompatible with a particular method: record those failures and continue
the other methods.

## CDR preservation policy

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
- Humatch: retain all family classifier outputs and pairing scores.
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

The main table must be easy to sort and
filter for manual selection for further characterization: one row per
parent/candidate, separate scalar numeric score and delta columns, explicit
error and CDR-preservation columns, mutation counts, sequences, baseline flag,
and generating-method provenance. Missing scores remain null, never zero or
error strings in numeric columns. No composite fitness score is imposed.

Keep full family distributions, residue-level outputs, and detailed mutations
in separate tables keyed by parent/candidate identity. Native regional scores
and percentiles remain available in companion evaluation tables; the main
selection table emphasizes chain/pair summaries and their parental changes.

## Recovery

Existing recovery distinguishes `resume` (continue suspended/state-unknown work,
without retrying failed tasks) from `restart` (an explicit successor run).
Conclusive failures require the successor-run path; do not add automatic retry
or replacement-call loops.

## Verified run controls

- HuDiff defaults to 10 sampling attempts per parent, seed 42. The exposed
  `candidate_count` is an attempt budget, not a guaranteed unique yield; current
  bounds are 1–25 attempts per parent and 10,000 per standalone request.
- Sapiens retains the paired VH/VL design after each of its 1–5 passes in
  `iteration_designs.csv`; `humanized.csv` remains the standalone final endpoint.
  Every pass enters the union with its iteration source ID, including no-ops
  and repeated endpoints, then exact-pair deduplication precedes evaluation.
  Retention adds no inference passes; additional unique designs add scoring work.
- Humatch returns one optimized endpoint per parent, possibly unchanged.
- `pabnativ2_num_seeds` is a strict integer from 1–25, default 1. At 1 the
  native app receives `pabnativ2_seed` unchanged. Above 1, use
  `random.Random(pabnativ2_seed).sample(range(2**32), pabnativ2_num_seeds)`
  to derive distinct roots. Reuse this ordered list for every parent; each
  native app derives its own pair seed from root, ID, VH and VL. Schedule
  each root/parent as an independent Task under the existing global limits.
  Retain successful siblings when a replica fails; the result remains partial.
  The manifest records actual roots in `generation_seeds.pabnativ2` and parameters.
  `generation.parquet` records each outcome's root and reported derived pair seed;
  native task evidence remains outside the downloaded archive.
  These are optimization replicates, not guaranteed novel candidates.
- For `I` Sapiens passes, `R` p-AbNatiV2 roots and `A` HuDiff attempts,
  the per-parent ceiling is `1 + I + 1 + R + A` including the baseline,
  before deduplication and failed/invalid generation. Defaults still give at
  most 14 rows per parent. HuDiff's root/count semantics are unchanged.
- Existing CSV parsers cap requests at 1,000 parents and 3 MiB. Method-specific
  length and alignment limits differ and must not become an implicit requirement
  that every accepted parent be processable by every generator.
- The CLI exposes shared `--max-containers` and `--max-gpu-containers` ceilings;
  GPU calls count toward the total. These are concurrency limits, not monetary
  or total-work budgets.

## Panel ordering

The main CSV adds nullable integer `quality_tier` and `panel_order`
after the parent and candidate IDs. There is no `selection_reason` column.
Every candidate and all original scores are retained. Within each parent, the
baseline appears first, followed by ranked candidates in panel order, followed
by unranked candidates in stable candidate-ID order. Select ranks 1 through N;
the parental control does not consume one of those N slots.

Ranking v2 is an explicit provisional selection heuristic, not experimental
validation or a calibrated confidence score:

- Require complete candidate and parental evaluations, preserved IMGT CDRs,
  complete framework mutation evidence, and positive parental pairing scores.
- More than 10% relative loss from the parent in either Humatch or p-AbNatiV2
  pairing score puts the candidate outside the default ranking pool. Both new
  fields remain null. This review guardrail is not a validated biological cutoff;
  affected rows remain available for manual exploratory selection. It does not
  alter generation settings, sequences, score values, or execution status.
- Compute Pareto layers within each parent's eligible candidates: maximize
  p-AbNatiV2 paired nativeness, both evaluators' pairing scores, and the arithmetic
  mean of Humatch VH/VL best-human-family probabilities; minimize total VH/VL
  changes. These are five separate objectives, not a composite score. The mean
  is `(humatch_vh_best_family_probability + humatch_vl_best_family_probability) / 2`,
  requires both chain values, and adds no parental family-probability guardrail.
  Layer 1 is nondominated, then repeat on the remaining candidates. Compute the
  mean from raw scores, then compare objectives rounded to six decimals to
  suppress numerical jitter, not to claim biological significance. Raw scores
  retain their original precision; no derived mean column is added to the table.
- Seed panel order by lowest tier, then descending paired nativeness, p-AbNatiV2
  pairing, Humatch pairing, mean Humatch best-family probability, fewer changes,
  and finally candidate ID. For subsequent
  picks, consider the best remaining tier and the immediately following tier;
  maximize minimum framework distance to already selected candidates. Break ties
  with the seed ordering. Thus a tier-2 alternative can precede a tier-1 near-copy,
  but diversity cannot immediately promote an arbitrarily worse tier.
- Framework distance counts differing IMGT positions across VH and VL using
  recorded mutations, including insertion codes and deletions. Two different
  replacements at one position count as one difference. Generator labels do not
  affect order. No new numbering/model call is needed.

Chain-level nativeness and Sapiens summaries remain visible for review but are
not ranking objectives. Humatch's two best-family scores contribute one mean
objective, not two independent votes; target-family probabilities are not added.
All ranks are per-run/per-parent and can change when the candidate pool changes.
The manifest records the full ranking policy and its version; that version also
participates in the workflow fingerprint. The full candidate table is retained.
Ranking v2 supersedes v1 by adding the mean-family objective. Previously published
v1 results remain unchanged; using v2 for new jobs requires an updated workflow
deployment. The change reuses existing scores and adds no model calls.

Ranking uses Polars filtering, grouped mutation counts, pairwise dominance
joins, and position-overlap aggregation on narrow frames. Pairwise comparisons
are bounded within each parent; sequences and unrelated score columns are not
copied into Python row dictionaries. Only Pareto-layer peeling and the inherently
sequential greedy panel picks use small control loops. Mutation JSON is parsed
once into typed Polars frames and reused for ranking and Parquet export.

## Seed and yield interpretation

Both seeded apps derive pair seeds from root, parental ID, VH and VL; changing
the ID changes the stream, whereas scheduling order does not. HuDiff seeds one
batched random stream, not one independent stream per attempt. Changing its
attempt count can change every sample; a larger batch need not preserve the
smaller batch's prefix. Invalid and duplicate attempts are not resampled.

p-AbNatiV2 replicates call the same greedy optimizer. Seeding runtime components
does not guarantee different endpoints or cross-device determinism. Humatch
has no sampling seed: a parent already satisfying its target thresholds can
successfully return unchanged. Raising its targets changes the optimization
objective rather than requesting stochastic diversity.

These semantics explain unequal method yields. Do not force mutations, repeat
parent IDs to simulate seeds, or present a missing method label on a parental
row as evidence that generation did not run. The generation ledger records
those no-ops. The [publication decision](../adr/0010-humanization-publication-contract.md)
retains the supporting observed-yield evidence.

## Result publication

- Keep `selection.csv`, `imgt_mutations.parquet`, and all four consolidated
  `scores/*.parquet` files. They answer final candidate selection and detailed
  evaluation questions without rerunning models.
- Keep a compact manifest containing run/schema/status, settings, scientific
  identity once, ranking policy, generation failure count, and delivered-file
  sizes/hashes. Remove nested scoring manifests and native-copy bookkeeping from
  the user archive. Do not weaken validation of upstream publications before
  their values enter consolidated tables.
- Consolidate generation accounting into one compact `generation.parquet`:
  parent, method, seed/iteration/attempt, outcome/rejection reason, and retained
  candidate ID (including parent/no-op and deduplicated origins). This replaces
  duplicated manifest candidate provenance rather than creating another copy.
- Omit `native/` from the standard download. Do not delete original app/Task
  publications in Modal: leave them as operational diagnostic evidence under
  the existing retention policy. No new debug-download endpoint is needed now.
- Deliberately omit predicted PDBs, detailed generation profiles/alignments and
  structural diagnostics from the standard download. If users need those for
  selection, preserve the requested metrics in a dedicated consolidated table
  or structure artifact rather than copying every opaque native result.

The download boundary deliberately omits some native evidence rather than
claiming it is all present in the Parquets. Result schema/scientific
identity is 3. The service continues to accept schema 2 and serve immutable
old archives. The ZIP transport remains humanization/1.

`generation.parquet` contains parent_id, method, source_id, root_seed, seed, iteration,
attempt_index, outcome, reason, and candidate_id. Outcomes are generated, no_op,
duplicate (a native HuDiff duplicate), rejected, no_candidates, or failed.
Multiple generated rows can reference the same exact union candidate; no origin
is arbitrarily selected as its owner. Rejected/failed/zero-yield outcomes have
null candidate IDs. Sequences live only in selection.csv. Failure rows describe
failed generator calls, not fictional sampling attempts. Manifest generation
failure counts summarize this ledger; evaluator errors remain in selection.csv.
`root_seed` is the seed passed to the app; `seed` is its reported derived
per-pair seed and remains null if the call failed without a publication.

The finalizer publishes one self-contained `humanization_results` directory.
Parental rows have null `generating_methods`; missing evaluations have explicit
`*_error` values, while null errors indicate success. `evaluation_complete`
requires every evaluator and common IMGT annotation. Detailed score tables
retain evaluator-native regions; `imgt_mutations.parquet` uses common IMGT
coordinates without repeating constant scheme/definition columns.

Humatch output families are `hv1`–`hv7`, `kv1`–`kv7`, or `lv1`–`lv10`.
The input value `auto` resolves the parent's best human family once and records
the resolved label, not `auto`.

Apps own parameter/publication validation. The scientific fingerprint includes
model, patch, PSSM and annotation identities plus ranking and result versions.
An incompatible fingerprint requires a new root, not reuse of an old plan.
The shared execution kernel owns scheduling and recovery; the workflow owns
candidate identity, scoring and terminal publication. See the
[workflow-development skill](../../.agents/skills/biomodals-workflow-development/SKILL.md)
for implementation rules and the [service spec](humanization-service.md#result-preparation)
for cache/download behavior.

## Verification and deferred work

Offline contracts live in `tests/workflow/test_humanization_*.py` and
`tests/service/test_humanization_*.py`: graph fanout/partial outcomes, exact
deduplication, score joins, CDR unknowns, ranking, generator seeds, archive
membership/hashes, cancellation and successor reuse.

The publication-envelope regression covers the website's 200-pair ceiling,
11,400 selection rows, maximum-length multibyte IDs/chains, and populated
numeric scores. It verifies the 32 MiB reader bound, the last result page and
a manifest below 64 KiB. This does not benchmark full-size detailed score
payloads or run inference.

Historical CLI and website smoke tests established end-to-end delivery for
older publications, not experimental suitability. Publication or ranking
changes require matching deployment and separately authorized live verification.
Do not infer the deployed version or current test count from this document.

Humatch's repeated parental-scoring optimization remains deferred: profiling
must show meaningful cost/runtime impact before changing the concurrent path.
