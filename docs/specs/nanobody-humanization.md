# Nanobody humanization

Status: implementation approved 22 September 2026; deployment remains separate.
Branch `feat/nanobody-humanization` starts at `d57de18`. The supplied research
report is design input, not an accepted implementation specification.

## Goal and agreed release boundary

Generate humanized single-domain antibody candidates for experimental
characterization. Keep human-VH resemblance, VHH compatibility, measured
function, developability and immunogenicity risk distinct. A higher model score
is not a measured improvement in any of the latter properties.

Accepted decisions:

- Deliver candidate generation, common evaluation, diverse experimental-panel
  ordering and integration with the existing local sequence analysis tool.
  Separate all-candidate structural ensembles, TNP, NanoMelt and HLA assessment
  are not first-release requirements. Structures required internally by a
  selected generator remain part of that generator's native execution.
- Provide a standalone website Tool and CLI workflow, sharing suitable
  components with antibody humanization. Do not introduce a second scheduler
  or represent single-domain candidates as artificial VH-VL pairs.
- Defer licensing review at the user's request. This records interview scope,
  not a conclusion about permissions.
- Accept any valid VH domain, not only camelid-derived VHH. Trim non-Fv
  flanking sequence and fill missing FR1/FR4 residues using closest germline
  information before feeding the models, under the preparation policy below.
- Include AbNatiV2's VHH humanization route and HuDiff-Nb only in the first
  release. Llamanade and the Sapiens comparator do not gate this release.
- Use manual ID plus sequence entry with an Add button and CSV `id,vhh`
  import. FASTA import is not part of the accepted first input surface.

The broader input scope supersedes the initial proposal to reject all flanks
and partial domains. A heavy-chain assignment or closest germline species does
not establish origin or functional single-domain behavior, and no camelid
species check is required. Acceptance of conventional VH is not evidence that
the VHH-oriented generators or VHH nativeness scores are equally validated for
every VH; outputs remain experimental candidates, not demonstrated autonomous
domains.

## Accepted preparation policy

- Use arpeggia's native terminal imputation: missing beginnings of FR1 and
  ends of FR4 only. Preserve observed domain residues and internal gaps; do
  not invent CDR sequence or replace ambiguous amino acids.
- Require one detected VH domain. Reject multiple detected variable domains
  rather than selecting one. Report a row-level error if recognition,
  reference coverage or native model compatibility remains insufficient.
- Use arpeggia's deterministic representative closest V reference for FR1
  and J reference for FR4 across available species. Imputation is automatic;
  do not show tie notes, per-reference notices or a separate imputation
  confirmation. Do not invent residues absent from the selected reference.
  Native reference IDs and source-position evidence remain internal
  reproducibility metadata, not extra user-facing output columns.
- Use the prepared VH domain as the parental baseline for scores, mutation
  counts and candidate comparison. Retain the original submitted sequence
  separately; trimming and imputation are not generator-proposed mutations.
- Show the prepared sequence inline before scientific submission, allowing
  input correction and review against the original. Submission stays
  explicit, with no separate confirmation page or mandatory imputation notice.
  Editing input requires a corresponding updated preparation preview.

The existing [glossary](../../CONTEXT.md) defines Humanization Candidate Union
and Complete VH-VL Pair around explicitly paired conventional antibodies.
Do not reinterpret those definitions, invent a light chain, or set pairing
scores to zero to accommodate single-domain candidates.

## Accepted generation and evaluation policy

- Preserve every residue classified as CDR by either generator's native
  convention, all cysteines in the prepared parent, and its existing residues
  at IMGT 42, 49, 50 and 52. Map the union through actual parent residue
  indices rather than combining AHo and IMGT numbers directly. Apply
  restrictions during generation and independently check them afterward.
  Other framework positions, including imputed termini, remain mutable.
  Do not expose a shared Allow CDR mutations control in this release.
- AbNatiV2 uses enhanced search and contributes its final endpoint per parent.
  Retaining internal intermediate states and exhaustive search are deferred.
  Preserve the native generator-required structure calculations.
- HuDiff-Nb defaults to 10 attempts per parent, with an accepted range of
  1–25 and root seed 0. Apply the seed explicitly; preserve native batch size
  10 and cross-batch token carry-over. A single parent budget is not split
  into independently remasked calls. Counts describe attempts, not guaranteed
  unique sequences; do not automatically sample replacements for duplicates.
- Run the two methods concurrently within the configured execution limits.
  Collect valid candidates and the prepared parent, deduplicating by exact
  sequence within each parent while retaining all method provenance. Do not
  merge distinct parent identities even when their sequences are identical.
- Evaluate every union member with AbNatiV2 human-VH (VH2) and VHH (VHH2)
  models. Report both scores separately. These are two model distributions
  from one method family, not independent validation; there is no native
  HuDiff candidate-quality score to fabricate as another vote.
- Preserve usable results when one generator fails and report a partial
  outcome with the failure. Missing evaluations remain null, never zero.
  A successful unchanged endpoint is a valid no-op; if no method introduces
  a change, say that no new designs were produced rather than treating the
  no-op itself as a generation failure. Terminal outcomes are specified below.

## Accepted ranking and controls

- Rank independently per parent using Pareto objectives that maximize VH2
  and VHH2 nativeness and minimize mutation count from the prepared parent.
  Within a Pareto tier, prefer distinct mutation patterns. Retain deterministic
  tie-breaking and reuse applicable paired-workflow ranking mechanics without
  inventing pairing scores. Detailed tie-breaking will be frozen in the
  implementation contract and tests.
- Publish each prepared parent first as an unranked reference, followed by
  ranked candidates, then candidates with incomplete evaluations and no
  numeric rank. Do not turn missing scores into zero or imply ranking by a
  partial objective set is equivalent to complete evaluation.
- Do not impose an absolute VHH nativeness floor or automatically exclude a
  valid candidate because its VHH score decreased. Show parent-relative
  deltas; protected-residue violations and invalid sequences remain hard
  failures. Model scores express computational tradeoffs, not biological
  acceptance criteria.
- Website admission defaults to 100 parents per Job, configurable on the
  server up to 200, with a 10 MiB browser CSV import limit. Prepared-domain
  eligibility uses native model position/coverage checks, not the paired
  workflow's Sapiens-derived 142-residue cap. Submitted-construct byte/length
  limits remain distinct from prepared-domain requirements.
- The new Tool's active-job limit N derives per-job GPU provider capacity
  `max(1, 2*N)` and total provider capacity `max(1, 8*N)`. N=0 pauses new
  admission. These are immutable admission snapshots, not one shared provider
  pool across all Jobs; existing Jobs retain their limits. CLI runs reuse
  global `--max-containers` and `--max-gpu-containers` arguments.
- Advanced controls expose AbNatiV's native humanness threshold, solvent-
  exposure threshold and allowed per-step VHH-score decrease, with concise
  explanations and explicit defaults, plus the accepted HuDiff attempt/seed
  controls. A per-step tolerance is not a maximum total loss from the parent.
  Keep the accepted residue policy fixed and defer arbitrary protected-site
  editing from the first release.

## Accepted presentation and failure outcomes

- The main single-chain table contains parent/candidate IDs, VH sequence, pI,
  V/J genes, panel order, Pareto tier, generating methods, mutation count,
  VH2/VHH2 scores and parent-relative deltas. Reuse bounded server paging and
  sorting; empty error columns are hidden based on the complete result.
- Sequence inspection reuses the shared numbering/liability/germline view
  with the exact saved prepared parent. The later requested antibody display
  fix supersedes the earlier five-row presentation: show the candidate's
  germline, bold Humanized sequence, Parental sequence and Germline (parental),
  with unlabeled difference strips and before/after V/J identities. Display
  CDR colors follow the chosen scheme, not the immutable generation mask.
- Keep selection.csv as the primary download. The standard ZIP adds compact
  provenance and consolidated generation, mutation, germline and detailed
  score evidence. Do not duplicate selection as Parquet or copy native trees.
  Native-generated structures remain operational artifacts outside this ZIP;
  no new structure viewer or extra structure inference is in scope.
- Selected sequences survive pagination and transfer to Group 1 of the
  existing analysis tool as standalone VH FASTA records, with no pairing
  cross-product or transferred scientific ranks. Enforce its advertised
  1,000-entry group limit without truncation or automatic partitioning.
  The destination keeps its existing no-parent inspection; parental
  comparison stays on the Job result page.
- Invalid rows must be corrected or removed before admission. Some failed
  parents/methods produce Partial usable results; if every generator fails
  for every parent, the Job is Failed. This deliberately differs from the
  paired workflow's parent-only Partial outcome and must not change it.
  Valid candidates with missing scores survive unranked; successful unchanged
  generation is not a failure.
- Reuse shared cancellation, explicit same-input rerun and local result-
  preparation recovery. Do not silently generate replacement attempts.

## Existing implementation: reusable and incompatible parts

- [Humanization contracts](../../src/biomodals/workflow/humanization/contracts.py)
  require both VH and VL; generator/evaluator names are closed literal sets.
  [Ranking](../../src/biomodals/workflow/humanization/ranking.py) includes two
  pairing guardrails and paired objectives. These are not VHH ranking rules.
- [Service Tool registration](../../src/biomodals/service/tools.py) and the
  shared execution kernel already provide owner-scoped Jobs, stage projection,
  cancellation, concurrency admission, pinned deployments and downloads. A new
  Tool needs explicit registration; it does not need a second scheduler.
- [p-AbNatiV2](../../src/biomodals/app/design/pabnativ2/app.py) calls paired
  humanization/scoring. Its [asset manifest](../../src/biomodals/app/design/pabnativ2/models.py)
  stages a paired model and ABodyBuilder3, not a complete VHH runtime.
- [HuDiff asset staging](../../src/biomodals/app/design/hudiff_ab/models.py)
  already manifests the released `checkpoints/nanobody/hudiffnb.pt` file. This
  is reusable asset evidence, not an implemented HuDiff-Nb generation operation.
- [Sapiens](../../src/biomodals/app/design/sapiens/app.py) has per-chain model
  operations internally, but its public scientific operations require paired
  CSV input. A heavy-chain comparator needs a truthful single-chain interface,
  not a dummy partner. It is not thereby a VHH-specific humanizer.
- [Local antibody analysis](../../src/biomodals/helper/antibody.py) and its
  [service](../../src/biomodals/service/antibody_sequence_analysis/analysis.py)
  already support unsuffixed single-domain FASTA records, native numbering,
  liabilities, pI, V/J matches and optional parental comparison. Existing
  therapeutic gene usage is not a VHH-only cohort and must not be relabeled so.

Keep reusable scientific analysis in the existing helper and general execution
in the kernel. VHH-specific generation, candidates, validation,
ranking and publication belong to a separate workflow package. Exact app
boundaries and any extraction from paired implementations remain undecided.

Frontend inspection at `0462dde` confirms that shared Job detail, server stages,
authentication, cancellation and downloads can support another Tool. Catalog
and lazy routes need explicit registration. The paired form, result-column
interpretations, seed/CDR fanout and parent lookup cannot be reused unchanged.
`AntibodySequenceDialog` already supports one sequence plus an optional parent.
Existing analysis transfer supports single heavy chains, but does not carry the
parental sequence; generic analysis still has pair-oriented labels and columns.
The accepted transfer keeps that destination's no-parent inspection, as above.

## Primary-source checks affecting implementation

### Domain validation is not a VHH classifier

The inspected [arpeggia 0.10.1 recognition source](https://github.com/y1zhou/arpeggia/blob/3c89295060457f20d218d2221cc1fadd868e5c73/src/ab_numbering/core.rs)
uses H/K/L profiles and rejects additional detected domains. It deliberately
retains unnumbered flanks and accepts some terminal truncations, reporting
`PARTIAL_DOMAIN`. Numbering success alone therefore does not establish a
complete isolated domain. Preparation must distinguish original input,
numbered observed residues and germline additions, then check compatibility
with each selected generator. Do not infer source species from H assignment.

A small offline public-sequence probe confirmed that Ozoralizumab numbers as a
complete domain; added His tags fall outside its numbered span; a linked tandem
produces a multiple-domain error; removal of its first five residues still
numbers the remaining full span but reports `PARTIAL_DOMAIN`. These examples
exercise checks, not sensitivity/specificity of a biological classifier.
Human-derived single-domain VH can pass mechanical eligibility. A VNAR may fail
recognition, but H/K/L recognition is not a dedicated VNAR exclusion test.

### Existing terminal imputation and its limits

Arpeggia already provides [native terminal imputation](https://github.com/y1zhou/arpeggia/blob/3c89295060457f20d218d2221cc1fadd868e5c73/src/ab_numbering/impute.rs).
It fills supported missing beginnings of FR1 from V references and ends of FR4
from J references without modifying observed residues, internal gaps or CDRs.
Added residues have no original input index and retain their reference IDs;
the original input and detected domain span remain available. Reuse this
operation, not the UI alignment grid, whose unmatched reference ends are
intentionally omitted. Existing general sequence analysis remains a faithful
analysis of supplied input; this preparation policy must not silently change it.

Default imputation requires tied best references to agree on each added amino
acid and its presence. Selecting exact reference IDs permits a representative
V/J choice instead. In a public Ozoralizumab truncation probe, default consensus
left the last FR4 position unresolved; selecting the display representatives
completed the tail. Best V and J representatives can come from different
species, especially with short J-region ties. Reference identity is provenance,
not a species assertion or recovery of experimentally observed missing residues.

Domain boundaries are inferred, not infallible. A probe combining terminal
truncation with His tags allowed some tag residues to align into the missing
framework, even though tags on the complete domain were excluded. Consequently,
automatic trimming cannot promise to remove every tag from every fragment.
This motivates the accepted prepared-sequence preview, not an unvalidated
tag-removal heuristic or a claim that the inferred residues were observed.

### Humanization is not conventional heavy-chain optimization

The [prospective NKp30 study](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.5176)
found parent-dependent effects of hallmark substitutions and noncanonical
disulfides on binding and CDR3 stabilization. Preserving IMGT 42/49/50/52 alone
does not establish functional preservation, and extra cysteines are not an
automatic defect. Its combined CDR protection convention also differs from a
universal IMGT-only mask. We must choose the shared protection policy explicitly.

### AbNatiV2

[Upstream documentation](https://pypi.org/project/abnativ/) distinguishes the
unpaired VH2/VHH2 models from the paired model used by the existing app.
It supports VHH generation as well as scoring. Its current package documentation
also mixes NbForge installation instructions with older predictor descriptions.

In the [pinned 2.0.8 CLI](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/__main__.py)
and [Python pipeline](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/vhh_humanisation_functions.py),
the CLI selects VH2/VHH2 but the Python humanization function defaults to VH/VHH.
The CLI and Python defaults also differ in VHH-decrease allowance and objective
weights. Do not allow these defaults to choose scientific behavior implicitly.

The [2.0.9 packaging update](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/commit/413ebd3995f9383bcb225638810d7fa0b5a3dbe4)
changes the version and package-data paths, not the audited VHH search logic.
It is a candidate pin for the new VHH app, not authorization to upgrade the
existing paired runtime. Native exhaustive search has no effective combination
cap: it materializes combinations, scores them and predicts frontier-member
structures. Enhanced endpoint generation is the accepted first scope;
publishing its internal intermediate states would need explicit instrumentation
and equivalence tests because no native trace/callback is exposed.

The [mutation-acceptance code](https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/blob/eb517f1f0b947084cb7e44a54ef34103e9692f5e/abnativ/humanisation/humanisation_utils.py)
applies VHH-loss tolerance relative to the current sequence at each accepted
step, not as a guaranteed total loss bound from the original parent. Forbidden
C/M substitution targets do not protect existing parental cysteines. Enhanced
search publishes an endpoint; exhaustive search returns a frontier rather than
a fixed number of seeded replicas. Additional whole-parent guardrails and
candidate-count controls must not be presented as native semantics.

In that source, native sequence-input VHH humanization invokes NbForge for
solvent-exposure selection and for final parental/candidate structures. Setting
the exposure threshold to zero does not eliminate final structure prediction.
Standalone unpaired VH2/VHH2 scoring does not require that structural pipeline.
A sequence-input interface is therefore not a promise of structure-free compute.
The exact source, predictor, weights and defaults are a later interview decision;
the new release must not silently upgrade the existing paired app.

The inspected [NbForge 0.1.1 dependency declarations](https://gitlab.doc.ic.ac.uk/sormanni-lab/nbforge/-/blob/f5c90aa6a81968759890ae269d4ba137d0a6a61b/setup.cfg)
require NumPy >=2.2.6 and Lightning >=2.5.6, conflicting with the existing paired
image's validated NumPy 1.26.4 and Lightning 2.5.5. A VHH runtime therefore needs
separate compatibility verification. Its native AbNatiV invocation also does
not request NbForge GPU mode; GPU allocation is not evidence every step uses it.

### HuDiff-Nb

The [upstream README](https://github.com/TencentAI4S/HuDiff#hudiff-nb) describes a
distinct nanobody checkpoint and inpainting sampler. In the
[released sampler](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/nanobody_scripts/sample_for_nano_cdr.py),
passing a seed does not suffice: the seed-initialization call is commented out.
Requested samples are not a guarantee of unique valid candidates. Any narrow
compatibility/reproducibility patches need explicit provenance and native tests,
as with the existing HuDiff-Ab integration.

The [native mask](https://github.com/TencentAI4S/HuDiff/blob/bb7636f182699f98c37855dad05a5c6c61b576bd/nanobody_scripts/nanosample.py)
uses a fixed IMGT grid. Inpainting protects native CDRs and four hallmark sites,
not every framework cysteine. Unsupported numbered positions must be rejected,
not silently omitted. Batch size also affects sampling behavior in the released
loop and cannot be treated as an unversioned throughput-only setting.

The native protected CDRs are IMGT 27–38, 56–65 and 105–117, including supported
insertions. Inpainting expands CDR2 to 55–66 and adds the four hallmark sites.
Turning inpainting off does not enable CDR mutation. AbNatiV instead uses AHo
27–42, 57–69 and 108–138 as its native CDRs. A shared conservative policy can
protect the union mapped onto actual prepared-parent residue indices, plus
parental cysteines and hallmark residues. This is an accepted additional
workflow restriction, not native equivalence of the two masks.
For HuDiff, extra protection must preserve parental tokens as well as remove
those sites from sampling; output validation alone is not a generation mask.

The [implementation protocol](https://bio-protocol.org/en/bpdetail?id=5816&type=0)
uses AbNatiV during nanobody fine-tuning. Consequently, favorable AbNatiV scores
are not fully independent validation of HuDiff outputs. Do not manufacture a
native HuDiff humanness score or treat score agreement as biological validation.

### Llamanade and supporting assessments

The paper's original GitHub repository returned 404 during this check, but its
[author-deposited release](https://zenodo.org/records/5575933) remains available.
Its archive is not yet audited for license, runtime, assets or protected-site
controls. This is an integration uncertainty, not evidence against the method.
The [published method](https://pmc.ncbi.nlm.nih.gov/articles/PMC11698024/)
can take sequence input and model structure internally; an uploaded PDB is not
necessarily required. Its burial/contact pipeline is not a small sequence-only
substitution helper.

[TNP](https://github.com/oxpig/TNP),
[NanoMelt](https://pubmed.ncbi.nlm.nih.gov/39772905/) and
[NetMHCIIpan](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/)
are separate assessment integrations, not obligatory consequences of adding
humanization. TNP needs structural processing; NanoMelt predictions have
meaningful uncertainty; HLA analysis needs a chosen panel and construct scope.
Do not submit private sequences to public web servers as an implementation
shortcut. Separate assessment integrations are deferred from the accepted
first-release scope; their future input scope and interpretation are undecided.

### Licensing research deferred

[AbNatiV](https://pypi.org/project/abnativ/) declares CC BY-NC-SA 4.0 and a
noncommercial-use notice. [HuDiff's LICENSE](https://github.com/TencentAI4S/HuDiff/blob/main/LICENSE)
is PolyForm Noncommercial 1.0.0. These notices conflict with the protocol's
software-table labels. Check the exact source, weight and dependency terms;
do not infer permission from public availability or an existing paired app.
The user asked to ignore licensing during this design cycle. Do not infer
permission or reintroduce it as an unresolved implementation interview question.

## Interview decision tree

Release depth, product surfaces, broad VH input scope, manual/CSV entry,
preparation, generation, evaluation, ranking and resource/control policy are
accepted above; licensing is deferred.

The presentation, transfer, failure policy and complete implementation plan
are approved. The preceding antibody display fix must land before nanobody
implementation.

## Approved implementation plan

### 1. Antibody display fix first

Both backend branches currently point to d57de18; only nanobody planning docs
are uncommitted. After approval, checkpoint these docs on the nanobody branch
without mixing them into the antibody fix, then work on
feat/antibody-sequence-analysis. The frontend is already on its corresponding
antibody branch and has no nanobody branch yet.

- New selection publications order vh, vl, vh_pI, vl_pI, vh_vl_pI, then the
  four existing V/J gene columns. Preserve the Polars deduplication/joins,
  pI values, gene assignments, ranking and row order. This change applies to
  newly generated CSVs and website presentation, not only header labels.
- Advance result schema 5 to 6 and preserve immutable schema 2–5 reads and
  their actual sort keys. Historical webpage presentation can show adjacent
  chains and pI casing without rewriting old archives.
- Extend the shared sequence-detail response to independently assign the
  original parent's V/J references and project its germline comparison through
  the native parent-to-humanized alignment. Keep both full sequences, exact
  indices, independent reference-only gaps and unavailable coverage; do not
  infer direct homology between the two germline references.
- Render four biological rows in the requested order, retaining three
  unlabeled difference strips with accessible comparison descriptions. Bold
  the Humanized row, label the final row Germline (parental), and identify both
  before/after V/J genes and species.
  Paired humanization uses the original submitted chain as parent; nanobody
  humanization will supply its saved prepared parent. No-parent standalone
  analysis retains its Input terminology and existing behavior.
- Advance analysis contract 4 to 5, export exact offline OpenAPI for frontend
  generation, and preserve candidate inspection when parent evidence is
  unavailable. No new parent-fetch endpoint or alignment dependency is needed.
- Test native alignment reconstruction across all five schemes, independent
  before/after assignments, indels/tails, missing parent evidence, historical
  sorting and new CSV column order/case. Run backend/frontend offline gates
  and commit the coherent fix in each antibody branch.

### 2. Rebase before nanobody implementation

Rebase the backend nanobody planning branch onto the antibody fix commit,
preserving its documentation. Create the frontend nanobody branch from its
fixed antibody tip. Verify clean worktrees and the intended ancestry; do not
push or force-update remote branches as part of this operation.

### 3. Build preparation and pinned model apps

Implement shared VH preparation using arpeggia, preserving original/prepared
identities and exact internal imputation evidence. Add version-bound local
preview/admission contracts so edited or stale previews cannot submit a
different sequence. Keep this distinct from general analysis's no-trimming
contract.

Add focused package-based AbNatiV2-VHH and HuDiff-Nb apps using existing
execution primitives and reusable assets. Plan to pin the audited AbNatiV
2.0.9 packaging fix and separate NbForge-compatible runtime, and the inspected
HuDiff nanobody source/checkpoint. Implement explicit model preparation,
native position checks, accepted masks, seed/batch semantics and content-bound
publications without upgrading existing paired apps. Verify observational
wrappers and restrictions against native behavior.

### 4. Build the workflow and publication

Implement preparation-bound parallel generation, per-parent exact union,
common VH2/VHH2 evaluation, mutation/germline/pI annotation, three-objective
Pareto ranking and mutation-pattern diversity. Keep tabular work Polars-native
and avoid duplicated reads/conversions. Reuse ranking mechanics only where
their scientific meaning agrees. Implement the accepted no-op/partial/failure
rules and lean result files with frozen scientific/publication identities.

### 5. Integrate service and frontend together

Register the new Tool with shared auth, idempotent admission, owner-scoped
inputs/results, limit snapshots, stages, cancellation, downloads, recovery and
billing attribution. Export exact contracts and deterministic offline fixtures
for the frontend agent. Build manual/CSV input, inline preparation, known
Advanced controls, bounded result table, corrected common popup and standalone
analysis transfer. Retain per-user/source isolation and stale-response guards.

### 6. Verify, document and request review

Commit self-contained milestones for preparation, each app, workflow, service
and coordinated frontend work. Test native mapping/protection/sampling, result
integrity, ranking/partial outcomes, authorization, idempotency and browser
integration; run discovery/help checks, type guidance and repository hooks.
Update specs, ADR/glossary where needed, human-facing instructions and deploy
guidance without duplicating content. Report exact test evidence and any
unverified native/deployment behavior, then request user review. No push,
deployment or paid cloud validation is implied by this plan approval; any
such validation needs an explicit budget/authorization first.

The supplied report's hundreds/thousands of candidates, illustrative
32-construct panel, structural ensembles and deimmunization branch are proposals,
not default counts or requirements.
