# Humanizing antibodies

The humanization workflow runs Sapiens, Humatch, p-AbNatiV2 and HuDiff independently
on complete VH–VL pairs, combines their distinct candidates, and evaluates every
candidate with the three available scorers. Results support experimental panel
selection; model scores do not establish preserved binding or reduced immunogenicity.
CUMAb and single-chain/VHH inputs are not included in this release.

## On the website

1. Open Antibody humanization and choose **Submit a job**.
2. Enter an ID, VH and VL, then Add the pair; or import a CSV with `id,vh,vl`
   columns. Edit or remove highlighted invalid rows before submitting. IDs must
   be unique. Whitespace is removed from sequences and letters are uppercased;
   unsupported amino-acid characters are rejected, not silently deleted.
3. Name the job above the batch editor. Use Advanced to change scientific
   controls. Root seed applies to both seeded methods; Allow CDR mutations
   applies only to methods that support that option.
4. Submit and follow each method's stage on the shared Job page. Models may
   require preparation before execution starts. Independent method failures can
   still yield useful partial results.

CSV imports may be up to 10 MiB, but the normalized batch must obey the limits
shown by the service. The default batch limit is 100 pairs; administrators may
configure up to 200. VH and VL limits are 142 and 126 residues respectively.
Shorter inputs can still be incompatible with an individual method's numbering.

Drafts stay in memory, not browser storage. Leaving or reloading the page can
discard them. After a lost submission response, use Check submission to recover
the same intent instead of creating a duplicate job. Rerun on a Job page loads
its retained inputs into an editable draft; nothing runs until you submit it.
Historical model-specific settings remain intact until you edit shared controls.

## From the CLI

Inspect controls or validate the checked-in example without inference:

```bash
uv run biomodals workflow help humanization
bash examples/workflow/humanization.sh
```

After staging the required model assets and deploying a compatible containing
workflow, submit a paid run with deliberately chosen concurrency limits:

```bash
uv run biomodals workflow run --max-containers 4 --max-gpu-containers 2 \
  humanization -- --input-csv examples/data/sapiens_pairs.csv \
  --hudiff-ab-candidate-count 10 --hudiff-ab-seed 42
```

The CLI prints deployment/run identities and result locations. Container limits
bound concurrent calls, not total work or cost. The website prepares required
runtime assets through its service adapter; direct CLI submission assumes its
documented model-staging prerequisites are already met.

## Sampling and outcomes

- Sapiens uses chain language models and contributes each successive greedy
  refinement (1–5 iterations). Increasing iterations explores a refinement path,
  not independent random replicas.
- Humatch edits toward human V-family and pairing-classifier targets. It returns
  one endpoint per parent, possibly unchanged if targets already hold. Increasing
  its edit limit or thresholds does not request more candidates.
- p-AbNatiV2 optimizes paired nativeness with accessibility/pairing constraints.
  Increase **Sampling attempts per parent** (1–25) for more independent optimizer
  runs; different seeds may still converge to the same sequence.
- HuDiff samples paired frameworks around protected CDRs. Increase its attempts
  (1–25) for more sampling, not guaranteed yield. Invalid and duplicate attempts
  reduce the number of distinct candidates. HuDiff has no native evaluation score.

Increasing these controls can increase generation and evaluation cost. There is
no requirement that each method contribute the same number of candidates.
Root seed affects p-AbNatiV2 and HuDiff, not all four methods. Position controls
remain model-specific: Humatch uses IMGT, p-AbNatiV2 AHo and Sapiens its configured
convention. Allowing CDR changes can alter binding determinants; none of these
controls or sequence scores establishes experimental suitability.

## Choosing candidates

The website opens a bounded page of `selection.csv`. Sort, filter by parent,
show additional columns, or download the original CSV. Sorting changes the view,
not the stored ranking or scientific scores.

Within each parent, `panel_order` suggests a diverse selection order; lower is
earlier. `quality_tier=1` is the first Pareto front. The parent is a control, not
one of the recommended N candidates. Null ranks mean baseline or ineligible,
not a worst numeric score; retain those rows for manual assessment.

Model metrics have different meanings. Positive deltas mean a higher score than
the parent, not proven biological improvement. Nativeness can be negative and
is not a probability; its display bars are scaled within the result, so bar
lengths are not comparable across jobs. Missing values remain missing. Check CDR
preservation and error columns before choosing a characterization panel.

## Downloaded results

New schema 5 archives contain the selection CSV with VH pI, VL pI, VH+VL pI
and four V/J gene columns immediately after `vh`,
one consolidated germline-evidence Parquet, detailed score Parquets,
IMGT mutations, a compact generation ledger, and a digest manifest. The ledger
links generation outcomes to retained candidates, including parental no-ops,
duplicates and rejected attempts. Native model payloads and predicted structures
are not copied into the standard archive. Historical archives keep their original
layout and stored ranking.

Hover genes for matched reference species and approved-therapeutic usage, with
source/date information once above the table. Click sequences for numbering,
CDR and potential-liability annotations, plus one Germline / Diffs / **Input** /
Diffs / Parental alignment. The parent is read from this job's retained inputs;
if unavailable, the germline display remains usable. pIs use the complete supplied
chains; VH+VL treats their direct concatenation as one chain without a linker.
Neither pIs nor genes change ranking. VH/VL checkboxes let you send selected
chains to [Antibody sequence analysis](antibody-sequence-analysis.md), pairing
only within each parental antibody. This immediate local analysis creates no Job
and does not transfer source ranks or pairing scores to new recombinations.

For exact scientific definitions and file fields, see the
[workflow contract](specs/humanization-workflow.md). Deployment operators should
also use the [runbook](deployment/mvp-runbook.md).
