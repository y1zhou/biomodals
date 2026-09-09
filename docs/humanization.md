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

- Sapiens contributes the design after each iteration, before exact deduplication.
- Humatch can successfully return an unchanged parent if its targets are met.
- p-AbNatiV2 Sampling attempts means independent optimizer runs; different seeds
  may converge to the same sequence.
- HuDiff's candidate count is a sampling-attempt budget, not guaranteed yield.
  Invalid and duplicate attempts reduce the number of distinct candidates.

Increasing these controls can increase generation and evaluation cost. There is
no requirement that each method contribute the same number of candidates.

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

Current schema 3 archives contain the selection CSV, detailed score Parquets,
IMGT mutations, a compact generation ledger, and a digest manifest. The ledger
links generation outcomes to retained candidates, including parental no-ops,
duplicates and rejected attempts. Native model payloads and predicted structures
are not copied into the standard archive. Historical archives keep their original
layout and stored ranking.

For exact scientific definitions and file fields, see the
[workflow contract](specs/humanization-workflow.md). Deployment operators should
also use the [runbook](deployment/mvp-runbook.md).
