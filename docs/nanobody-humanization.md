# Nanobody humanization

Generate a diverse set of single-domain candidates with **AbNatiV2-VHH** and
**HuDiff-Nb**, then evaluate every unique candidate with both human-VH and VHH
nativeness models. The unchanged prepared parent is included as a reference.
Higher scores do not establish retained binding, solubility, autonomous
single-domain behavior or reduced clinical immunogenicity.

## Submit sequences

Use the Nanobody humanization Tool to add an ID and VH sequence manually, or
upload a CSV with exactly `id,vhh` columns. The website defaults to 100 parents
per Job; the server can raise this to 200. CSV uploads are limited to 10 MiB
and original sequences to 512 residues each.

**Prepare sequences** shows the domain that will be sent to the models. Check
it before submitting. Input edits invalidate that review. Correct or remove
invalid rows; preparing sequences is local and does not submit a scientific
Job. Any recognizable, compatible VH domain is eligible, not only camelid VHH.
The VHH-oriented models are not equally validated for every possible VH.

The workflow protects its fixed CDR-mask union, parental cysteines and IMGT
hallmark positions 42, 49, 50 and 52. Selecting another numbering scheme in a
sequence popup changes the annotation display, not this design policy.

## Tune candidate generation

- **HuDiff-Nb sampling attempts per parent** defaults to 10 and accepts 1–25.
  More attempts explore more designs but increase compute. Duplicates, no-ops
  and rejected samples are not replaced, so this is not a guaranteed yield.
- **Root seed** controls HuDiff-Nb's unchanged sampling and an independent,
  deterministic AbNatiV exploration stream per parent.
- **AbNatiV2-VHH** defaults to enhanced search: one best-effort endpoint per
  parent, possibly unchanged. Turn on **Explore more candidates** to evaluate
  combinations of native-allowed substitutions. Small spaces are evaluated
  completely; larger ones are sampled across mutation counts. All passing
  designs enter the common ranked table, not a separate top-N shortlist.
- **Exploration budget per parent** defaults to 1,000 distinct nonparent
  evaluations, with a range of 1–5,000. Parents × budget must not exceed
  10,000 per Job. For example, 100 parents can use a budget of at most 100
  each. This allowance is not a predicted yield, runtime or cost: rejected
  designs are not replaced, and HuDiff plus final evaluation add work.
- **Solvent-exposure screening** is on by default at threshold 0.15. Turning
  it off permits consideration of buried positions and bypasses structure
  prediction; the protected CDR/cysteine/hallmark policy still applies.
  The workflow does not calculate unused endpoint structural reports.
- **Allowed VHH-score decrease** is relative to the current sequence at each
  enhanced-search step, but relative to the prepared parent during exploration.
  It is not an absolute VHH score floor and is not imposed on HuDiff outputs.

Both generators run independently within the Job's snapshotted provider limits.
With a large batch and limited GPU slots, one method may start before the other;
the shared scheduler does not guarantee simultaneous progress from both.
Required model files are prepared and verified on CPU before GPU work starts.
Initial preparation may take time; cancelling does not submit replacement jobs.

## Interpret results

The main `selection.csv` lists the VH sequence, pI, V/J genes, mutation count,
raw VH2/VHH2 scores, parent-relative deltas and per-parent panel ordering.
The parent appears first and is unranked. Complete candidates are ordered by
three-objective Pareto tiers: higher VH2, higher VHH2 and fewer mutations.
Within each tier, mutation-pattern diversity determines panel order, with
deterministic score/identity tie-breaks. No weighted composite score is used.

Scores are not probabilities and are not clipped to zero or one. Missing
scores are null; candidates missing either objective remain available but
unranked. Negative score deltas are visible rather than automatically rejected.
Sorting the webpage changes the view, not the scientific ranking.

Open a VH cell for numbering, CDR/liability annotations, germline alignment
and comparison with the **saved prepared parent**. Select VH sequences across
pages and send them to [Antibody sequence analysis](antibody-sequence-analysis.md)
as standalone entries. This transfer does not create VH–VL combinations or
carry humanization ranks. Its advertised entry limit is enforced without
truncating selections.

The ZIP contains the selection CSV, consolidated generation, germline and
IMGT mutation tables, detailed scores under `scores/`, and a content-bound
manifest. Exploration adds compact per-parent search counts, coverage and seed
identity to that manifest, not extra columns or rejected-design tables. Parent
structure is used only when screening needs it; the ZIP contains no native trees.

If one generator or evaluation fails, usable results remain available with a
Partial outcome. If every generator fails, the Job fails before scoring.
An unchanged successful design is a valid no-op. When the whole result contains
only prepared parents, the page says no new designs were produced. Rerunning loads original
inputs into an editable form and requires a fresh preparation review and
explicit submission. **Retry fetching results** rebuilds local result packaging
only; it does not rerun models.

## CLI and rollout

Install the API extra for local native preparation and use the normal workflow
CLI. This example performs no cloud work:

```bash
uv sync --extra api
uv run biomodals workflow run --dry-run nanobody_humanization -- \
  --input-csv examples/data/nanobody.csv
```

For a scientific run, remove `--dry-run` and choose the containing deployment
with the normal workflow options. The existing global `--max-containers` and
`--max-gpu-containers` control provider concurrency. The CLI stages required
models before launching the scientific coordinator.

Roll out the matching API and frontend together. Deploy
`nanobody_humanization`, pin its exact version with
`BIOMODALS_NANOBODY_HUMANIZATION_APP_VERSION`, and restart the API before
exposing the new frontend. Its dedicated AbNatiV runtime does not upgrade the
paired workflow. `BIOMODALS_NANOBODY_HUMANIZATION_MAX_PARENTS` controls the
website batch limit; the active Job limit is managed through the shared
Administrator settings. Per-Job provider limits are documented in the
[service contract](specs/api-tool-service.md#provider-call-limits).
The exploration release changes scientific identity and requires a new containing
workflow deployment even for enhanced mode. Admission checks for its new
`abnativ2_vhh_generate` entrypoint without starting compute; an older pin is
rejected with deployment guidance. Historical outputs remain unchanged and
downloadable; do not resume old scientific plans against the changed implementation.

Offline tests do not validate native GPU inference or image installation. A
separately authorized deployment and smallest native smoke run are required
before claiming production readiness. See the
[deployment guide](../deploy/README.md) and
[implementation specification](specs/nanobody-humanization.md).
