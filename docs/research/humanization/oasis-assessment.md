# OASis assessment for the antibody-humanization app stack

Research date: 2026-09-02

## Recommendation: defer, do not exclude

OASis does **not** need to be included for BioPhi-equivalent Sapiens
humanization. Sapiens generates mutations from its own residue score matrix;
BioPhi invokes OASis separately to evaluate humanness. Omitting OASis therefore
does not change Sapiens mutations or Sapiens score matrices
([BioPhi humanization code](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanization.py),
[OASis code](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanness.py),
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).

Defer OASis from the initial `feat/antibody-humanization` PR stack and consider
it later as a separate, method-agnostic evaluation app. This is a scope and
operational recommendation, not a judgment that the metric lacks scientific
value:

- OASis is a transparent repertoire-prevalence humanness measure and can
  localize non-human-like 9-mers. That makes it useful for comparing outputs
  from different humanizers with one common metric
  ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).
- It neither proposes mutations nor models VH/VL pairing, so it adds no
  humanization method to a stack already containing Sapiens, p-AbNatiV2,
  HuMatch, CUMAb, and HuDiff-Ab
  ([implementation](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanness.py)).
- Full BioPhi OASis adds a 6.48 GB download and roughly 22 GB of unpacked SQLite
  storage. A much smaller official implementation may cover the common relaxed
  score, but exact equivalence needs validation before it can replace BioPhi
  ([database record](https://zenodo.org/records/5164685),
  [BioPhi README](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/README.md),
  [`promb` README](https://github.com/MSDLLCpapers/promb/blob/d11c24b1557d3368d6508ecb4f14b51facaf8863/README.md)).
- Calling the metric “immunogenicity” would overstate its evidence. It is a
  humanness/exposure proxy and explains only part of the variation in a
  heterogeneous clinical ADA dataset
  ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).

Reconsider inclusion when the product explicitly needs a common cross-method
score in the first release. If that need appears, add a separate `oasis`
scoring branch rather than coupling OASis to every humanizer.

## Reproducible upstream snapshot and licenses

- Current BioPhi `main` is
  [`bc59cd17b690a634553ac50a60840b9a89bd21b0`](https://github.com/Merck/BioPhi/tree/bc59cd17b690a634553ac50a60840b9a89bd21b0),
  dated 2025-05-13. BioPhi code is MIT licensed
  ([license](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/LICENSE)).
- The OASis database is immutable Zenodo record
  [`10.5281/zenodo.5164685`](https://doi.org/10.5281/zenodo.5164685), version
  `v1`, published 2021-08-05 under CC BY 4.0. Its one file,
  `OASis_9mers_v1.db.gz`, is 6,484,781,782 bytes with MD5
  `28fa18135b3ef1710970f72b5953cbc5`; BioPhi documents approximately 22 GB
  after decompression
  ([Zenodo record](https://zenodo.org/records/5164685),
  [setup instructions](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/README.md)).
- Zenodo describes the database as generated from 118 million human antibody
  sequences in Observed Antibody Space. The paper says the source OAS snapshot
  was downloaded in November 2019, so it is a fixed historical reference, not a
  live view of OAS
  ([database record](https://zenodo.org/records/5164685),
  [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).
- The article is CC BY-NC 4.0; that article license is distinct from the MIT
  code and CC BY database licenses
  ([paper copyright](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).

Any app must retain the MIT and CC BY notices and attribute the database. These
licenses do not remove the need to review the terms of any optional external
predictor such as netMHCIIpan. This note is not legal advice.

## What OASis computes

For each numbered variable-region chain, BioPhi enumerates every overlapping
9-mer and queries exact matches in the SQLite `peptides` table. For the relevant
heavy or light class, its denominator counts OAS subjects having at least
10,000 complete sequences. The peptide-hit query additionally excludes the
`Corcoran_2016` study, although the denominator query in current code does not
apply that exclusion. A peptide's prevalence is the resulting subject-hit count
divided by the heavy- or light-specific denominator; occurrence count is
retained separately
([implementation](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanness.py)).

OASis identity at threshold `t` is:

```text
number of query 9-mers observed in at least fraction t of eligible subjects
--------------------------------------------------------------------------
                         total query 9-mers
```

BioPhi names four prevalence thresholds: loose 1%, relaxed 10%, medium 50%,
and strict 90%. Its default CLI threshold is relaxed 10%, while accepting any
percentage from 1 through 90
([thresholds](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanness.py),
[CLI](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/cli/oasis.py)).

For a complete antibody, BioPhi combines heavy and light by summing their
human and total peptide counts. This is an aggregate of two chain-local
lookups, not paired-sequence conditioning and not a model of the VH/VL
interface. BioPhi also reports chain and pair percentiles relative to a
reference set of 544 therapeutic antibodies, peptide-level subject and
occurrence counts, non-human peptide localization, closest human germlines,
germline content, and repertoire residue frequencies
([data model](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanness.py),
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).

The BioPhi CLI accepts FASTA or PDB input, supports Kabat, Chothia, IMGT, and
AHo numbering plus several CDR definitions, uses a multiprocessing pool, and
writes an XLSX report. It warns and skips invalid or duplicate inputs, and its
worker wrapper catches per-antibody exceptions and removes failed results
([CLI](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/cli/oasis.py)).
A future Biomodals app should instead use strict wide-CSV pair validation and
all-or-nothing accounting so a failed row cannot disappear.

## Scientific usefulness

OASis has three useful properties beside learned humanizers:

1. **Interpretability.** Every score decomposes into exact 9-mer membership and
   observed subject prevalence, so a scientist can see which local windows
   drive the result.
2. **Method-independent comparison.** The same frozen database can score the
   parental and final sequences from every humanizer without using that
   humanizer's own objective.
3. **Broad repertoire support.** It was built from 118 million sequences rather
   than a small germline library, retaining somatically mutated human patterns
   ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/),
   [database record](https://zenodo.org/records/5164685)).

In the paper's human-versus-nonhuman benchmark, relaxed OASis identity achieved
94.4% accuracy and ROC AUC 97.2%. On 217 therapeutics with reported treatment-
emergent anti-drug-antibody incidence, medium OASis identity had Pearson
`r=-0.53` and `R²=0.28`; this was comparable to the best methods in that study
but left most clinical variation unexplained
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)). Those results
support screening and ranking utility, not a stand-alone safety decision.

## Humanness is not immunogenicity

OASis asks whether short antibody sequence windows have been observed across
human repertoires. It does not directly model antigen processing, HLA binding,
T-cell activation, aggregation, formulation, dose, route, patient disease,
immune status, assay sensitivity, or treatment duration. All of those can
affect clinical ADA observations, and the paper emphasizes that the available
therapeutic studies are heterogeneous
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).

An exact 9-mer absent from many OAS subjects is therefore “non-human-like under
this repertoire threshold,” not “immunogenic.” Conversely, a common human
9-mer is not proof of safety. Appropriate output labels are `oasis_identity`,
`oasis_percentile`, and `oas_subject_prevalence`; avoid `immunogenicity_score`
or `risk_probability`.

The distinction also matters for Sapiens and p-AbNatiV2: their learned
nativeness scores and OASis repertoire identity are related humanness signals,
but none is a clinical immunogenicity probability. Experimental binding,
function, developability, and immunogenicity assessment remain downstream.

## Incremental value beside the planned apps

| Planned app | What OASis adds | What OASis cannot add |
| --- | --- | --- |
| Sapiens | A transparent score that is external to Sapiens' mutation objective; the BioPhi paper used it to evaluate generated variants. | It is not needed to reproduce the Sapiens humanization sequence or score matrices, and both methods ultimately draw on OAS-derived human repertoires. |
| p-AbNatiV2 | Exact peptide explanations and one common metric across tools. | No pair-conditioned reconstruction or learned pairing likelihood; the pair score merely pools chain-local 9-mers. |
| HuMatch, CUMAb, HuDiff-Ab | A frozen, common post-hoc humanness yardstick could reduce reliance on each method's self-score. | No primary head-to-head evidence identified here establishes that OASis improves these methods' candidate selection, and it supplies no mutations, structures, binding, or developability prediction. |

The BioPhi paper directly compares OASis with its contemporary humanness
benchmarks, but it predates some of the planned methods. The absence of a
published head-to-head comparison for a newer method should be recorded as
uncertainty, not converted into a claim of superiority or redundancy
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/)).

## Operational cost of full BioPhi OASis

Full OASis requires no neural-network weights or GPU. Its costs are database
distribution, SQLite random access, and per-chain numbering/query work:

- 6.48 GB network transfer for the published gzip and about 22 GB persistent
  storage after decompression;
- a checksum-verified setup step and a read-only staged Volume, because baking
  22 GB into every application image is disproportionate;
- database-version attribution and lifecycle management; and
- CPU parallelism. The paper reports 1,000 antibodies in 14 minutes on an
  eight-core personal computer, about 1.19 antibodies/second in aggregate on
  that reference system. Modal cold start and Volume access are not included in
  that figure
  ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837241/),
  [database record](https://zenodo.org/records/5164685)).

For expected workloads below 100 pairs, steady-state query time is likely
modest, but moving and mounting the full database dominates the complexity.
There is no result-cache requirement: exact lookup is deterministic for a
fixed database, AbNumber version, and sequence.

## Lightweight official path: `promb`

BioPhi's current README points users who only need relaxed (>10% subject)
OASis identity to the authors' `promb` package
([BioPhi README](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/README.md)).
Current `promb` `main` is
[`d11c24b1557d3368d6508ecb4f14b51facaf8863`](https://github.com/MSDLLCpapers/promb/tree/d11c24b1557d3368d6508ecb4f14b51facaf8863),
and PyPI release [`promb==1.0.2`](https://pypi.org/project/promb/1.0.2/) has a
39.3 MB wheel with SHA-256
`7c14f54ccde18975fced953c54f7acd3c55918b2b14b80f20a0b35f3c721b46f`.
The code is MIT licensed
([license](https://github.com/MSDLLCpapers/promb/blob/d11c24b1557d3368d6508ecb4f14b51facaf8863/LICENSE)).

`promb` bundles a 14,018,806-byte gzip containing 5,725,303 unique 9-mers
reported in at least 10% of human OAS subjects. It loads them into a Python
`frozenset`; `compute_peptide_content` returns the fraction of overlapping
query 9-mers found by exact membership
([database implementation](https://github.com/MSDLLCpapers/promb/blob/d11c24b1557d3368d6508ecb4f14b51facaf8863/promb/db.py),
[bundled peptide file](https://github.com/MSDLLCpapers/promb/blob/d11c24b1557d3368d6508ecb4f14b51facaf8863/promb/resources/OASis_9mers_v1_10perc_subjects.txt.gz),
[usage](https://github.com/MSDLLCpapers/promb/blob/d11c24b1557d3368d6508ecb4f14b51facaf8863/README.md)).
The 39.3 MB wheel is larger because it also includes unrelated human proteome
resources.

This removes almost all deployment cost for the common relaxed identity, but
it is not a drop-in implementation of the full BioPhi report:

- it cannot change the OAS subject threshold to 1%, 50%, 90%, or an arbitrary
  value;
- the flattened peptide set does not expose subject prevalence, occurrence
  count, or BioPhi's database-backed evidence rows;
- its `oasis` CLI outputs one identity per input sequence rather than BioPhi's
  paired antibody aggregation and percentile report; and
- BioPhi's full query applies heavy/light-specific subject eligibility, while
  the packaged set is not documented as preserving chain-specific metadata
  ([`promb` CLI](https://github.com/MSDLLCpapers/promb/blob/d11c24b1557d3368d6508ecb4f14b51facaf8863/promb/cli.py),
  [BioPhi query](https://github.com/Merck/BioPhi/blob/bc59cd17b690a634553ac50a60840b9a89bd21b0/biophi/humanization/methods/humanness.py)).

The authors describe `promb oasis` as fast OASis computation, but the Python
README also calls `compute_peptide_content` “OASis-like.” Before claiming
BioPhi identity, run an oracle corpus through full BioPhi at the relaxed
threshold and `promb`, covering heavy, kappa, lambda, identical 9-mers appearing
in different chain classes, short/invalid sequences, and complete pair
aggregation. Record any discrepancy and, if necessary, version the lightweight
metric as `promb_relaxed_oasis_like_identity` rather than `oasis_identity`.

## Conditions for a later OASis app

Add OASis to the stack only if users want a shared cross-humanizer evaluation
artifact. Before implementation:

1. Decide whether relaxed identity alone satisfies the product need. If yes,
   validate and prefer the small pinned `promb` resource. If no, stage the full
   checksum-verified SQLite database on a dedicated read-only Volume.
2. Define strict wide CSV `id,vh,vl` input with complete pairs and a 1,000-pair
   ceiling, matching the humanization apps.
3. Return chain and pair identity, threshold, percentile where scientifically
   valid, and a long peptide-evidence table. Never silently skip a row.
4. Label it humanness, not immunogenicity, and preserve the fixed database
   version in the run manifest.
5. Benchmark one pair and 100 pairs on CPU; do not add GPU or result caching.
6. Keep OASis scoring independent of each generator so a failure or database
   cold start cannot prevent the requested humanization result.

On current evidence, those capabilities are worth a later focused app but do
not justify delaying or enlarging the first antibody-humanization PR stack.
