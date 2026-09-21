# Antibody sequence analysis and humanization annotations

Status: accepted and implemented for offline verification, 21 September 2026.
Deployment and paid scientific verification require separate authorization.
The [user guide](../antibody-sequence-analysis.md) owns website instructions;
this document owns scientific definitions and implementation boundaries.

## Scope and ownership

Antibody Sequence Analysis is an immediate authenticated operation, not a Service
Job. It computes sequence metrics, nearest germline references and display
annotations for single variable domains or explicit VH/VL pairs. It neither
humanizes sequences nor measures binding, immunogenicity or developability.
Nanobody humanization is deferred. Recognition is H/K/L, not proof of VHH.

- `helper/antibody.py` owns provider-neutral ordinary Python results. Optional
  native imports stay inside scientific functions, never `helper/__init__.py`.
  analyze_chain and analyze_pair are the simple Python entry points; batch
  consumers share lower-level chain and pair-pI work.
- service/antibody_sequence_analysis owns FASTA, authentication, public reference
  I/O and ephemeral HTTP responses. No ToolRegistration, Modal preflight,
  execution ledger, saved user results or Job admission.
- workflow/humanization/germlines.py owns frozen candidate-bound evidence and
  post-ranking joins. Only the existing CPU annotation image installs arpeggia;
  unrelated app images do not acquire that dependency.
- API extras pin arpeggia 0.10.1 and Biopython 1.86. Arpeggia requires
  polars>=1.44.2,<2; no dependency override or Polars downgrade is used.

## Input and failure contract

POST /api/v1/antibody-sequence-analysis/analyze accepts one or two independent
groups, each containing an id and fasta. Supported forms:

- One unsuffixed record containing VH:VL, in that order.
- Two records with matching _vh and _vl ID suffixes, in either record order.
- An unsuffixed single domain. Recognized H fills VH; K/L fills VL. An
  unrecognized canonical sequence retains metrics under unassigned.

The ID is the first whitespace-delimited header token; descriptions are ignored.
IDs must be unique within a group; the same ID in independent groups is allowed.
Group identifiers are distinct. Missing suffixed partners stay errors, not
standalone entries. Wrong chain roles are reported without swapping or assigning
a misleading combined pI.

Only uppercase conversion and whitespace removal are allowed. Accept the 20
canonical amino acids, reject other symbols, and never silently trim tails.
Limits: **1000 output entries per group**, **512 residues per supplied chain**,
**4 MiB total HTTP request before parsing**. Count Cartesian outputs before
frontend expansion. The API independently limits records/output entries before
native computation; these are not Sapiens's 142/126-residue admission limits.

Responses retain per-entry and per-group issues. Bad entries do not discard valid
ones. An unsplittable group gets a group error while the other remains analyzable.
Numbering/germline failures make those annotations unavailable but retain physical
metrics for valid canonical sequences.

GET /options supplies authoritative limits, schemes and versions.
POST /sequence accepts one sequence/scheme for lazy details. All routes use
existing authentication; POST also uses Origin/CSRF protection. Responses are
private/no-store. No endpoint logs or persists user sequences.

## Physicochemical metrics

Use the pinned Biopython implementation on the full supplied sequence, including
tags/tails. Values remain full precision in data and CSV.

| Field | Definition |
| --- | --- |
| pi | Theoretical sequence pI |
| molecular_weight_kda | Average unmodified mass, not monoisotopic, divided by 1000 |
| gravy | Kyte–Doolittle GRAVY, dimensionless |
| extinction_reduced | Reduced-chain molar ε280, M⁻¹ cm⁻¹ |
| extinction_oxidized | ε280 with Biopython's cystine assumption, M⁻¹ cm⁻¹ |
| vh_vl_pi | Literal VH followed by VL, one continuous sequence without a linker |

Label the combined value **VH+VL sequence pI**. It is not mean chain pI,
two-chain charge-sum pI, full-IgG pI or an unspecified scFv/linker estimate.
Extinction assumptions do not establish disulfide bonds. No unprovided constant
regions, glycans or conjugates are included.
Sources: [Biopython 1.86 ProtParam](https://github.com/biopython/biopython/blob/biopython-186/Bio/SeqUtils/ProtParam.py),
[terminal accounting](https://github.com/biopython/biopython/blob/biopython-186/Bio/SeqUtils/IsoelectricPoint.py).

## Germline assignment and frequency

Use arpeggia's all-species, independent V/J local BLOSUM62 comparisons with fixed
IMGT matching. These are nearest-reference similarities, not proven ancestry or
the antibody's organism. Humatch target-family probabilities are a different
model output, not these individual gene assignments.

Keep all best hits with species/gene/allele/accession/reference identity, score,
known-pair counts, coverage and query/IMGT spans. Main columns contain gene names
without alleles, deduplicated and slash-joined in deterministic order. Hover shows
all tied **matched reference species** and gene-level usage. Never pick an
arbitrary winner or turn missing assignments into a zero score.

Reference identity: IMGT 202636-7 plus arpeggia's small llama supplement.
Coverage is human/mouse/rat/rabbit H/K/L and alpaca/llama H. Llama has six V and
five J references, not a complete repertoire. Native APIs/conventions were
researched in [0.10.0](https://github.com/y1zhou/arpeggia/tree/08b32197e9622ededa4b1254007be0b208181c7a)
and exercised with compatible [0.10.1](https://github.com/y1zhou/arpeggia/tree/v0.10.1).
The old release's Polars 1.38.1 constraint was not overridden.

### Approved therapeutic cohort

The [Thera-SAbDab CSV](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/static/downloads/TheraSAbDab_SeqStruc_OnlineDownload.csv)
has sequences/clinical labels, not V/J assignments. Select Highest_Clin_Trial
(Feb '25) exactly Approved. Exclude Approved (w) and Approved (withdrawn);
do not additionally require Est. Status Active. This is marked-approved in a
snapshot, not a current regulatory assertion.

Pool primary/secondary bispecific fields. Treat native na values as missing
before uppercasing; otherwise they become a false canonical NA peptide.
Remove whitespace, uppercase and exact-deduplicate supplied sequences separately
for heavy and pooled light, not by trimmed numbered domains.

For each chain/segment, collapse reference/allele duplicates to distinct
(full reference species, gene) assignments. Split one chain's credit equally
among these ties, then divide by the role's total unique canonical chains.
Unassignable chains remain in the denominator without invented credit. An
assigned gene absent from a usable reference has frequency zero; missing
reference data yield unavailable frequencies.

These are not natural repertoire usage, clinical success, prescription prevalence
or experimentally validated ancestry. Cite
[Thera-SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/about/).

### Build-once cache

The file under BIOMODALS_CACHE_DIR/antibody-sequence-analysis/
therapeutic-gene-usage.json contains public counts plus download time, source
SHA-256, engine/reference and counting-policy identities. No user sequences or
results enter it; the source CSV is not distributed with this repository.

Build only when absent. Share overlapping initialization and atomically publish a
completed file. No TTL, scheduled refresh, HTTP revalidation or silent replacement.
Initial failure leaves usage unavailable; a later request may retry. Unreadable
or incompatible existing data remain untouched with an explicit error, requiring
deliberate operator repair. Reference I/O/CPU work stays outside the archive-cache
executor and job polling.

Show source, cohort, download date/hash, engine/reference version and counting
explanation once in the FAQ above results, shared by both tables. Do not repeat
dates/source labels in gene cells.

## Sequence inspector

Clicking a sequence opens a wide dialog; load details only on opening or scheme
change. Default IMGT; offer IMGT, Kabat, Chothia, Martin and AHo with matching CDR
conventions. No independent CDR selector. Display changes do not reinterpret
workflow IMGT preservation or ranking.

Use native residue order and zero-based input_index, not sorted numeric labels.
domain_span is half-open. Keep original sequences and visibly identify unnumbered
tails. Show CDR1/2/3 separately; do not impute, graft or edit residues.

Liability intervals are zero-based, half-open and overlap-preserving across the
entire supplied chain. Follow active
[LAMBS v0.12.0 rules](https://github.com/dcroote/lambs/blob/61d6f28f3c666778fc06ed05a8a2d50faef4d715/index.html#L3052):

| Kind | Detection / highlighted positions |
| --- | --- |
| Odd cysteine count | All C positions when the total count is odd |
| Methionine | Every M |
| N-glycosylation | N-X-S/T where X is not P; all three residues |
| Asn deamidation | NG, NS, NT, NN, NH; both residues |
| Asp isomerization | DG, DS; both residues |
| Acid cleavage | DP; both residues |
| N-terminal glutamine | Q at the supplied sequence's first position only |
| Hydrophobic patch | All positions in seven-residue KD windows with mean ≥1.6 |

No W oxidation rule: it is disabled upstream. Label **potential sequence
liabilities**, not measured modifications, accessibility or aggregation risk.
Odd cysteine count cannot identify an unpaired bond. See
[third-party notices](../third-party/antibody-analysis.md).

## Workflow publication and compatibility

New schema **4** adds vh_v_gene, vh_j_gene, vl_v_gene and vl_j_gene to ordinary
selection.csv, plus one typed germlines.parquet. Each evidence row binds parent,
candidate, chain role and exact sequence SHA-256. Scientific versions record
arpeggia/reference identity; the manifest binds the sidecar bytes. CLI and website
download the same scientific tables.

The existing per-candidate CPU Provider Call batches independent ANARCI and
arpeggia Tasks. Each retains its own publication and failure status; a Successor
reuses successful evidence and runs only missing work. No extra Provider Call,
generator, GPU or therapeutic-network dependency is added to a normal run.
Labels join **after ranking**. Missing genes do not change evaluation_complete;
unavailable IMGT checks retain partial-result/eligibility behavior while
independent gene evidence remains available.

Keep historical schema 2/3 readers and original CSV/archive bytes. Do not backfill
genes or rerank old jobs. Explicit local inspection or transfer to analysis is
allowed on historical rows. New SelectionPage metadata contains only its page's
candidate assignments and one reference provenance block. Gene sorting remains
full-table Polars sorting before paging. Verify sidecar evidence against selected
sequence identities. The [publication ADR](../adr/0010-humanization-publication-contract.md)
records why frozen assignments and service-owned frequencies are separate.

## Frontend interaction

Replace VH/VL copy buttons with checkboxes; retain candidate-ID copying.
Selection identity is (parent_id, role, exact sequence), not row/candidate index.
Keep selections across pages/filters with per-parent totals and Clear.

For each parent with both roles selected, emit its VH×VL product. Never cross
parents. With only one role selected, emit standalone chains without invented
partners. Count products/singletons before expansion, maximum 1000. Keep origins
locally but never transfer source ranks or pairing scores to recombinations.

Analyze selected sequences transfers FASTA in memory and runs directly, without
confirmation. Normal standalone input requires explicit Analyze. No sequences in
URLs, browser storage or My Jobs. Mounted reauthentication preserves input;
reload loses it. Recompute on each explicit new analysis request.

Two groups have independent tables, not matched-ID comparisons/deltas. Share
column visibility, with independent sorts and 50-row browser pages. Order ID,
VH pI, VL pI, VH+VL pI, four genes, then remaining chain metrics. Unassigned
sequences retain honestly labeled metrics. Display pI/kDa to two decimals, GRAVY
to three, EC as integers and usage percentages to one; retain raw sorting/CSV.
Humanization keeps backend sorting/paging. No mobile-specific scope.

Configuration guidance explains independent generation, exact-pair union and
cross-evaluation. Describe Sapiens successive greedy refinements (1–5), Humatch's
single possibly unchanged endpoint, p-AbNatiV2 optimizer attempts (1–25), and
HuDiff stochastic attempts (1–25). More sampling costs more but guarantees
neither unique candidates nor quality. Root seed affects p-AbNatiV2/HuDiff;
CDR controls only supported models; position conventions stay model-specific.
See the [humanization guide](../humanization.md#sampling-and-outcomes).
No scientific defaults or allowed knobs change.

## Verification and rollout

Offline tests cover native numbering, metric units, overlapping motifs/tails,
roles, species/gene ties, exact Approved filtering, missing na, build-once/corrupt
caches, partial inputs, auth/body bounds, no Job admission, identity rejection,
historical schemas and unchanged ranks. The 11,400-candidate website envelope
remains below the 32 MiB CSV bound with the four new scalar columns.

Local measurements on 21 September 2026, not deployment latency promises:

- Cached public CSV: 636,357 bytes, 1,133 rows, SHA-256
  cc9402788297ed7b6ab3d676daa21953a6d28ebbd9cc78ec7248b19f74d49d6d.
  Exact Approved: 197 therapeutics, 204 unique VH, 194 VL. Including withdrawn
  labels would incorrectly give 207 VH/197 VL.
- Arpeggia 0.10.1 built 222 accepted-cohort gene-usage records in 0.408 s from the
  already downloaded CSV, excluding network transfer.
- Two 1000-entry groups of repeated public pairs, three runs: analysis median
  0.039 s, serialization median 0.019 s, response 8,024,355 bytes.
- Two 1000-entry groups of diverse public sequences: analysis median 2.074 s,
  serialization median 0.024 s, response 7,190,767 bytes. Peak process RSS
  224,288 KiB; observed maximum event-loop lag 0.081 s. Includes assembly,
  excludes HTTP transfer and browser rendering.

Benchmark harnesses/data stay temporary. Rollout requires updated API extras and
service, frontend build, and a new containing humanization workflow deployment
plus compatible version pin. No historical migration or standalone generator
redeploy is needed. No automatic reference refresh or paid run is included.
