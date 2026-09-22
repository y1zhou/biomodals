# Antibody sequence analysis and humanization annotations

Status: accepted and implemented for offline verification, amended 22 September 2026.
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

GET /options supplies authoritative limits, schemes and versions. The frontend
requires analysis version 4 before analysis or sequence-detail requests, so an
older API cannot be mislabeled as the current scientific/display contract.
Full-sequence copying remains available during a version mismatch.
POST /sequence accepts one sequence/scheme and optional parental_sequence for
lazy details. Both supplied sequences use the same canonical-residue/length
validation. All routes use
existing authentication; POST also uses Origin/CSRF protection. Responses are
private/no-store. No endpoint logs or persists user sequences.

## Physicochemical metrics

Use the pinned Biopython implementation on the full supplied sequence, including
tags/tails. Values remain full precision in data and CSV.

| Field | Definition |
| --- | --- |
| pi | Theoretical sequence pI |
| molecular_weight_kda | Average unmodified mass, not monoisotopic, divided by 1000 |
| gravy | Black–Mould GRAVY via Biopython's BlackMould scale, dimensionless |
| vh_vl_pi | Literal VH followed by VL, one continuous sequence without a linker |
| germline_pi | Each chain's full representative V-reference followed by J-reference, with no D/junction sequence or linker |

Label the combined value **VH+VL pI**. It is not mean chain pI,
two-chain charge-sum pI, full-IgG pI or an unspecified scFv/linker estimate.
No unprovided constant regions, glycans or conjugates are included. Analysis
version 4 retains BlackMould, germline pI and native local alignments, with a
common-axis parental comparison in the sequence inspector;
extinction coefficients are not calculated.
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
arbitrary winner for those assignments or turn missing assignments into a zero score.

The scalar germline pI and displayed alignment use arpeggia's deterministic
display representative (hits[0]) independently for V and J. This is a display
choice among equal scores, not stronger evidence of ancestry; disclose it near
the table and identify the selected references and tie counts in the inspector.
Compute pI on alignment.reference for the full V and J segments, including
unaligned reference ends, not on the gapped alignment strings. Do not average
ties or chain pIs. Missing V/J assignments or noncanonical reference residues
yield null rather than trimming residues or fabricating a sequence. All tied
reference evidence remains available in the gene popovers.

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
workflow IMGT preservation or ranking. Include Copy sequence for the full
normalized input; clicking outside the dialog dismisses it. Place the CDR1/2/3
and potential-liability legends to the right of Copy sequence, without the
repeated numbering-convention paragraph.

Use native residue order and zero-based input_index, not sorted numeric labels.
domain_span is half-open. Keep original sequences and visibly identify unnumbered
tails. Show CDR1/2/3 separately; do not impute, graft or edit residues.

Use one common-axis table labeled Germline, Diffs, **Input**, with both V/J
reference names/species and tie notes above it. The backend projects native local
V/J columns onto the full original input, preserving native gaps and operations;
it does not parse terminal output or realign the references. Unmatched junction
positions are implicit gaps with blank differences, not an inferred D gene.
Outer uncovered positions are blank. Input is bold and retains its numbering,
CDRs, liabilities and visibly unnumbered tails without duplicating the sequence.

On humanization results append Diffs and Parental rows. Fetch the exact same
parent ID/chain from the existing owner-scoped retained-input endpoint lazily
when inspecting a sequence, not on table opening or pagination. Keep that query
only for the mounted result's lifetime. Missing retained input leaves ordinary
inspection/copying usable with a parental-comparison-unavailable notice; shared
authentication failures still trigger reauthentication.

Compare the full parent/input with arpeggia's native global BLOSUM62 alignment
(gap-open 10, extension 0.5). Both Diffs rows describe Input relative to their
reference. Project the two comparisons onto one input axis; keep independent
germline-only/parent-only columns separate rather than implying reference-to-
reference homology. This display does not replace the workflow's IMGT mutation
counts or preservation policy. No browser alignment calculation is needed.

SequenceDetail contains germlines (reference identities/names and tie counts)
and alignment (equal-width germline/germline_diffs/input strings, nullable
parental/parental_diffs, and input_indices). Input indices are zero-based in the
full supplied sequence, null for reference-only gap columns. Numbering failures
return a null alignment rather than fabricate one. Operations are blank for exact matches, + for input
insertions, - for deletions, : for positive-BLOSUM62 substitutions and x for
other mismatches, following the pinned
[arpeggia display contract](https://github.com/y1zhou/arpeggia/blob/v0.10.1/docs/antibody-numbering.md#display-and-antibody-alignments).

Liability intervals are zero-based, half-open and overlap-preserving across the
entire supplied chain. Use the following selected
[LAMBS v0.12.0 rules](https://github.com/dcroote/lambs/blob/61d6f28f3c666778fc06ed05a8a2d50faef4d715/index.html#L3052):

| Kind | Detection / highlighted positions |
| --- | --- |
| Odd cysteine count | Non-conserved C positions when the full-input count is odd |
| Methionine | Every M |
| N-glycosylation | N-X-S/T where X is not P; all three residues |
| Asn deamidation | NG, NS, NT, NN, NH; both residues |
| Asp isomerization | DG, DS; both residues |
| Acid cleavage | DP; both residues |

Exclude conserved Cys at native IMGT 23 and 104 from odd-count markers,
using their original input indices even when another display scheme is selected.
These conserved positions apply to both H and L variable domains; see
[IMGT V-domain anchors](https://pmc.ncbi.nlm.nih.gov/articles/PMC3358611/).
If numbering fails, no conserved positions are inferred. N-terminal glutamine,
hydrophobic patches and W oxidation are not screened. Label **potential sequence
liabilities**, not measured modifications, accessibility or aggregation risk.
Odd cysteine count cannot identify an unpaired bond. See
[third-party notices](../third-party/antibody-analysis.md).

## Workflow publication and compatibility

New schema **5** places vh_pi, vl_pi, vh_vl_pi, vh_v_gene, vh_j_gene, vl_v_gene
and vl_j_gene immediately after vh in ordinary selection.csv. Germline pIs remain
exclusive to standalone analysis. Physical pIs use the full input and literal
VH+VL definitions above, computed once per unique chain/pair and joined after
ranking with Polars. Missing germline assignments do not suppress physical pI.
The coordinator image pins Biopython and records its scientific version.

Retain the typed germlines.parquet introduced in schema 4. Each evidence row binds parent,
candidate, chain role and exact sequence SHA-256. Scientific versions record
arpeggia/reference identity; the manifest binds the sidecar bytes. CLI and website
download the same scientific tables.

The existing per-candidate CPU Provider Call batches independent ANARCI and
arpeggia Tasks. Each retains its own publication and failure status; a Successor
reuses successful evidence and runs only missing work. No extra Provider Call,
generator, GPU or therapeutic-network dependency is added to a normal run.
Annotations join **after ranking**. Missing genes do not change evaluation_complete;
unavailable IMGT checks retain partial-result/eligibility behavior while
independent gene evidence remains available.

Keep historical schema 2/3/4 readers and original CSV/archive bytes. Do not backfill
genes/pIs or rerank old jobs. Explicit local inspection or transfer to analysis is
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

Load example sequences populates Group 1 with the user-supplied pembrolizumab
and OKT3 pairs and Ozoralizumab single domain, retaining the optional second
group. Loading is an input edit, not an analysis request; it invalidates any
pending file read for that group. After explicit analysis returns new results,
bring the results heading into view with accessible focus and reduced-motion
support. Sorting/paging or inspecting a sequence must not retrigger scrolling.

Two groups have independent tables, not matched-ID comparisons/deltas. Share
column visibility, with independent sorts and 50-row browser pages. Order ID,
VH pI, VL pI, VH+VL pI, four genes, VH germline pI, VL germline pI, then remaining
chain metrics. The two germline columns follow VL J gene. Unassigned
sequences retain honestly labeled metrics. Display pI/kDa to two decimals, GRAVY
to three and usage percentages to one; retain raw sorting/CSV. Use compact VH/VL
sequence buttons and a Columns (shown/all) visibility button, shared across groups.
Gene popovers show evidence without repeating generic V/J match descriptions.
The Issues column contains input/annotation diagnostics, not liability flags or
quality assessments.

Each analysis group has independent row selection, retained through sorting and
paging and reset with new results. Download selected pairs exports FASTA in
original input order: VH:VL for pairs, one sequence for standalone entries.
Only entries with analyzed sequences are selectable; never invent missing
partners. This is distinct from humanization's within-parent recombination.
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
remains below the 32 MiB CSV bound with the seven added scalar columns.

Local measurements on 21 September 2026 (analysis version 1, before the metric
amendments), not deployment latency or current response-size promises:

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
