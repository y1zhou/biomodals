# Analyzing antibody sequences

Antibody sequence analysis runs locally on the API server. It does not submit
Modal work, require a confirmation step or create a My Jobs entry. Use it for
VH/VL pairs or individual variable domains, including single heavy domains.
An H-chain assignment alone does not establish that a sequence is a nanobody.

## Enter sequences

Open **Antibody sequence analysis**, enter FASTA and choose **Analyze sequences**.
You can provide a second independent group for side-by-side comparison. Each group
accepts up to 1000 entries, each supplied chain up to 512 residues, with a 4 MiB
total request limit. The page displays the service's current limits.

**Load example sequences** replaces Group 1 with pembrolizumab and OKT3 VH:VL
pairs and an Ozoralizumab single-domain example. It leaves Group 2 unchanged and
does not run analysis. Choose **Analyze sequences** when ready; newly generated
results are brought into view below the input form.

Use one of these forms (replace the placeholders with amino-acid sequences):

```text
>clone_a
VH_SEQUENCE:VL_SEQUENCE
```

Or provide matching suffixed records in either order:

```text
>clone_a_vh
VH_SEQUENCE
>clone_a_vl
VL_SEQUENCE
```

For one chain, use an ordinary unsuffixed FASTA record. IDs are the first header
word and must be unique within each group. A pair needs both correctly labeled
chains; missing partners and swapped roles are reported, not guessed. Spaces
and line breaks are removed from sequences and letters uppercased. Other
noncanonical characters are rejected. Invalid entries stay visible while valid
entries are analyzed. Numbering failures do not discard valid physical metrics.

Inputs and results stay in memory. Reloading loses them; download a group's CSV
if you want to retain its table. Sort or change pages without recomputing results.
The two tables sort independently and share column visibility controls.
Use **Columns (shown/all)** to choose visible metrics. Select rows and choose
**Download selected pairs** to save FASTA for that group; pairs use VH:VL and
standalone entries retain their single sequence. Selections survive sorting and
paging, but a new analysis starts a fresh selection.

## Understand the results

The table starts with ID, VH pI, VL pI, **VH+VL pI**, then V/J genes,
**VH germline pI**, **VL germline pI**, and the remaining metrics.
The combined pI treats VH followed directly by VL as one
continuous sequence, with no linker. It is not a whole-antibody pI or the mean
of the two chain values.

Chain mass is average unmodified mass in kDa. GRAVY uses the **BlackMould**
hydrophobicity scale (higher means more hydrophobic).
All metrics include supplied tails/tags, not just the numbered domain. Unprovided
constant regions, glycans, linkers and conjugates are not included.

Germline pI concatenates the closest V and J reference segments for that chain.
Where matches tie, it uses arpeggia's default display representative, as shown
in the sequence inspector; gene popovers still retain every tie. It includes
the full reference segments, not just locally aligned residues. This estimate
does not reconstruct D/junction residues or a complete ancestral antibody.
If either segment is unavailable or contains unresolved residues, no pI is shown.

**Issues** reports input or annotation problems, such as duplicate IDs, missing
partners, invalid amino-acid symbols, swapped VH/VL roles, or unavailable
numbering/germline assignments. These are not quality scores or liability flags.

Hover a V or J gene for its matched reference species and usage among unique
chains of therapeutics marked approved in the cached Thera-SAbDab snapshot.
All tied assignments are retained. This is gene-level rather than allele-level
usage; ties share one chain's credit. The FAQ above the results records the
source, date, hash and cohort. These frequencies are not clinical success rates
or natural repertoire frequencies. Missing reference data are shown as unavailable.

Click **VH** or **VL** to open its numbered display. Choose IMGT, Kabat, Chothia,
Martin or AHo; each uses its matching CDR definition. CDR1/2/3 and unnumbered
tails are identified. Hover potential liability markers for motif explanations.
**Copy sequence** copies the full supplied sequence; click outside to close.
Motifs are screened over the full sequence, including tails. Odd-count cysteine
markers exclude the conserved framework cysteines (IMGT 23 and 104) regardless
of display scheme. N-terminal glutamine and hydrophobic patches are not screened.
They flag sequence patterns, not measured modifications, accessibility, binding
or developability. Changing display numbering does not change humanization
scores or its stored IMGT-preservation checks.

The Numbered domain section shows one Germline / Diffs / **Input** alignment,
with the selected V/J reference genes, species and tie counts above it.
Blank differences in covered positions mean exact matches;
`+` means an input insertion, `-` a deletion, `:` a similar substitution and
`x` another mismatch. These are local matches: omitted flanks and the unknown
V/J junction are not called insertions or mutations. Humanization results show
Germline (humanized), **Humanized**, Parental and Germline (parental), separated
by unlabeled comparison strips. Both before/after V/J genes and species appear
above the alignment. If retained parent inputs or their germline assignments
are unavailable, the rest of the inspector remains usable.

## Analyze humanization selections

On a humanization result, select VH and VL sequences using the checkboxes.
Selections survive pages and parent filters. **Analyze selected sequences**
opens analysis and runs it directly, without confirmation or a saved Job.

Two VH and three VL selections from the same parent create six pairs. Sequences
are never paired across different parental antibodies. A parent with only one
selected chain role contributes standalone chains instead. Recombined pairs do
not inherit source rankings or pairing scores and are not proven compatible.
Candidate-ID copying remains available.

New humanization publications also include four gene columns in their ordinary
selection CSV. Old results are not backfilled or reranked; you can still inspect
or explicitly analyze their sequences.

## Python and deployment

Install the API extra to use the shared Python helpers:

```python
from biomodals.helper.antibody import analyze_chain, analyze_pair

single = analyze_chain(sequence)
paired = analyze_pair(vh, vl)
```

The helpers return ordinary Python dictionaries and perform no network I/O.
The service separately builds its public therapeutic frequency cache on first
use, only when its file is absent. It records the download date/hash and does not
schedule updates. An unreadable existing file needs deliberate administrator
repair; other sequence metrics remain usable. The cache lives under
BIOMODALS_CACHE_DIR/antibody-sequence-analysis/therapeutic-gene-usage.json.

Roll out the updated API extras/service and frontend together. The page requires
analysis version 5; an update notice blocks calculations against an older API
rather than mislabeling its metric scale or liability rules. Sequence copying
remains available. New humanization
gene publications additionally require deploying and pinning the updated
containing workflow; existing jobs need no migration. Standalone generators do
not need redeployment for this feature. See the
[implementation specification](specs/antibody-sequence-analysis.md) and
[third-party acknowledgements](third-party/antibody-analysis.md).
