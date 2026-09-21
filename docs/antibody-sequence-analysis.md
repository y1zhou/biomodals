# Analyzing antibody sequences

Antibody sequence analysis runs locally on the API server. It does not submit
Modal work, require a confirmation step or create a My Jobs entry. Use it for
VH/VL pairs or individual variable domains, including single heavy domains.
An H-chain assignment alone does not establish that a sequence is a nanobody.

## Enter sequences

Open **Antibody sequence analysis**, enter FASTA and choose **Analyze**. You can
provide a second independent group for side-by-side comparison. Each group
accepts up to 1000 entries, each supplied chain up to 512 residues, with a 4 MiB
total request limit. The page displays the service's current limits.

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

## Understand the results

The table starts with ID, VH pI, VL pI, **VH+VL sequence pI**, then V/J genes and
the remaining metrics. The combined pI treats VH followed directly by VL as one
continuous sequence, with no linker. It is not a whole-antibody pI or the mean
of the two chain values.

Chain mass is average unmodified mass in kDa. GRAVY is a sequence hydrophobicity
index. Reduced and oxidized-assumption ε280 values are molar extinction
coefficients; the latter does not prove that the expected disulfide bonds form.
All metrics include supplied tails/tags, not just the numbered domain. Unprovided
constant regions, glycans, linkers and conjugates are not included.

Hover a V or J gene for its matched reference species and usage among unique
chains of therapeutics marked approved in the cached Thera-SAbDab snapshot.
All tied assignments are retained. This is gene-level rather than allele-level
usage; ties share one chain's credit. The FAQ above the results records the
source, date, hash and cohort. These frequencies are not clinical success rates
or natural repertoire frequencies. Missing reference data are shown as unavailable.

Click a sequence to open its numbered display. Choose IMGT, Kabat, Chothia,
Martin or AHo; each uses its matching CDR definition. CDR1/2/3 and unnumbered
tails are identified. Hover potential liability markers for motif explanations.
They flag sequence patterns, not measured modifications, accessibility, binding
or developability. Changing display numbering does not change humanization
scores or its stored IMGT-preservation checks.

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

Roll out the updated API extras/service and frontend together. New humanization
gene publications additionally require deploying and pinning the updated
containing workflow; existing jobs need no migration. Standalone generators do
not need redeployment for this feature. See the
[implementation specification](specs/antibody-sequence-analysis.md) and
[third-party acknowledgements](third-party/antibody-analysis.md).
