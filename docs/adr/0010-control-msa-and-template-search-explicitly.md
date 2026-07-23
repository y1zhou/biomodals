# Control MSA and template search explicitly

Status: accepted.

The AlphaFold3 entrypoint exposes `search_msa` and
`search_protein_templates`, both defaulting to true. They define field
population as follows:

| MSA search | Protein-template search | Behavior |
| --- | --- | --- |
| On | On | Preserve each non-empty caller-supplied MSA or template field and search to populate every missing or empty field. |
| On | Off | Preserve each non-empty caller-supplied MSA field and search to populate missing or empty MSA fields. Preserve non-empty protein templates and set missing or null template fields to `[]`. |
| Off | Either value | Run no MSA or template searches. Preserve supplied fields, set unset protein and RNA MSA fields to `""`, and set missing or null protein template fields to `[]`. |

For protein inputs, the MSA fields are unpaired and paired; RNA has only an
unpaired MSA. With MSA search disabled, `search_protein_templates` cannot start
template work because the entire data stage is disabled. Empty sentinels are
therefore deliberate consequences of the chosen policy, not silent inference
fallbacks.
