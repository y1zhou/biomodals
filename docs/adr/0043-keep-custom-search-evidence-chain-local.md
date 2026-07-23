# Keep custom search evidence chain-local

Status: accepted.

Canonical database work is deduplicated by `(polymer_type, sequence)` so one
MSA Search Subject can serve every chain whose corresponding field is missing.
Generated canonical results may therefore populate multiple identical-sequence
chains.

Caller-Supplied Search Evidence remains attached only to the chain where it was
provided. The coordinator does not copy a custom MSA or template list into an
identical sibling's missing field. That sibling resolves independently from
canonical cache/search results.

Request-local template work may be deduplicated only when both the protein
sequence and resolved unpaired-MSA digest match. This preserves chain-specific
biological intent while retaining safe reuse of genuinely equivalent generated
and template-search work.
