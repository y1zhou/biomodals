# Separate MSA caches by polymer

Status: accepted.

The MSA cache continues to define `sequence_hash` from the validated sequence
text alone. Production adds a Polymer Cache Namespace above the existing hash
fanout:

- `/Protein/{sequence_hash[:2]}/{sequence_hash}/`
- `/RNA/{sequence_hash[:2]}/{sequence_hash}/`

Raw database results, combined MSAs, markers, and protein templates live below
the corresponding sequence root. Search Build Claim keys include the polymer
type wherever they protect paths below these namespaces.

This preserves a simple sequence-only digest and homomer reuse while preventing
the same short letter sequence from making protein and RNA overwrite a shared
`unpaired.a3m`. Legacy unnamespaced cache paths are ignored rather than
migrated.
