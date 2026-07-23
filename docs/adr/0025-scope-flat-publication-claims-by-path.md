# Scope flat publication claims by path

Status: accepted.

Search Build Claims follow the exclusive cache path they protect. Versioned
Raw Database MSA paths retain claims keyed by
`(polymer_type, sequence_hash, database_id, search_identity)`. The mutable
combined-MSA files use one claim namespace per
`(polymer_type, sequence_hash)`, and the mutable template file uses a separate
claim namespace per protein `sequence_hash`.

For each flat publication, the desired raw-dependency or template identity is
stored in the claim generation and completion marker rather than in the claim
key. A writer revalidates that dependency identity before replacing files and
publishing its marker. A waiter that needs another identity validates the
completed marker, then advances to a later generation if it still needs to
replace the flat publication.

Identity-scoped claims would allow two scientifically valid identities to write
the same flat paths concurrently. Path-scoped serialization prevents mixed
files and markers while preserving the decision to retain only the latest
combined-MSA and template publication.
