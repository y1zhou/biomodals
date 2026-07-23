# Use one combined MSA publication per sequence

Status: accepted.

Combined MSAs remain at the existing sequence-root paths:
`/{polymer}/{prefix}/{sequence_hash}/unpaired.a3m` and, for proteins,
`paired.a3m`. Production does not add an assembly-identity directory. A
`combined.done.json` manifest written last binds those files to the exact Raw
Database MSA completion digests, pinned upstream merge semantics, file sizes,
and file digests.

Existing unmarked cache files are not automatic cache hits because their
database provenance is unknown and they may contain caller-supplied data. They
remain untouched until the new pipeline has every required valid raw result and
has constructed and validated replacement files. Publication then replaces the
top-level files and writes the manifest; no migration or legacy archive is
required.

A missing or mismatched completion manifest causes reconstruction from the
versioned raw database results. This deliberately retains only the latest
combined publication at the convenient legacy paths. Older combinations remain
reconstructable from their raw results but are not preserved as separate
combined-cache versions.
