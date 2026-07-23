# Summarize all completed seeds

Status: accepted.

The canonical Inference Run Summary covers the accumulated union of every
validated Seed Prediction beneath its seed-independent AlphaFold Run Root. A
later submission may add seeds but never removes earlier completed seeds. After
new seed outputs are promoted, a serialized finalizer rebuilds the global
ranking table and replaces the top-ranked AlphaFold files only when the union
has a better-ranked sample.

Seed Predictions remain immutable publications; the summary is derived,
mutable state. Its completion record binds the exact included seed set and
artifact digests so concurrent or stale finalizers cannot regress a summary
that already covers a larger validated set.
