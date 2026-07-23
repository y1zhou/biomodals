# Derive HMMER search space from the source

Status: accepted.

The generic shard builder derives each profile's Database Search Space from the
validated source FASTA, following the prototype MSA app. For protein databases,
`Z` and `domZ` equal the exact source sequence count. For RNA databases, `Z`
equals the exact source nucleotide count divided by 1,000,000 and has megabase
units; Nhmmer has no separate `domZ` setting.

The builder runs full source and aggregate-shard SeqKit statistics and requires
their sequence and residue totals to match. The profile manifest records those
integer totals, the derived search-space value, and its unit. Runtime search
configuration reads the manifest value rather than duplicating it in app code,
and Search Identity binds the manifest.

Code-owned expected values from the supported upstream database snapshot are
guards, not runtime truth. A mismatch fails publication for inspection instead
of silently searching with a stale E-value scale.
