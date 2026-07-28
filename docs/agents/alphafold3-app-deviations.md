# AlphaFold3 App-Development Deviations

`src/biomodals/app/fold/alphafold3_app.py` intentionally differs from the
default app-development contract in the following reviewed ways.

## Cache integrity

AlphaFold3 Seed Completion Markers do not inventory or rehash every prediction
artifact on reuse. A worker validates the complete expected seed output before
publishing its marker. Request publication and retrieval hash the bytes they
observe, but have no earlier artifact digest against which to detect
valid-looking post-publication corruption. Marker-only reconciliation avoids
scanning a potentially large and growing prediction tree before every
additional seed request.

The tradeoff is explicit: corruption after seed publication is not detected
automatically. Missing or structurally invalid files fail later publication,
but other changes may be returned as current seed output. Operators must remove
the affected seed marker and outputs before rerunning the seed.

The model checkpoint and upstream template store are also treated as immutable
operator-managed infrastructure:

- inference identity uses a code-owned checkpoint label rather than hashing
  `af3.bin`;
- template identity excludes inventories and digests for `pdb_seqres` and
  `mmcif_files/`.

Replacing either store in place is unsupported. A model replacement requires a
new declared model identity or explicit removal of affected run caches. A
template-store replacement requires explicit template-cache removal or a new
template identity policy.

## Run layout

AlphaFold3 does not use the generic `AppRunLayout`. Its seed-independent
`/{run_id[:2]}/{run_id}/` root is a scientific cache shared by multiple seed
requests, workers, summaries, and stable request views. Upstream's per-seed
directory names remain intact below `outputs/`, while app-owned completion
markers and manifest-only request views provide the durable publication
boundaries. Exact pre-enrichment invocation receipts live separately under
`/invocations/{invocation_id[:2]}/` and bind immutable invocation identity to a
completed request manifest. Presentation-only ranking files and best aliases
are generated in the downloaded archive instead of persisted in request views.

This custom layout is confined to the AlphaFold3 supporting modules and is
documented in `docs/adr/0005-alphafold3-msa-sharding.md`.
