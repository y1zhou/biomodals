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

The model checkpoint and required sharded MSA profiles are automatically
provisioned once per Modal Environment when absent. A completed checkpoint or
profile manifest is then treated as immutable infrastructure until an operator
explicitly repairs it:

- inference identity uses a code-owned checkpoint label rather than hashing
  `af3.bin`;
- search identity excludes inventories and digests for completed sharded
  profiles and the upstream template stores.

Replacing completed assets in place is unsupported. A model replacement
requires a new declared model identity or explicit removal of affected run
caches. A profile or template-store replacement requires explicit cache repair
or a new identity policy. See
[`ADR 0005`](../adr/0005-alphafold3-msa-sharding.md) for setup coordination and
readiness boundaries.

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

## Generation claims

AlphaFold3 may expire an old generation claim before electing a new scientific
publisher. This is workload-level coordination across Execution Runs, not a
kernel Task lease. It does not authorize retry or replacement inside one Run;
only a new root Run or explicit compatible Successor may schedule missing work.
Completion markers and validated publications remain the reuse authority.

For publication-compatible API resubmissions, the service delays the new root
Run until matching Service Jobs are conclusively terminal. It may then name
those Execution Run IDs in the new request. AlphaFold3 derives stable Task
generations from the Execution Run ID, Node key, and Task key and fences only
those generations; unrelated or unknown owners keep the ordinary conservative
claim behavior. This exception is service-local workload repair, not kernel
Successor inference.
