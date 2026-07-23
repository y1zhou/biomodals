# Hash the normalized enriched input

Status: accepted.

The app-local `hash_sequences` helper derives `run_id` from an Inference
Identity View plus seed-independent inference identity fragments. The view is
created by validating the Enriched AlphaFold Input through `AF3Config`, dumping
all defaults explicitly, removing only `name` and `modelSeeds`, and replacing
Staged Custom Template paths with their content digests.

The view preserves sequence order, chain IDs, descriptions, modifications,
bonds, MSA content, templates and mappings, custom CCD content, dialect, schema
version, and every other validated input field. The additional hash fragments
cover recycle count, diffusion-sample count, pinned app/upstream identity, the
Declared Model Identity, and the run-identity schema version.

Search-policy flags, search and GPU worker counts, container partitioning,
local/remote output paths, and other scheduling controls are excluded.
Conservatively hashing the normalized model avoids a result-affecting field
being omitted by a hand-maintained biological-field whitelist while still
allowing different seed requests and display names to share a run.
