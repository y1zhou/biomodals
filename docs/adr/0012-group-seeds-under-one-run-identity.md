# Group seeds under one run identity

Status: accepted.

An AlphaFold3 `run_id` identifies the Enriched AlphaFold Input and
seed-independent inference configuration, not one submission or seed list. An
app-local `hash_sequences` helper builds the stable identifier from the
canonical covered fields: enriched biological input content, caller-template
content digests, declared AlphaFold/app/model identity, recycle count, and
diffusion-sample count. It excludes model seeds, display name, local output
directory, and operational container parallelism.

A GPU container may receive a list of seeds, matching upstream
`process_fold_input`. Upstream publishes each diffusion sample beneath
`<run-root>/outputs/seed-<seed>_sample-<sample-index>/` and may also publish
seed-specific embeddings or distograms. The scheduler must assign disjoint seed
sets to containers and must not submit a seed whose complete output already
exists. A seed completion marker is valid only after all of its expected sample
directories and optional seed artifacts validate.

Submitting a different seed set for the same run identity therefore reuses
completed Seed Predictions and schedules only missing seeds without forcing one
container per seed or creating another run root.
