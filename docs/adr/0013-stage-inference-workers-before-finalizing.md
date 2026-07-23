# Stage inference workers before finalizing

Status: accepted.

`predict_structures` may assign a disjoint list of model seeds to each bounded
GPU container, but it does not point concurrent upstream processes at the same
output directory. Each invocation writes under
`<run-root>/outputs/.workers/<worker-id>/` on the output Volume. This isolates
upstream's shared data JSON, ranking CSV, top-ranked files, and non-empty-output
directory handling while retaining its native `seed-{seed}_sample-{index}`
layout.

After a worker succeeds, the wrapper validates the complete expected output for
each assigned seed and promotes only those seed-specific sample, embedding, and
distogram directories into the shared `<run-root>/outputs/` tree. The scheduler
must never assign the same seed to concurrent workers. Worker-local shared
files are not canonical.

Once the required Seed Predictions are present, one finalizer reloads the
Volume and exclusively publishes the Inference Run Summary: the enriched data
JSON, global ranking table, and top-ranked AlphaFold files. It writes completion
state only after validating and committing those shared artifacts. This keeps
all concurrent heavy output writes on distinct Volume paths without changing
upstream inference semantics.
