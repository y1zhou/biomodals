# Group seeds under one run identity

Status: accepted.

An AlphaFold3 `run_id` identifies the Enriched AlphaFold Input and
seed-independent inference configuration, not one submission or seed list. An
app-local `hash_sequences` helper builds the stable identifier from the
canonical covered fields: enriched biological input content, caller-template
content digests, declared AlphaFold/app/model identity, recycle count, and
diffusion-sample count. It excludes model seeds, display name, local output
directory, and operational container parallelism.

Each model seed publishes independently beneath
`<run-root>/outputs/seed-<seed>/` with its own completion state. Submitting a
different seed set for the same run identity therefore reuses completed seed
predictions and schedules only missing seeds. This matches the existing
`predict_structures` responsibility for distributing seeds while preventing
seed-list changes from creating duplicate run roots.
