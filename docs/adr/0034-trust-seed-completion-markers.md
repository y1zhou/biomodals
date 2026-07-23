# Trust seed completion markers

Status: accepted.

A GPU worker waits for the pinned upstream process to exit successfully, moves
the assigned seed-specific directories from Inference Worker Staging into
their canonical output paths, closes them, and commits the output Volume. It
then writes and commits one small Seed Completion Marker per assigned seed.

The marker binds the `run_id`, seed, and owning claim generation, but it is not
an artifact inventory. Cache reconciliation trusts a matching marker without
walking sample directories, parsing confidence files, or recomputing sizes and
digests. Directory existence without the marker remains incomplete.

If preemption or another failure occurs before marker publication, a later
request reruns that entire seed. This deliberately favors a cheap, simple
completion check over defending against post-publication output corruption;
rerunning an occasional partial Seed Prediction is acceptable.
