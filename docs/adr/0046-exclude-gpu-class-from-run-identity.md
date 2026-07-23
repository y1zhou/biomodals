# Exclude GPU class from run identity

Status: accepted.

Inference Run Identity does not include the Modal GPU accelerator class.
Predictions produced with the same enriched input, model seed, and scientific
inference settings are cache-interchangeable across supported accelerator
classes.

The app may record the actual accelerator class in seed provenance and logs,
but changing from an L40S to an H100 or another supported GPU does not create a
new run root or invalidate existing Seed Predictions. GPU count, worker
partitioning, and concurrency likewise remain operational settings outside the
identity.

This accepts that hardware-dependent floating-point behavior can produce small
numerical differences. A run may therefore contain different seeds computed
on different supported accelerator classes; the cache does not promise
bitwise reproducibility across GPU types.
