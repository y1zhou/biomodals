# Reconcile seeds before scheduling

Status: accepted.

Different model-seed requests for the same seed-independent input and inference
settings share one `run_id` and AlphaFold Run Root. An Inference Request has a
separate stable identity built with `hash_sequences` from the `run_id` and its
sorted requested seed set; that request identity names a return view and never
serves as the seed cache key.

The submitted seed list must be non-empty and is normalized to a sorted unique
set before request identity, reconciliation, or worker partitioning. The
request manifest preserves both the submitted list and normalized set, and the
entrypoint logs any removed duplicates. Repeated seeds are scientifically
redundant and never create duplicate GPU work.

Before partitioning GPU work, the coordinator validates each requested seed's
completion marker. It divides the request into the intersection of requested
and valid completed seeds, which are reused, and the requested seeds with no
valid publication, which are missing. Only the missing set is partitioned into
disjoint worker seed lists. Directory existence, an earlier request manifest,
or an earlier worker grouping is insufficient evidence of completion.

The Inference Request Result contains every requested Seed Prediction,
regardless of whether it was reused or newly computed. Its ranking table and
top-ranked AlphaFold files cover only the requested set. Its manifest records
the two seed subsets and references the current accumulated Inference Run
Summary, including the global best seed. It does not pull unrelated completed
seeds into the request package or duplicate their durable source artifacts.
