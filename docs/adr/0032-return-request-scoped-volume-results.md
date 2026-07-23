# Return request-scoped Volume results

Status: accepted.

Inference functions return compact primitive metadata and paths relative to the
AlphaFold3 output Volume; they do not return prediction archives as function
bytes. Every Inference Request durably owns a small request directory containing
its manifest, requested-seed ranking, and requested best-output files. The
manifest references the canonical Seed Prediction directories beneath the
shared run rather than copying them.

The local entrypoint retrieves only manifest-declared artifacts and builds a
Request Retrieval Archive locally. That archive includes every requested
seed/sample directory, requested optional embeddings and distograms,
request-specific ranking and best files, the Enriched AlphaFold Input, and the
request manifest. It excludes unrelated completed seeds and Inference Worker
Staging.

No request archive is retained on the output Volume. Keeping full seed outputs
canonical and singular avoids multiplying storage across overlapping seed
requests, while local packaging preserves the existing convenient `.tar.zst`
user experience without Modal function-result size pressure.
