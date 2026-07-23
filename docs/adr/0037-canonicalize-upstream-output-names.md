# Canonicalize upstream output names

Status: accepted.

The caller's display name is preserved in the Enriched AlphaFold Input and
Inference Request manifest but remains excluded from Inference Run Identity.
Before invoking the pinned upstream inference process, the app clones the input
with the Canonical Output Name `af3-{run_id[:16]}`.

The pinned model discards `target_name` before inference, so this substitution
does not change scientific computation. Upstream does use its sanitized name as
the prefix for model CIF, confidence, ranking, data-JSON, embedding, and
distogram filenames. Durable files on the output Volume therefore deliberately
use the canonical run-derived prefix.

A later request with another display name reuses those files without renaming
or duplicating them. This avoids first-writer-dependent filenames beneath a
seed-independent shared run while retaining the caller's label as presentation
metadata.
