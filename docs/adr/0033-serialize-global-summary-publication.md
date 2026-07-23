# Serialize global summary publication

Status: accepted.

Every Inference Run Identity has one path-scoped Summary Build Claim in the
persistent Modal Dict. The atomic generation protocol used for other claims
elects one finalizer at a time. After acquiring ownership, the finalizer reloads
the output Volume, validates the current Seed Prediction markers, and builds a
global ranking from that exact completed-seed union in generation-exclusive
staging.

The finalizer may publish only if its seed set contains every seed bound by the
current Inference Run Summary marker. It promotes the ranking and global-best
files before writing a new marker last. A stale Volume view or finalizer can
therefore never regress the accumulated summary.

A finalizer does not wait for unrelated seeds whose GPU work is still in
flight. Each owning request finalizes after its own requested seeds complete,
so a later claimant incorporates newly available publications. Every request
manifest records the exact global-summary marker digest it observed, while its
request-specific ranking remains independently durable.
