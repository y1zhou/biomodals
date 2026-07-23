# Claim missing seeds atomically

Status: accepted.

Overlapping Inference Requests coordinate each missing `(run_id, seed)` through
a generation-scoped Seed Build Claim in a persistent Modal Dict. A request uses
the Dict's atomic `put(..., skip_if_exists=True)` operation to elect exactly one
claimant before scheduling that seed. Other requests wait for, reload, and
validate the seed's output-Volume completion marker rather than launching
duplicate GPU work.

The claim is scheduling state, not cache evidence. The elected generation
writes to exclusive Inference Worker Staging and may promote the seed only
while that generation remains current. Failed or conservatively expired claims
advance through append-only generations; they are not blindly deleted, and a
superseded worker cannot publish after a replacement generation has taken
ownership. The validated per-seed marker remains the sole reusable completion
authority.

Claims are acquired per seed before missing seeds are grouped into bounded
container assignments. Thus one container may still process several seeds, but
each seed in that list is owned by the same request and cannot appear in a
concurrent worker list.
