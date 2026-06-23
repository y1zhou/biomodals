# Filter PPIFlow candidates through manifests

PPIFlow filter stages narrow the active candidate set by emitting a retained-candidate manifest, not just a filtered score CSV. Rejected candidates are preserved in an audit table with filter outcomes and reasons so downstream stages consume only retained candidates while users can still inspect what was removed.
