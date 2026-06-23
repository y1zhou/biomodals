# Recover stale node attempts according to placement

The workflow runtime will recover a stale orchestrator-placed attempt through its declared rerun or resume policy, while a stale remote attempt without a recorded Modal function-call identity remains blocked. Orchestrator work has no independent execution after its owner is gone, but blindly replacing an untracked remote call could duplicate work that is still writing deterministic outputs.
