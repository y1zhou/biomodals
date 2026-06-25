# Prune workflow runs from terminal nodes

The workflow runtime will decide run completion and resume scope from terminal
workflow nodes. If every terminal node has durable completion and non-missing
recorded outputs, the run succeeds without scheduling intermediate nodes, even
when stale failed, running, or incomplete intermediate state remains. If some
terminal nodes are incomplete, the scheduler only considers those terminals and
their ancestor closure.

This keeps resume output-driven and avoids recomputing expensive intermediate
work when the externally relevant workflow outputs already exist. Missing
terminal artifacts invalidate completion; unknown external artifact availability
continues to warn but does not force recomputation. `force=True` still discards
the workflow run root before scheduling, so terminal pruning only affects
non-forced resumes.
