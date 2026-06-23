# Use a shared PPIFlow candidate concurrency default

PPIFlow uses one shared `candidate_concurrency` default for candidate-wide stage coordinators, initially `4`, with optional per-stage overrides. This keeps the workflow config small while preserving a tuning escape hatch for AF3Score, ReFold, Rosetta, LigandMPNN, and Partial stages.
