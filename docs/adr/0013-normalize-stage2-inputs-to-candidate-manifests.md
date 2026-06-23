# Normalize stage-2 PPIFlow inputs to candidate manifests

Stage-2-only PPIFlow runs start from a candidate manifest so downstream scoring, refolding, DockQ pairing, ranking, and reporting can use deterministic candidate identity. A plain user-provided structures path remains a convenience input, but the workflow normalizes it into a minimal manifest with synthetic candidate ids before stage-2 nodes run.
