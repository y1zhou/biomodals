# Biomodals app coordination

For app changes, use the [app-development skill](../../.agents/skills/biomodals-app-development/SKILL.md).
It owns standard workflow, reference routing and verification. This file owns
only approved deviations; add a focused note here when a new deviation is
explicitly approved.

## Approved App-Specific Deviations

- [AlphaFold3 cache integrity and run layout](alphafold3-app-deviations.md)
  documents the intentionally trusted immutable stores, marker-only seed cache,
  and multi-request run layout used by `alphafold3_app.py`.
- [GROMACS analysis publication](gromacs-analysis-checkpoints.md) documents
  content-bound statistics and the remaining native PBC postprocessing
  timestamp exception. Do not extend that exception to new cache stages.
- [p-AbNatiV2 upstream compatibility patches](pabnativ2-app-deviations.md)
  document the two guarded Python 3.12 compatibility edits applied to pinned
  AbNatiV and ABodyBuilder3 sources.
