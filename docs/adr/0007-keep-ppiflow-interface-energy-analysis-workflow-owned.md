# Keep PPIFlow interface-energy analysis workflow-owned

PPIFlow-specific Rosetta interface-energy analysis stays in the workflow for now. The workflow owns Rosetta script generation, expected `residue_energy.csv` discovery, fixed-position derivation, and candidate identity preservation for this stage. The generic Rosetta app remains a command runner instead of growing a PPIFlow-specific workflow-compatible API until that contract proves reusable outside PPIFlow.
