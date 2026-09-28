#!/usr/bin/env bash
set -euo pipefail

# Only measured substitution identities are needed; no parental FASTA required.
# Labels must already be normalized and corrected for batch/plate effects.
# Deploy mutation_ridge first; change paths and pin version for your environment.
uv run biomodals app run mutation_ridge --environment main --version 1 -- \
  --input-csv measurements.csv \
  --max-mutations 2 \
  --candidate-budget 1000000 \
  --output-csv candidates.csv
