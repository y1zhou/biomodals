#!/usr/bin/env bash
set -euo pipefail

# Measurement positions are one-based offsets in the supplied parent chains.
# Deploy mutation_ridge first; change paths and pin version for your environment.
uv run biomodals app run mutation_ridge --environment main --version 1 -- \
  --input-csv measurements.csv \
  --parental-fasta parental.fasta \
  --max-mutations 2 \
  --candidate-budget 1000000 \
  --output-csv candidates.csv
