#!/usr/bin/env bash
# Combination requires only mutations,label CSV with already-normalized labels.
# Exploration additionally requires --parental-fasta with matching chain IDs.
# Optional --settings-json carries explicit position masks/budgets for Exploration.
set -euo pipefail

uv run biomodals workflow run --environment main --version 1 protein_optimization -- \
    --input-csv measurements.csv \
    --mode combination
