#!/usr/bin/env bash
# Parental FASTA headers must match the chain IDs in mutations,label CSV.
# Optional --settings-json carries explicit position masks/budgets for Exploration.
set -euo pipefail

uv run biomodals workflow run --environment main --version 1 protein_optimization -- \
    --input-csv measurements.csv \
    --parental-fasta parents.fasta \
    --mode combination
