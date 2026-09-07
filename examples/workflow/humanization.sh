#!/usr/bin/env bash
set -euo pipefail

# Validate the paired CSV and graph without launching inference.
uv run biomodals workflow run --dry-run humanization -- \
  --input-csv examples/data/sapiens_pairs.csv \
  --hudiff-ab-candidate-count 2

# After model assets are staged and the containing workflow is deployed:
# uv run biomodals workflow run --max-containers 4 --max-gpu-containers 2 \
#   humanization -- --input-csv examples/data/sapiens_pairs.csv \
#   --hudiff-ab-candidate-count 2 --hudiff-ab-seed 42
#
# Models are not downloaded or deployed by workflow submission. Results include
# selection.csv, candidates.fasta, detailed Parquet tables, and manifest.json.
# HuDiff's candidate count is an attempt budget, not a guaranteed unique yield.
