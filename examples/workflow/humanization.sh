#!/usr/bin/env bash
set -euo pipefail

# Validate the paired CSV and graph without launching inference.
uv run biomodals workflow run --dry-run humanization -- \
  --input-csv examples/data/sapiens_pairs.csv \
  --hudiff-ab-candidate-count 25

# After model assets are staged and the containing workflow is deployed:
# uv run biomodals workflow run --max-containers 1000 --max-gpu-containers 40 \
#   humanization -- --input-csv examples/data/sapiens_pairs.csv \
#   --hudiff-ab-candidate-count 25 --hudiff-ab-seed 42
#
# Models are not downloaded or deployed by workflow submission. Results include
# selection.csv, scores/ Parquet tables, imgt_mutations.parquet,
# generation.parquet, and manifest.json.
# HuDiff's candidate count is an attempt budget, not a guaranteed unique yield.
