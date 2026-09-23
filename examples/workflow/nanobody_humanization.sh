#!/usr/bin/env bash
set -euo pipefail

# Local preparation and graph validation only; no model download or cloud call.
uv run biomodals workflow run --dry-run nanobody_humanization -- \
  --input-csv examples/data/nanobody.csv

# After authorized deployment and exact pinning, an explicit scientific run:
# uv run biomodals workflow run --max-containers 8 --max-gpu-containers 2 \
#   nanobody_humanization -- --input-csv examples/data/nanobody.csv \
#   --hudiff-nb-candidate-count 10 --root-seed 0
# The CLI verifies/stages models on CPU before any GPU generation.
