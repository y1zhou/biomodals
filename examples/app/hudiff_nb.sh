#!/usr/bin/env bash
# One-call development probe. Production users should use the nanobody workflow.
# First stage assets via stage_hudiff_nb_models in the same environment.
set -euo pipefail
input_json="${1:?Provide a prepared sequence/protected_indices JSON file}"
output_file="${2:?Provide a new generation JSON output file}"
uv run biomodals app run --development hudiff_nb -- \
    --input-json "$input_json" --output-file "$output_file" \
    --candidate-count 10 --seed 0
