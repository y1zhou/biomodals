#!/usr/bin/env bash
# One-call development probe. Production users should use the nanobody workflow.
# First stage assets via stage_abnativ2_vhh_models in the same environment.
set -euo pipefail
input_json="${1:?Provide a prepared sequence/protected_indices JSON file}"
output_dir="${2:?Provide a new output directory}"
uv run biomodals app run --development abnativ2_vhh -- \
    --input-json "$input_json" --output-dir "$output_dir" \
    --residue-score-threshold 0.98 --rasa-threshold 0.15 \
    --max-relative-vhh-score-decrease 0.05
