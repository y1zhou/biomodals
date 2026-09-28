#!/usr/bin/env bash
# schema.json: {"target":"label","identifier":"id","features":[{"name":"dose","kind":"numeric"},{"name":"group","kind":"categorical"}]}
# Provisioning runs inside the same tracked execution; review model license first.
set -euo pipefail

uv run biomodals app run --environment main --version 1 tabpfn -- \
    --training-csv training.csv \
    --inference-csv inference.csv \
    --schema-json schema.json \
    --output-csv predictions.csv
