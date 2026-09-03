#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")
INPUT_CSV="${SCRIPT_DIR}/../data/pabnativ2_pairs.csv"

temp_dir=$(mktemp -d)

"${ENTRY_BIN}" app r pabnativ2 -- \
    --input-csv "${INPUT_CSV}" \
    --output-dir "${temp_dir}" \
    --run-name biomodals_pabnativ2_example
