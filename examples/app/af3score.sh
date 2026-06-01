#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")

pembro_pdb="${SCRIPT_DIR}/../data/5B8C.pdb.gz"

temp_dir=$(mktemp -d)
mkdir -p "${temp_dir}/inputs"
gunzip -c "${pembro_pdb}" > "${temp_dir}/inputs/5B8C.pdb"
"${ENTRY_BIN}" app r af3score -- \
    --input-dir "${temp_dir}/inputs" \
    --output-dir "${temp_dir}/outputs" \
    --run-name 'biomodals-af3score-test' \
    --max-batches 2
