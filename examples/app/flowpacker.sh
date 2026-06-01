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

gunzip -c "${pembro_pdb}" > "${temp_dir}/5B8C.pdb"
"${ENTRY_BIN}" app r flowpacker -- \
    --input-path "${temp_dir}/5B8C.pdb" \
    --out-dir "${temp_dir}" \
    --use-confidence
