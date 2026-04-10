#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/run.sh")

pembro_pdb="${SCRIPT_DIR}/../data/5B8C.pdb.gz"

temp_dir=$(mktemp -d)

gunzip -c "${pembro_pdb}" > "${temp_dir}/5B8C.pdb"
"${ENTRY_BIN}" r antifold -- \
    --run-name biomodals_antifold_example \
    --struct-file "${temp_dir}/5B8C.pdb" \
    --out-dir "${temp_dir}" \
    --heavy-chain B \
    --light-chain A \
    --antigen-chain C \
    --num-seq-per-target 2
