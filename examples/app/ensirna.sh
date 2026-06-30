#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")

temp_dir=$(mktemp -d)

"${ENTRY_BIN}" app r ensirna -- \
    --mrna-fasta "${SCRIPT_DIR}/../data/sirna_target.fa" \
    --out-dir "${temp_dir}" \
    --run-name biomodals_ensirna_example
