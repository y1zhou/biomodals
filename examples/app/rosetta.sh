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
cd "${temp_dir}" || exit 1

gunzip -c "${pembro_pdb}" > "${temp_dir}/5B8C.pdb"
cp -an "${SCRIPT_DIR}/../data/rosetta_example.csv" "${temp_dir}/rosetta_example.csv"
"${ENTRY_BIN}" r rosetta -- \
    --input-csv "${temp_dir}/rosetta_example.csv" \
    --max-num-pods 2
