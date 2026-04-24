#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/run.sh")

pembro_pdb="${SCRIPT_DIR}/../data/5B8C.pdb.gz"
vhh_pdb="${SCRIPT_DIR}/../data/7eow_nanobody_framework.pdb.gz"

temp_dir=$(mktemp -d)
cd "${temp_dir}" || exit 1

gunzip -c "${pembro_pdb}" > "${temp_dir}/5B8C.pdb"
gunzip -c "${vhh_pdb}" > "${temp_dir}/7eow_nanobody_framework.pdb"
"${ENTRY_BIN}" r ppiflow -- \
    --input-yaml "${SCRIPT_DIR}/../data/ppiflow_vhh.yaml"
