#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")

pembro_pdb="${SCRIPT_DIR}/../data/5B8C.pdb.gz"

temp_dir=${OUTPUT_DIR:-$(mktemp -d)}
mkdir -p "${temp_dir}"
temp_dir=$(realpath "${temp_dir}")

gunzip -c "${pembro_pdb}" > "${temp_dir}/5B8C.pdb"
"${ENTRY_BIN}" app r --development flowpacker -- \
    --input-path "${temp_dir}/5B8C.pdb" \
    --run-name flowpacker_example \
    --out-dir "${temp_dir}" \
    --use-confidence

cd "${BIOMODALS_ROOT}"
uv run --frozen python examples/validation/model_outputs.py flowpacker \
    "${temp_dir}/flowpacker_example.tar.zst" \
    --report "${temp_dir}/validation.json"
echo "FlowPacker artifacts retained in ${temp_dir}"
