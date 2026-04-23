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
"${ENTRY_BIN}" r ligandmpnn -- \
    --run-name biomodals_ligandmpnn_score_example \
    --input-pdb "${temp_dir}/5B8C.pdb" \
    --out-dir "${temp_dir}" \
    --script-mode score \
    --model-type ligand_mpnn

"${ENTRY_BIN}" r ligandmpnn -- \
    --run-name biomodals_ligandmpnn_design_example \
    --input-pdb "${temp_dir}/5B8C.pdb" \
    --out-dir "${temp_dir}" \
    --script-mode run \
    --pack-side-chains \
    --model-type soluble_mpnn
