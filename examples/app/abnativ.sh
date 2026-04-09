#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/run.sh")

pembro_vh='QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS'
pembro_vl='EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIKTSENLYFQ'

temp_dir=$(mktemp -d)

uv run --project "${SCRIPT_DIR}" "${ENTRY_BIN}" r abnativ -- \
    --run-name biomodals_abnativ_example \
    --out-dir "${temp_dir}" \
    --input-vh-seq "${pembro_vh}" \
    --input-vl-seq "${pembro_vl}" \
    --model-type 'paired'
