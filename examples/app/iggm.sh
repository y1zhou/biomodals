#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" = "1" ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")
DATA_DIR="${SCRIPT_DIR}/../data/iggm"

TASK="${IGGM_EXAMPLE_TASK:-design}"
OUT_DIR="${IGGM_OUT_DIR:-${PWD}}"
EPITOPE="7,8,9,10,11,12,13,14,108,109,110,111,112,113,114,115,116,118,157,158,160,161,162,163,164,167"

case "${TASK}" in
    download_models)
        "${ENTRY_BIN}" r iggm -- \
            --download-models
        ;;
    force_download_models)
        "${ENTRY_BIN}" r iggm -- \
            --download-models \
            --force-redownload
        ;;
    inverse_design)
        "${ENTRY_BIN}" r iggm -- \
            --input-fasta "${DATA_DIR}/8hpu_M_N_A_CDR_H3.fasta" \
            --antigen "${DATA_DIR}/8hpu_M_N_A.pdb" \
            --epitope "${EPITOPE}" \
            --task inverse_design \
            --run-name biomodals_iggm_inverse_design_example \
            --out-dir "${OUT_DIR}"
        ;;
    fr_design)
        "${ENTRY_BIN}" r iggm -- \
            --input-fasta "${DATA_DIR}/1vfb_B_A_C.fasta" \
            --antigen "${DATA_DIR}/1vfb_B_A_C.pdb" \
            --task fr_design \
            --run-name biomodals_iggm_fr_design_example \
            --out-dir "${OUT_DIR}"
        ;;
    affinity_maturation)
        "${ENTRY_BIN}" r iggm -- \
            --input-fasta "${DATA_DIR}/8hpu_M_N_A_CDR_H3.fasta" \
            --antigen "${DATA_DIR}/8hpu_M_N_A.pdb" \
            --fasta-origin "${DATA_DIR}/8hpu_M_N_A.fasta" \
            --task affinity_maturation \
            --num-samples 3 \
            --run-name biomodals_iggm_affinity_maturation_example \
            --out-dir "${OUT_DIR}"
        ;;
    cdr_h3_design|design)
        "${ENTRY_BIN}" r iggm -- \
            --input-fasta "${DATA_DIR}/8hpu_M_N_A_CDR_H3.fasta" \
            --antigen "${DATA_DIR}/8hpu_M_N_A.pdb" \
            --task design \
            --run-name biomodals_iggm_cdr_h3_design_example \
            --out-dir "${OUT_DIR}"
        ;;
    cdr_all_design)
        "${ENTRY_BIN}" r iggm -- \
            --input-fasta "${DATA_DIR}/8hpu_M_N_A_CDR_All.fasta" \
            --antigen "${DATA_DIR}/8hpu_M_N_A.pdb" \
            --task design \
            --run-name biomodals_iggm_cdr_all_design_example \
            --out-dir "${OUT_DIR}"
        ;;
    epitope_design)
        "${ENTRY_BIN}" r iggm -- \
            --input-fasta "${DATA_DIR}/8hpu_M_N_A_CDR_All.fasta" \
            --antigen "${DATA_DIR}/8hpu_M_N_A.pdb" \
            --epitope "${EPITOPE}" \
            --task design \
            --run-name biomodals_iggm_epitope_design_example \
            --out-dir "${OUT_DIR}"
        ;;
    merge_chains)
        "${ENTRY_BIN}" r iggm -- \
            --task merge_chains \
            --antigen "${DATA_DIR}/8ucd.pdb" \
            --antibody-ids H_L \
            --merge-ids A_B_C \
            --run-name biomodals_iggm_merge_chains_example \
            --out-dir "${OUT_DIR}"
        ;;
    *)
        cat <<EOF
Unknown IGGM_EXAMPLE_TASK: ${TASK}

Supported values:
  download_models
  force_download_models
  inverse_design
  fr_design
  affinity_maturation
  cdr_h3_design
  cdr_all_design
  epitope_design
  merge_chains

Default:
  design

Example:
  IGGM_EXAMPLE_TASK=epitope_design IGGM_OUT_DIR="\${PWD}" ./examples/app/iggm.sh
EOF
        exit 2
        ;;
esac
