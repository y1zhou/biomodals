#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" = "1" ]; then
    set -x
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")

TASK="${CALIBY_EXAMPLE_TASK:-ensemble_design}"
DESIGN_YAML="${CALIBY_DESIGN_YAML:-${SCRIPT_DIR}/../data/caliby/caliby_design.yaml}"
ENSEMBLE_YAML="${CALIBY_ENSEMBLE_YAML:-${SCRIPT_DIR}/../data/caliby/caliby_ensemble_design.yaml}"
OUT_DIR="${CALIBY_OUT_DIR:-${PWD}}"

# Caliby upstream exposes many shell scripts, but most examples are parameter
# presets around a few core Python entrypoints. The Biomodals example keeps the
# common pipeline in YAML and lets caliby_app.py validate the YAML with Pydantic.
#
# Advanced sampling controls such as batch size, dataloader workers, sampling
# temperature, globally omitted amino acids, raw sampling overrides, baseline
# scoring, and input cleaning are intentionally kept at app defaults for a
# reproducible first-pass pipeline.

case "${TASK}" in
    design)
        "${ENTRY_BIN}" app r caliby -- \
            --input-yaml "${DESIGN_YAML}" \
            --task design \
            --out-dir "${OUT_DIR}"
        ;;
    ensemble_design)
        "${ENTRY_BIN}" app r caliby -- \
            --input-yaml "${ENSEMBLE_YAML}" \
            --task ensemble_design \
            --out-dir "${OUT_DIR}"
        ;;
    *)
        cat <<EOF
Unknown CALIBY_EXAMPLE_TASK: ${TASK}

Supported values:
  design
  ensemble_design
EOF
        exit 2
        ;;
esac
