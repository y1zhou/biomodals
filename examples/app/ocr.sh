#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" = "1" ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")

if [ -z "${OCR_EXAMPLE_PDF:-}" ]; then
    echo "Set OCR_EXAMPLE_PDF to a local PDF path before running this example." >&2
    exit 2
fi

OUT_DIR="${OCR_OUT_DIR:-${PWD}}"

"${ENTRY_BIN}" app r ocr -- \
    --input-pdf "${OCR_EXAMPLE_PDF}" \
    --out-dir "${OUT_DIR}" \
    --effort "${OCR_EFFORT:-high}" \
    ${OCR_RUN_POPO:+--run-popo}
