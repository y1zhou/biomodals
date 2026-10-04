#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")

temp_dir=${OUTPUT_DIR:-$(mktemp -d)}
mkdir -p "${temp_dir}"
temp_dir=$(realpath "${temp_dir}")
if [ "${ACCEPTANCE:-0}" = "1" ]; then
    : "${ORACLE_DIR:?Set ORACLE_DIR to independently generated corrected-upstream tables}"
    ORACLE_DIR=$(realpath "${ORACLE_DIR}")
fi

"${ENTRY_BIN}" app r --development oligoformer -- \
    --mrna-fasta "${SCRIPT_DIR}/../data/sirna_target.fa" \
    --out-dir "${temp_dir}" \
    --run-name biomodals_oligoformer_example \
    --toxicity --force

validation_flags=()
if [ "${ACCEPTANCE:-0}" = "1" ]; then
    "${ENTRY_BIN}" app r --development oligoformer -- \
        --mrna-fasta "${SCRIPT_DIR}/../data/sirna_target.fa" \
        --out-dir "${temp_dir}/repeat" \
        --run-name biomodals_oligoformer_example \
        --toxicity --force
    validation_flags=(--repeat "${temp_dir}/repeat/biomodals_oligoformer_example_oligoformer.tar.zst" --oracle "${ORACLE_DIR}")
fi
cd "${BIOMODALS_ROOT}"
uv run --frozen python examples/validation/model_outputs.py oligoformer \
    "${temp_dir}/biomodals_oligoformer_example_oligoformer.tar.zst" \
    "${validation_flags[@]}" --report "${temp_dir}/validation.json"
echo "OligoFormer artifacts retained in ${temp_dir}"
