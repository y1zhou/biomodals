#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BIOMODALS_ROOT=$(realpath "${SCRIPT_DIR}/../../")
ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")

task_yaml="${SCRIPT_DIR}/../data/ppiflow_workflow_task.yaml"
steps_yaml="${SCRIPT_DIR}/../data/ppiflow_workflow_steps.yaml"
antigen_pdb="${SCRIPT_DIR}/../data/5B8C.pdb.gz"
framework_pdb="${SCRIPT_DIR}/../data/7eow_nanobody_framework.pdb.gz"

temp_dir=$(mktemp -d)
trap 'rm -rf "${temp_dir}"' EXIT

gunzip -c "${antigen_pdb}" > "${temp_dir}/5B8C.pdb"
gunzip -c "${framework_pdb}" > "${temp_dir}/7eow_nanobody_framework.pdb"
cd "${temp_dir}" || exit 1

workflow_flags=()
if [ "${DRY_RUN:-1}" != "0" ]; then
    workflow_flags+=(--dry-run)
fi

"${ENTRY_BIN}" workflow r ppiflow "${workflow_flags[@]}" -- \
    --task-yaml "${task_yaml}" \
    --steps-yaml "${steps_yaml}" \
    --run-id ppiflow-vhh-example \
    --max-parallel 2
