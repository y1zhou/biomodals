#!/bin/bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_BIN_NAME='biomodals'

if [ $# -eq 0 ] || { [ $# -eq 1 ] && [[ "$1" == "${DEFAULT_BIN_NAME}" ]] ; }; then
  uv run --project "${SCRIPT_DIR}" "${DEFAULT_BIN_NAME}" --help

elif [ $# -gt 1 ] && [[ "$1" == "${DEFAULT_BIN_NAME}" ]]; then
  uv run --project "${SCRIPT_DIR}" "$@"

elif [ $# -eq 1 ] && [[ "$1" == 'python' ]]; then
  uv run --project "${SCRIPT_DIR}" python

else
  uv run --project "${SCRIPT_DIR}" "${DEFAULT_BIN_NAME}" "$@"
fi
