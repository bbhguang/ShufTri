#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SHUFTRI_PYTHON:-}" ]]; then
    PYTHON_BIN="${SHUFTRI_PYTHON}"
elif [[ -x /opt/anaconda3/bin/python3 ]]; then
    PYTHON_BIN=/opt/anaconda3/bin/python3
else
    PYTHON_BIN=python3
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/src/run_experiments.py" "$@"
