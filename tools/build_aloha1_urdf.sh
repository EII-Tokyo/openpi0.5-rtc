#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CONFIG_PATH="${ALOHA1_XACRO_CONFIG:-${PROJECT_ROOT}/configs/aloha1_xacro_args.yaml}"
PYTHON_BIN="${ALOHA1_BUILD_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'ERROR: Python executable is unavailable: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'ERROR: Xacro config is unavailable: %s\n' "${CONFIG_PATH}" >&2
  exit 2
fi

cd -- "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m tools.aloha1_mapping.build_urdf \
  --config "${CONFIG_PATH}" \
  --project-root "${PROJECT_ROOT}"
