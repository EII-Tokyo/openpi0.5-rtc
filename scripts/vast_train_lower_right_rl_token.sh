#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/openpi0.5-rtc-reward-learning}"
S3_VLA_URI="${S3_VLA_URI:-s3://openpi-tokyo/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000/}"
LOCAL_VLA_DIR="${LOCAL_VLA_DIR:-${ROOT_DIR}/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000}"
RUN_SMALL="${RUN_SMALL:-1}"
RUN_4LAYER="${RUN_4LAYER:-1}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required. Export it before running this script." >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required. Export it before running this script." >&2
  exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required on the vast.ai machine." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the vast.ai machine." >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "Using repository: ${ROOT_DIR}"
echo "Downloading VLA checkpoint from ${S3_VLA_URI}"
mkdir -p "${LOCAL_VLA_DIR}"
aws s3 sync "${S3_VLA_URI}" "${LOCAL_VLA_DIR}/"

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
fi

uv sync --frozen

if [[ "${RUN_SMALL}" == "1" ]]; then
  uv run scripts/train.py eii_rinse_11repo_cam4_fullft_rl_token_lower_right_small_query
fi

if [[ "${RUN_4LAYER}" == "1" ]]; then
  uv run scripts/train.py eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer
fi
