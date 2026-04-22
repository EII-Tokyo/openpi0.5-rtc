#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"

uv run python scripts/prepare_qwen_twist_swift_dataset.py \
  --train-repo-id lyl472324464/twist_subset_balanced_100k_448_multi_repo_300mb \
  --val-repo-id lyl472324464/twist_rotate_pickup_eval_dataset \
  --val-repo-id lyl472324464/twist_nine_class_eval_dataset \
  --max-train-parquet-files-per-repo "${MAX_TRAIN_PARQUET_FILES_PER_REPO:-0}" \
  --limit-per-val-repo 1000 \
  --max-val-parquet-files-per-repo "${MAX_VAL_PARQUET_FILES_PER_REPO:-8}" \
  --output-dir "${DATA_DIR:-qwen_twist_swift_data_full}"

PATH="$PWD/.venv-qwen35/bin:$PATH" \
DATA_DIR="${DATA_DIR:-qwen_twist_swift_data_full}" \
OUTPUT_DIR=output/qwen35-2b-twist-lora \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}" \
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}" \
LAZY_TOKENIZE="${LAZY_TOKENIZE:-true}" \
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}" \
NUM_TRAIN_EPOCHS=1 \
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-output/qwen35-2b-twist-lora/v3-20260421-091009/checkpoint-797}" \
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-true}" \
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-true}" \
bash scripts/train_qwen35_2b_twist_ms_swift.sh
