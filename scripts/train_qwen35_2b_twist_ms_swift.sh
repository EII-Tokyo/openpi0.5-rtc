#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
# 默认全量 Swift jsonl；子集请显式 DATA_DIR=qwen_twist_swift_data
DATA_DIR="${DATA_DIR:-qwen_twist_swift_data_full}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen35-2b-twist-lora}"

# 从头训练：不要设置 RESUME_FROM_CHECKPOINT / RESUME_ONLY_MODEL / IGNORE_DATA_SKIP
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
REPORT_TO="${REPORT_TO:-wandb}"
DEEPSPEED="${DEEPSPEED:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-false}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
LAZY_TOKENIZE="${LAZY_TOKENIZE:-true}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen35-2b-twist}"
export WANDB_NAME="${WANDB_NAME:-qwen35-2b-twist-lora-$(date +%Y%m%d-%H%M%S)}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
  sft
  --model "${MODEL}" \
  --use_hf true \
  --tuner_type lora \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${VAL_JSONL}" \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --group_by_length "${GROUP_BY_LENGTH}" \
  --lazy_tokenize "${LAZY_TOKENIZE}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --logging_steps 5 \
  --report_to "${REPORT_TO}" \
  --max_length "${MAX_LENGTH}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --model_author eii \
  --model_name qwen35-2b-twist
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  args+=(
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
    --resume_only_model "${RESUME_ONLY_MODEL}"
    --ignore_data_skip "${IGNORE_DATA_SKIP}"
  )
fi

if [ -n "${DEEPSPEED}" ]; then
  args+=(--deepspeed "${DEEPSPEED}")
fi

swift "${args[@]}"
