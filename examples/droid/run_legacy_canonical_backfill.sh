#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT_DEFAULT="$ROOT_DIR/examples/droid/data/success"
ANNOTATIONS_DEFAULT="$ROOT_DIR/examples/droid/speed/annotations_tmp.jsonl"
MONGO_URL_DEFAULT="mongodb://localhost:27017/eii_data_system"
MONGO_DB_DEFAULT="eii_data_system"
DEST_ROOT_BASE_DEFAULT="/tmp"
S3_SUCCESS_ROOT_DEFAULT="s3://openpi-tokyo/droid_xxjd_data/success"
RAW_CACHE_ROOT_DEFAULT="/tmp/droid_xxjd_raw_success_cache"

RAW_ROOT="${RAW_ROOT:-$RAW_ROOT_DEFAULT}"
ANNOTATIONS_PATH="${ANNOTATIONS_PATH:-$ANNOTATIONS_DEFAULT}"
MONGO_URL="${MONGO_URL:-$MONGO_URL_DEFAULT}"
MONGO_DB_NAME="${MONGO_DB_NAME:-$MONGO_DB_DEFAULT}"
DEST_ROOT_BASE="${DEST_ROOT_BASE:-$DEST_ROOT_BASE_DEFAULT}"
S3_SUCCESS_ROOT="${S3_SUCCESS_ROOT:-$S3_SUCCESS_ROOT_DEFAULT}"
RAW_CACHE_ROOT="${RAW_CACHE_ROOT:-$RAW_CACHE_ROOT_DEFAULT}"
PUSH_TO_HUB="${PUSH_TO_HUB:-1}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
DOWNLOAD_FROM_S3="${DOWNLOAD_FROM_S3:-1}"

REPOS=(
  "michios/droid_xxjd_2"
  "michios/droid_xxjd_3"
  "michios/droid_xxjd_4"
  "michios/droid_xxjd_5"
  "michios/droid_xxjd_6"
  "michios/droid_xxjd_7"
  "michios/droid_xxjd_8_2"
)

date_filters_for_repo() {
  case "$1" in
    "michios/droid_xxjd")
      printf '%s\n' "2025-10-07 2025-10-08 2025-10-09"
      ;;
    "michios/droid_xxjd_2")
      printf '%s\n' "2025-10-22 2025-10-23"
      ;;
    "michios/droid_xxjd_3")
      printf '%s\n' "2025-10-28 2025-10-29"
      ;;
    "michios/droid_xxjd_4")
      printf '%s\n' "2025-10-31 2025-11-04"
      ;;
    "michios/droid_xxjd_5")
      printf '%s\n' "2025-11-06"
      ;;
    "michios/droid_xxjd_7")
      printf '%s\n' "2025-12-03 2025-12-04 2025-12-05 2025-12-08 2025-12-09 2025-12-11"
      ;;
    "michios/droid_xxjd_8_2")
      printf '%s\n' "2025-12-22 2025-12-23 2026-01-14 2026-01-15"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ ! -f "$ANNOTATIONS_PATH" ]]; then
  echo "ANNOTATIONS_PATH does not exist: $ANNOTATIONS_PATH" >&2
  exit 1
fi

if [[ "$DOWNLOAD_FROM_S3" != "1" && ! -d "$RAW_ROOT" ]]; then
  echo "RAW_ROOT does not exist and DOWNLOAD_FROM_S3=0: $RAW_ROOT" >&2
  exit 1
fi

if [[ "$DOWNLOAD_FROM_S3" == "1" ]] && ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required when DOWNLOAD_FROM_S3=1" >&2
  exit 1
fi

cd "$ROOT_DIR"

for repo in "${REPOS[@]}"; do
  date_filters="$(date_filters_for_repo "$repo")"
  dest_repo="${repo}_canonical"
  dest_dir="$DEST_ROOT_BASE/$(basename "$dest_repo")"
  repo_raw_root="$RAW_ROOT"
  downloaded_dates=()

  echo
  echo "=== Converting $repo -> $dest_repo ==="
  echo "Dates: $date_filters"
  echo "Output: $dest_dir"

  if [[ "$CLEAN_OUTPUT" == "1" && -d "$dest_dir" ]]; then
    rm -rf "$dest_dir"
  fi

  cmd=(
    uv run python -m examples.droid.convert_legacy_lerobot_to_canonical
    --source-repo-ids "$repo"
    --destination-repo-id "$dest_repo"
    --destination-root "$dest_dir"
  )

  if [[ "$repo" != "michios/droid_xxjd_6" ]]; then
    if [[ "$DOWNLOAD_FROM_S3" == "1" ]]; then
      repo_raw_root="$RAW_CACHE_ROOT"
      mkdir -p "$repo_raw_root"
      for date in $date_filters; do
        echo "Syncing $S3_SUCCESS_ROOT/$date/ -> $repo_raw_root/$date/"
        aws s3 sync "$S3_SUCCESS_ROOT/$date/" "$repo_raw_root/$date/" --no-progress
        downloaded_dates+=("$date")
      done
    fi

    cmd+=(
      --raw-data-roots "$repo_raw_root"
      --annotations-path "$ANNOTATIONS_PATH"
      --mongo-url "$MONGO_URL"
      --mongo-db-name "$MONGO_DB_NAME"
      --mongo-project-date-filters
    )

    for date in $date_filters; do
      cmd+=("$date")
    done

    cmd+=(--allow-task-length-fallback)
  fi

  if [[ "$PUSH_TO_HUB" == "1" ]]; then
    cmd+=(--push-to-hub)
  fi

  "${cmd[@]}"

  if [[ "$DOWNLOAD_FROM_S3" == "1" && "$repo" != "michios/droid_xxjd_6" ]]; then
    for date in "${downloaded_dates[@]}"; do
      if [[ -d "$repo_raw_root/$date" ]]; then
        echo "Deleting cached raw data: $repo_raw_root/$date"
        rm -rf "$repo_raw_root/$date"
      fi
    done
  fi
done

echo
echo "Finished all requested legacy DROID canonical conversions."
