#!/usr/bin/env bash
set -euo pipefail

project_root="/home/eii/project/openpi0.5-rtc-reward-learning"
report_root="${project_root}/reports/aloha_bottle_cap_report"
artifact="${report_root}/artifacts/project_inventory.txt"

{
  echo "# generated_at"
  date --iso-8601=seconds
  echo "# git"
  git -C "${project_root}" remote -v
  git -C "${project_root}" branch --show-current
  git -C "${project_root}" rev-parse HEAD
  git -C "${project_root}" status --short --branch
  echo "# recent_commits"
  git -C "${project_root}" log -n 20 --date=iso-strict --pretty=format:'%H%x09%ad%x09%s'
  echo
  echo "# top_level"
  find "${project_root}" -mindepth 1 -maxdepth 1 -printf '%y\t%f\n' | sort
  echo "# candidate_source_config_docs"
  rg --files "${project_root}" \
    -g '*.py' -g '*.sh' -g '*.yaml' -g '*.yml' -g '*.toml' -g '*.json' -g '*.jsonl' -g '*.md' \
    -g '!reports/aloha_bottle_cap_report/**' \
    -g '!.git/**' -g '!.venv*/**' -g '!wandb/**' -g '!local_rlt_runs/**' \
    -g '!local_eval_assets/**' -g '!checkpoints/**' -g '!external/**' \
    | sort
  echo "# checkpoint_files"
  find "${project_root}/checkpoints" -maxdepth 7 -type f \
    -printf '%s\t%T@  %p\n' | sort -k3
  echo "# run_log_candidates"
  find "${project_root}/local_rlt_runs" "${project_root}/logs" "${project_root}/outputs" \
    -maxdepth 5 -type f \
    \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.yaml' -o -name '*.yml' -o -name 'metrics*' -o -name 'config*' \) \
    -printf '%s\t%T@  %p\n' | sort -k3
  echo "# media_candidates"
  find "${project_root}/local_eval_assets" "${project_root}/outputs" "${project_root}/assets" \
    -maxdepth 6 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.webm' \) \
    -printf '%s\t%T@  %p\n' | sort -k3
  echo "# external_data_roots"
  for root in \
    "/home/eii/project/bottles_data" \
    "/home/eii/data/openpi0.5-rtc-reward-learning"; do
    if [[ -d "${root}" ]]; then
      find "${root}" -maxdepth 4 -type f \
        \( -name '*.hdf5' -o -name '*.h5' -o -name '*.npz' -o -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' -o -name '*.mp4' \) \
        -printf '%s\t%T@  %p\n' | sort -k3
    fi
  done
} > "${artifact}"

line_count="$(wc -l < "${artifact}")"
file_size="$(stat -c '%s' "${artifact}")"
printf 'artifact=%s\nlines=%s\nbytes=%s\n' "${artifact}" "${line_count}" "${file_size}"
