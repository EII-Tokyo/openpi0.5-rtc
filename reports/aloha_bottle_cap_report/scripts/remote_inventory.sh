#!/usr/bin/env bash
set -euo pipefail

report_root="/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report"
artifact="${report_root}/artifacts/remote_103_inventory.txt"

ssh -o BatchMode=yes -o ConnectTimeout=8 192.168.1.103 > "${artifact}" <<'REMOTE'
set -euo pipefail
cd /home/eii/openpi0.5-rlt

echo "# identity"
pwd
git rev-parse --show-toplevel
git remote -v
git branch --show-current
git rev-parse HEAD
git status --short --branch
echo "# recent_commits"
git log -n 50 --date=iso-strict --pretty=format:'%H%x09%ad%x09%s'
echo

echo "# root_sizes"
du -sh checkpoints data logs parameter_analysis_round08 rlt_outputs rlt_runs src scripts wandb 2>/dev/null || true

echo "# source_config_inventory"
find src scripts -maxdepth 6 -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' \) \
  -printf '%s\t%T@  %p\n' | sort -k3

echo "# checkpoint_inventory"
find checkpoints rlt_runs rlt_outputs -maxdepth 7 \
  \( -type f -o -type l \) \
  \( -name '*.msgpack' -o -name '*.npz' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
     -o -name '*.pkl' -o -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' \
     -o -name '_CHECKPOINT_METADATA' -o -name '_METADATA' -o -name 'checkpoint' -o -name 'params' \
     -o -name 'LATEST' -o -name 'BEST' \) \
  -printf '%y\t%s\t%T@  %p -> %l\n' | sort -k4

echo "# logs_results_inventory"
find logs parameter_analysis_round08 rlt_outputs rlt_runs wandb -maxdepth 7 -type f \
  \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o -name '*.csv' \
     -o -name '*.yaml' -o -name '*.yml' -o -name 'wandb-summary.json' \
     -o -name 'wandb-metadata.json' -o -name 'output.log' \) \
  -printf '%s\t%T@  %p\n' | sort -k3

echo "# data_inventory"
find data -maxdepth 6 -type f \
  \( -name '*.hdf5' -o -name '*.h5' -o -name '*.npz' -o -name '*.json' \
     -o -name '*.jsonl' -o -name '*.parquet' -o -name '*.mp4' \) \
  -printf '%s\t%T@  %p\n' | sort -k3

echo "# media_inventory"
find data logs parameter_analysis_round08 rlt_outputs rlt_runs -maxdepth 7 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
     -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.webm' \) \
  -printf '%s\t%T@  %p\n' | sort -k3
REMOTE

printf 'artifact=%s\nlines=%s\nbytes=%s\n' \
  "${artifact}" "$(wc -l < "${artifact}")" "$(stat -c '%s' "${artifact}")"
