#!/usr/bin/env bash
# Serve one locally downloaded run 4 checkpoint; does not start the robot client.
set -euo pipefail

usage() {
  printf '%s\n' 'Usage: bash examples/fastumi/serve_run4.sh {final|75k|50k|25k|mid} [port]' \
    'Default port: 8001. mid selects the final 200k-step midtraining checkpoint.'
}

case "${1:-}" in
  final|100k) phase=posttrain; step=99999 ;;
  75k) phase=posttrain; step=75000 ;;
  50k) phase=posttrain; step=50000 ;;
  25k) phase=posttrain; step=25000 ;;
  mid) phase=midtrain; step=199999 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
checkpoint="checkpoints/pi05_fastumi_${phase}/umi_pi05_run4_${phase}/${step}"
if [[ ! -d "$checkpoint/params" || ! -f "$checkpoint/assets/fastumi_mid_post/norm_stats.json" ]]; then
  printf 'Checkpoint is not ready: %s\n' "$repo_root/$checkpoint" >&2
  exit 1
fi

exec .venv/bin/python scripts/serve_policy.py --port="${2:-8001}" policy:checkpoint \
  --policy.config="pi05_fastumi_${phase}" \
  --policy.dir="$checkpoint"
