#!/usr/bin/env bash
set -euo pipefail

out="/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report/artifacts/dataset_statistics.json"

ssh -o BatchMode=yes 192.168.1.103 'bash -s' > "${out}" <<'REMOTE'
cd /home/eii/openpi0.5-rlt
.venv/bin/python - <<'PY'
import json
import math
import re
from pathlib import Path

import numpy as np

raw_root = Path("data/rlt_online_replay/rinse_smoke")
replay_root = Path("data/rlt_replay_buffer/rinse_smoke_rtc_window10_35_stride5_h25_next25_s4_reward1_clean_plus_online339_840")
raw_files = sorted(raw_root.glob("episode_*.npz"))
replay_files = sorted(replay_root.glob("episode_*.npz"))

lengths = []
replay_lengths = []
timestamp_bad = 0
missing_trim = 0
success = 0
failure = 0
unlabeled = 0
success_episode_ids = []
failure_episode_ids = []
nan_action = 0
zero_action_rows = 0
action_rows = 0
trim_lengths = []
task_values = set()
subtask_values = set()
camera_counts = {}

for path in raw_files:
    with np.load(path, allow_pickle=False) as data:
        ts = data["timestamps"]
        lengths.append(len(ts))
        if len(ts) > 1 and np.any(np.diff(ts) <= 0):
            timestamp_bad += 1
        task_values.update(map(str, np.unique(data["task"])))
        subtask_values.update(map(str, np.unique(data["subtask"])))
    trim = Path(str(path) + ".trim.json")
    if not trim.exists():
        missing_trim += 1
    else:
        meta = json.loads(trim.read_text())
        a = int(meta.get("trim_start_step", 0))
        b = int(meta.get("trim_end_step", len(ts) - 1))
        trim_lengths.append(max(0, b - a + 1))
        rewards = meta.get("frame_rewards", {})
        total = sum(float(v) for v in rewards.values())
        if total > 0:
            success += 1
            success_episode_ids.append(int(re.search(r"(\d+)$", path.stem).group(1)))
        elif "frame_rewards" in meta:
            failure += 1
            failure_episode_ids.append(int(re.search(r"(\d+)$", path.stem).group(1)))
        else:
            unlabeled += 1
    stem = path.stem
    for cam in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
        camera_counts[cam] = camera_counts.get(cam, 0) + int((raw_root / f"{stem}.{cam}.mp4").exists())

for path in replay_files:
    with np.load(path, allow_pickle=False) as data:
        replay_lengths.append(len(data["td_reward"]))
        act = data["normalized_executed_action_chunk"]
        nan_action += int(np.isnan(act).sum())
        rows = act.reshape(-1, act.shape[-1])
        zero_action_rows += int(np.all(rows == 0, axis=1).sum())
        action_rows += int(rows.shape[0])

def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "count": int(a.size), "sum": int(a.sum()), "min": int(a.min()),
        "max": int(a.max()), "mean": float(a.mean()), "median": float(np.median(a)),
        "std": float(a.std()), "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
    }

result = {
    "evidence_root": "/home/eii/openpi0.5-rlt",
    "raw_dataset": {
        "path": str(raw_root), "format": "NPZ + per-camera MP4 + trim JSON",
        "episodes": len(raw_files), "frames": stats(lengths),
        "episode_frame_lengths": lengths,
        "trimmed_frames": stats(trim_lengths), "missing_trim_json": missing_trim,
        "episode_trimmed_lengths": trim_lengths,
        "success_by_positive_terminal_reward": success,
        "failure_by_zero_reward": failure, "unlabeled": unlabeled,
        "success_episode_ids": success_episode_ids,
        "failure_episode_ids": failure_episode_ids,
        "timestamp_non_monotonic_episodes": timestamp_bad,
        "camera_video_counts": camera_counts,
        "task_values": sorted(task_values), "subtask_values": sorted(subtask_values),
    },
    "training_replay": {
        "path": str(replay_root), "episodes": len(replay_files),
        "transitions": stats(replay_lengths), "state_dim_effective": 14,
        "episode_transition_lengths": replay_lengths,
        "stored_normalized_state_dim": 32, "action_dim_effective": 14,
        "stored_model_action_dim": 32, "action_horizon": 25,
        "rlt_token_dim": 2048, "nan_action_values": nan_action,
        "all_zero_action_rows": zero_action_rows, "action_rows": action_rows,
    },
    "split": {
        "validation_episode_ids": [19,103,115,196,238,304,625,634,641,714,728],
        "validation_episodes": 11,
        "nominal_training_episodes": len(replay_files) - 11,
        "test_episodes": 0,
    },
    "limitations": [
        "Success/failure is inferred from manually authored frame_rewards, not an automatic physical success detector.",
        "Duplicate trajectory detection was not performed because no invariant episode-level content hash is stored.",
        "Image decode integrity is checked separately on selected videos; all 3340 videos were not fully decoded.",
        "No independent test split is present in the effective round-32 config."
    ],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
REMOTE

python3 -m json.tool "${out}" >/dev/null
printf 'artifact=%s bytes=%s\n' "${out}" "$(stat -c '%s' "${out}")"
