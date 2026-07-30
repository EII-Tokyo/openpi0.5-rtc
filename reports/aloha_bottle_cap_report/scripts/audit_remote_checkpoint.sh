#!/usr/bin/env bash
set -euo pipefail

out="/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report/artifacts/checkpoint_metadata.json"

ssh -o BatchMode=yes 192.168.1.103 'bash -s' > "${out}" <<'REMOTE'
cd /home/eii/openpi0.5-rlt
.venv/bin/python - <<'PY'
import json
import pickle
from pathlib import Path

import numpy as np

run = Path("rlt_runs/rlt_rtc_window10_35_s4_reward1_clean_835eps_widedeep1536x8_online_round32_constantlr3e-4_coef10_tdposbalanced_val11")
ckpt = run / "rlt_actor_critic"
config = json.loads((ckpt / "config.json").read_text())
with (ckpt / "params.pkl").open("rb") as f:
    params = pickle.load(f)

entries = []
def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            walk(value, f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            walk(value, f"{prefix}/{i}")
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        entries.append({"name": prefix, "shape": list(obj.shape), "dtype": str(obj.dtype), "count": int(np.prod(obj.shape))})
walk(params)

by_root = {}
for item in entries:
    root = item["name"].split("/", 1)[0]
    by_root[root] = by_root.get(root, 0) + item["count"]

token_ckpt = Path("rlt_runs/rlt_token_rinse_9000_bs64_nw4_warmup2000_10000_abs/9999")
result = {
    "evidence_host": "192.168.1.103",
    "repository": "/home/eii/openpi0.5-rlt",
    "selected_checkpoint": str(ckpt),
    "format": "Python pickle parameter tree + optimizer_state.pkl + JSON config/split/target_diff",
    "saved_step": 300,
    "epoch": 0,
    "parameter_tensors": len(entries),
    "total_parameters": sum(x["count"] for x in entries),
    "trainable_parameters": sum(x["count"] for x in entries),
    "parameters_by_root": by_root,
    "parameter_entries": entries,
    "optimizer_state_exists": (ckpt / "optimizer_state.pkl").exists(),
    "target_parameters_exist": (ckpt / "target_params.pkl").exists(),
    "ema_parameters_exist": False,
    "normalization_statistics_in_checkpoint": False,
    "config": config,
    "token_encoder_checkpoint": {
        "path": str(token_ckpt), "format": "Orbax",
        "saved_step": 9999,
        "checkpoint_metadata_exists": (token_ckpt / "_CHECKPOINT_METADATA").exists(),
        "config": json.loads((token_ckpt / "rlt_token_config.json").read_text()),
        "training": json.loads((token_ckpt / "training.json").read_text()),
    },
    "load_check": "params.pkl successfully unpickled in the remote project .venv",
    "limitations": [
        "The VLA checkpoint configured under /data/openpi0.5-rtc/checkpoints is outside the user-authorized /home/eii/openpi0.5-rlt audit boundary and was not loaded.",
        "Epoch is inferred from the final training log (epoch=0); the checkpoint config does not persist an explicit epoch field.",
        "All actor/critic parameter arrays are treated as trainable because the offline trainer exposes no frozen parameter mask for this checkpoint."
    ]
}
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
REMOTE

python3 -m json.tool "${out}" >/dev/null
printf 'artifact=%s bytes=%s\n' "${out}" "$(stat -c '%s' "${out}")"
