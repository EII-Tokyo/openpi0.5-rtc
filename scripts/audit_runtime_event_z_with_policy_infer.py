#!/usr/bin/env python3
"""Compare runtime event z_rl with offline Policy.infer same-forward z_rl.

This script is stricter than a manual token-block converter audit. It loads the
same cam4 VLA policy plus same-forward lower/right RLToken encoder, then runs
the public Policy methods on the exact rollout frames where runtime recorded a
real VLA policy-forward event.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import h5py
import numpy as np

from openpi.policies import policy_config, same_forward_rl_token
from openpi.training import config as train_config
from scripts.compare_vla_same_forward_vs_sidecar_tokens import (
    DEFAULT_CAM4_CHECKPOINT,
    DEFAULT_CAM4_CONFIG,
    DEFAULT_SIDECAR_CHECKPOINT,
    DEFAULT_SIDECAR_CONFIG,
)
from scripts.reencode_clean_no_actor_z_rl import _VideoFrameReader, _load_qpos


DEFAULT_ROLLOUT_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/"
    "twist_off_the_bottle_cap/2026-07-07/rl"
)
DEFAULT_OUTPUT_DIR = Path("local_rlt_reports/runtime_event_z_with_policy_infer_20260707")
DEFAULT_PROMPT = "Twist off the bottle cap."


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _h5_attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def discover_hdf5_with_policy_events(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("episode.hdf5")):
        try:
            with h5py.File(path, "r") as h5:
                if "rlt_policy_forward_events/z_rl" in h5 and "rlt_policy_forward_events/step_index" in h5:
                    paths.append(path)
        except OSError:
            logging.exception("failed opening %s", path)
    return paths


def create_policy(*, vla_config: str, vla_checkpoint: Path, rl_token_config: str, rl_token_checkpoint: Path):
    logging.info("loading same-forward encoder config=%s checkpoint=%s", rl_token_config, rl_token_checkpoint)
    encoder = same_forward_rl_token.load_same_forward_rl_token_encoder(
        config_name=rl_token_config,
        checkpoint_dir=rl_token_checkpoint,
    )
    logging.info("loading VLA policy config=%s checkpoint=%s", vla_config, vla_checkpoint)
    cfg = train_config.get_config(vla_config)
    return policy_config.create_trained_policy(
        cfg,
        vla_checkpoint,
        default_prompt=DEFAULT_PROMPT,
        same_forward_rl_token_encoder=encoder,
    )


def audit_rollout(path: Path, policy: Any, *, convert_bgr_to_rgb: bool) -> list[dict[str, Any]]:
    rollout_dir = path.parent
    qpos = _load_qpos(path)
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        group = h5["rlt_policy_forward_events"]
        steps = np.asarray(group["step_index"], dtype=np.int64)
        runtime_z = np.asarray(group["z_rl"], dtype=np.float32)
        runtime_proprio = np.asarray(group["proprio"], dtype=np.float32)
        key_region_id = _h5_attr_str(h5.attrs.get("key_region_id", path.parent.name)).removeprefix("key_region_")
        reward = float(h5.attrs.get("reward", 0.0))
        z_source = _h5_attr_str(group.attrs.get("z_rl_source", ""))

    reader = _VideoFrameReader(rollout_dir, convert_bgr_to_rgb=convert_bgr_to_rgb)
    try:
        for event_index, step in enumerate(steps):
            frame = int(step)
            obs = {
                "images": reader.read_all(frame),
                "state": np.asarray(qpos[frame], dtype=np.float32),
                "prompt": DEFAULT_PROMPT,
            }
            infer_result = policy.infer(obs, prev_action=None, use_rtc=True)
            token_result = policy.infer_rl_token(obs)
            for source_name, result in (("policy_infer_use_rtc", infer_result), ("policy_infer_rl_token", token_result)):
                offline_z = np.asarray(result["z_rl"], dtype=np.float32)
                offline_proprio = np.asarray(result["state"], dtype=np.float32)
                rows.append(
                    {
                        "key_region_id": key_region_id,
                        "hdf5_path": str(path),
                        "reward": reward,
                        "event_index": int(event_index),
                        "event_step": frame,
                        "runtime_z_source": z_source,
                        "offline_source": source_name,
                        "offline_z_source": str(result.get("z_rl_source", "")),
                        "cosine": _cosine(runtime_z[event_index], offline_z),
                        "l2": float(np.linalg.norm(runtime_z[event_index] - offline_z)),
                        "runtime_norm": float(np.linalg.norm(runtime_z[event_index])),
                        "offline_norm": float(np.linalg.norm(offline_z)),
                        "proprio_l2": float(np.linalg.norm(runtime_proprio[event_index] - offline_proprio)),
                        "proprio_max_abs": float(np.max(np.abs(runtime_proprio[event_index] - offline_proprio))),
                    }
                )
    finally:
        reader.close()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "num_events": len(rows) // 2,
        "by_source": {},
    }
    for source in sorted({row["offline_source"] for row in rows}):
        subset = [row for row in rows if row["offline_source"] == source]
        cos = np.asarray([row["cosine"] for row in subset], dtype=np.float32)
        l2 = np.asarray([row["l2"] for row in subset], dtype=np.float32)
        prop = np.asarray([row["proprio_l2"] for row in subset], dtype=np.float32)
        summary["by_source"][source] = {
            "count": len(subset),
            "cos_mean": float(np.mean(cos)) if cos.size else None,
            "cos_min": float(np.min(cos)) if cos.size else None,
            "l2_mean": float(np.mean(l2)) if l2.size else None,
            "proprio_l2_mean": float(np.mean(prop)) if prop.size else None,
            "equivalent": bool(cos.size and float(np.min(cos)) >= 0.999),
        }
    return summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runtime_event_z_with_policy_infer.csv"
    json_path = output_dir / "runtime_event_z_with_policy_infer.json"
    md_path = output_dir / "runtime_event_z_with_policy_infer_report.md"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Runtime event z vs offline Policy.infer z",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- `policy_infer_use_rtc` tests the same public inference path used to create a runtime action chunk.",
        "- `policy_infer_rl_token` tests the deterministic z-only path on the same frame.",
        "- If both are far from runtime event z, saved runtime event z and offline reconstruction are not numerically equivalent.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vla-config", default=DEFAULT_CAM4_CONFIG)
    parser.add_argument("--vla-checkpoint", type=Path, default=DEFAULT_CAM4_CHECKPOINT)
    parser.add_argument("--rl-token-config", default=DEFAULT_SIDECAR_CONFIG)
    parser.add_argument("--rl-token-checkpoint", type=Path, default=DEFAULT_SIDECAR_CHECKPOINT)
    parser.add_argument("--convert-bgr-to-rgb", action="store_true")
    parser.add_argument("--limit-rollouts", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    paths = discover_hdf5_with_policy_events(args.rollout_root)
    if args.limit_rollouts is not None:
        paths = paths[: args.limit_rollouts]
    logging.info("found %d HDF5 rollouts with policy events under %s", len(paths), args.rollout_root)
    policy = create_policy(
        vla_config=args.vla_config,
        vla_checkpoint=args.vla_checkpoint,
        rl_token_config=args.rl_token_config,
        rl_token_checkpoint=args.rl_token_checkpoint,
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        logging.info("auditing %s", path)
        rows.extend(audit_rollout(path, policy, convert_bgr_to_rgb=args.convert_bgr_to_rgb))
    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
