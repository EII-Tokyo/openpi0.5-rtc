#!/usr/bin/env python3
"""Audit runtime policy-forward z_rl against offline same-forward z_rl.

This script answers a narrow question:

    Does a z_rl stored by the robot at a real VLA policy-forward event match
    the z_rl we later recompute offline from the same rollout frame?

It deliberately does not compare against sidecar tokens. The only two sources
are:

* runtime /rlt_policy_forward_events/z_rl from the HDF5 rollout
* offline cam4 VLA same-forward low/right tokens encoded by the lower-right
  RLToken autoencoder
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
import jax
import numpy as np

from scripts.compare_vla_same_forward_vs_sidecar_tokens import (
    DEFAULT_CAM4_CHECKPOINT,
    DEFAULT_CAM4_CONFIG,
    DEFAULT_SIDECAR_CHECKPOINT,
    DEFAULT_SIDECAR_CONFIG,
)
from scripts.rebuild_online_rollout_paper_anchor_replay import VLAAnchorExtractor, encode_blocks
from scripts.reencode_clean_no_actor_z_rl import _VideoFrameReader, _load_qpos


DEFAULT_ROLLOUT_ROOT = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/"
    "twist_off_the_bottle_cap/2026-07-07/rl"
)
DEFAULT_OUTPUT_DIR = Path("local_rlt_reports/policy_event_vs_offline_same_forward_z_20260707")
DEFAULT_PROMPT = "Twist off the bottle cap."


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / (norms + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(_normalize_rows(a[None, :]) * _normalize_rows(b[None, :]), axis=-1)[0])


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


def _select_balanced(paths: list[Path], *, max_success: int, max_failure: int) -> list[Path]:
    success: list[Path] = []
    failure: list[Path] = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            reward = float(h5.attrs.get("reward", 0.0))
        if reward > 0:
            success.append(path)
        else:
            failure.append(path)
    return success[:max_success] + failure[:max_failure]


def _read_policy_events(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        group = h5["rlt_policy_forward_events"]
        step_index = np.asarray(group["step_index"], dtype=np.int64)
        z_rl = np.asarray(group["z_rl"], dtype=np.float32)
        proprio = np.asarray(group["proprio"], dtype=np.float32)
        z_rl_source = _h5_attr_str(group.attrs.get("z_rl_source", ""))
        reward = float(h5.attrs.get("reward", 0.0))
        key_region_id = _h5_attr_str(h5.attrs.get("key_region_id", path.parent.name))
        action_len = int(h5["action"].shape[0]) if "action" in h5 else int(len(_load_qpos(path)))
    return {
        "step_index": step_index,
        "z_rl": z_rl,
        "proprio": proprio,
        "z_rl_source": z_rl_source,
        "reward": reward,
        "key_region_id": key_region_id.removeprefix("key_region_"),
        "action_len": action_len,
    }


def _load_autoencoder(config_name: str, checkpoint: Path):
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    cfg = train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt=DEFAULT_PROMPT)
    autoencoder = getattr(policy._model, "rl_token_autoencoder", None)  # noqa: SLF001
    if autoencoder is None:
        raise ValueError(f"{config_name} checkpoint {checkpoint} does not contain rl_token_autoencoder")
    return autoencoder


def audit_one_rollout(
    *,
    hdf5_path: Path,
    extractor: VLAAnchorExtractor,
    autoencoder: Any,
    shifts: tuple[int, ...],
    vla_batch_size: int,
    encode_batch_size: int,
) -> list[dict[str, Any]]:
    events = _read_policy_events(hdf5_path)
    rollout_dir = hdf5_path.parent
    qpos = _load_qpos(hdf5_path)
    action_len = int(events["action_len"])
    frame_requests: list[tuple[int, int, int]] = []
    for event_idx, step in enumerate(events["step_index"]):
        for shift in shifts:
            frame = int(step) + int(shift)
            if 0 <= frame < min(len(qpos), action_len):
                frame_requests.append((event_idx, int(shift), frame))

    unique_frames = sorted({frame for _, _, frame in frame_requests})
    by_frame: dict[int, dict[str, np.ndarray]] = {}
    reader = _VideoFrameReader(rollout_dir, convert_bgr_to_rgb=False)
    try:
        for start in range(0, len(unique_frames), vla_batch_size):
            batch_frames = unique_frames[start : start + vla_batch_size]
            observations = [
                {
                    "images": reader.read_all(frame),
                    "state": np.asarray(qpos[frame], dtype=np.float32),
                    "prompt": DEFAULT_PROMPT,
                }
                for frame in batch_frames
            ]
            for frame, features in zip(batch_frames, extractor.extract_batch(observations), strict=True):
                by_frame[int(frame)] = features
    finally:
        reader.close()

    low = np.stack([by_frame[frame]["low_tokens"] for frame in unique_frames]).astype(np.float32)
    right = np.stack([by_frame[frame]["right_tokens"] for frame in unique_frames]).astype(np.float32)
    offline_z = encode_blocks(autoencoder, low, right, batch_size=encode_batch_size)
    offline_by_frame = {frame: offline_z[index] for index, frame in enumerate(unique_frames)}

    rows: list[dict[str, Any]] = []
    for event_idx, shift, frame in frame_requests:
        event_z = np.asarray(events["z_rl"][event_idx], dtype=np.float32)
        event_proprio = np.asarray(events["proprio"][event_idx], dtype=np.float32)
        offline = np.asarray(offline_by_frame[frame], dtype=np.float32)
        offline_proprio = np.asarray(by_frame[frame]["proprio"], dtype=np.float32)
        rows.append(
            {
                "key_region_id": events["key_region_id"],
                "hdf5_path": str(hdf5_path),
                "reward": events["reward"],
                "event_index": event_idx,
                "event_step": int(events["step_index"][event_idx]),
                "shift": shift,
                "frame_index": frame,
                "z_rl_source": events["z_rl_source"],
                "cosine": _cosine(event_z, offline),
                "l2": float(np.linalg.norm(event_z - offline)),
                "event_norm": float(np.linalg.norm(event_z)),
                "offline_norm": float(np.linalg.norm(offline)),
                "proprio_l2": float(np.linalg.norm(event_proprio - offline_proprio)),
                "proprio_max_abs": float(np.max(np.abs(event_proprio - offline_proprio))),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "status": "blocked", "reason": "no comparable policy-forward events"}
    by_event: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault((str(row["key_region_id"]), int(row["event_index"])), []).append(row)

    same_rows = [row for row in rows if int(row["shift"]) == 0]
    best_rows = [max(items, key=lambda row: float(row["cosine"])) for items in by_event.values()]
    same_cos = np.asarray([row["cosine"] for row in same_rows], dtype=np.float32)
    best_cos = np.asarray([row["cosine"] for row in best_rows], dtype=np.float32)
    best_shifts = [int(row["shift"]) for row in best_rows]
    proprio_same = np.asarray([row["proprio_l2"] for row in same_rows], dtype=np.float32)
    return {
        "num_rollouts": len({row["key_region_id"] for row in rows}),
        "num_events": len(by_event),
        "num_comparisons": len(rows),
        "same_frame_cos_mean": float(np.mean(same_cos)) if same_cos.size else None,
        "same_frame_cos_min": float(np.min(same_cos)) if same_cos.size else None,
        "same_frame_cos_p05": float(np.quantile(same_cos, 0.05)) if same_cos.size else None,
        "best_shift_cos_mean": float(np.mean(best_cos)) if best_cos.size else None,
        "best_shift_cos_min": float(np.min(best_cos)) if best_cos.size else None,
        "best_shift_zero_fraction": float(np.mean(np.asarray(best_shifts) == 0)) if best_shifts else None,
        "best_shift_counts": {str(shift): int(best_shifts.count(shift)) for shift in sorted(set(best_shifts))},
        "same_frame_proprio_l2_mean": float(np.mean(proprio_same)) if proprio_same.size else None,
        "same_frame_proprio_l2_max": float(np.max(proprio_same)) if proprio_same.size else None,
        "is_offline_same_forward_equivalent_to_runtime_event": bool(
            same_cos.size > 0 and float(np.min(same_cos)) >= 0.999 and float(np.max(proprio_same)) <= 1e-4
        ),
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "policy_event_vs_offline_same_forward_z.csv"
    json_path = output_dir / "policy_event_vs_offline_same_forward_z.json"
    report_path = output_dir / "policy_event_vs_offline_same_forward_z_report.md"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(summary, csv_path=csv_path, json_path=json_path), encoding="utf-8")


def _render_report(summary: dict[str, Any], *, csv_path: Path, json_path: Path) -> str:
    ok = summary.get("is_offline_same_forward_equivalent_to_runtime_event")
    if ok:
        verdict = "离线 same-forward z 与 runtime policy-forward event z 在本样本上等价。"
    else:
        verdict = "离线 same-forward z 与 runtime policy-forward event z 在本样本上不满足严格等价阈值。"
    return "\n".join(
        [
            "# Policy-forward event z vs offline same-forward z 审计",
            "",
            "## 结论",
            "",
            verdict,
            "",
            "## 核心指标",
            "",
            f"- rollouts: {summary.get('num_rollouts')}",
            f"- events: {summary.get('num_events')}",
            f"- same-frame cosine mean: {summary.get('same_frame_cos_mean')}",
            f"- same-frame cosine min: {summary.get('same_frame_cos_min')}",
            f"- best-shift cosine mean: {summary.get('best_shift_cos_mean')}",
            f"- best-shift zero fraction: {summary.get('best_shift_zero_fraction')}",
            f"- best-shift counts: {summary.get('best_shift_counts')}",
            f"- same-frame proprio L2 max: {summary.get('same_frame_proprio_l2_max')}",
            "",
            "## 判读",
            "",
            "- 如果 same-frame cosine 接近 0.999+ 且 proprio 差异接近 0，说明离线 same-forward 转换能复现 runtime actor 实际看到的 z。",
            "- 如果 best-shift 明显不是 0，说明存在 frame 对齐偏移。",
            "- 如果 same-frame 和 best-shift 都低，说明离线转换路径、图像预处理、checkpoint/config 或 runtime 保存语义至少有一处不一致。",
            "",
            "## 输出",
            "",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-success", type=int, default=20)
    parser.add_argument("--max-failure", type=int, default=20)
    parser.add_argument("--shift-window", type=int, default=5)
    parser.add_argument("--vla-batch-size", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--vla-config", default=DEFAULT_CAM4_CONFIG)
    parser.add_argument("--vla-checkpoint", type=Path, default=DEFAULT_CAM4_CHECKPOINT)
    parser.add_argument("--rl-token-config", default=DEFAULT_SIDECAR_CONFIG)
    parser.add_argument("--rl-token-checkpoint", type=Path, default=DEFAULT_SIDECAR_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    paths = discover_hdf5_with_policy_events(args.rollout_root)
    selected = _select_balanced(paths, max_success=args.max_success, max_failure=args.max_failure)
    if not selected:
        summary = summarize([])
        write_outputs([], summary, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    logging.info("selected %d/%d HDF5 rollouts with policy-forward events", len(selected), len(paths))
    extractor = VLAAnchorExtractor(config_name=args.vla_config, checkpoint=args.vla_checkpoint, prompt=DEFAULT_PROMPT)
    autoencoder = _load_autoencoder(args.rl_token_config, args.rl_token_checkpoint)
    shifts = tuple(range(-int(args.shift_window), int(args.shift_window) + 1))
    rows: list[dict[str, Any]] = []
    for index, hdf5_path in enumerate(selected, start=1):
        logging.info("auditing %d/%d %s", index, len(selected), hdf5_path)
        rows.extend(
            audit_one_rollout(
                hdf5_path=hdf5_path,
                extractor=extractor,
                autoencoder=autoencoder,
                shifts=shifts,
                vla_batch_size=args.vla_batch_size,
                encode_batch_size=args.encode_batch_size,
            )
        )
    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
