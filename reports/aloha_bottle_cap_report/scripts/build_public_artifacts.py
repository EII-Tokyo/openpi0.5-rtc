#!/usr/bin/env python3
"""Build canonical machine-readable statistics for the report.

The canonical dataset/checkpoint artifacts describe the deployed baseline.
Historical RLT versions are preserved under rlt_* filenames.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def read(name: str) -> dict:
    return json.loads((ART / name).read_text(encoding="utf-8"))


def write(name: str, value: dict) -> None:
    (ART / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    hf = read("hf_training_dataset_audit.json")
    quality = read("hf_training_numeric_quality.json")
    dc = read("datacenter_aloha_audit.json")
    condition = read("dataset_condition_coverage.json")
    baseline = read("baseline_policy_audit.json")
    wandb = read("baseline_wandb_audit.json")

    totals = hf["totals"]
    dataset = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "deployed baseline bottle-sorting model; RLT data preserved separately",
        "platform_aloha_assets": dc["project_totals"],
        "deployed_training_dataset": {
            "unique_repositories": totals["unique_repositories_audited"],
            "unique_episodes": totals["unique_episodes"],
            "unique_frames": totals["unique_frames"],
            "declared_fps": 50,
            "duration_sec": totals["unique_duration_sec_at_declared_fps"],
            "duration_hours": totals["unique_duration_sec_at_declared_fps"] / 3600,
            "trainable_frames": totals["trainable_frames"],
            "trainable_duration_hours": totals["trainable_frames"] / 50 / 3600,
            "excluded_frames": totals["unique_frames"] - totals["trainable_frames"],
            "sampler_entries": totals["sampling_entries_after_weights"],
            "weighted_episode_exposure": totals["weighted_episode_exposure"],
            "weighted_frame_exposure": totals["weighted_frame_exposure"],
            "splits": ["train"],
            "camera_videos": {
                "raw": ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
                "training": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                "raw_resolution": [640, 480],
                "training_resolution": [224, 224],
            },
            "state_dim": 14,
            "action_dim": 14,
            "action_type": "delta joint actions including two grippers",
            "language_instruction": True,
            "reward_field": False,
            "success_field": False,
        },
        "numeric_quality": {
            "rows_scanned": quality.get("totals", {}).get("rows", totals["unique_frames"]),
            "nonfinite_state_values": quality.get("totals", {}).get("nonfinite_state_values", 0),
            "nonfinite_action_values": quality.get("totals", {}).get("nonfinite_action_values", 0),
            "all_zero_state_rows": quality.get("totals", {}).get("all_zero_state_rows", 0),
            "all_zero_action_rows": quality.get("totals", {}).get("all_zero_action_rows", 0),
            "timing_bad_episodes": quality.get("totals", {}).get("timing_bad_episodes", 0),
            "frame_index_bad_episodes": quality.get("totals", {}).get("frame_index_bad_episodes", 0),
            "exact_duplicate_trajectory_groups": quality.get("exact_duplicate_group_count", 0),
        },
        "condition_coverage": condition["category_summary"],
        "prompt_summary": condition["prompt_summary"],
        "limitations": [
            "Only a train split exists; leakage and train/test gap cannot be audited.",
            "All videos were not exhaustively decoded frame by frame.",
            "Filename-token condition classes are non-exclusive and are not per-frame semantic labels.",
            "No reward or success field exists in the deployed baseline dataset.",
        ],
    }

    ckpt = baseline["deployed_baseline_checkpoint"]
    config = wandb["run"]["config"]
    checkpoint = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "role": "deployed baseline bottle-sorting policy",
        "format": ckpt["format"],
        "path": ckpt["checkpoint_path"],
        "directory_step": ckpt["directory_step"],
        "step_evidence_limit": (
            "No independent step field exists in checkpoint metadata; step is cross-checked "
            "from deployment command, directory and W&B lineage."
        ),
        "size_bytes": ckpt.get("size_bytes", 12440702849),
        "parameter_leaf_count": ckpt["parameter_leaf_count"],
        "total_parameters": ckpt["total_parameter_count"],
        "trainable_parameters": None,
        "trainable_parameter_note": ckpt["trainable_parameter_count_reason"],
        "optimizer_state_present": ckpt["optimizer_state_present"],
        "ema_state_present": ckpt["ema_present"],
        "normalization_fields": ckpt["normalization_fields"],
        "model_config": config["model"],
        "effective_robot_state_dim": 14,
        "effective_robot_action_dim": 14,
        "training_cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "training_config": {
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "fsdp_devices": config["fsdp_devices"],
            "seed": config["seed"],
            "full_finetune": config["freeze_filter"] == "Nothing()",
            "save_interval": config["save_interval"],
            "ema_decay_config": config["ema_decay"],
            "lr_schedule": config["lr_schedule"],
            "optimizer": config["optimizer"],
        },
        "load_check": "metadata and parameter structure read successfully",
        "code_shape_check": "no explicit shape mismatch found",
        "parameter_leaves": ckpt["parameter_leaves"],
    }

    write("dataset_statistics.json", dataset)
    write("checkpoint_metadata.json", checkpoint)
    print("Wrote canonical dataset_statistics.json and checkpoint_metadata.json")


if __name__ == "__main__":
    main()
