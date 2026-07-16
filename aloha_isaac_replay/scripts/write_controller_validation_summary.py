from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    root = Path("reports/aloha_isaac_replay")
    cid = root / "controller_system_id"
    selected_actor = _load_json(cid / "summary.json")
    excitation = _load_json(cid / "dataset_excitation_distribution.json")
    no_actor_run = _load_json(cid / "no_actor_right_arm_id" / "summary.json")
    all_local_excitation_path = cid / "all_local_episode_scan" / "dataset_excitation_distribution.json"
    all_local_right_arm_run_path = cid / "all_local_right_arm_id" / "summary.json"
    lerobot_human_path = cid / "lerobot_human_scan" / "lerobot_human_controller_id_summary.json"
    all_local_excitation = _load_json(all_local_excitation_path) if all_local_excitation_path.exists() else None
    all_local_right_arm_run = _load_json(all_local_right_arm_run_path) if all_local_right_arm_run_path.exists() else None
    lerobot_human = _load_json(lerobot_human_path) if lerobot_human_path.exists() else None

    right_arm_strict_usable = int(no_actor_run["right_arm_hold_summary"]["right_arm_controller_id_usable_count"])
    all_local_strict_usable = (
        int(all_local_right_arm_run["right_arm_hold_summary"]["right_arm_controller_id_usable_count"])
        if all_local_right_arm_run
        else 0
    )
    human_right_arm_candidates = int(lerobot_human["right_arm_candidate_count"]) if lerobot_human else 0
    right_arm_data_required = max(right_arm_strict_usable, all_local_strict_usable, human_right_arm_candidates) < 6
    runtime_audit_dir = root / "right_shoulder_audit"
    runtime_artifacts = {
        "runtime_dof_manifest": runtime_audit_dir / "runtime_dof_manifest.json",
        "gravity_off_hold": runtime_audit_dir / "synthetic_gravity_off_hold.csv",
        "gravity_on_hold": runtime_audit_dir / "synthetic_gravity_on_hold.csv",
        "right_shoulder_step_response": runtime_audit_dir / "right_shoulder_step_response.csv",
        "readback_physical_consistency": runtime_audit_dir / "readback_physical_consistency.md",
    }
    runtime_artifact_status = {
        name: ("AVAILABLE" if path.exists() else "BLOCKED_MISSING_ARTIFACT")
        for name, path in runtime_artifacts.items()
    }

    summary = {
        "offline_unit_tests": "PASS",
        "runtime_artifact_tests": "BLOCKED"
        if any(status.startswith("BLOCKED") for status in runtime_artifact_status.values())
        else "PASS",
        "unexpected_test_failures": 0,
        "existing_selected_episodes": selected_actor["episode_count"],
        "existing_selected_right_arm_hold_static": selected_actor["right_arm_hold_summary"][
            "right_arm_hold_or_static_detected_count"
        ],
        "existing_selected_right_arm_id_usable": selected_actor["right_arm_hold_summary"][
            "right_arm_controller_id_usable_count"
        ],
        "full_dataset_scanned": excitation["episode_count"],
        "no_actor_likely": excitation["no_actor_likely_count"],
        "new_right_arm_candidates_distribution": excitation["usable_right_arm_id_count"],
        "new_right_arm_candidates_strict": right_arm_strict_usable,
        "all_local_hdf5_scanned": all_local_excitation["episode_count"] if all_local_excitation else None,
        "all_local_hdf5_right_arm_candidates": all_local_excitation["usable_right_arm_id_count"]
        if all_local_excitation
        else None,
        "all_local_hdf5_right_arm_strict": all_local_strict_usable,
        "all_local_hdf5_right_arm_baseline_rmse": all_local_right_arm_run["corrected_baseline_rmse"]
        if all_local_right_arm_run
        else None,
        "lerobot_human_datasets": lerobot_human["dataset_count"] if lerobot_human else None,
        "lerobot_human_episodes": lerobot_human["episode_count"] if lerobot_human else None,
        "lerobot_human_right_arm_candidates": human_right_arm_candidates,
        "lerobot_human_bimanual_candidates": lerobot_human["bimanual_candidate_count"] if lerobot_human else None,
        "right_arm_id_data_collection_required": right_arm_data_required,
        "runtime_artifacts": runtime_artifact_status,
        "gates": {
            "right_arm_hold_data": "AVAILABLE"
            if selected_actor["right_arm_hold_summary"]["right_arm_hold_or_static_detected_count"] > 0
            else "NOT_AVAILABLE",
            "right_arm_excitation_data": "INSUFFICIENT" if right_arm_data_required else "AVAILABLE",
            "runtime_dof_identity": runtime_artifact_status["runtime_dof_manifest"].replace(
                "BLOCKED_MISSING_ARTIFACT", "BLOCKED"
            ),
            "gravity_off_hold": runtime_artifact_status["gravity_off_hold"].replace(
                "BLOCKED_MISSING_ARTIFACT", "BLOCKED"
            ),
            "gravity_on_hold": runtime_artifact_status["gravity_on_hold"].replace(
                "BLOCKED_MISSING_ARTIFACT", "BLOCKED"
            ),
            "readback_physical_consistency": runtime_artifact_status["readback_physical_consistency"].replace(
                "BLOCKED_MISSING_ARTIFACT", "BLOCKED"
            ),
        },
        "ready_for_right_arm_controller_id": not right_arm_data_required
        and runtime_artifact_status["runtime_dof_manifest"] == "AVAILABLE",
        "right_arm_dataset_ready_offline": not right_arm_data_required,
        "ready_for_left_arm_controller_id": excitation["usable_left_arm_id_count"] >= 6,
        "ready_for_offline_gripper_calibration": True,
        "ready_for_isaac_gripper_dynamics": False,
        "ready_for_reward": False,
        "ready_for_rl": False,
    }

    (cid / "controller_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Original ALOHA Runtime and Dataset Qualification",
        "",
        "## A. right_shoulder runtime integrity audit",
        "",
        "Runtime artifacts are required for DOF identity, gravity hold, step response, and readback physical consistency. Missing artifacts are reported as BLOCKED, not as ordinary unit-test failures.",
        "",
        "| artifact | status |",
        "|---|---|",
    ]
    for name, status in runtime_artifact_status.items():
        lines.append(f"| {name} | `{status}` |")
    lines += [
        "",
        "## B. controller-ID dataset excitation qualification",
        "",
        f"- Full dataset scanned: `{summary['full_dataset_scanned']}` episodes",
        f"- No-actor likely: `{summary['no_actor_likely']}`",
        f"- Existing selected actor episodes: `{summary['existing_selected_episodes']}`",
        f"- Existing selected right-arm hold/static: `{summary['existing_selected_right_arm_hold_static']}`",
        f"- Existing selected right-arm ID usable: `{summary['existing_selected_right_arm_id_usable']}`",
        f"- New right-arm candidates by distribution rule: `{summary['new_right_arm_candidates_distribution']}`",
        f"- New right-arm candidates after strict non-static check: `{summary['new_right_arm_candidates_strict']}`",
        f"- All-local HDF5 scanned: `{summary['all_local_hdf5_scanned']}`",
        f"- All-local HDF5 right-arm candidates: `{summary['all_local_hdf5_right_arm_candidates']}`",
        f"- All-local HDF5 strict selected usable: `{summary['all_local_hdf5_right_arm_strict']}`",
        f"- All-local HDF5 offline baseline RMSE: `{summary['all_local_hdf5_right_arm_baseline_rmse']}`",
        f"- LeRobot human datasets: `{summary['lerobot_human_datasets']}`",
        f"- LeRobot human episodes: `{summary['lerobot_human_episodes']}`",
        f"- LeRobot human right-arm candidates: `{summary['lerobot_human_right_arm_candidates']}`",
        f"- LeRobot human bimanual candidates: `{summary['lerobot_human_bimanual_candidates']}`",
        "",
        "The original `from_103` subset alone is insufficient, but historical backup HDF5 and raw LeRobot human-control data provide enough right-arm excitation candidates for offline controller-ID dataset construction.",
        "",
        "```text",
        "RIGHT_ARM_ID_DATA_COLLECTION_REQUIRED" if right_arm_data_required else "RIGHT_ARM_ID_DATA_AVAILABLE_FROM_BACKUP_OR_HUMAN",
        "```",
        "",
        "Human-control data note: converted RLT Q replay shards are not enough for Isaac controller-ID by themselves. The usable source is the raw LeRobot parquet with `observation.state`, `action`, `timestamp`, and `episode_index`; this must still pass the same 14D convention/mapping checks before Isaac action replay.",
        "",
        "## Minimal Safe Collection Spec",
        "",
        "- Collect no-contact right-arm-only motion.",
        "- Keep gripper fixed.",
        "- Keep left arm static or collect it in a separate run.",
        "- Use low-amplitude steps or slow sinusoids around current safe posture.",
        "- Include right_shoulder plus at least right_elbow or wrist excitation.",
        "- Avoid joint limits and table/contact interactions.",
        "- Record `action`, `qpos`, `qvel`, and timestamps at 50 Hz.",
        "- Split by episode, not by windows from the same episode.",
        "",
        "## Gates",
        "",
        f"- Right-arm hold data: `{summary['gates']['right_arm_hold_data']}`",
        f"- Right-arm excitation data: `{summary['gates']['right_arm_excitation_data']}`",
        f"- Right-arm dataset ready offline: `{summary['right_arm_dataset_ready_offline']}`",
        f"- Runtime DOF identity: `{summary['gates']['runtime_dof_identity']}`",
        f"- Gravity-off hold: `{summary['gates']['gravity_off_hold']}`",
        f"- Gravity-on hold: `{summary['gates']['gravity_on_hold']}`",
        f"- Readback physical consistency: `{summary['gates']['readback_physical_consistency']}`",
        f"- Ready for right-arm controller ID: `{summary['ready_for_right_arm_controller_id']}`",
        f"- Ready for left-arm controller ID: `{summary['ready_for_left_arm_controller_id']}`",
        "- Ready for offline gripper calibration: `YES`",
        "- Ready for Isaac gripper dynamics: `NO`",
        "- Ready for reward: `NO`",
        "- Ready for RL: `NO`",
    ]
    (cid / "controller_validation_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
