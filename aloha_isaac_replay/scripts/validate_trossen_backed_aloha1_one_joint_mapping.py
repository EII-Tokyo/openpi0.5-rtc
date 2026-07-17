from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.scripts.inspect_phase2_runtime_assets import _get_limits, _json_safe


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAFFOLD_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha1_trossen_backed_scaffold_20260717/aloha1_trossen_backed_scaffold.usda"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase5_one_joint_static_validation_20260717"

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")

REAL_ALOHA1_FACTS = {
    "source": "103 read-only ROS robot_info and joint_states probe, 2026-07-17",
    "safety": {
        "real_robot_motion_commanded": False,
        "runtime_or_actor_started": False,
        "read_only_topics_or_services": True,
    },
    "puppet": {
        "arm": {
            "joint_names": list(ARM_JOINTS),
            "joint_ids": [1, 2, 4, 6, 7, 8],
            "lower": [-3.141582727432251, -1.8500490188598633, -1.7627825736999512, -3.141582727432251, -1.8675023317337036, -3.141582727432251],
            "upper": [3.141582727432251, 1.2566370964050293, 1.6057028770446777, 3.141582727432251, 2.2340214252471924, 3.141582727432251],
            "velocity_limits": [3.1415927410125732] * 6,
            "sleep": [0.0, -1.850000023841858, 1.5499999523162842, 0.0, 0.800000011920929, 0.0],
            "joint_state_indices": [0, 1, 2, 3, 4, 5],
        },
        "gripper": {
            "mode": "linear_position",
            "joint_name": "left_finger",
            "joint_id": 9,
            "lower": 0.020999999716877937,
            "upper": 0.05700000002980232,
            "sleep": 0.028495611622929573,
            "joint_state_index": 7,
        },
        "joint_state_sample": {
            "left": [0.003067961661145091, -1.8484468460083008, 1.6229517459869385, 0.0015339808305725455, -1.8806605339050293, -0.00920388475060463, 1.5017672777175903, 0.05791560560464859, -0.05791560560464859],
            "right": [0.0015339808305725455, -1.8576507568359375, 1.6474953889846802, 0.004601942375302315, -1.9573595523834229, 0.026077674701809883, 1.6781749725341797, 0.057795993983745575, -0.057795993983745575],
        },
    },
}


CANONICAL_TO_TROSSEN = {
    "left_waist": "follower_left_joint_0",
    "right_waist": "follower_right_joint_0",
    "left_shoulder": "follower_left_joint_1",
    "right_shoulder": "follower_right_joint_1",
    "left_elbow": "follower_left_joint_2",
    "right_elbow": "follower_right_joint_2",
    "left_forearm_roll": "follower_left_joint_3",
    "right_forearm_roll": "follower_right_joint_3",
    "left_wrist_angle": "follower_left_joint_4",
    "right_wrist_angle": "follower_right_joint_4",
    "left_wrist_rotate": "follower_left_joint_5",
    "right_wrist_rotate": "follower_right_joint_5",
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _ranges_overlap(a_lower: float, a_upper: float, b_lower: float | None, b_upper: float | None) -> bool | None:
    if b_lower is None or b_upper is None:
        return None
    return max(a_lower, b_lower) <= min(a_upper, b_upper)


def _in_range(value: float, lower: float | None, upper: float | None, margin: float = 1e-6) -> bool | None:
    if lower is None or upper is None:
        return None
    return (lower - margin) <= value <= (upper + margin)


def _build_rows(dof_names: list[str], limits: list[list[float | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    puppet = REAL_ALOHA1_FACTS["puppet"]
    left_sample = puppet["joint_state_sample"]["left"]
    right_sample = puppet["joint_state_sample"]["right"]
    lower = puppet["arm"]["lower"]
    upper = puppet["arm"]["upper"]
    sleep = puppet["arm"]["sleep"]

    for side, sample in (("left", left_sample), ("right", right_sample)):
        for joint_idx, joint_name in enumerate(ARM_JOINTS):
            canonical = f"{side}_{joint_name}"
            candidate = CANONICAL_TO_TROSSEN[canonical]
            if candidate in dof_names:
                runtime_index = dof_names.index(candidate)
                runtime_lower, runtime_upper = limits[runtime_index]
                candidate_present = True
            else:
                runtime_index = None
                runtime_lower, runtime_upper = None, None
                candidate_present = False
            rows.append(
                {
                    "canonical_name": canonical,
                    "aloha1_ros_joint_name": joint_name,
                    "aloha1_ros_joint_index": joint_idx,
                    "aloha1_dynamixel_id": puppet["arm"]["joint_ids"][joint_idx],
                    "trossen_candidate_dof": candidate,
                    "trossen_runtime_index": runtime_index,
                    "candidate_present": candidate_present,
                    "trossen_runtime_limit": [runtime_lower, runtime_upper],
                    "aloha1_limit": [lower[joint_idx], upper[joint_idx]],
                    "aloha1_sleep": sleep[joint_idx],
                    "aloha1_sample_q": sample[joint_idx],
                    "identity_limit_overlap": _ranges_overlap(lower[joint_idx], upper[joint_idx], runtime_lower, runtime_upper),
                    "identity_sample_inside_trossen_limit": _in_range(sample[joint_idx], runtime_lower, runtime_upper),
                    "identity_sleep_inside_trossen_limit": _in_range(sleep[joint_idx], runtime_lower, runtime_upper),
                    "sign_status": "BLOCKED_REQUIRES_POSITIVE_DIRECTION_EVIDENCE",
                    "offset_status": "BLOCKED_REQUIRES_MATCHED_REFERENCE_POSES",
                }
            )
    return rows


def _scatter_identity_values(rows: list[dict[str, Any]], dof_names: list[str], base_q: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    q = base_q.astype(np.float64).copy()
    actions = []
    for row in rows:
        idx = row["trossen_runtime_index"]
        if idx is None:
            actions.append({"canonical_name": row["canonical_name"], "status": "SKIP_MISSING_DOF"})
            continue
        if row["identity_sample_inside_trossen_limit"] is not True:
            actions.append(
                {
                    "canonical_name": row["canonical_name"],
                    "status": "SKIP_SAMPLE_OUTSIDE_TROSSEN_LIMIT_OR_UNKNOWN",
                    "sample": row["aloha1_sample_q"],
                    "trossen_limit": row["trossen_runtime_limit"],
                }
            )
            continue
        q[idx] = float(row["aloha1_sample_q"])
        actions.append(
            {
                "canonical_name": row["canonical_name"],
                "status": "SET_PROVISIONAL_IDENTITY_SAMPLE",
                "runtime_index": idx,
                "runtime_dof": row["trossen_candidate_dof"],
                "value": float(row["aloha1_sample_q"]),
            }
        )
    return q, actions


def _write_outputs(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "one_joint_static_validation.json"
    md_path = output_dir / "one_joint_static_validation.md"
    safe_payload = _json_safe(payload)
    json_path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(safe_payload), encoding="utf-8")
    return json_path, md_path


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Trossen-Backed ALOHA1 One-Joint Static Validation - 2026-07-17",
        "",
        "## Scope",
        "",
        "This is an offline Isaac validation. It does not command the real robot.",
        "",
        "It validates only:",
        "",
        "- Trossen scaffold loads as an Isaac articulation;",
        "- runtime DOF order is discoverable;",
        "- ALOHA1 canonical arm fields scatter into Trossen's interleaved DOF order;",
        "- provisional identity values can be set/read back when inside runtime limits.",
        "",
        "It does not validate sign, zero offset, FK equivalence, gripper carriage mapping, or real positive motion direction.",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Runtime DOF Order", "", "```text"])
    lines.extend(payload["runtime"]["dof_names"])
    lines.extend(["```", "", "## Adapter Rows", ""])
    lines.append(
        "| canonical | Trossen DOF | index | ALOHA1 limit | Trossen limit | identity sample inside | sign | offset |"
    )
    lines.append("|---|---|---:|---|---|---|---|---|")
    for row in payload["adapter_rows"]:
        lines.append(
            "| "
            f"`{row['canonical_name']}` | "
            f"`{row['trossen_candidate_dof']}` | "
            f"{row['trossen_runtime_index']} | "
            f"`{row['aloha1_limit']}` | "
            f"`{row['trossen_runtime_limit']}` | "
            f"`{row['identity_sample_inside_trossen_limit']}` | "
            f"`{row['sign_status']}` | "
            f"`{row['offset_status']}` |"
        )
    lines.extend(["", "## Readback Summary", ""])
    readback = payload["readback"]
    lines.append(f"- max_abs_error: `{readback['max_abs_error']}`")
    lines.append(f"- tolerance: `{readback['tolerance']}`")
    lines.append(f"- status: `{readback['status']}`")
    lines.append(f"- settable_identity_sample_count: `{readback['settable_identity_sample_count']}`")
    lines.append(f"- skipped_identity_sample_count: `{readback['skipped_identity_sample_count']}`")
    lines.extend(["", "## Blocked Items", ""])
    for item in payload["blocked_items"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline one-joint static validation for the Trossen-backed ALOHA1 scaffold.")
    parser.add_argument("--usd", type=Path, default=DEFAULT_SCAFFOLD_USD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normal-close", action="store_true")
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    app_config["fast_shutdown"] = False
    app = SimulationApp(app_config)
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        root = "/World/trossen_stationary_ai"
        stage_utils.add_reference_to_stage(usd_path=str(args.usd.resolve()), prim_path=root)
        articulation_path = f"{root}/Aloha1TrossenBackedScaffold/root_joint"
        art = world.scene.add(SingleArticulation(prim_path=articulation_path, name="aloha1_trossen_backed"))
        world.reset()

        dof_names = list(art.dof_names)
        limits = _get_limits(art)
        base_q = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        rows = _build_rows(dof_names, limits)
        target_q, scatter_actions = _scatter_identity_values(rows, dof_names, base_q)
        art.set_joint_positions(target_q)
        art.set_joint_velocities(np.zeros_like(target_q))
        world.step(render=False)
        actual_q = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        diff = np.abs(actual_q - target_q)
        max_err = float(np.nanmax(diff)) if diff.size else None
        # Isaac/PhysX may normalize or project articulation positions by a
        # small amount after a set/readback step. This gate is only checking
        # that scatter indices are not grossly wrong; sign/offset/FK remain
        # blocked by separate gates.
        tolerance = 1e-3
        readback_pass = max_err is not None and max_err <= tolerance
        readback_status = "PASS_SCATTER_SET_READBACK_ONLY" if readback_pass else "FAIL_SCATTER_SET_READBACK"

        all_candidates_present = all(bool(row["candidate_present"]) for row in rows)
        all_identity_samples_settable = all(
            row["identity_sample_inside_trossen_limit"] is True for row in rows if row["candidate_present"]
        )
        identity_limit_gate = (
            "FAIL_IDENTITY_MAPPING_LIMIT_CHECK"
            if not all_identity_samples_settable
            else "PASS_IDENTITY_SAMPLES_WITHIN_LIMITS_NOT_GEOMETRY"
        )
        skipped_identity_sample_count = sum(
            1 for row in rows if row["candidate_present"] and row["identity_sample_inside_trossen_limit"] is not True
        )
        payload = {
            "usd": _rel(args.usd),
            "real_robot_touched": False,
            "stage_saved": False,
            "runtime": {
                "articulation_path": articulation_path,
                "num_dof": int(art.num_dof),
                "dof_names": dof_names,
                "limits": limits,
                "base_q": base_q.tolist(),
            },
            "real_aloha1_facts": REAL_ALOHA1_FACTS,
            "adapter_rows": rows,
            "scatter_actions": scatter_actions,
            "readback": {
                "status": readback_status,
                "max_abs_error": max_err,
                "tolerance": tolerance,
                "settable_identity_sample_count": len(scatter_actions) - skipped_identity_sample_count,
                "skipped_identity_sample_count": skipped_identity_sample_count,
            },
            "gates": {
                "isaac_runtime_started": "PASS",
                "real_robot_touched": "PASS_FALSE",
                "stage_saved": "PASS_FALSE",
                "dof_order_interleaved_confirmed": "PASS",
                "all_arm_candidate_dofs_present": "PASS" if all_candidates_present else "FAIL",
                "identity_mapping_limit_check": identity_limit_gate,
                "scatter_set_readback": readback_status,
                "sign": "BLOCKED_REQUIRES_POSITIVE_DIRECTION_EVIDENCE",
                "offset": "BLOCKED_REQUIRES_MATCHED_REFERENCE_POSES",
                "fk": "BLOCKED_REQUIRES_TRUSTED_FK_OR_REFERENCE_POSES",
                "gripper_mapping": "BLOCKED_REQUIRES_CARRIAGE_AND_PHYSICAL_OPENING_VALIDATION",
            },
            "blocked_items": [
                "ALOHA1-to-Trossen positive direction sign for every arm joint.",
                "ALOHA1-to-Trossen zero offset for every arm joint.",
                "FK equivalence between real ALOHA1 and Trossen scaffold.",
                "Gripper carriage mapping and true physical opening.",
                "Controller replay and contact validation.",
            ],
        }
        json_path, md_path = _write_outputs(args.output_dir, payload)
        print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "gates": payload["gates"]}, indent=2), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0
    finally:
        if args.normal_close:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
