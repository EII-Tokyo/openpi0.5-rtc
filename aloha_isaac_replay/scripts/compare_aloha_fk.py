from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers
from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


ARM_NAMES = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def _load_qpos(path: Path, max_frames: int | None) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        qpos = h5["observations/qpos"][:]
    if max_frames is not None:
        qpos = qpos[:max_frames]
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected /observations/qpos shape (T, 14), got {qpos.shape}")
    return qpos


def _pin_model(urdf: Path):
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    return model, data


def _pin_q(model, side: str, qpos_frame: np.ndarray) -> np.ndarray:
    q = pin.neutral(model)
    side_offset = 0 if side == "left" else 7
    gripper_value = float(qpos_frame[6 if side == "left" else 13])
    fingers = standard_gripper_qpos_to_isaac_fingers(gripper_value, side=side)
    values = {
        "waist": float(qpos_frame[side_offset + 0]),
        "shoulder": float(qpos_frame[side_offset + 1]),
        "elbow": float(qpos_frame[side_offset + 2]),
        "forearm_roll": float(qpos_frame[side_offset + 3]),
        "wrist_angle": float(qpos_frame[side_offset + 4]),
        "wrist_rotate": float(qpos_frame[side_offset + 5]),
        "gripper": 0.0,
        "left_finger": float(fingers[f"{side}/left_finger"]),
        "right_finger": float(fingers[f"{side}/right_finger"]),
    }
    for joint_name, value in values.items():
        if not model.existJointName(joint_name):
            continue
        joint_id = model.getJointId(joint_name)
        idx = model.joints[joint_id].idx_q
        nq = model.joints[joint_id].nq
        q[idx : idx + nq] = value
    return q


def _pin_ee(model, data, side: str, qpos_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = _pin_q(model, side, qpos_frame)
    pin.framesForwardKinematics(model, data, q)
    frame_name = f"puppet_{side}/ee_gripper_link"
    if not model.existFrame(frame_name):
        raise ValueError(f"Pinocchio model is missing EE frame {frame_name}; frames={list(model.frames)}")
    placement = data.oMf[model.getFrameId(frame_name)]
    quat_xyzw = Rotation.from_matrix(np.asarray(placement.rotation)).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return np.asarray(placement.translation, dtype=np.float64), quat_wxyz


def _side_values(qpos_frame: np.ndarray, side: str, mapping: dict) -> tuple[np.ndarray, list[str]]:
    targets = arm_only_targets_from_standard_qpos(qpos_frame, mapping)
    values = [target.value for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    names = [target.isaac_dof_name.split("/", 1)[1] for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    fingers = standard_gripper_qpos_to_isaac_fingers(qpos_frame[6 if side == "left" else 13], side=side)
    for finger_name in ("left_finger", "right_finger"):
        names.append(finger_name)
        values.append(float(fingers[f"{side}/{finger_name}"]))
    return np.asarray(values, dtype=np.float64), names


def _indices(actual_dof_names: list[str], names: list[str]) -> list[int]:
    missing = [name for name in names if name not in actual_dof_names]
    if missing:
        raise ValueError(f"Missing DOF names {missing}; actual={actual_dof_names}")
    return [actual_dof_names.index(name) for name in names]


def _link_transform(art, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    view = art._articulation_view
    body_idx = view.get_body_index(body_name)
    raw = np.asarray(view._physics_view.get_link_transforms())
    if raw.ndim == 3:
        raw = raw[0]
    raw = raw.reshape((-1, 7))
    pose = raw[body_idx]
    pos = pose[:3].astype(np.float64)
    quat_xyzw = pose[3:7].astype(np.float64)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return pos, quat_wxyz


def _orientation_error_deg(q_ref_wxyz: np.ndarray, q_isaac_wxyz: np.ndarray) -> float:
    ref = Rotation.from_quat([q_ref_wxyz[1], q_ref_wxyz[2], q_ref_wxyz[3], q_ref_wxyz[0]])
    isaac = Rotation.from_quat([q_isaac_wxyz[1], q_isaac_wxyz[2], q_isaac_wxyz[3], q_isaac_wxyz[0]])
    return float((ref.inv() * isaac).magnitude() * 180.0 / np.pi)


def _write_rows(path: Path, header: list[str], rows: list[list[float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare independent Pinocchio FK against Isaac runtime link transforms.")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--mapping", default="configs/aloha/original_stationary_aloha_mapping.yaml")
    parser.add_argument("--left-urdf", default="assets/isaac/original_stationary_aloha/generated/puppet_left_vx300s_resolved.urdf")
    parser.add_argument("--right-urdf", default="assets/isaac/original_stationary_aloha/generated/puppet_right_vx300s_resolved.urdf")
    parser.add_argument("--left-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_left.usd")
    parser.add_argument("--right-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_right.usd")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        qpos = _load_qpos(Path(args.episode), args.max_frames)
        mapping = load_mapping(args.mapping)
        left_model, left_data = _pin_model(Path(args.left_urdf))
        right_model, right_data = _pin_model(Path(args.right_urdf))

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        left = world.scene.add(SingleArticulation(prim_path="/World/left/root_joint/root_joint", name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path="/World/right/root_joint/root_joint", name="right_vx300s"))
        world.reset()

        left_indices = _indices(list(left.dof_names), _side_values(qpos[0], "left", mapping)[1])
        right_indices = _indices(list(right.dof_names), _side_values(qpos[0], "right", mapping)[1])

        side_rows = {"left": [], "right": []}
        position_errors = {"left": [], "right": []}
        orientation_errors = {"left": [], "right": []}
        for frame_idx, qpos_frame in enumerate(qpos):
            left_values, _ = _side_values(qpos_frame, "left", mapping)
            right_values, _ = _side_values(qpos_frame, "right", mapping)
            left.set_joint_positions(left_values, joint_indices=np.asarray(left_indices, dtype=np.int64))
            right.set_joint_positions(right_values, joint_indices=np.asarray(right_indices, dtype=np.int64))
            for side, art, model, data in (
                ("left", left, left_model, left_data),
                ("right", right, right_model, right_data),
            ):
                ref_pos, ref_quat = _pin_ee(model, data, side, qpos_frame)
                isaac_pos, isaac_quat = _link_transform(art, f"puppet_{side}_ee_gripper_link")
                pos_err = float(np.linalg.norm(isaac_pos - ref_pos))
                ori_err = _orientation_error_deg(ref_quat, isaac_quat)
                position_errors[side].append(pos_err)
                orientation_errors[side].append(ori_err)
                side_rows[side].append(
                    [
                        frame_idx,
                        *ref_pos.tolist(),
                        *isaac_pos.tolist(),
                        pos_err,
                        ori_err,
                    ]
                )

        header = ["frame", "ref_x", "ref_y", "ref_z", "isaac_x", "isaac_y", "isaac_z", "position_error_m", "orientation_error_deg"]
        _write_rows(output / "left_ee_position.csv", header, side_rows["left"])
        _write_rows(output / "right_ee_position.csv", header, side_rows["right"])
        metrics = {
            "status": "PASS"
            if max(max(position_errors["left"]), max(position_errors["right"])) <= 0.005
            and max(max(orientation_errors["left"]), max(orientation_errors["right"])) <= 2.0
            else "FAIL",
            "episode": args.episode,
            "frames": int(qpos.shape[0]),
            "reference_fk_source": "Pinocchio built from generated URDF resolved from archived 103 robot_description",
            "isaac_fk_source": "PhysX articulation get_link_transforms for puppet_{side}_ee_gripper_link",
            "frame_alignment": "asset-local root frame; audited 103 base static transform is recorded but not applied to this generated USD yet",
            "left_position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors["left"])))),
            "right_position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors["right"])))),
            "left_position_max_m": float(max(position_errors["left"])),
            "right_position_max_m": float(max(position_errors["right"])),
            "left_orientation_max_deg": float(max(orientation_errors["left"])),
            "right_orientation_max_deg": float(max(orientation_errors["right"])),
            "warning_threshold_position_m": 0.005,
            "warning_threshold_orientation_deg": 2.0,
        }
        (output / "fk_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        (output / "frame_alignment.json").write_text(
            json.dumps(
                {
                    "comparison_frame": "asset-local root frame",
                    "real_103_base_transform_applied": False,
                    "reason": "Current generated USD side assets are rooted at the imported URDF base; applying measured left/right workcell base transforms is a later calibration step.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0 if metrics["status"] == "PASS" else 2
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
