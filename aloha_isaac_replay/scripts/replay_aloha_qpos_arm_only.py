from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.replay.arm_only_mapping import ARM_ONLY_NAMES
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.runtime.eula_status import probe_isaac_simulation_app


def _side_dof_name(logical_name: str, side: str) -> str:
    prefix = f"{side}/"
    if not logical_name.startswith(prefix):
        raise ValueError(f"Expected {logical_name!r} to start with {prefix!r}")
    return logical_name[len(prefix) :]


def _resolve_indices(actual_dof_names: list[str], mapped_names: list[str], side: str) -> list[int]:
    stripped = [_side_dof_name(name, side) for name in mapped_names]
    missing = [name for name in stripped if name not in actual_dof_names]
    if missing:
        raise ValueError(
            f"{side} articulation is missing mapped DOFs {missing}; "
            f"actual DOFs are {actual_dof_names}"
        )
    indices = [actual_dof_names.index(name) for name in stripped]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{side} mapping resolves duplicate DOF indices: {indices}")
    return indices


def _write_csv(path: Path, header: list[str], rows: list[list[float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_joint_error(path: Path, header: list[str], errors: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(errors.shape[0])
    for idx, name in enumerate(header[1:]):
        ax.plot(x, errors[:, idx], linewidth=1.0, label=name)
    ax.axhline(1e-5, color="red", linestyle="--", linewidth=1, label="1e-5 rad gate")
    ax.axhline(-1e-5, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("readback - expected")
    ax.set_title("Arm-only qpos readback error")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_qpos(episode: Path, max_frames: int | None) -> np.ndarray:
    with h5py.File(episode, "r") as h5:
        qpos = h5["observations/qpos"][:]
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected /observations/qpos shape (T, 14), got {qpos.shape}")
    if max_frames is not None:
        qpos = qpos[:max_frames]
    if not np.isfinite(qpos).all():
        raise ValueError(f"Episode qpos contains NaN/Inf: {episode}")
    return qpos


def _run_isaac_replay(
    *,
    episode: Path,
    mapping_path: Path,
    output: Path,
    left_usd: Path,
    right_usd: Path,
    left_prim_path: str,
    right_prim_path: str,
    max_frames: int | None,
    include_gripper: bool,
) -> int:
    print("ARM_ONLY_REPLAY_STAGE=before_simulation_app", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        print("ARM_ONLY_REPLAY_STAGE=after_simulation_app", flush=True)
        import omni.kit.app
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        print("ARM_ONLY_REPLAY_STAGE=loading_qpos", flush=True)
        qpos = _load_qpos(episode, max_frames=max_frames)
        mapping = load_mapping(mapping_path)
        output.mkdir(parents=True, exist_ok=True)

        print("ARM_ONLY_REPLAY_STAGE=creating_world", flush=True)
        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        print("ARM_ONLY_REPLAY_STAGE=adding_references", flush=True)
        stage_utils.add_reference_to_stage(usd_path=str(left_usd.resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(right_usd.resolve()), prim_path="/World/right")
        left = world.scene.add(SingleArticulation(prim_path=left_prim_path, name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path=right_prim_path, name="right_vx300s"))
        print("ARM_ONLY_REPLAY_STAGE=world_reset", flush=True)
        world.reset()
        omni.kit.app.get_app().update()

        print("ARM_ONLY_REPLAY_STAGE=resolving_dofs", flush=True)
        left_names = list(left.dof_names)
        right_names = list(right.dof_names)
        first_targets = arm_only_targets_from_standard_qpos(qpos[0], mapping)
        left_target_names = [target.isaac_dof_name for target in first_targets if target.isaac_dof_name.startswith("left/")]
        right_target_names = [target.isaac_dof_name for target in first_targets if target.isaac_dof_name.startswith("right/")]
        if include_gripper:
            left_target_names.extend(["left/left_finger", "left/right_finger"])
            right_target_names.extend(["right/left_finger", "right/right_finger"])
        left_indices = _resolve_indices(left_names, left_target_names, "left")
        right_indices = _resolve_indices(right_names, right_target_names, "right")

        gripper_names = (
            "left_gripper_left_finger",
            "left_gripper_right_finger",
            "right_gripper_left_finger",
            "right_gripper_right_finger",
        )
        canonical_header = ["frame", *ARM_ONLY_NAMES, *(gripper_names if include_gripper else ())]
        expected_rows: list[list[float | int]] = []
        readback_rows: list[list[float | int]] = []
        error_rows: list[list[float | int]] = []
        errors = []

        print(f"ARM_ONLY_REPLAY_STAGE=writing_frames count={qpos.shape[0]}", flush=True)
        for frame_idx, qpos_frame in enumerate(qpos):
            targets = arm_only_targets_from_standard_qpos(qpos_frame, mapping)
            left_values = np.array([target.value for target in targets if target.isaac_dof_name.startswith("left/")])
            right_values = np.array([target.value for target in targets if target.isaac_dof_name.startswith("right/")])
            if include_gripper:
                left_fingers = standard_gripper_qpos_to_isaac_fingers(qpos_frame[6], side="left")
                right_fingers = standard_gripper_qpos_to_isaac_fingers(qpos_frame[13], side="right")
                left_values = np.concatenate(
                    [
                        left_values,
                        np.array(
                            [
                                float(left_fingers["left/left_finger"]),
                                float(left_fingers["left/right_finger"]),
                            ],
                            dtype=np.float64,
                        ),
                    ]
                )
                right_values = np.concatenate(
                    [
                        right_values,
                        np.array(
                            [
                                float(right_fingers["right/left_finger"]),
                                float(right_fingers["right/right_finger"]),
                            ],
                            dtype=np.float64,
                        ),
                    ]
                )
            left.set_joint_positions(left_values, joint_indices=np.array(left_indices, dtype=np.int64))
            right.set_joint_positions(right_values, joint_indices=np.array(right_indices, dtype=np.int64))
            left_readback = left.get_joint_positions(joint_indices=np.array(left_indices, dtype=np.int64))
            right_readback = right.get_joint_positions(joint_indices=np.array(right_indices, dtype=np.int64))
            expected = np.concatenate([left_values, right_values])
            readback = np.concatenate([left_readback, right_readback])
            error = readback - expected
            expected_rows.append([frame_idx, *expected.tolist()])
            readback_rows.append([frame_idx, *readback.tolist()])
            error_rows.append([frame_idx, *error.tolist()])
            errors.append(error)

        error_array = np.asarray(errors, dtype=np.float64)
        abs_error = np.abs(error_array)
        max_abs_error = float(abs_error.max()) if abs_error.size else 0.0
        metrics = {
            "status": "PASS" if max_abs_error < 1e-5 else "FAIL",
            "episode": str(episode),
            "mapping": str(mapping_path),
            "left_usd": str(left_usd),
            "right_usd": str(right_usd),
            "frames": int(qpos.shape[0]),
            "left_prim_path": left.prim_path,
            "right_prim_path": right.prim_path,
            "left_dof_names": left_names,
            "right_dof_names": right_names,
            "left_indices": left_indices,
            "right_indices": right_indices,
            "include_gripper": include_gripper,
            "gripper_source": "HDF5 observations/qpos[6] and observations/qpos[13]; action gripper values are intentionally unused",
            "ignored_isaac_dofs": ["left/gripper", "right/gripper"] if include_gripper else ["left/gripper", "right/gripper", "left/left_finger", "left/right_finger", "right/left_finger", "right/right_finger"],
            "max_abs_readback_error": max_abs_error,
            "mean_abs_readback_error": float(abs_error.mean()) if abs_error.size else 0.0,
            "gate_max_abs_error_rad": 1e-5,
            "video_status": "BLOCKED_VISUAL_MESH_IMPORT_HAS_ZERO_MESHES",
        }
        _write_csv(output / "expected_qpos.csv", canonical_header, expected_rows)
        _write_csv(output / "readback_qpos.csv", canonical_header, readback_rows)
        _write_csv(output / "joint_error.csv", canonical_header, error_rows)
        _plot_joint_error(output / "joint_error.png", canonical_header, error_array)
        (output / "replay_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        (output / "replay_config.json").write_text(
            json.dumps(
                {
                    "mode": "full_qpos_readback" if include_gripper else "arm_only_qpos_readback",
                    "uses_action": False,
                    "uses_controller": False,
                    "uses_dynamics": False,
                    "qpos_indices": {
                        "left_arm": list(range(0, 6)),
                        "right_arm": list(range(7, 13)),
                        "gripper": [6, 13] if include_gripper else [],
                        "ignored": [] if include_gripper else [6, 13],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0 if metrics["status"] == "PASS" else 2
    except BaseException as exc:
        print(f"ARM_ONLY_REPLAY_EXCEPTION={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        print("ARM_ONLY_REPLAY_STAGE=closing_simulation_app", flush=True)
        app.close(skip_cleanup=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run arm-only qpos replay/readback in Isaac.")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=".venv_issac/bin/python")
    parser.add_argument("--left-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_left.usd")
    parser.add_argument("--right-usd", default="assets/isaac/original_stationary_aloha/generated/vx300s_right.usd")
    parser.add_argument("--left-prim-path", default="/World/left/root_joint")
    parser.add_argument("--right-prim-path", default="/World/right/root_joint")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--probe-runtime", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    with h5py.File(args.episode, "r") as h5:
        first_qpos = h5["observations/qpos"][0]
    targets = arm_only_targets_from_standard_qpos(first_qpos, mapping)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "arm_only_mapping_preview.json").write_text(
        json.dumps([target.__dict__ for target in targets], ensure_ascii=False, indent=2) + "\n"
    )

    if args.probe_runtime:
        probe = probe_isaac_simulation_app(args.python)
        (output / "runtime_probe.json").write_text(json.dumps(probe.__dict__, ensure_ascii=False, indent=2) + "\n")
        if probe.manual_action_required:
            print("MANUAL EULA ACCEPTANCE REQUIRED")
            return 3
        if probe.returncode != 0:
            print("ISAAC RUNTIME PROBE FAILED")
            return 4
    if args.preview_only:
        print("arm-only mapping preview written.")
        return 0
    return _run_isaac_replay(
        episode=Path(args.episode),
        mapping_path=Path(args.mapping),
        output=output,
        left_usd=Path(args.left_usd),
        right_usd=Path(args.right_usd),
        left_prim_path=args.left_prim_path,
        right_prim_path=args.right_prim_path,
        max_frames=args.max_frames,
        include_gripper=args.include_gripper,
    )


if __name__ == "__main__":
    raise SystemExit(main())
