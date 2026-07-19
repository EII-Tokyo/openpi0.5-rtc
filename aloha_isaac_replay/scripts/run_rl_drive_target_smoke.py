from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from aloha_isaac_replay.rl.drive_target_env import DriveTargetReplayConfig
from aloha_isaac_replay.rl.drive_target_env import load_hdf5_qpos
from aloha_isaac_replay.rl.drive_target_env import summarize_step
from aloha_isaac_replay.rl.drive_target_env import targets_from_hdf5_qpos
from aloha_isaac_replay.rl.drive_target_env import tracking_groups
from aloha_isaac_replay.rl.readiness import build_rl_readiness_report
from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.validation.contact_proxy_profiles import contact_proxy_profile_names
from aloha_isaac_replay.validation.contact_proxy_profiles import finger_dof_names_for_side
from aloha_isaac_replay.validation.contact_proxy_profiles import finger_qpos_limits_for_side
from aloha_isaac_replay.validation.contact_proxy_profiles import resolve_contact_proxy_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/"
    "aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda"
)
DEFAULT_EPISODE = (
    REPO_ROOT
    / "local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/"
    "key_region_15c193959d7d449783517a9c9d257529/episode.hdf5"
)
DEFAULT_MAPPING = REPO_ROOT / "configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports/aloha_isaac_replay/rl_drive_target_smoke/latest.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _apply_target_and_step(
    world: Any,
    art: Any,
    target: np.ndarray,
    *,
    hold_steps: int,
    render: bool,
    set_full_target: Any,
) -> np.ndarray:
    if hold_steps <= 0:
        raise ValueError(f"target_hold_steps must be positive, got {hold_steps}")
    pre_step_qpos: np.ndarray | None = None
    for _ in range(hold_steps):
        set_full_target(art, target)
        if pre_step_qpos is None:
            pre_step_qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
        world.step(render=render)
    assert pre_step_qpos is not None
    return pre_step_qpos


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal ALOHA1 Isaac drive-target smoke test shaped like a future RL step loop. "
            "This is not a training environment yet; it verifies reset/target/step/metric semantics."
        )
    )
    parser.add_argument("--stage-usd", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--contact-proxy-profile", choices=contact_proxy_profile_names(), default="scene_base_link")
    parser.add_argument("--stage-units-in-meters", type=float, default=1.0)
    parser.add_argument("--physics-dt", type=float, default=0.02)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--arm-kp", type=float, default=1600.0)
    parser.add_argument("--arm-kd", type=float, default=100.0)
    parser.add_argument("--finger-kp", type=float, default=200.0)
    parser.add_argument("--finger-kd", type=float, default=50.0)
    parser.add_argument("--start-frame", type=int, default=143)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--target-hold-steps", type=int, default=1)
    parser.add_argument("--max-controlled-error", type=float, default=0.02)
    parser.add_argument(
        "--causality-probe",
        action="store_true",
        help="After the replay tracking smoke, reset to the same state and apply two different actions.",
    )
    parser.add_argument("--causality-delta", type=float, default=0.03)
    parser.add_argument("--min-causality-state-delta", type=float, default=1e-4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open an Isaac window for visual debugging. Headless remains the default for regression and RL checks.",
    )
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "status": "STARTED",
        "purpose": "rl_drive_target_smoke",
        "inputs": {
            "stage_usd": str(args.stage_usd),
            "episode": str(args.episode),
            "mapping": str(args.mapping),
            "side": args.side,
            "contact_proxy_profile": args.contact_proxy_profile,
            "stage_units_in_meters": args.stage_units_in_meters,
            "physics_dt": args.physics_dt,
            "gravity": args.gravity,
            "arm_kp": args.arm_kp,
            "arm_kd": args.arm_kd,
            "finger_kp": args.finger_kp,
            "finger_kd": args.finger_kd,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "target_hold_steps": args.target_hold_steps,
            "max_controlled_error": args.max_controlled_error,
            "causality_probe": args.causality_probe,
            "causality_delta": args.causality_delta,
            "min_causality_state_delta": args.min_causality_state_delta,
        },
        "rows": [],
    }
    _write_json(args.output_json, payload)

    app = None
    try:
        qpos = load_hdf5_qpos(args.episode, start=args.start_frame, end=args.end_frame)
        if args.max_frames is not None:
            qpos = qpos[: int(args.max_frames)]
        if qpos.shape[0] < 2:
            raise ValueError("Need at least two qpos frames after max-frame slicing")

        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["headless"] = not bool(args.gui)
        app_config["disable_viewport_updates"] = not bool(args.render)
        app_config["fast_shutdown"] = False
        app = SimulationApp(app_config)

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import isaacsim.core.utils.stage as stage_utils

        from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
        from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_gravity
        from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_named_dof_gains
        from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
        from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
        from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target

        stage_utils.open_stage(str(args.stage_usd.resolve()))
        World.clear_instance()
        world = World(stage_units_in_meters=args.stage_units_in_meters, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        paths = resolve_contact_proxy_paths(args.contact_proxy_profile)[args.side]
        finger_dof_names = finger_dof_names_for_side(args.contact_proxy_profile, args.side)
        finger_qpos_limits = finger_qpos_limits_for_side(args.contact_proxy_profile, args.side)
        art = world.scene.add(SingleArticulation(prim_path=paths["articulation"], name=f"{args.side}_vx300s"))
        world.reset()
        _apply_gravity(world, args.gravity)
        _apply_arm_gains(art, args.arm_kp, args.arm_kd)
        _apply_named_dof_gains(
            art,
            [finger_dof_names["left_finger"], finger_dof_names["right_finger"]],
            args.finger_kp,
            args.finger_kd,
        )

        dof_names = list(art.dof_names)
        initial = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
        cfg = DriveTargetReplayConfig(
            side=args.side,
            replay_mode="left_arm_and_gripper",
            target_hold_steps=args.target_hold_steps,
            max_controlled_error=args.max_controlled_error,
        )
        targets = targets_from_hdf5_qpos(
            initial_target=initial,
            dof_names=dof_names,
            side=cfg.side,
            qpos=qpos,
            mapping_path=args.mapping,
            replay_mode=cfg.replay_mode,
            finger_dof_names=finger_dof_names,
            finger_qpos_limits=finger_qpos_limits,
        )
        groups = tracking_groups(
            dof_names,
            side=cfg.side,
            replay_mode=cfg.replay_mode,
            finger_dof_names=finger_dof_names,
        )
        limits = _get_limits(art)

        _set_full_state(art, targets[0])
        _set_full_target(art, targets[0])
        world.step(render=bool(args.render))

        rows = []
        for step_index, target in enumerate(targets[1:], start=1):
            pre_step_qpos = _apply_target_and_step(
                world,
                art,
                target,
                hold_steps=cfg.target_hold_steps,
                render=bool(args.render),
                set_full_target=_set_full_target,
            )
            actual = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            metrics = summarize_step(
                step_index=step_index,
                target=target,
                actual=actual,
                limits=limits,
                groups=groups,
                max_controlled_error=cfg.max_controlled_error,
            )
            pre_metrics = summarize_step(
                step_index=step_index,
                target=target,
                actual=pre_step_qpos,
                limits=limits,
                groups=groups,
                max_controlled_error=cfg.max_controlled_error,
            )
            rows.append(
                {
                    "step_index": metrics.step_index,
                    "controlled_max_abs_error": metrics.controlled_max_abs_error,
                    "controlled_rms_error": metrics.controlled_rms_error,
                    "target_limit_controlled_max_violation": metrics.target_limit_controlled_max_violation,
                    "reward_ready": metrics.reward_ready,
                    "pre_step_controlled_max_abs_error": pre_metrics.controlled_max_abs_error,
                }
            )

        controlled_errors = [row["controlled_max_abs_error"] for row in rows]
        reward_ready_count = sum(1 for row in rows if row["reward_ready"])
        causality_probe: dict[str, Any] = {
            "enabled": bool(args.causality_probe),
            "pass": False,
            "status": "NOT_EVALUATED",
        }
        if args.causality_probe:
            arm_indices = groups.get("arm") or groups["controlled"]
            probe_dof_index = int(arm_indices[0])
            base_target = targets[0].copy()
            plus_target = base_target.copy()
            minus_target = base_target.copy()
            lower, upper = [float(x) for x in limits[probe_dof_index]]
            plus_target[probe_dof_index] = float(np.clip(base_target[probe_dof_index] + args.causality_delta, lower, upper))
            minus_target[probe_dof_index] = float(np.clip(base_target[probe_dof_index] - args.causality_delta, lower, upper))

            _set_full_state(art, base_target)
            _set_full_target(art, base_target)
            world.step(render=bool(args.render))
            _apply_target_and_step(
                world,
                art,
                plus_target,
                hold_steps=cfg.target_hold_steps,
                render=bool(args.render),
                set_full_target=_set_full_target,
            )
            plus_actual = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)

            _set_full_state(art, base_target)
            _set_full_target(art, base_target)
            world.step(render=bool(args.render))
            _apply_target_and_step(
                world,
                art,
                minus_target,
                hold_steps=cfg.target_hold_steps,
                render=bool(args.render),
                set_full_target=_set_full_target,
            )
            minus_actual = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
            state_delta = float(np.linalg.norm(plus_actual - minus_actual))
            causality_probe = {
                "enabled": True,
                "pass": bool(state_delta >= args.min_causality_state_delta),
                "status": "PASS" if state_delta >= args.min_causality_state_delta else "FAILED_GATE",
                "probe_dof_index": probe_dof_index,
                "probe_dof_name": dof_names[probe_dof_index],
                "target_delta": float(abs(plus_target[probe_dof_index] - minus_target[probe_dof_index])),
                "state_delta_l2": state_delta,
                "min_state_delta_l2": float(args.min_causality_state_delta),
            }
        payload.update(
            {
                "status": "PASS" if reward_ready_count == len(rows) else "FAILED_GATE",
                "rl_training_readiness": build_rl_readiness_report(
                    drive_gate_pass=reward_ready_count == len(rows),
                    drive_gate_evidence=(
                        f"{reward_ready_count}/{len(rows)} drive-target replay steps passed tracking and limit gates"
                    ),
                    causality_gate_pass=bool(causality_probe["pass"]),
                ),
                "frame_count": int(qpos.shape[0]),
                "step_count": len(rows),
                "reward_ready_count": reward_ready_count,
                "reward_ready_fraction": float(reward_ready_count / max(len(rows), 1)),
                "max_controlled_error_observed": float(max(controlled_errors)) if controlled_errors else None,
                "mean_controlled_error_observed": float(np.mean(controlled_errors)) if controlled_errors else None,
                "groups": groups,
                "dof_names": dof_names,
                "causality_probe": causality_probe,
                "rows": rows,
                "next_gate": "add_reset_action_observation_reward_api" if reward_ready_count == len(rows) else "fix_drive_tracking_before_rl",
            }
        )
        _write_json(args.output_json, payload)
        print(json.dumps({"status": payload["status"], "json": str(args.output_json)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if payload["status"] == "PASS" else 3)
    except Exception as exc:
        payload.update({"status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()})
        _write_json(args.output_json, payload)
        print(json.dumps({"status": "ERROR", "json": str(args.output_json), "error": str(exc)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    raise SystemExit(main())
