#!/usr/bin/env python3
"""One-shot, readback-driven dual-follower Sleep alignment on machine 103.

The runner is inert unless both ``--execute-real`` and
``--allow-startup-sleep-align`` are supplied.  It reads the current arm pose,
interpolates to the frozen Sleep target at 50 Hz, and holds the latest measured
pose if interrupted.  It never sends an all-zero command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from tools.aloha1_mapping.startup_sleep_alignment import DEFAULT_MOVE_SECONDS
from tools.aloha1_mapping.startup_sleep_alignment import DEFAULT_RATE_HZ
from tools.aloha1_mapping.startup_sleep_alignment import interpolate_targets
from tools.aloha1_mapping.startup_sleep_alignment import max_step_velocity
from tools.aloha1_mapping.startup_sleep_alignment import validate_sleep_manifest


EXPECTED = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate-hz", type=int, default=DEFAULT_RATE_HZ)
    parser.add_argument("--move-seconds", type=float, default=DEFAULT_MOVE_SECONDS)
    parser.add_argument("--start-delay-s", type=float, default=2.0)
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--allow-startup-sleep-align", action="store_true")
    return parser.parse_args()


def _run_live(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    import rospy
    from interbotix_xs_msgs.msg import JointGroupCommand
    from sensor_msgs.msg import JointState

    sleep_target, joint_order = validate_sleep_manifest(manifest)
    states: dict[str, dict[str, Any]] = {
        "left": {"names": [], "position": [], "stamp": None},
        "right": {"names": [], "position": [], "stamp": None},
    }

    def callback(side: str):
        def receive(message: JointState) -> None:
            states[side]["names"] = list(message.name)
            states[side]["position"] = [float(value) for value in message.position]
            states[side]["stamp"] = {
                "secs": int(message.header.stamp.secs),
                "nsecs": int(message.header.stamp.nsecs),
            }

        return receive

    rospy.init_node("aloha1_startup_sleep_alignment", anonymous=False, disable_signals=True)
    for side, role in (("left", "puppet_left"), ("right", "puppet_right")):
        rospy.Subscriber(f"/{role}/joint_states", JointState, callback(side), queue_size=10)
    publishers = {
        "left": rospy.Publisher("/puppet_left/commands/joint_group", JointGroupCommand, queue_size=1),
        "right": rospy.Publisher("/puppet_right/commands/joint_group", JointGroupCommand, queue_size=1),
    }
    deadline = time.monotonic() + max(5.0, float(args.start_delay_s))
    while not rospy.is_shutdown():
        ready = all(
            states[side]["names"][:6] == EXPECTED and len(states[side]["position"]) >= 6
            for side in ("left", "right")
        ) and all(publishers[side].get_num_connections() >= 1 for side in ("left", "right"))
        if ready:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("dual readback or command subscribers did not become ready")
        time.sleep(0.01)

    starts = {side: list(states[side]["position"][:6]) for side in ("left", "right")}
    trajectories = {
        side: interpolate_targets(
            starts[side], sleep_target, rate_hz=int(args.rate_hz), move_seconds=float(args.move_seconds)
        )
        for side in ("left", "right")
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "mode": "STARTUP_SLEEP_ALIGNMENT",
        "joint_order": joint_order,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "target_sleep_rad": sleep_target,
        "initial_readback_rad": starts,
        "rate_hz": int(args.rate_hz),
        "move_seconds": float(args.move_seconds),
        "max_command_velocity_rad_s": {
            side: max_step_velocity(starts[side], sleep_target, rate_hz=int(args.rate_hz), move_seconds=float(args.move_seconds))
            for side in ("left", "right")
        },
        "commands_published": {"puppet_left": 0, "puppet_right": 0},
        "telemetry": [],
        "abort_reason": None,
    }
    start_at = time.monotonic() + max(0.0, float(args.start_delay_s))
    try:
        for index in range(len(trajectories["left"])):
            target_at = start_at + index / float(args.rate_hz)
            while time.monotonic() < target_at and not rospy.is_shutdown():
                time.sleep(0.0005)
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")
            for side, role in (("left", "puppet_left"), ("right", "puppet_right")):
                message = JointGroupCommand()
                message.name = "arm"
                message.cmd = list(trajectories[side][index])
                publishers[side].publish(message)
            result["commands_published"]["puppet_left"] += 1
            result["commands_published"]["puppet_right"] += 1
            result["telemetry"].append({
                "index": index,
                "target_left_rad": list(trajectories["left"][index]),
                "target_right_rad": list(trajectories["right"][index]),
                "readback_left_rad": list(states["left"]["position"][:6]),
                "readback_right_rad": list(states["right"]["position"][:6]),
                "stamp": {"left": states["left"]["stamp"], "right": states["right"]["stamp"]},
                "host_monotonic_ns": time.monotonic_ns(),
            })
    except BaseException as exc:
        result["status"] = "ABORTED_WITH_READBACK_HOLD"
        result["abort_reason"] = f"{type(exc).__name__}: {exc}"
        for side, role in (("left", "puppet_left"), ("right", "puppet_right")):
            if len(states[side]["position"]) >= 6:
                hold = JointGroupCommand()
                hold.name = "arm"
                hold.cmd = list(states[side]["position"][:6])
                for _ in range(10):
                    publishers[side].publish(hold)
                    time.sleep(0.02)
    else:
        result["status"] = "PASS_STARTUP_SLEEP_ALIGNMENT"
    _write(args.output.resolve(), result)
    return 0 if result["status"] == "PASS_STARTUP_SLEEP_ALIGNMENT" else 2


def main() -> int:
    args = _parse()
    manifest = json.loads(args.manifest.resolve(strict=True).read_text(encoding="utf-8"))
    if not (args.execute_real and args.allow_startup_sleep_align):
        _write(args.output.resolve(), {
            "status": "NOT_RUN_EXECUTE_REAL_AND_ALLOW_STARTUP_SLEEP_ALIGN_REQUIRED",
            "commands_published": {"puppet_left": 0, "puppet_right": 0},
        })
        return 2
    return _run_live(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
