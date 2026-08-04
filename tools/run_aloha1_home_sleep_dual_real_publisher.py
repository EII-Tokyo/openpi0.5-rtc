#!/usr/bin/env python3
"""Synchronized dual-follower publisher; fail-closed by default.

This entry point is suitable for deployment into the approved project path on
103.  It performs no ROS import or network mutation unless both
``--execute-real`` and ``--allow-dual-real-motion`` are supplied.  The current
workflow intentionally uses its dry-run path until a fresh physical-motion
authorization is granted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from tools.aloha1_mapping.dual_real_publisher import build_dual_dry_run_report
from tools.aloha1_mapping.dual_real_publisher import validate_dual_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, report: dict[str, Any]) -> None:
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    path.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-manifest", type=Path, required=True)
    parser.add_argument("--right-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-delay-s", type=float, default=3.0)
    parser.add_argument("--left-role", default="puppet_left")
    parser.add_argument("--right-role", default="puppet_right")
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--allow-dual-real-motion", action="store_true")
    return parser.parse_args()


def _run_live(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_path: Path,
    right_path: Path,
    output: Path,
    start_delay_s: float,
) -> int:
    """Publish both arms from one monotonic start barrier after readiness gates."""
    from interbotix_xs_msgs.msg import JointGroupCommand
    import rospy
    from sensor_msgs.msg import JointState

    expected = [
        "waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"
    ]
    states: dict[str, dict[str, Any]] = {
        "left": {"names": [], "position": [], "stamp": None},
        "right": {"names": [], "position": [], "stamp": None},
    }

    def callback(side: str):
        def receive(message: JointState) -> None:
            states[side]["names"] = list(message.name)
            states[side]["position"] = [float(value) for value in message.position]
            states[side]["stamp"] = {"secs": int(message.header.stamp.secs), "nsecs": int(message.header.stamp.nsecs)}
        return receive

    rospy.init_node("aloha1_home_sleep_dual_real_publisher", anonymous=False, disable_signals=True)
    rospy.Subscriber("/puppet_left/joint_states", JointState, callback("left"), queue_size=10)
    rospy.Subscriber("/puppet_right/joint_states", JointState, callback("right"), queue_size=10)
    publishers = {
        "left": rospy.Publisher("/puppet_left/commands/joint_group", JointGroupCommand, queue_size=1),
        "right": rospy.Publisher("/puppet_right/commands/joint_group", JointGroupCommand, queue_size=1),
    }
    ready_deadline = time.monotonic() + max(5.0, float(start_delay_s))
    while not rospy.is_shutdown():
        ready = all(
            states[side]["names"][:6] == expected and len(states[side]["position"]) >= 9
            for side in ("left", "right")
        ) and all(publishers[side].get_num_connections() >= 1 for side in ("left", "right"))
        if ready:
            break
        if time.monotonic() >= ready_deadline:
            raise RuntimeError("dual readback or command subscribers did not become ready")
        time.sleep(0.01)
    # The manifests contain side-specific Sleep references.  This gate rejects
    # an implicit large preposition before the synchronized trajectory.
    for side, manifest in (("left", left), ("right", right)):
        reference = [float(v) for v in manifest["initial_arm_rad"]]
        actual = states[side]["position"][:6]
        error = max(abs(a - b) for a, b in zip(reference, actual))  # noqa: B905
        if error > 0.05:
            raise RuntimeError(f"{side} initial Sleep readback error {error:.6f} rad exceeds 0.05 rad gate")
    start_ns = time.monotonic_ns() + int(float(start_delay_s) * 1_000_000_000)
    telemetry: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "IN_PROGRESS",
        "left_manifest": str(left_path),
        "right_manifest": str(right_path),
        "left_manifest_sha256": _sha256(left_path),
        "right_manifest_sha256": _sha256(right_path),
        "topics": {
            "left": "/puppet_left/commands/joint_group",
            "right": "/puppet_right/commands/joint_group",
        },
        "joint_order": expected,
        "sample_count": len(left["samples"]),
        "commands_published": {"puppet_left": 0, "puppet_right": 0},
        "telemetry": telemetry,
        "real_motion_authorized": True,
        "abort_reason": None,
    }
    try:
        for left_sample, right_sample in zip(left["samples"], right["samples"]):  # noqa: B905
            target_ns = start_ns + int(left_sample["time_ns"])
            while time.monotonic_ns() < target_ns and not rospy.is_shutdown():
                time.sleep(0.0005)
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")
            messages: dict[str, JointGroupCommand] = {}
            for side, sample in (("left", left_sample), ("right", right_sample)):
                message = JointGroupCommand()
                message.name = "arm"
                message.cmd = [float(value) for value in sample["q_rad"]]
                messages[side] = message
            publishers["left"].publish(messages["left"])
            publishers["right"].publish(messages["right"])
            result["commands_published"]["puppet_left"] += 1
            result["commands_published"]["puppet_right"] += 1
            telemetry.append({
                "sample_index": int(left_sample["index"]),
                "time_ns": int(left_sample["time_ns"]),
                "left_target_q_rad": list(messages["left"].cmd),
                "right_target_q_rad": list(messages["right"].cmd),
                "left_readback_q_rad": list(states["left"]["position"][:6]),
                "right_readback_q_rad": list(states["right"]["position"][:6]),
                "host_monotonic_ns": time.monotonic_ns(),
            })
    except BaseException as exc:
        result["status"] = "ABORTED"
        result["abort_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        result["status"] = "PASS_REAL_DUAL_MANIFEST_PUBLISHED"
    _write(output, result)
    return 0 if result["status"] == "PASS_REAL_DUAL_MANIFEST_PUBLISHED" else 2


def main() -> int:
    args = _parse_args()
    left_path = args.left_manifest.resolve(strict=True)
    right_path = args.right_manifest.resolve(strict=True)
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    validated = validate_dual_manifest(left, right)
    report = build_dual_dry_run_report(
        left_sha256=_sha256(left_path),
        right_sha256=_sha256(right_path),
        sample_count=validated["sample_count"],
    )
    report["left_manifest"] = str(left_path)
    report["right_manifest"] = str(right_path)
    report["validation"] = validated
    report["execute_real_requested"] = bool(args.execute_real)
    report["dual_motion_flag_present"] = bool(args.allow_dual_real_motion)
    report["roles"] = {"left": args.left_role, "right": args.right_role}
    report["status_reason"] = "A separate reviewed live implementation is required before any dual publish"
    if args.execute_real and args.allow_dual_real_motion:
        return _run_live(
            left,
            right,
            left_path=left_path,
            right_path=right_path,
            output=args.output.resolve(),
            start_delay_s=args.start_delay_s,
        )
    _write(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
