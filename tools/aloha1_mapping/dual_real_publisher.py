"""Fail-closed planning helpers for synchronized dual-follower replay.

The module is intentionally transport-free.  It validates that both arm
manifests share one sample clock before a separate, explicitly authorized ROS
entry point is allowed to construct publishers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import shlex
from typing import Any

EXPECTED_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)


def _samples(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = manifest.get("samples")
    if not isinstance(value, list) or not value:
        raise ValueError("manifest samples must be a non-empty list")
    return [item if isinstance(item, Mapping) else {} for item in value]


def validate_dual_manifest(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an identical timebase and six-DOF command shape."""

    left_samples = _samples(left)
    right_samples = _samples(right)
    if len(left_samples) != len(right_samples):
        raise ValueError("left/right sample count mismatch")
    for side, manifest in (("left", left), ("right", right)):
        if tuple(manifest.get("joint_order", ())) != EXPECTED_JOINT_ORDER:
            raise ValueError(f"{side} joint order is not the pinned arm order")
        if int(manifest.get("sample_count", -1)) != len(left_samples):
            raise ValueError(f"{side} sample_count does not match samples")
        if int(manifest.get("command_rate_hz", 0)) != 50:
            raise ValueError(f"{side} command rate must be 50 Hz")
    for index, (left_sample, right_sample) in enumerate(
        zip(left_samples, right_samples)  # noqa: B905  # Python 3.8 remote runtime
    ):
        if int(left_sample.get("index", -1)) != int(right_sample.get("index", -2)):
            raise ValueError(f"sample clock index mismatch at {index}")
        if int(left_sample.get("time_ns", -1)) != int(right_sample.get("time_ns", -2)):
            raise ValueError(f"sample clock time mismatch at {index}")
        for side, sample in (("left", left_sample), ("right", right_sample)):
            q_rad = sample.get("q_rad")
            if not isinstance(q_rad, Sequence) or isinstance(q_rad, (str, bytes)) or len(q_rad) != 6:  # noqa: UP038
                raise ValueError(f"{side} sample {index} must contain six q_rad values")
            if not all(math.isfinite(float(value)) for value in q_rad):
                raise ValueError(f"{side} sample {index} contains non-finite q_rad")
    return {
        "sample_count": len(left_samples),
        "command_rate_hz": 50,
        "joint_order": list(EXPECTED_JOINT_ORDER),
        "left_command_signature": str(left.get("command_signature", "")),
        "right_command_signature": str(right.get("command_signature", "")),
        "sample_clock_signature": [
            {"index": int(sample["index"]), "time_ns": int(sample["time_ns"])}
            for sample in left_samples
        ],
    }


def build_dual_command(
    *, left_manifest: str, right_manifest: str, output: str, start_delay_s: float = 3.0
) -> list[str]:
    """Build a remote command that remains dry-run unless flags are added later."""

    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "192.168.1.103",
        "cd /home/eii/openpi0.5-rtc-reward-learning && "
        "C=$(docker ps --format '{{.Names}}' | grep aloha_ros_nodes | head -1) && "
        'docker exec -i "$C" bash -lc '
        f'"source /opt/ros/noetic/setup.bash; python3 /app/tools/'
        f"run_aloha1_home_sleep_dual_real_publisher.py --left-manifest {left_manifest} "
        f"--right-manifest {right_manifest} --output {output} "
        f"--start-delay-s {float(start_delay_s):g} --left-role puppet_left "
        f'--right-role puppet_right"',
    ]


def build_remote_dual_publisher_command(
    *,
    left_local: str,
    right_local: str,
    script_local: str,
    module_local: str,
    left_remote: str,
    right_remote: str,
    output_remote: str,
    output_local: str,
    start_delay_s: float = 4.0,
) -> list[str]:
    """Stage GUI-generated manifests and run one explicitly authorized bridge."""

    local_left = shlex.quote(str(left_local))
    local_right = shlex.quote(str(right_local))
    local_script = shlex.quote(str(script_local))
    local_module = shlex.quote(str(module_local))
    staged_left_name = shlex.quote(Path(str(left_local)).name)
    staged_right_name = shlex.quote(Path(str(right_local)).name)
    remote_left = shlex.quote(str(left_remote))
    remote_right = shlex.quote(str(right_remote))
    output = shlex.quote(str(output_remote))
    remote_script = "/tmp/aloha1_dual_real_script.py"
    output_local_q = shlex.quote(str(output_local))
    command = (
        "set -e; "
        "ssh -o BatchMode=yes 192.168.1.103 'mkdir -p /tmp/aloha1_gui_manifest_stage'; "
        f"scp {local_left} 192.168.1.103:/tmp/aloha1_gui_manifest_stage/; "
        f"scp {local_right} 192.168.1.103:/tmp/aloha1_gui_manifest_stage/; "
        f"scp {local_script} 192.168.1.103:/tmp/aloha1_dual_real_script.py; "
        f"scp {local_module} 192.168.1.103:/tmp/aloha1_dual_real_module.py; "
        "ssh -o BatchMode=yes 192.168.1.103 "
        f"'set -e; C=$(docker ps --format \"{{{{.Names}}}}\" | grep aloha_ros_nodes | head -1); "
        'test -n "$C"; '
        f'docker cp /tmp/aloha1_dual_real_script.py "$C":{remote_script}; '
        'docker exec "$C" mkdir -p /app/tools/aloha1_mapping; '
        'docker cp /tmp/aloha1_dual_real_module.py "$C":/app/tools/aloha1_mapping/dual_real_publisher.py; '
        f'docker cp /tmp/aloha1_gui_manifest_stage/{staged_left_name} "$C":{remote_left}; '
        f'docker cp /tmp/aloha1_gui_manifest_stage/{staged_right_name} "$C":{remote_right}; '
        'docker exec "$C" bash -lc "cd /app; source /opt/ros/noetic/setup.bash; '
        "source /root/interbotix_ws/devel/setup.bash; "
        f"/usr/bin/python3 {remote_script} --left-manifest {remote_left} "
        f"--right-manifest {remote_right} --output {output} --start-delay-s {float(start_delay_s):g} "
        '--left-role puppet_left --right-role puppet_right --execute-real --allow-dual-real-motion"; '
        f"docker cp \"$C\":{output} /tmp/aloha1_gui_result.json'; "
        f"scp 192.168.1.103:/tmp/aloha1_gui_result.json {output_local_q}"
    )
    return [
        "bash",
        "-lc",
        command,
    ]


def build_dual_dry_run_report(
    *, left_sha256: str, right_sha256: str, sample_count: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "NOT_RUN_AUTHORIZATION_REQUIRED",
        "left_manifest_sha256": left_sha256,
        "right_manifest_sha256": right_sha256,
        "sample_count": int(sample_count),
        "commands_published": {"puppet_left": 0, "puppet_right": 0},
        "ros_transport_instantiated": False,
        "serial_device_opened": False,
        "torque_changed": False,
        "real_motion_authorized": False,
    }
