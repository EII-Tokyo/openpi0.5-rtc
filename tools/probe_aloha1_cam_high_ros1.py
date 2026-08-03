#!/usr/bin/env python3
"""Fail-closed ROS1 cam_high-only runtime probe for the real ALOHA cell."""

from __future__ import annotations

import argparse
import hashlib
from itertools import pairwise
import json
from pathlib import Path
import statistics
import time
from typing import Any

CAM_HIGH_SERIAL = "130322270656"


def camera_contract() -> dict[str, object]:
    return {
        "name": "cam_high",
        "serial": CAM_HIGH_SERIAL,
        "width": 640,
        "height": 480,
        "fps": 60,
        "realsense_format": "bgr8",
        "ros_encoding": "bgr8",
        "topic": "/cam_high",
        "message_type": "aloha.msg/RGBGrayscaleImage",
        "serial_evidence": "REMOTE_DEPLOYED_CONFIG_READBACK",
    }


def build_dry_run_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "NOT_RUN_EXPLICIT_CAMERA_FLAG_REQUIRED",
        "contract": camera_contract(),
        "camera_opened": False,
        "ros_publisher_constructed": False,
        "robot_command_publishers": 0,
        "hardware_resets": 0,
        "frames_captured": 0,
    }


def classify_capture(
    *,
    frame_count: int,
    publisher_count: int,
    width: int,
    height: int,
    ros_encoding: str,
    serial: str,
) -> str:
    expected = camera_contract()
    if frame_count <= 0:
        return "FAIL_NO_CAMERA_FRAMES"
    if publisher_count != frame_count:
        return "FAIL_CAMERA_PUBLISH_COUNT"
    if serial != expected["serial"]:
        return "FAIL_WRONG_CAMERA_SERIAL"
    if (width, height) != (expected["width"], expected["height"]):
        return "FAIL_CAMERA_RESOLUTION"
    if ros_encoding != expected["ros_encoding"]:
        return "FAIL_CAMERA_ENCODING"
    return "PASS_CAM_HIGH_SINGLE_CAMERA_RUNTIME"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_live(*, max_frames: int, output: Path, snapshot: Path | None) -> int:
    # Live-only imports remain behind the explicit hardware flag.
    from aloha.msg import RGBGrayscaleImage
    import cv2
    from cv_bridge import CvBridge
    import numpy as np
    import pyrealsense2 as rs
    import rospy

    contract = camera_contract()
    context = rs.context()
    devices = context.query_devices()
    discovered = [
        device.get_info(rs.camera_info.serial_number) for device in devices
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "contract": contract,
        "discovered_serials": discovered,
        "camera_opened": False,
        "ros_publisher_constructed": False,
        "robot_command_publishers": 0,
        "hardware_resets": 0,
        "frames_captured": 0,
        "frames_published": 0,
    }
    if CAM_HIGH_SERIAL not in discovered:
        report["status"] = "FAIL_CAM_HIGH_SERIAL_MISSING"
        _write_report(output, report)
        return 2

    rospy.init_node("aloha1_cam_high_probe", anonymous=False, disable_signals=True)
    publisher = rospy.Publisher(
        str(contract["topic"]), RGBGrayscaleImage, queue_size=1
    )
    report["ros_publisher_constructed"] = True
    bridge = CvBridge()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(CAM_HIGH_SERIAL)
    config.enable_stream(
        rs.stream.color,
        int(contract["width"]),
        int(contract["height"]),
        rs.format.bgr8,
        int(contract["fps"]),
    )
    host_monotonic_ns: list[int] = []
    sensor_timestamps_ms: list[float] = []
    first_frame_number: int | None = None
    last_frame_number: int | None = None
    last_pixels_sha256: str | None = None
    started = False
    try:
        pipeline.start(config)
        started = True
        report["camera_opened"] = True
        for _ in range(max_frames):
            if rospy.is_shutdown():
                break
            frameset = pipeline.wait_for_frames(timeout_ms=1000)
            color = frameset.get_color_frame()
            if not color:
                continue
            pixels = np.asanyarray(color.get_data())
            if pixels.shape != (int(contract["height"]), int(contract["width"]), 3):
                report["status"] = "FAIL_CAMERA_FRAME_SHAPE"
                report["observed_shape"] = list(pixels.shape)
                break
            frame_number = int(color.get_frame_number())
            first_frame_number = (
                frame_number if first_frame_number is None else first_frame_number
            )
            last_frame_number = frame_number
            host_monotonic_ns.append(time.monotonic_ns())
            sensor_timestamps_ms.append(float(color.get_timestamp()))
            last_pixels_sha256 = hashlib.sha256(pixels.tobytes()).hexdigest()
            image = bridge.cv2_to_imgmsg(pixels, encoding="bgr8")
            message = RGBGrayscaleImage()
            message.header.stamp = rospy.Time.now()
            image.header = message.header
            message.images.append(image)
            publisher.publish(message)
            report["frames_captured"] += 1
            report["frames_published"] += 1
            if report["frames_captured"] == 1 and snapshot is not None:
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(snapshot), pixels):
                    report["status"] = "FAIL_SNAPSHOT_WRITE"
                    break
    except Exception as exc:  # runtime evidence must survive camera failures
        report["status"] = "FAIL_CAMERA_RUNTIME_EXCEPTION"
        report["exception_type"] = type(exc).__name__
        report["exception"] = str(exc)
    finally:
        if started:
            pipeline.stop()

    intervals_ms = [
        (after - before) / 1_000_000
        for before, after in pairwise(host_monotonic_ns)
    ]
    if report["status"] == "IN_PROGRESS":
        report["status"] = classify_capture(
            frame_count=int(report["frames_captured"]),
            publisher_count=int(report["frames_published"]),
            width=int(contract["width"]),
            height=int(contract["height"]),
            ros_encoding=str(contract["ros_encoding"]),
            serial=CAM_HIGH_SERIAL,
        )
    report.update(
        {
            "first_frame_number": first_frame_number,
            "last_frame_number": last_frame_number,
            "last_pixels_sha256": last_pixels_sha256,
            "host_interval_ms_mean": (
                statistics.fmean(intervals_ms) if intervals_ms else None
            ),
            "host_interval_ms_max": max(intervals_ms) if intervals_ms else None,
            "sensor_timestamp_ms_first": (
                sensor_timestamps_ms[0] if sensor_timestamps_ms else None
            ),
            "sensor_timestamp_ms_last": (
                sensor_timestamps_ms[-1] if sensor_timestamps_ms else None
            ),
            "snapshot": str(snapshot) if snapshot is not None else None,
        }
    )
    _write_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "frames_captured": report["frames_captured"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS_CAM_HIGH_SINGLE_CAMERA_RUNTIME" else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-camera-hardware", action="store_true")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if not args.execute_camera_hardware:
        _write_report(output, build_dry_run_report())
        print(json.dumps(build_dry_run_report(), sort_keys=True))
        return 2
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    return run_live(
        max_frames=args.max_frames,
        output=output,
        snapshot=args.snapshot.resolve() if args.snapshot else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
