from __future__ import annotations

import cv2
import numpy as np

from calibration_workbench.apriltag_analyzer import AprilTagAnalyzer
from calibration_workbench.intrinsics_capture import FramePacket
from calibration_workbench.models import FactoryIntrinsics


def test_detects_tag36h11_id0_and_returns_explicit_tag_to_camera_transform():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[140:340, 220:420] = marker
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    analyzer = AprilTagAnalyzer(tag_id=0, tag_size_m=0.080)

    result = analyzer.analyze(
        FramePacket(rgb=rgb, frame_number=42, device_timestamp_ms=123.0),
        FactoryIntrinsics(
            width=640,
            height=480,
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0,
            distortion_model="none",
            distortion_coefficients=[0.0] * 5,
        ),
    )

    assert result.detected is True
    assert result.tag_id == 0
    assert result.sample is not None
    assert result.sample.camera_from_tag.source_frame == "tag"
    assert result.sample.camera_from_tag.target_frame == "camera_high_optical"
    assert np.asarray(result.sample.camera_from_tag.matrix)[2, 3] > 0
    assert result.sample.reprojection_rms_px < 0.5
    assert result.png.startswith(b"\x89PNG")
    assert result.jpeg.startswith(b"\xff\xd8")


def test_rejects_frames_without_the_configured_tag():
    analyzer = AprilTagAnalyzer(tag_id=0, tag_size_m=0.080)
    result = analyzer.analyze(
        FramePacket(rgb=np.full((480, 640, 3), 255, dtype=np.uint8), frame_number=1, device_timestamp_ms=1.0),
        FactoryIntrinsics(
            width=640,
            height=480,
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0,
            distortion_model="none",
            distortion_coefficients=[0.0] * 5,
        ),
    )

    assert result.detected is False
    assert result.sample is None
