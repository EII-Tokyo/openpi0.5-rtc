from __future__ import annotations

import cv2
import numpy as np

from calibration_workbench.apriltag_analyzer import AprilTagAnalyzer
from calibration_workbench.intrinsics_capture import FramePacket
from calibration_workbench.models import FactoryIntrinsics


def test_detects_tag36h11_id0_and_returns_explicit_tag_to_camera_transform():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    source = np.float32([[0, 0], [199, 0], [199, 199], [0, 199]])
    # Corners projected from a physically valid, slightly tilted square pose.
    target = np.float32(
        [
            [290.4231, 215.9056],
            [376.5220, 221.9158],
            [372.0575, 308.4819],
            [284.1640, 302.7110],
        ]
    )
    image = cv2.warpPerspective(
        marker,
        cv2.getPerspectiveTransform(source, target),
        (640, 480),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
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


def test_refines_detected_tag_corners_beyond_integer_pixel_contours():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 160)
    source = np.float32([[0, 0], [159, 0], [159, 159], [0, 159]])
    target = np.float32(
        [
            [278.35, 191.70],
            [358.65, 194.20],
            [356.40, 274.80],
            [276.10, 272.30],
        ]
    )
    homography = cv2.getPerspectiveTransform(source, target)
    image = cv2.warpPerspective(
        marker,
        homography,
        (640, 480),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    result = AprilTagAnalyzer(tag_id=0, tag_size_m=0.080).analyze(
        FramePacket(rgb=rgb, frame_number=7, device_timestamp_ms=234.0),
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
    corners = np.asarray(result.corners_px)
    assert np.max(np.abs(corners - np.round(corners))) > 0.05
