from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .intrinsics_capture import FramePacket
from .models import FactoryIntrinsics
from .workflow import TagPoseSample
from .workflow import TransformRecord
from .workflow import _normalized_points
from .workflow import _project_pixels


@dataclass(frozen=True)
class AprilTagDetection:
    detected: bool
    tag_id: int
    frame_number: int
    device_timestamp_ms: float
    sample: TagPoseSample | None
    corners_px: tuple[tuple[float, float], ...]
    jpeg: bytes
    png: bytes


class AprilTagAnalyzer:
    """Detect one exact AprilTag and return tag -> OpenCV optical pose candidates."""

    def __init__(self, *, tag_id: int, tag_size_m: float) -> None:
        if tag_id < 0:
            raise ValueError("tag_id must be non-negative")
        if not 0.01 <= tag_size_m <= 0.30:
            raise ValueError("tag_size_m is outside the supported physical range")
        self._tag_id = tag_id
        self._tag_size_m = tag_size_m
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector = cv2.aruco.ArucoDetector(dictionary)

    def analyze(self, frame: FramePacket, intrinsics: FactoryIntrinsics) -> AprilTagDetection:
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        if rgb.shape != (intrinsics.height, intrinsics.width, 3):
            raise ValueError(
                f"Frame shape {rgb.shape!r} does not match intrinsics "
                f"{intrinsics.width}x{intrinsics.height} RGB"
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = self._detector.detectMarkers(gray)
        selected: np.ndarray | None = None
        if marker_ids is not None:
            for corners, marker_id in zip(marker_corners, marker_ids.reshape(-1), strict=True):
                if int(marker_id) == self._tag_id:
                    selected = np.asarray(corners, dtype=np.float64).reshape(4, 2)
                    break

        overlay = bgr.copy()
        sample: TagPoseSample | None = None
        corners_tuple: tuple[tuple[float, float], ...] = ()
        if selected is not None:
            corners_tuple = tuple((float(point[0]), float(point[1])) for point in selected)
            half = self._tag_size_m / 2.0
            object_points = np.asarray(
                [
                    [-half, half, 0.0],
                    [half, half, 0.0],
                    [half, -half, 0.0],
                    [-half, -half, 0.0],
                ],
                dtype=np.float64,
            )
            normalized = _normalized_points(selected, intrinsics)
            solved, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points,
                normalized,
                np.eye(3, dtype=np.float64),
                None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
            if solved:
                for rvec, tvec in zip(rvecs, tvecs, strict=True):
                    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
                    if translation[2] <= 0:
                        continue
                    rotation, _ = cv2.Rodrigues(rvec)
                    points_camera = (rotation @ object_points.T).T + translation
                    projected = _project_pixels(points_camera, intrinsics)
                    rms = float(np.sqrt(np.mean(np.sum((projected - selected) ** 2, axis=1))))
                    candidates.append((rms, rotation, translation))
            if candidates:
                rms, rotation, translation = min(candidates, key=lambda item: item[0])
                camera_from_tag = np.eye(4)
                camera_from_tag[:3, :3] = rotation
                camera_from_tag[:3, 3] = translation
                sample = TagPoseSample(
                    camera_from_tag=TransformRecord(
                        source_frame="tag",
                        target_frame="camera_high_optical",
                        matrix=camera_from_tag.tolist(),
                    ),
                    reprojection_rms_px=rms,
                )
                cv2.polylines(overlay, [selected.astype(np.int32)], True, (30, 220, 30), 2)
                center = tuple(np.mean(selected, axis=0).astype(int))
                cv2.circle(overlay, center, 4, (0, 255, 255), -1)
                cv2.putText(
                    overlay,
                    f"tag36h11:{self._tag_id} RMS={rms:.2f}px",
                    (max(5, center[0] - 100), max(20, center[1] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (30, 220, 30),
                    1,
                    cv2.LINE_AA,
                )
        if sample is None:
            cv2.putText(
                overlay,
                f"AprilTag {self._tag_id} not detected",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 40, 230),
                2,
                cv2.LINE_AA,
            )
        png_ok, png = cv2.imencode(".png", overlay)
        jpeg_ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not png_ok or not jpeg_ok:
            raise RuntimeError("OpenCV could not encode AprilTag evidence image")
        return AprilTagDetection(
            detected=sample is not None,
            tag_id=self._tag_id,
            frame_number=frame.frame_number,
            device_timestamp_ms=frame.device_timestamp_ms,
            sample=sample,
            corners_px=corners_tuple,
            jpeg=jpeg.tobytes(),
            png=png.tobytes(),
        )
