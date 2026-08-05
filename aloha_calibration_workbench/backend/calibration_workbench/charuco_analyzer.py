from __future__ import annotations

import math

import cv2
import numpy as np
import pyrealsense2 as rs

from .intrinsics_capture import FramePacket
from .models import CharucoObservation
from .models import FactoryIntrinsics


class CharucoAnalyzer:
    """Analyze the exact printed target without altering camera controls or calibration."""

    def __init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self._board = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.022, dictionary)
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def analyze(
        self,
        frame: FramePacket,
        intrinsics: FactoryIntrinsics,
    ) -> tuple[CharucoObservation, bytes, bytes]:
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        if rgb.shape != (intrinsics.height, intrinsics.width, 3):
            raise ValueError(
                f"Frame shape {rgb.shape!r} does not match intrinsics {intrinsics.width}x{intrinsics.height} RGB"
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self._detector.detectBoard(gray)
        marker_count = 0 if marker_ids is None else len(marker_ids)
        corner_count = 0 if charuco_ids is None else len(charuco_ids)
        detected = corner_count >= 4

        centroid_x: float | None = None
        centroid_y: float | None = None
        board_area_percent: float | None = None
        reprojection_rms_px: float | None = None
        overlay = bgr.copy()
        if marker_count:
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        if corner_count:
            cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids)
            points = charuco_corners.reshape(-1, 2)
            centroid = points.mean(axis=0)
            centroid_x = float(centroid[0] / intrinsics.width)
            centroid_y = float(centroid[1] / intrinsics.height)
            hull = cv2.convexHull(points.astype(np.float32))
            board_area_percent = float(100.0 * cv2.contourArea(hull) / (intrinsics.width * intrinsics.height))

        camera_matrix = np.array(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)
        if detected:
            object_points, image_points = self._board.matchImagePoints(charuco_corners, charuco_ids)
            if intrinsics.distortion_model == "inverse_brown_conrady":
                success, rvec, tvec, projected = self._solve_inverse_brown_pose(
                    object_points,
                    image_points,
                    intrinsics,
                )
            elif intrinsics.distortion_model in {"brown_conrady", "none"}:
                solve_distortion = distortion if intrinsics.distortion_model == "brown_conrady" else None
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    solve_distortion,
                )
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    solve_distortion,
                )
            else:
                success = False
                rvec = tvec = projected = None
            if success:
                assert projected is not None
                residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
                reprojection_rms_px = float(math.sqrt(np.mean(np.sum(residual * residual, axis=1))))
                if intrinsics.distortion_model == "inverse_brown_conrady":
                    self._draw_inverse_brown_axes(overlay, rvec, tvec, intrinsics, 0.045)
                else:
                    cv2.drawFrameAxes(overlay, camera_matrix, distortion, rvec, tvec, 0.045)

        observation = CharucoObservation(
            board_detected=detected,
            marker_count=marker_count,
            charuco_corner_count=corner_count,
            blur_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            black_clip_percent=float(100.0 * np.count_nonzero(gray <= 5) / gray.size),
            white_clip_percent=float(100.0 * np.count_nonzero(gray >= 250) / gray.size),
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            board_area_percent=board_area_percent,
            reprojection_rms_px=reprojection_rms_px,
            frame_number=frame.frame_number,
            device_timestamp_ms=frame.device_timestamp_ms,
        )
        ok_jpeg, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        ok_png, png = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok_jpeg or not ok_png:
            raise RuntimeError("OpenCV failed to encode calibration frame")
        return observation, jpeg.tobytes(), png.tobytes()

    @staticmethod
    def _rs_intrinsics(intrinsics: FactoryIntrinsics) -> rs.intrinsics:
        result = rs.intrinsics()
        result.width = intrinsics.width
        result.height = intrinsics.height
        result.fx = intrinsics.fx
        result.fy = intrinsics.fy
        result.ppx = intrinsics.cx
        result.ppy = intrinsics.cy
        result.model = rs.distortion.inverse_brown_conrady
        result.coeffs = intrinsics.distortion_coefficients
        return result

    @classmethod
    def _solve_inverse_brown_pose(cls, object_points, image_points, intrinsics: FactoryIntrinsics):
        rs_intrinsics = cls._rs_intrinsics(intrinsics)
        normalized_points = np.asarray(
            [
                rs.rs2_deproject_pixel_to_point(rs_intrinsics, point.tolist(), 1.0)[:2]
                for point in image_points.reshape(-1, 2)
            ],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            normalized_points,
            np.eye(3, dtype=np.float64),
            None,
        )
        if not success:
            return False, rvec, tvec, None
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation @ object_points.reshape(-1, 3).T + tvec).T
        projected = np.asarray(
            [rs.rs2_project_point_to_pixel(rs_intrinsics, point.tolist()) for point in camera_points],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        return True, rvec, tvec, projected

    @classmethod
    def _draw_inverse_brown_axes(
        cls,
        overlay: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        intrinsics: FactoryIntrinsics,
        length: float,
    ) -> None:
        axes = np.asarray(
            [[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
            dtype=np.float64,
        )
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation @ axes.T + tvec).T
        rs_intrinsics = cls._rs_intrinsics(intrinsics)
        pixels = [rs.rs2_project_point_to_pixel(rs_intrinsics, point.tolist()) for point in camera_points]
        origin = tuple(round(value) for value in pixels[0])
        for endpoint, color in zip(pixels[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)], strict=True):
            cv2.line(overlay, origin, tuple(round(value) for value in endpoint), color, 2, cv2.LINE_AA)
