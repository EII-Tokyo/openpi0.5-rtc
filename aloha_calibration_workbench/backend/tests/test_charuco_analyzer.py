from calibration_workbench.charuco_analyzer import CharucoAnalyzer
from calibration_workbench.intrinsics_capture import FramePacket
from calibration_workbench.models import FactoryIntrinsics
import cv2
import numpy as np


def test_detects_the_exact_7x5_dict_5x5_100_target_and_reports_factory_reprojection():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.022, dictionary)
    target = board.generateImage((490, 350), marginSize=18, borderBits=1)
    gray = np.full((480, 640), 255, dtype=np.uint8)
    gray[65:415, 75:565] = target
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    intrinsics = FactoryIntrinsics(
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        distortion_model="brown_conrady",
        distortion_coefficients=[0.0] * 5,
    )

    observation, jpeg, png = CharucoAnalyzer().analyze(
        FramePacket(rgb=rgb, frame_number=9, device_timestamp_ms=41.25),
        intrinsics,
    )

    assert observation.board_detected is True
    assert observation.charuco_corner_count == 24
    assert observation.marker_count == 17
    assert observation.reprojection_rms_px is not None
    assert observation.reprojection_rms_px < 1.0
    assert 0.45 < observation.centroid_x < 0.55
    assert 0.45 < observation.centroid_y < 0.55
    assert jpeg.startswith(b"\xff\xd8")
    assert png.startswith(b"\x89PNG")
