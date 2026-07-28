from __future__ import annotations

from PIL import Image
import pytest

from tools.aloha1_mapping.cad_gripper_screenshot_review import camera_matrix_mm
from tools.aloha1_mapping.cad_gripper_screenshot_review import color_bbox
from tools.aloha1_mapping.cad_gripper_screenshot_review import remap_point
from tools.aloha1_mapping.cad_gripper_screenshot_review import review_status


def test_color_bbox_finds_blue_and_orange_finger_pixels() -> None:
    image = Image.new("RGB", (100, 80), (30, 35, 42))
    for x in range(10, 31):
        for y in range(12, 42):
            image.putpixel((x, y), (18, 100, 180))
    for x in range(62, 91):
        for y in range(20, 71):
            image.putpixel((x, y), (190, 78, 15))

    assert color_bbox(image, role="cad_positive_x_finger") == (10, 12, 30, 41)
    assert color_bbox(image, role="cad_negative_x_finger") == (62, 20, 90, 70)


def test_remap_point_preserves_relative_location_between_bboxes() -> None:
    assert remap_point(
        point=(25.0, 75.0),
        source_bbox=(0.0, 50.0, 100.0, 150.0),
        target_bbox=(200.0, 300.0, 400.0, 500.0),
    ) == pytest.approx((250.0, 350.0))


def test_review_status_only_passes_when_all_sixteen_files_are_reviewed() -> None:
    accepted = [{"visual_self_review": "PASS"} for _ in range(8)]
    rejected = accepted[:-1] + [{"visual_self_review": "FAIL"}]

    assert review_status(accepted) == "PASS"
    assert review_status(rejected) == "FAIL"
    assert review_status(accepted[:-1]) == "PARTIAL"


def test_camera_matrix_uses_image_axes_and_negative_forward_as_columns() -> None:
    matrix = camera_matrix_mm(
        camera={
            "image_right": [1.0, 0.0, 0.0],
            "image_up": [0.0, 1.0, 0.0],
            "camera_forward": [0.0, 0.0, -1.0],
            "camera_location_mm": [10.0, 20.0, 30.0],
        }
    )

    assert matrix == [
        [1.0, 0.0, -0.0, 10.0],
        [0.0, 1.0, -0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
