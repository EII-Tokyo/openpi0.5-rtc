from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from tools.aloha1_mapping.screenshot_manifest import ScreenshotEvidenceError
from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot


def _write_nonblank_png(path: Path, size: tuple[int, int] = (1280, 900)) -> None:
    image = Image.new("RGB", size, (8, 16, 24))
    for x in range(0, size[0], 64):
        for y in range(0, size[1], 64):
            image.putpixel((x, y), ((x // 4) % 255, (y // 3) % 255, 180))
    image.save(path)


def test_validate_screenshot_records_absolute_path_and_both_hashes(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "screenshots").resolve()
    root.mkdir()
    screenshot = root / "open_closing_axis.png"
    _write_nonblank_png(screenshot)

    record = validate_screenshot(
        screenshot,
        artifact_root=root,
        phase="asset_preflight",
        capture_name="open_closing_axis",
        gate_status="PASS",
        camera={"position": [0.4, 0.0, 0.5], "orientation_wxyz": [1, 0, 0, 0]},
        simulation={"time_s": 0.0, "joint_positions": [0.057, -0.057]},
    )

    assert record["status"] == "PASS"
    assert record["absolute_path"] == str(screenshot)
    assert record["resolution"] == [1280, 900]
    assert record["file_sha256"] == hashlib.sha256(screenshot.read_bytes()).hexdigest()
    assert len(record["decoded_pixel_sha256"]) == 64
    assert record["phase"] == "asset_preflight"
    assert record["capture_name"] == "open_closing_axis"
    assert record["camera"]["position"] == [0.4, 0.0, 0.5]
    assert record["simulation"]["joint_positions"] == [0.057, -0.057]


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("relative", "absolute"),
        ("outside", "outside"),
        ("missing", "does not exist"),
        ("wrong_size", "resolution"),
        ("blank", "blank"),
    ],
)
def test_validate_screenshot_rejects_invalid_evidence(
    tmp_path: Path,
    case: str,
    expected_message: str,
) -> None:
    root = (tmp_path / "screenshots").resolve()
    root.mkdir()
    screenshot = root / "capture.png"
    if case == "relative":
        screenshot = Path("capture.png")
    elif case == "outside":
        screenshot = (tmp_path / "outside.png").resolve()
        _write_nonblank_png(screenshot)
    elif case == "missing":
        pass
    elif case == "wrong_size":
        _write_nonblank_png(screenshot, (640, 480))
    elif case == "blank":
        Image.new("RGB", (1280, 900), (0, 0, 0)).save(screenshot)

    with pytest.raises(ScreenshotEvidenceError, match=expected_message):
        validate_screenshot(
            screenshot,
            artifact_root=root,
            phase="asset_preflight",
            capture_name="capture",
            gate_status="PASS",
            camera={},
            simulation={},
        )


def test_build_manifest_fails_when_required_capture_is_missing(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "screenshots").resolve()
    root.mkdir()
    screenshot = root / "open_closing_axis.png"
    _write_nonblank_png(screenshot)
    capture = validate_screenshot(
        screenshot,
        artifact_root=root,
        phase="asset_preflight",
        capture_name="open_closing_axis",
        gate_status="PASS",
        camera={},
        simulation={},
    )

    manifest = build_screenshot_manifest(
        captures=[capture],
        required_captures={
            "asset_preflight": ["open_closing_axis", "open_isometric"]
        },
        artifact_root=root,
    )

    assert manifest["status"] == "FAIL"
    assert manifest["missing_required_captures"] == [
        "asset_preflight/open_isometric"
    ]


def test_build_manifest_requires_capture_gate_pass_and_unique_names(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "screenshots").resolve()
    root.mkdir()
    screenshot = root / "closed_isometric.png"
    _write_nonblank_png(screenshot)
    capture = validate_screenshot(
        screenshot,
        artifact_root=root,
        phase="asset_preflight",
        capture_name="closed_isometric",
        gate_status="FAIL",
        camera={},
        simulation={},
    )

    manifest = build_screenshot_manifest(
        captures=[capture, dict(capture)],
        required_captures={"asset_preflight": ["closed_isometric"]},
        artifact_root=root,
    )

    assert manifest["status"] == "FAIL"
    assert manifest["duplicate_capture_keys"] == [
        "asset_preflight/closed_isometric"
    ]
    assert manifest["failed_capture_gates"] == [
        "asset_preflight/closed_isometric"
    ]
