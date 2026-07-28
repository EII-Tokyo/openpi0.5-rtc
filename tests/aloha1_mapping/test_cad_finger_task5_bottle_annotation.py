from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.annotate_aloha_viper_cad_finger_task5_bottle import _bottle_bbox

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tools/annotate_aloha_viper_cad_finger_task5_bottle.py"
)


def test_green_bottle_bbox_is_detected_without_using_file_hash() -> None:
    image = Image.new("RGB", (100, 80), (210, 210, 210))
    for x in range(30, 71):
        for y in range(10, 76):
            image.putpixel((x, y), (80, 200, 110))
    assert _bottle_bbox(image.convert("RGBA")) == (30, 10, 70, 75)


def test_annotations_use_runtime_contact_projection_and_explicit_boundaries() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "contact_projection" in source
    assert "Blue = left_finger / CAD +X" in source
    assert "Orange = right_finger / CAD -X" in source
    assert "PHYSICAL BILATERAL CONTACT" in source
    assert "FIXED BOTTLE; NOT HOLD" in source
    assert "20/20 STATIC HOLD GATE" in source
    assert "TEMPORARY_UNCALIBRATED" in source
    assert "machine runtime data are authoritative" in source
