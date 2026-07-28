from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from tools.render_aloha_viper_gripper_review import validate_render_outputs


def test_validate_render_outputs_rejects_blender_zero_exit_without_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="render metadata is missing"):
        validate_render_outputs(tmp_path)


def test_validate_render_outputs_requires_exactly_eight_captures(
    tmp_path: Path,
) -> None:
    (tmp_path / "render_metadata.json").write_text(
        json.dumps({"capture_count": 0, "captures": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected 8 captures"):
        validate_render_outputs(tmp_path)


def test_validate_render_outputs_accepts_eight_full_resolution_pngs(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "screenshots_raw"
    raw_root.mkdir()
    captures = []
    for index in range(8):
        path = raw_root / f"capture_{index}.png"
        Image.new("RGB", (1280, 900), (index, 10, 20)).save(path)
        captures.append({"raw_path": str(path.resolve())})
    (tmp_path / "render_metadata.json").write_text(
        json.dumps({"capture_count": 8, "captures": captures}),
        encoding="utf-8",
    )

    result = validate_render_outputs(tmp_path)

    assert result["capture_count"] == 8
    assert result["resolution"] == [1280, 900]
