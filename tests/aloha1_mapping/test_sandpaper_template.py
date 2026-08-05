from __future__ import annotations

from pathlib import Path

import pytest

from tools.aloha1_mapping.sandpaper_template import EXPECTED_SOURCE_SHA256
from tools.aloha1_mapping.sandpaper_template import FINGER_CONTRACTS
from tools.aloha1_mapping.sandpaper_template import render_flat_pattern_svg
from tools.aloha1_mapping.sandpaper_template import validate_review_report


def _panel(name: str, points: list[list[float]]) -> dict[str, object]:
    return {
        "name": name,
        "wires_2d_mm": [points],
    }


def _report() -> dict[str, object]:
    sides: dict[str, object] = {}
    for side, contract in FINGER_CONTRACTS.items():
        sides[side] = {
            "object_name": contract["object_name"],
            "main_face_index_1_based": contract["main_face_index_1_based"],
            "main_face_area_mm2": 2020.89740871246,
            "folds": [
                {
                    "name": fold["name"],
                    "main_edge_index_1_based": fold["main_edge_index_1_based"],
                    "adjacent_face_index_1_based": fold["adjacent_face_index_1_based"],
                    "line_2d_mm": [[10.0 + i, 10.0], [10.0 + i, 50.0]],
                    "normal_alignment_residual": 1e-14,
                    "shared_edge_residual_mm": 1e-13,
                }
                for i, fold in enumerate(contract["folds"])
            ],
            "flat_pattern": {
                "panels": [
                    _panel("main", [[0.0, 0.0], [80.0, 0.0], [80.0, 50.0], [0.0, 50.0]]),
                    _panel("outer_z_min", [[0.0, -8.0], [80.0, -8.0], [80.0, 0.0], [0.0, 0.0]]),
                    _panel("outer_z_max", [[0.0, 50.0], [80.0, 50.0], [80.0, 58.0], [0.0, 58.0]]),
                    _panel("inner_z_min", [[15.0, 12.0], [50.0, 12.0], [50.0, 18.0], [15.0, 18.0]]),
                    _panel("inner_z_max", [[15.0, 32.0], [50.0, 32.0], [50.0, 38.0], [15.0, 38.0]]),
                ],
                "bounds_mm": [0.0, -8.0, 80.0, 50.0],
                "maximum_panel_plane_residual_mm": 1e-12,
                "relief_cut_lines_2d_mm": [[[30.0, 12.0], [30.0, 38.0]]],
            },
        }
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "LOCAL_ONLY_ZERO_THICKNESS_SANDPAPER_REVIEW",
        "source": {
            "sha256": EXPECTED_SOURCE_SHA256,
            "license_status": "UNKNOWN_HARD_BLOCKER_LOCAL_ONLY",
            "read_only": True,
        },
        "toolchain": {
            "freecad_version": "1.1.1",
            "opencascade_version": "7.8.1",
        },
        "design": {
            "material_total_thickness_mm": 0.0,
            "one_piece_per_finger": True,
            "overlap_tabs": False,
            "fold_count_per_finger": 4,
            "coverage": "FULL_INNER_PROFILE_PLUS_FOUR_ADJACENT_LONGITUDINAL_PANELS",
        },
        "sides": sides,
    }


def test_contract_uses_two_installed_handed_fingers_without_mirroring() -> None:
    assert FINGER_CONTRACTS["left"]["object_name"] == "Part__Feature007"
    assert FINGER_CONTRACTS["right"]["object_name"] == "Part__Feature008"
    assert FINGER_CONTRACTS["left"]["main_face_index_1_based"] == 117
    assert FINGER_CONTRACTS["right"]["main_face_index_1_based"] == 128
    assert all(len(contract["folds"]) == 4 for contract in FINGER_CONTRACTS.values())
    assert FINGER_CONTRACTS["left"]["mirror_applied"] is False
    assert FINGER_CONTRACTS["right"]["mirror_applied"] is False


def test_review_report_contract_accepts_zero_thickness_local_only_design() -> None:
    validate_review_report(_report())


def test_review_report_rejects_wrong_source_hash() -> None:
    report = _report()
    report["source"]["sha256"] = "0" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="source hash"):
        validate_review_report(report)


def test_svg_is_physical_mm_and_distinguishes_cut_and_fold_lines(tmp_path: Path) -> None:
    output = tmp_path / "left.svg"
    render_flat_pattern_svg(_report(), side="left", output_path=output)

    text = output.read_text(encoding="utf-8")
    assert 'width="210mm"' in text
    assert 'height="297mm"' in text
    assert 'class="cut"' in text
    assert 'class="fold"' in text
    assert 'class="relief"' in text
    assert "ZERO-THICKNESS REVIEW" in text
    assert "NOT FINAL PRINT TEMPLATE" in text
    assert "LEFT FINGER" in text
