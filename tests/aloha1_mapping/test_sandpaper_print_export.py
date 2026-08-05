from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.export_sandpaper_print_templates import export_print_template_set
from tools.aloha1_mapping.sandpaper_template import EXPECTED_SOURCE_SHA256
from tools.aloha1_mapping.sandpaper_template import FINGER_CONTRACTS


def _report() -> dict[str, object]:
    sides = {}
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
                    "line_2d_mm": [[10.0 + index, 10.0], [10.0 + index, 50.0]],
                    "normal_alignment_residual": 1e-14,
                    "shared_edge_residual_mm": 1e-13,
                }
                for index, fold in enumerate(contract["folds"])
            ],
            "flat_pattern": {
                "panels": [
                    {"name": "main", "wires_2d_mm": [[[0.0, 0.0], [80.0, 0.0], [80.0, 50.0], [0.0, 50.0]]]},
                    {
                        "name": "outer_z_min",
                        "wires_2d_mm": [[[0.0, -8.0], [80.0, -8.0], [80.0, 0.0], [0.0, 0.0]]],
                    },
                    {
                        "name": "outer_z_max",
                        "wires_2d_mm": [[[0.0, 50.0], [80.0, 50.0], [80.0, 58.0], [0.0, 58.0]]],
                    },
                ],
                "bounds_mm": [0.0, -8.0, 80.0, 58.0],
                "maximum_panel_plane_residual_mm": 1e-12,
                "relief_cut_lines_2d_mm": [],
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
        "toolchain": {"freecad_version": "1.1.1", "opencascade_version": "7.8.1"},
        "design": {
            "material_total_thickness_mm": 0.0,
            "one_piece_per_finger": True,
            "overlap_tabs": False,
            "fold_count_per_finger": 2,
            "coverage": "FULL_INNER_PROFILE_PLUS_TWO_OUTER_LONGITUDINAL_PANELS",
        },
        "sides": sides,
    }


def test_export_print_template_set_is_a4_mm_layered_and_deterministic(tmp_path: Path) -> None:
    report_path = tmp_path / "review.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = export_print_template_set(report_path=report_path, output_dir=first_dir)
    second = export_print_template_set(report_path=report_path, output_dir=second_dir)

    assert first["status"] == "PASS"
    assert first["classification"] == "LOCAL_ONLY_APPROVED_ZERO_THICKNESS_PRINT_TEMPLATE"
    assert first["final_print_template"] is True
    assert first["print_scale"] == 1.0
    assert first["bend_compensation_mm"] == 0.0
    assert first["calibration_square_mm"] == [50.0, 50.0]
    assert first["material_assumption"] == "USER_APPROVED_VERY_THIN_ZERO_THICKNESS"

    for side in ("left", "right"):
        side_artifacts = first["sides"][side]["artifacts"]
        pdf_path = Path(side_artifacts["pdf"]["absolute_path"])
        dxf_path = Path(side_artifacts["dxf"]["absolute_path"])
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        dxf = dxf_path.read_text(encoding="ascii")
        assert "$INSUNITS\n70\n4" in dxf
        assert "\nCUT\n" in dxf
        assert "\nFOLD\n" in dxf
        assert "\nREFERENCE\n" in dxf
        assert "inner_z" not in dxf
        assert side_artifacts["pdf"]["page_size_mm"] == [210.0, 297.0]
        assert side_artifacts["dxf"]["units"] == "mm"
        assert side_artifacts["dxf"]["fold_line_count"] == 2
        for artifact_type in ("pdf", "dxf"):
            assert side_artifacts[artifact_type]["sha256"] == second["sides"][side]["artifacts"][artifact_type][
                "sha256"
            ]
