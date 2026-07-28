from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np

from tools.aloha1_mapping.finger_palm_orientation import apply_orientation_candidate
from tools.aloha1_mapping.finger_palm_orientation import mirror_pair_residual_m
from tools.aloha1_mapping.finger_palm_orientation import required_capture_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_STATE_ROOT = (
    PROJECT_ROOT
    / ".codex/artifacts/20260728-aloha1-gripper-orientation"
)


def _obj_points(path: Path) -> np.ndarray:
    return np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )


def _binary_stl_points(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    points = np.empty((triangle_count * 3, 3), dtype=np.float64)
    for index in range(triangle_count):
        points[index * 3 : index * 3 + 3] = np.asarray(
            struct.unpack_from("<9f", payload, 84 + index * 50 + 12),
            dtype=np.float64,
        ).reshape((3, 3))
    return points


def test_fixed_custom_fingers_are_a_geometric_mirror_pair() -> None:
    asset_root = (
        PROJECT_ROOT / "external/gym-aloha/gym_aloha/assets"
    )
    left = _binary_stl_points(
        asset_root / "vx300s_10_custom_finger_left.stl"
    )
    right = _binary_stl_points(
        asset_root / "vx300s_10_custom_finger_right.stl"
    )

    assert mirror_pair_residual_m(left, right) < 1.0e-5


def test_candidates_preserve_mount_center_and_c_preserves_symmetry() -> None:
    left = _obj_points(HISTORICAL_STATE_ROOT / "closed/physical_left.obj")
    right = _obj_points(HISTORICAL_STATE_ROOT / "closed/physical_right.obj")
    original_centers = {
        "left": 0.5 * (left.min(axis=0) + left.max(axis=0)),
        "right": 0.5 * (right.min(axis=0) + right.max(axis=0)),
    }
    baseline_residual = mirror_pair_residual_m(left, right)

    candidate_b = apply_orientation_candidate(
        "B_LEFT_ONLY_DIAGNOSTIC",
        left,
        right,
    )
    candidate_c = apply_orientation_candidate(
        "C_BILATERAL_SYMMETRIC",
        left,
        right,
    )
    candidate_d = apply_orientation_candidate(
        "D_RIGHT_ONLY_DIAGNOSTIC",
        left,
        right,
    )

    for candidate in (candidate_b, candidate_c, candidate_d):
        for side in ("left", "right"):
            actual_center = 0.5 * (
                candidate[side].min(axis=0) + candidate[side].max(axis=0)
            )
            assert np.allclose(
                actual_center,
                original_centers[side],
                atol=1.0e-12,
                rtol=0.0,
            )

    assert np.array_equal(candidate_b["right"], right)
    assert not np.array_equal(candidate_b["left"], left)
    assert not np.array_equal(candidate_c["left"], left)
    assert not np.array_equal(candidate_c["right"], right)
    assert np.array_equal(candidate_d["left"], left)
    assert not np.array_equal(candidate_d["right"], right)
    assert mirror_pair_residual_m(
        candidate_c["left"],
        candidate_c["right"],
    ) <= baseline_residual + 1.0e-12


def test_screenshot_contract_requires_open_and_closed_top_views() -> None:
    names = required_capture_names()

    for candidate in (
        "A_CURRENT_REJECTED",
        "B_LEFT_ONLY_DIAGNOSTIC",
        "D_RIGHT_ONLY_DIAGNOSTIC",
        "C_BILATERAL_SYMMETRIC",
    ):
        for view in (
            "tip_from_upper",
            "tip_from_lower",
            "tip_from_left",
            "tip_from_right",
        ):
            assert f"{candidate}/open_{view}.png" in names
            assert f"{candidate}/closed_{view}.png" in names

    assert len(names) == 32


def test_authoritative_cad_visual_report_contains_reviewed_pairs() -> None:
    report_path = (
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_viper_gripper_screenshot_review.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["gate"] == "CAD_INSTALLATION_VISUAL_EVIDENCE_ONLY"
    assert report["scope_boundaries"]["isaac_runtime"] == "NOT_RUN"
    assert report["scope_boundaries"]["bottle_hold"] == "NOT_RUN"
    assert report["scope_boundaries"]["task_8"] == "NOT_RUN"
    records = {
        (record["state_id"], record["view_id"]): record
        for record in report["captures"]
    }
    assert len(records) == 8

    for key, record in records.items():
        assert Path(record["raw"]["absolute_path"]).is_file(), key
        assert Path(record["annotated"]["absolute_path"]).is_file(), key
        assert record["raw"]["visual_self_review"] == "PASS"
        assert record["annotated"]["visual_self_review"] == "PASS"
        assert record["visual_self_review"] == "PASS"
        assert record["camera"]
        assert record["target"] == (
            "supplier-CAD handed finger installation, center-facing inner "
            "surfaces, and open/closed aperture"
        )

    for view in ("true_top", "true_bottom", "tip_end", "base_oblique"):
        opened = records[("open", view)]
        closed = records[("closed", view)]
        assert opened["camera"] == closed["camera"]
        assert (
            opened["raw"]["pixel_sha256"]
            != closed["raw"]["pixel_sha256"]
        )
