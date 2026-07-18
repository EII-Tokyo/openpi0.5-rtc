from __future__ import annotations

import csv
from pathlib import Path

import yaml
import pytest

from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_calibration_config
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_evidence_record
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _audit_required_table_frame
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _load_support_plane_config
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _resolve_support_plane_options
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _summarize_contact_pairs
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _write_csv


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_passive_contact_csv_writer_preserves_late_diagnostic_columns(tmp_path) -> None:
    path = tmp_path / "contact.csv"
    _write_csv(
        path,
        [
            {"phase": "settle", "step": 0, "object_center_x": 0.0},
            {"phase": "close", "step": 0, "tracking_controlled_max_abs_error": 0.12},
        ],
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert "tracking_controlled_max_abs_error" in rows[0]
    assert rows[1]["tracking_controlled_max_abs_error"] == "0.12"


def test_contact_summary_classifies_diagnostic_support_contacts() -> None:
    object_path = "/World/object"
    finger_path = "/World/left_finger"
    support_path = "/World/support"

    summary = _summarize_contact_pairs(
        contact_pair_rows=[
            {
                "phase": "close",
                "step": 1,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{object_path}/body",
                "collider1": support_path,
                "sorted_pair": [f"{object_path}/body", support_path],
            },
            {
                "phase": "close",
                "step": 2,
                "type_name": "CONTACT_FOUND",
                "collider0": support_path,
                "collider1": f"{finger_path}/proxy",
                "sorted_pair": [support_path, f"{finger_path}/proxy"],
            },
            {
                "phase": "close",
                "step": 3,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{object_path}/body",
                "collider1": f"{finger_path}/proxy",
                "sorted_pair": [f"{object_path}/body", f"{finger_path}/proxy"],
            },
        ],
        object_path=object_path,
        expected_finger_paths=[finger_path],
        diagnostic_contact_paths=[support_path],
    )

    support_summary = summary["diagnostic_contact_summaries"][support_path]
    assert summary["target_contact_pair_found"] is True
    assert support_summary["contact_pair_count"] == 2
    assert support_summary["object_contact_pair_count"] == 1
    assert support_summary["expected_finger_contact_pair_count"] == 1
    assert support_summary["other_contact_pair_count"] == 0


def test_phase63_fixed_table_candidate_config_is_explicit_and_diagnostic() -> None:
    cfg = _load_support_plane_config(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml")

    assert cfg["mode"] == "fixed_box"
    assert cfg["center"] == [0.593227851197621, 0.7853100288947757, -0.3171450733686908]
    assert cfg["size"] == [1.22, 0.625, 0.04]
    assert cfg["provenance"]["table_size"]["source"] == "user_measured"
    assert cfg["provenance"]["center_xy"]["source"] == "phase60_diagnostic_object_bottom"
    assert cfg["table_frame"]["T_table_left_base"]["status"] == "not_calibrated"
    assert cfg["table_frame"]["T_table_right_base"]["status"] == "not_calibrated"


def test_support_plane_config_resolves_fixed_box_options() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        support_plane_mode="none",
        support_plane_center=None,
        support_plane_size=2.0,
        support_plane_size_x=None,
        support_plane_size_y=None,
        support_plane_thickness=0.02,
    )

    resolved = _resolve_support_plane_options(args)

    assert resolved["mode"] == "fixed_box"
    assert resolved["center"] == [0.593227851197621, 0.7853100288947757, -0.3171450733686908]
    assert resolved["size_x"] == 1.22
    assert resolved["size_y"] == 0.625
    assert resolved["thickness"] == 0.04
    assert resolved["config"] == "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"
    assert resolved["table_frame"]["T_world_table"]["status"] == "diagnostic_candidate"


def test_support_plane_config_rejects_object_bottom_mix() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        support_plane_mode="object_bottom",
        support_plane_center=None,
        support_plane_size=2.0,
        support_plane_size_x=None,
        support_plane_size_y=None,
        support_plane_thickness=0.02,
    )

    try:
        _resolve_support_plane_options(args)
    except ValueError as exc:
        assert "object_bottom" in str(exc)
    else:
        raise AssertionError("expected object_bottom/config combination to be rejected")


def test_require_calibrated_table_frame_requires_config() -> None:
    import argparse

    args = argparse.Namespace(require_calibrated_table_frame=True, support_plane_config=None)

    with pytest.raises(ValueError, match="requires --support-plane-config"):
        _audit_required_table_frame(args)


def test_require_calibrated_table_frame_rejects_diagnostic_config() -> None:
    import argparse

    args = argparse.Namespace(
        require_calibrated_table_frame=True,
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
    )

    with pytest.raises(ValueError, match="BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"):
        _audit_required_table_frame(args)


def test_require_calibrated_table_frame_accepts_measured_config(tmp_path: Path) -> None:
    import argparse

    evidence_path = tmp_path / "measurement_evidence.yaml"
    evidence_path.write_text("measurement: synthetic\n")
    cfg = build_calibration_config(
        table_top_center=[1.0, 2.0, 0.5],
        table_size=[1.22, 0.625, 0.04],
        table_yaw_deg=0.0,
        left_base_in_table=[-0.3, 0.1, 0.0],
        left_yaw_deg=0.0,
        right_base_in_table=[0.3, 0.1, 0.0],
        right_yaw_deg=180.0,
        source="user_measured",
        status="measured",
        calibration_evidence=build_evidence_record(
            evidence_path,
            evidence_type="unit_test",
            real_robot_touched=False,
            remote_103_touched=False,
        ),
    )
    path = tmp_path / "measured.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = argparse.Namespace(require_calibrated_table_frame=True, support_plane_config=str(path))

    audit = _audit_required_table_frame(args)

    assert audit is not None
    assert audit["status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"
