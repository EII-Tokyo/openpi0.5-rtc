from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_calibration_config
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_evidence_record
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _audit_required_table_frame
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _guard_final_contact_stage_namespace
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _guard_support_plane_calibration_mode
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _passive_contact_geometry_sanity
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _load_support_plane_config
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _resolve_support_plane_options
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _should_disable_workcell_environment_collision
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _summarize_contact_pairs
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _tracking_groups
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


def test_tracking_groups_accept_scene_base_link_prefixed_left_arm_dofs() -> None:
    groups = _tracking_groups(
        [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_left_finger",
            "left_right_finger",
        ],
        replay_mode="left_arm_and_gripper",
        finger_dof_names={"left_finger": "left_left_finger", "right_finger": "left_right_finger"},
        side="left",
    )

    assert groups["left_arm"] == [0, 1, 2, 3, 4, 5]
    assert groups["controlled"] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_workcell_environment_collision_filter_is_diagnostic_and_does_not_match_robot_or_target_paths() -> None:
    assert _should_disable_workcell_environment_collision("/scene/worldBody/table/collisions/table/table/table")
    assert _should_disable_workcell_environment_collision("/scene/worldBody/__27/collisions/__27/__27/extrusion_1000")
    assert _should_disable_workcell_environment_collision("/World/Table/collisions/table")

    assert not _should_disable_workcell_environment_collision("/scene/left_base_link/left_wrist_link/collisions/wrist")
    assert not _should_disable_workcell_environment_collision(
        "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy"
    )
    assert not _should_disable_workcell_environment_collision(
        "/World/phase43_passive_contact_cube/Collisions/COL_Body_14/COL_Body_14Mesh"
    )
    assert not _should_disable_workcell_environment_collision("/World/phase58_static_support_plane/Collision")


def test_passive_contact_geometry_sanity_rejects_implausible_open_gap() -> None:
    sanity = _passive_contact_geometry_sanity(
        finger_surface_gap_stage_units=0.745959177295105,
        object_side_length_stage_units=0.44757550637706295,
        stage_units_in_meters=1.0,
        max_finger_surface_gap_meters=0.12,
        max_generated_object_side_meters=0.08,
    )

    assert sanity["status"] == "FAIL_IMPLAUSIBLE_FINGER_GAP"
    assert sanity["pass"] is False
    assert sanity["finger_surface_gap_open_meters"] == pytest.approx(0.745959177295105)


def test_passive_contact_geometry_sanity_rejects_implausible_object_size() -> None:
    sanity = _passive_contact_geometry_sanity(
        finger_surface_gap_stage_units=0.09,
        object_side_length_stage_units=0.081,
        stage_units_in_meters=1.0,
        max_finger_surface_gap_meters=0.12,
        max_generated_object_side_meters=0.08,
    )

    assert sanity["status"] == "FAIL_IMPLAUSIBLE_OBJECT_SIZE"
    assert sanity["pass"] is False
    assert sanity["object_side_length_meters"] == pytest.approx(0.081)


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


def test_contact_summary_accepts_collision_descendants_under_finger_link() -> None:
    object_path = "/World/object"
    finger_link = "/scene/left_base_link/left_left_finger_link"

    summary = _summarize_contact_pairs(
        contact_pair_rows=[
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{finger_link}/collisions/left_left_g0/left_left_g0",
                "collider1": object_path,
                "sorted_pair": [object_path, f"{finger_link}/collisions/left_left_g0/left_left_g0"],
            }
        ],
        object_path=object_path,
        expected_finger_paths=[finger_link],
    )

    assert summary["target_contact_pair_found"] is True
    assert summary["target_contact_found_event"] is True
    assert summary["target_contact_finger_hits"][finger_link] is True


def test_contact_summary_classifies_non_target_object_contacts() -> None:
    object_path = "/World/object"
    finger_link = "/scene/left_base_link/left_left_finger_link"
    support_path = "/World/support_plane"

    summary = _summarize_contact_pairs(
        contact_pair_rows=[
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{finger_link}/collisions/left_left_g0/left_left_g0",
                "collider1": object_path,
                "sorted_pair": [object_path, f"{finger_link}/collisions/left_left_g0/left_left_g0"],
            },
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": "/scene/left_base_link/left_wrist_link/collisions/wrist",
                "collider1": object_path,
                "sorted_pair": [object_path, "/scene/left_base_link/left_wrist_link/collisions/wrist"],
            },
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": support_path,
                "collider1": object_path,
                "sorted_pair": [object_path, support_path],
            },
        ],
        object_path=object_path,
        expected_finger_paths=[finger_link],
        diagnostic_contact_paths=[support_path],
        same_side_robot_root="/scene/left_base_link",
        other_side_robot_root="/scene/right_base_link",
    )

    assert summary["object_contact_categories"]["target_finger"]["contact_pair_count"] == 1
    assert summary["object_contact_categories"]["same_side_robot_non_target"]["contact_pair_count"] == 1
    assert summary["object_contact_categories"]["diagnostic_support"]["contact_pair_count"] == 1
    assert summary["non_target_object_contact_found"] is True
    assert summary["non_target_object_contact_pair_count"] == 2


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


def test_support_plane_config_requires_calibrated_gate_or_explicit_diagnostic_opt_in() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        require_calibrated_table_frame=False,
        allow_diagnostic_support_plane_config=False,
    )

    with pytest.raises(ValueError, match="--allow-diagnostic-support-plane-config"):
        _guard_support_plane_calibration_mode(args)


def test_support_plane_config_allows_explicit_diagnostic_opt_in() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        require_calibrated_table_frame=False,
        allow_diagnostic_support_plane_config=True,
    )

    _guard_support_plane_calibration_mode(args)


def test_require_calibrated_table_frame_rejects_diagnostic_config() -> None:
    import argparse

    args = argparse.Namespace(
        require_calibrated_table_frame=True,
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        stage_units_in_meters=1.0,
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
    args = argparse.Namespace(
        require_calibrated_table_frame=True, support_plane_config=str(path), stage_units_in_meters=1.0
    )

    audit = _audit_required_table_frame(args)

    assert audit is not None
    assert audit["status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"


def test_require_calibrated_table_frame_rejects_support_plane_cli_overrides(tmp_path: Path) -> None:
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
    args = argparse.Namespace(
        require_calibrated_table_frame=True,
        support_plane_config=str(path),
        stage_units_in_meters=1.0,
        support_plane_center=[0.593227851197621, 0.7853100288947757, -0.3171450733686908],
        support_plane_size=2.0,
        support_plane_size_x=None,
        support_plane_size_y=None,
        support_plane_thickness=0.02,
    )

    with pytest.raises(ValueError, match="cannot combine --support-plane-config with support-plane CLI overrides"):
        _audit_required_table_frame(args)


def test_require_calibrated_table_frame_rejects_legacy_centimeter_world_units(tmp_path: Path) -> None:
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
    args = argparse.Namespace(
        require_calibrated_table_frame=True, support_plane_config=str(path), stage_units_in_meters=0.01
    )

    with pytest.raises(ValueError, match="requires --stage-units-in-meters 1.0"):
        _audit_required_table_frame(args)


def test_final_contact_validation_rejects_scene_overlay_with_legacy_puppet_proxy_paths(tmp_path: Path) -> None:
    import argparse

    stage = tmp_path / "scene_overlay.usda"
    stage.write_text(
        "\n".join(
            [
                "#usda 1.0",
                'over "scene"',
                "{",
                '    over "left_base_link" {}',
                '    over "right_base_link" {}',
                "}",
            ]
        )
    )
    args = argparse.Namespace(
        require_calibrated_table_frame=True, stage_usd=str(stage), contact_proxy_profile="legacy_puppet"
    )

    with pytest.raises(ValueError, match="contact validator uses legacy /puppet_"):
        _guard_final_contact_stage_namespace(args)


def test_final_contact_validation_allows_scene_overlay_with_scene_proxy_profile(tmp_path: Path) -> None:
    import argparse

    stage = tmp_path / "scene_overlay.usda"
    stage.write_text(
        "\n".join(
            [
                "#usda 1.0",
                'over "scene"',
                "{",
                '    over "left_base_link" {}',
                '    over "right_base_link" {}',
                '    def Cube "bbox_collision_proxy" {}',
                "}",
            ]
        )
    )
    args = argparse.Namespace(
        require_calibrated_table_frame=True, stage_usd=str(stage), contact_proxy_profile="scene_base_link"
    )

    summary = _guard_final_contact_stage_namespace(args)

    assert summary["stage_namespace_hints"]["uses_scene_namespace"] is True
    assert summary["finger_proxy_namespace_roots"] == ["scene"]


def test_final_contact_validation_allows_legacy_puppet_runtime_stage_namespace(tmp_path: Path) -> None:
    import argparse

    stage = tmp_path / "legacy_puppet_runtime.usda"
    stage.write_text(
        "\n".join(
            [
                "#usda 1.0",
                'over "puppet_left_vx300s" {}',
                'over "puppet_right_vx300s" {}',
                'def Cube "bbox_collision_proxy" {}',
            ]
        )
    )
    args = argparse.Namespace(
        require_calibrated_table_frame=True, stage_usd=str(stage), contact_proxy_profile="legacy_puppet"
    )

    summary = _guard_final_contact_stage_namespace(args)

    assert summary["stage_namespace_hints"]["uses_legacy_puppet_namespace"] is True
