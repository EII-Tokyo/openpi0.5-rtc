from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_cad_finger_task5_drive_diagnostics.yaml"


def test_max_force_profile_changes_only_one_authored_drive_variable() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    baseline = document["profiles"]["baseline"]
    diagnostic = document["profiles"]["max_force_only"]

    assert diagnostic["changed_variable"] == "drive:linear:physics:maxForce"
    assert diagnostic["max_force_n"] == {"left": 5.0, "right": 5.0}
    assert diagnostic["stiffness"] == baseline["stiffness"] == 200.0
    assert diagnostic["damping"] == baseline["damping"] == 0.0
    assert diagnostic["drive_type"] == baseline["drive_type"] == "force"
    assert document["frozen"]["bottle"] == "NOT_PRESENT"
    assert document["boundaries"]["final_collider_modified"] is False


def test_root_frame_profile_is_computed_and_drive_frozen() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    diagnostic = document["profiles"]["root_frame_only"]
    evidence = document["parameter_evidence"]["root_frame"]

    assert diagnostic["joint"].endswith("/rootJoint_vx300s_left")
    assert diagnostic["local_pos0_m"] == [-0.4690000116825104, 0.5, 0.0]
    assert diagnostic["local_rot0_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert diagnostic["source"] == (
        "computed from body1 local-to-world transform"
    )
    assert evidence["initial_frame_translation_mismatch_m"] > 0.68
    assert document["frozen"]["stiffness"] == 200.0
    assert document["frozen"]["damping"] == 0.0


def test_combined_profile_adds_only_root_frame_to_max_force_parent() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    diagnostic = document["profiles"]["max_force_plus_root_frame"]

    assert diagnostic["parent_profile"] == "max_force_only"
    assert diagnostic["inherited_max_force_n"] == {
        "left": 5.0,
        "right": 5.0,
    }
    assert diagnostic["changed_variable"] == [
        "physics:localPos0",
        "physics:localRot0",
    ]
    assert diagnostic["joint"].endswith("/rootJoint_vx300s_left")
    assert document["frozen"]["collider"] == (
        "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC"
    )
    assert document["boundaries"]["bottle_contact_grasp"] == "NOT_RUN"


def test_arm_max_force_profile_uses_urdf_effort_values_only() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    diagnostic = document["profiles"]["arm_max_force_over_combined"]

    assert diagnostic["parent_profile"] == "max_force_plus_root_frame"
    assert diagnostic["changed_variable"] == (
        "drive:angular:physics:maxForce"
    )
    assert diagnostic["arm_max_force"] == {
        "vx300s_left_waist": 10.0,
        "vx300s_left_shoulder": 20.0,
        "vx300s_left_elbow": 15.0,
        "vx300s_left_forearm_roll": 2.0,
        "vx300s_left_wrist_angle": 5.0,
        "vx300s_left_wrist_rotate": 1.0,
    }
    assert diagnostic["stiffness_policy"] == "INHERIT_UNCHANGED"
    assert diagnostic["damping_policy"] == "INHERIT_UNCHANGED"
    assert document["boundaries"]["final_collider_modified"] is False
