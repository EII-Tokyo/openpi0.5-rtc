from __future__ import annotations

from pathlib import Path

from tools.aloha1_mapping.cad_mount_registration import build_mount_registration_report

ROOT = Path(__file__).resolve().parents[2]
PROBE = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "viper_gripper/mounting_reference_probe_v4"
)
MESH_ROOT = (
    ROOT
    / "external/ros2-essentials/aloha_ws/src/"
    "interbotix_ros_manipulators/interbotix_ros_xsarms/"
    "interbotix_xsarm_descriptions/meshes/aloha_vx300s_meshes"
)


def test_mount_registration_uses_four_nonzero_planar_datums() -> None:
    report = build_mount_registration_report(
        probe_manifest_path=PROBE / "manifest.json",
        cad_shell_obj_path=PROBE / "Part__Feature006.obj",
        follower_urdf_path=ROOT / "generated/urdf/follower_left.urdf",
        gripper_stl_path=MESH_ROOT / "gripper.stl",
        gripper_bar_stl_path=MESH_ROOT / "gripper_bar.stl",
    )

    assert report["status"] == "PASS"
    assert report["method"] == "CONTROLLED_ORTHOGONAL_PLANAR_DATUM_REGISTRATION"
    assert report["decision_boundary"]["full_surface_icp_used"] is False
    assert report["threshold_m"] == 0.0002
    assert set(report["datums"]) == {"x_min", "y_min", "y_max", "z_max"}
    for datum in report["datums"].values():
        assert datum["cad"]["triangle_count"] > 0
        assert datum["cad"]["area_m2"] > 0.0
        assert datum["stage"]["triangle_count"] > 0
        assert datum["stage"]["area_m2"] > 0.0
        assert datum["absolute_coordinate_residual_m"] <= 0.0002
        assert datum["status"] == "PASS"
    assert report["gates"]["proper_rotation_no_mirror"] is True
    assert report["gates"]["four_nonzero_planar_datums"] is True
