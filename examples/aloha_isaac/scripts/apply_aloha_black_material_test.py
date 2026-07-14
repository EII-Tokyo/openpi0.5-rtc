import pytest

from examples.aloha_isaac.scripts.apply_aloha_black_material import (
    qpos_to_usd_joint_positions,
    should_apply_robot_prototype_material,
    should_apply_robot_material,
)


def test_material_applies_to_left_and_right_vx300s_meshes() -> None:
    assert should_apply_robot_material("/scene/left_base_link/left_upper_arm/visuals")
    assert should_apply_robot_material("/scene/right_base_link/right_wrist/mesh")


def test_material_does_not_apply_to_table_frame_or_cameras() -> None:
    assert not should_apply_robot_material("/scene/tabletop")
    assert not should_apply_robot_material("/scene/overhead_camera")
    assert not should_apply_robot_material("/scene/left_d405_camera")
    assert not should_apply_robot_material("/scene/left_base_link/left_upper_arm_link/collisions")


def test_material_applies_to_vx300s_prototype_meshes_but_not_cameras() -> None:
    assert should_apply_robot_prototype_material("/__Prototype_7/vx300s_3_upper_arm/vx300s_3_upper_arm")
    assert should_apply_robot_prototype_material("/__Prototype_58/vx300s_1_base/vx300s_1_base")
    assert not should_apply_robot_prototype_material("/__Prototype_28/d405_solid/d405_solid")
    assert not should_apply_robot_prototype_material("/__Prototype_1/extrusion_2040_880")


def test_neutral_qpos_converts_revolute_joints_to_degrees_and_keeps_fingers_in_meters() -> None:
    converted = qpos_to_usd_joint_positions(
        (0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.0084, 0.0084) * 2
    )

    assert converted[0] == 0.0
    assert converted[1] == pytest.approx(-55.0039483321)
    assert converted[2] == pytest.approx(66.4631042352)
    assert converted[4] == pytest.approx(-17.1887338539)
    assert converted[6] == pytest.approx(0.0084)
    assert converted[14] == pytest.approx(0.0084)
