from __future__ import annotations

from tools.build_aloha_viper_cad_finger_task5_arm_max_force_diagnostic import ARM_MAX_FORCE_N
from tools.build_aloha_viper_cad_finger_task5_arm_max_force_diagnostic import render_configuration_layer
from tools.build_aloha_viper_cad_finger_task5_arm_max_force_diagnostic import render_diagnostic_stage


def test_arm_force_layer_authors_only_six_angular_max_force_values() -> None:
    text = render_configuration_layer(ARM_MAX_FORCE_N)

    assert text.count("drive:angular:physics:maxForce") == 6
    for joint_name, effort in ARM_MAX_FORCE_N.items():
        assert f'over "{joint_name}"' in text
        assert f"= {effort:g}" in text
    assert "stiffness" not in text
    assert "damping" not in text
    assert "collision" not in text.lower()
    assert "material" not in text.lower()


def test_arm_force_stage_references_frozen_combined_parent() -> None:
    text = render_diagnostic_stage()

    assert (
        "@../cad_finger_task5_max_force_plus_root_frame/"
        "aloha_viperx_supplier_cad_max_force_plus_root_frame.usda@"
        "</workcell>"
    ) in text
    assert (
        "@configuration/supplier_cad_arm_max_force_only.usda@"
    ) in text
