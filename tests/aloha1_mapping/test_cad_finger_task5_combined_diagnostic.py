from __future__ import annotations

from tools.build_aloha_viper_cad_finger_task5_combined_diagnostic import render_configuration_layer
from tools.build_aloha_viper_cad_finger_task5_combined_diagnostic import render_diagnostic_stage


def test_combined_layer_authors_only_root_joint_frame() -> None:
    text = render_configuration_layer(
        position=(-0.4690000116825104, 0.5, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
    )

    assert 'over "rootJoint_vx300s_left"' in text
    assert "physics:localPos0" in text
    assert "physics:localRot0" in text
    assert "drive:" not in text
    assert "collision" not in text.lower()
    assert "material" not in text.lower()


def test_combined_stage_references_frozen_max_force_parent() -> None:
    text = render_diagnostic_stage()

    assert (
        "@../cad_finger_task5_max_force_only/"
        "aloha_viperx_supplier_cad_max_force_only.usda@</workcell>"
    ) in text
    assert (
        "@configuration/supplier_cad_root_frame_over_max_force.usda@"
    ) in text
