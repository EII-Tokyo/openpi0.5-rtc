from __future__ import annotations

from pathlib import Path

import pytest

from tools.aloha1_mapping.gripper_force_diagnosis import audit_material_pair
from tools.aloha1_mapping.gripper_force_diagnosis import combine_material_value
from tools.aloha1_mapping.gripper_force_diagnosis import friction_scan_gate

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_average_material_combine_is_explicit_and_numeric() -> None:
    assert combine_material_value(0.7, 0.5, "average") == pytest.approx(0.6)
    assert combine_material_value(0.7, 0.5, "min") == pytest.approx(0.5)
    assert combine_material_value(0.7, 0.5, "multiply") == pytest.approx(0.35)
    assert combine_material_value(0.7, 0.5, "max") == pytest.approx(0.7)


def test_material_pair_audit_reports_effective_friction() -> None:
    finger = {
        "material_path": "/World/Materials/temporary_fingertip",
        "binding_strength": "weakerThanDescendants",
        "binding_source": "direct_or_inherited_physics_binding",
        "static_friction": 0.7,
        "dynamic_friction": 0.7,
        "restitution": 0.0,
        "friction_combine_mode": "average",
        "restitution_combine_mode": "average",
    }
    bottle = {
        **finger,
        "material_path": "/World/Materials/temporary_bottle",
    }

    result = audit_material_pair(
        finger,
        bottle,
        expected_friction=0.7,
        expected_restitution=0.0,
        contact_materials={
            "material0": "/World/Materials/temporary_fingertip",
            "material1": "/World/Materials/temporary_bottle",
        },
    )

    assert result["material_applied"] is True
    assert result["combine_mode_consistent"] is True
    assert result["effective_static_friction"] == pytest.approx(0.7)
    assert result["effective_dynamic_friction"] == pytest.approx(0.7)
    assert result["effective_restitution"] == pytest.approx(0.0)
    assert result["expected_values_match"] is True
    assert result["contact_materials_match_binding"] is True


def test_material_pair_rejects_requested_mu_not_applied() -> None:
    finger = {
        "material_path": "/Finger",
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
        "friction_combine_mode": "average",
        "restitution_combine_mode": "average",
    }
    bottle = {**finger, "material_path": "/Bottle"}

    result = audit_material_pair(
        finger,
        bottle,
        expected_friction=0.7,
        expected_restitution=0.0,
        contact_materials={
            "material0": "/Wrong",
            "material1": "/Bottle",
        },
    )

    assert result["expected_values_match"] is False
    assert result["contact_materials_match_binding"] is False


def test_different_combine_modes_are_not_guessed() -> None:
    finger = {
        "material_path": "/Finger",
        "static_friction": 0.7,
        "dynamic_friction": 0.7,
        "restitution": 0.0,
        "friction_combine_mode": "average",
        "restitution_combine_mode": "average",
    }
    bottle = {
        **finger,
        "material_path": "/Bottle",
        "friction_combine_mode": "max",
    }

    result = audit_material_pair(finger, bottle)

    assert result["combine_mode_consistent"] is False
    assert result["effective_static_friction"] is None


def test_friction_scan_requires_measured_sufficient_normal_force() -> None:
    assert friction_scan_gate({"NORMAL_FORCE_STATUS": "SUFFICIENT"})["run"] is True
    gated = friction_scan_gate({"NORMAL_FORCE_STATUS": "INSUFFICIENT"})
    assert gated == {
        "run": False,
        "status": "PARTIAL",
        "reason": "stable_sufficient_normal_force_not_confirmed",
    }


def test_isaac_5_1_physics_material_purpose_uses_string_token() -> None:
    source = (PROJECT_ROOT / "tools/aloha1_mapping/gripper_force_runtime.py").read_text(encoding="utf-8")

    assert 'materialPurpose="physics"' in source
    assert "UsdShade.Tokens.physics" not in source
