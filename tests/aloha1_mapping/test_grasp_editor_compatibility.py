from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROBE = Path("tools/probe_aloha1_grasp_editor_compatibility.py")
LAUNCHER = Path("tools/open_aloha1_grasp_editor_diagnostic.py")


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("grasp_editor_probe", PROBE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_uses_local_grasp_editor_and_frozen_stage() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert "isaacsim.robot_setup.grasp_editor" in source
    assert "2.0.20" in source
    assert "aloha1_signal_correspondence_workcell.usda" in source
    assert (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
        in source
    )
    assert '"left_finger"' in source
    assert '"right_finger"' in source
    assert '"waist"' in source


def test_launcher_never_saves_the_source_stage() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "session_layer" in source
    assert "save_as_stage" not in source
    assert "save_stage" not in source


def test_classification_accepts_only_exact_full_articulation_contract() -> None:
    module = _load_probe_module()
    expected = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ]
    assert (
        module.classify_compatibility(
            extension_version="2.0.20",
            dof_names=expected,
            active_joint_names=["left_finger", "right_finger"],
            arm_joint_mutation=False,
            settings_roundtrip=True,
            stage_immutable=True,
        )
        == "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"extension_version": "2.2.0"}, "INCOMPATIBLE"),
        ({"dof_names": ["left_finger", "right_finger"]}, "INCOMPATIBLE"),
        ({"active_joint_names": ["left_finger"]}, "INCOMPATIBLE"),
        ({"arm_joint_mutation": True}, "REQUIRES_DIAGNOSTIC_GRIPPER_ONLY"),
        ({"settings_roundtrip": False}, "INCOMPATIBLE"),
        ({"stage_immutable": False}, "INCOMPATIBLE"),
    ],
)
def test_classification_fails_closed(
    overrides: dict[str, object],
    expected: str,
) -> None:
    module = _load_probe_module()
    values: dict[str, object] = {
        "extension_version": "2.0.20",
        "dof_names": list(module.EXPECTED_DOF_NAMES),
        "active_joint_names": ["left_finger", "right_finger"],
        "arm_joint_mutation": False,
        "settings_roundtrip": True,
        "stage_immutable": True,
    }
    values.update(overrides)
    assert module.classify_compatibility(**values) == expected
