from __future__ import annotations

from pathlib import Path

from tools.diagnose_aloha_viper_cad_finger_task5_drive_runtime import ARM_DOF_NAMES
from tools.diagnose_aloha_viper_cad_finger_task5_drive_runtime import _arm_joint_paths

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "tools/diagnose_aloha_viper_cad_finger_task5_drive_runtime.py"
)


def test_drive_probe_keeps_acceptance_and_bottle_gates_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "NUMERIC_DIAGNOSTIC_PROBE_NOT_ACCEPTANCE_TEST" in source
    assert '"screenshot_acceptance": (' in source
    assert '"PENDING_VISUAL_MODEL_REVIEW"' in source
    assert 'else "NOT_RUN"' in source
    assert '"bottle_contact_grasp": "NOT_RUN"' in source
    assert '"task8": "NOT_RUN"' in source
    assert "set_solve_articulation_contact_last(True)" in source
    assert '"arm_drive_readback": _arm_drive_snapshot(stage)' in source


def test_arm_drive_probe_uses_explicit_urdf_order_not_sorting() -> None:
    assert ARM_DOF_NAMES == (
        "vx300s_left_waist",
        "vx300s_left_shoulder",
        "vx300s_left_elbow",
        "vx300s_left_forearm_roll",
        "vx300s_left_wrist_angle",
        "vx300s_left_wrist_rotate",
    )
    assert _arm_joint_paths() == {
        name: f"/workcell/joints/{name}" for name in ARM_DOF_NAMES
    }
