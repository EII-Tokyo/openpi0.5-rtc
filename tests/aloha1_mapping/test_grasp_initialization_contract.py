from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from tools.aloha1_mapping.grasp_initialization_contract import canonical_initialization_signature
from tools.aloha1_mapping.grasp_initialization_contract import evaluate_finger_initialization

SOURCE_LIMITS = {
    "left_finger": {"lower": 0.021, "upper": 0.057},
    "right_finger": {"lower": -0.057, "upper": -0.021},
}
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs/aloha1_grasp_20cm_five_pose_cad_derived_colliders.yaml"
)


def test_initialization_rejects_unsolved_zero_fingers() -> None:
    result = evaluate_finger_initialization(
        reset_complete=False,
        dof_order=["left_finger", "right_finger"],
        targets=[0.0, 0.0],
        readback=[0.0, 0.0],
        source_limits=SOURCE_LIMITS,
        overlap_volume_m3=3.1833401720316014e-5,
    )

    assert result["status"] == "FAIL"
    assert "FAIL_INITIALIZATION_CONTRACT" in result["failure_codes"]
    assert "FINGER_LIMIT_VIOLATION" in result["failure_codes"]
    assert "FINGER_PAIR_OVERLAP" in result["failure_codes"]


def test_initialization_accepts_legal_open_pair_after_reset() -> None:
    result = evaluate_finger_initialization(
        reset_complete=True,
        dof_order=["left_finger", "right_finger"],
        targets=[0.057, -0.057],
        readback=[0.057, -0.057],
        source_limits=SOURCE_LIMITS,
        overlap_volume_m3=0.0,
    )

    assert result["status"] == "PASS"
    assert result["failure_codes"] == []
    assert result["limit_margins_m"]["left_finger"]["readback_upper"] == 0.0
    assert result["limit_margins_m"]["right_finger"]["readback_lower"] == 0.0


def test_initialization_signature_excludes_process_and_output_identity() -> None:
    base = {
        "status": "PASS",
        "target_m": [0.057, -0.057],
        "readback_m": [0.057, -0.057],
        "process_id": 100,
        "output_path": "/tmp/primary.json",
    }
    repeat = {
        **base,
        "process_id": 200,
        "output_path": "/tmp/repeat.json",
    }

    assert canonical_initialization_signature(base) == (
        canonical_initialization_signature(repeat)
    )


def test_formal_config_freezes_source_finger_limits_and_urdf_hash() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    safety = config["finger_safety"]
    urdf_path = ROOT / safety["source_urdf"]["path"]

    assert safety["dof_names"] == ["left_finger", "right_finger"]
    assert safety["dof_indices"] == [7, 8]
    assert safety["source_limits_m"] == SOURCE_LIMITS
    assert safety["source_urdf"]["right_mimic"] == {
        "joint": "left_finger",
        "multiplier": -1.0,
        "offset": 0.0,
    }
    assert safety["source_urdf"]["sha256"] == hashlib.sha256(
        urdf_path.read_bytes()
    ).hexdigest()
    assert safety["require_world_reset"] is True
    assert safety["abort_on_first_runtime_violation"] is True
