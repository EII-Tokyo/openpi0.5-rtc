from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from tools.aloha1_mapping.grasp_initialization_contract import canonical_initialization_signature
from tools.aloha1_mapping.grasp_initialization_contract import evaluate_finger_initialization
from tools.aloha1_mapping.grasp_initialization_contract import evaluate_finger_runtime_frame

SOURCE_LIMITS = {
    "left_finger": {"lower": 0.021, "upper": 0.057},
    "right_finger": {"lower": -0.057, "upper": -0.021},
}
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs/aloha1_grasp_20cm_five_pose_cad_derived_colliders.yaml"
)
RUNTIME_CONFIG_PATH = (
    ROOT / "configs/aloha1_grasp_20cm_gui_cad_derived_colliders.yaml"
)
FINGER_PATHS = {
    "left_finger": "/World/follower_left/vx300s_left/follower_left_left_finger_link",
    "right_finger": "/World/follower_left/vx300s_left/follower_left_right_finger_link",
}


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


def test_runtime_config_exposes_same_finger_safety_contract() -> None:
    formal = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    runtime = yaml.safe_load(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))

    assert runtime["finger_safety"] == formal["finger_safety"]


def test_runtime_frame_rejects_right_finger_outside_source_limit() -> None:
    result = evaluate_finger_runtime_frame(
        frame=223,
        phase="OPEN_PREGRASP",
        targets=[0.057, -0.057],
        readback=[0.057, -0.0138],
        source_limits=SOURCE_LIMITS,
        pair_overlap_volume_m3=0.0,
        contacts=[],
        finger_paths=FINGER_PATHS,
    )

    assert result["status"] == "FAIL"
    assert result["failure_codes"] == ["FINGER_LIMIT_VIOLATION"]
    assert result["first_failure"]["frame"] == 223


def test_runtime_frame_classifies_environment_forced_limit_violation() -> None:
    result = evaluate_finger_runtime_frame(
        frame=223,
        phase="OPEN_PREGRASP",
        targets=[0.057, -0.057],
        readback=[0.057, -0.017872605472803116],
        source_limits=SOURCE_LIMITS,
        pair_overlap_volume_m3=0.0,
        contacts=[
            {
                "actor0_path": FINGER_PATHS["right_finger"],
                "actor1_path": "/World/environment/worldBody/__13",
                "collider0_path": f"{FINGER_PATHS['right_finger']}/collisions/finger",
                "collider1_path": (
                    "/World/environment/worldBody/__13/collisions/__13/"
                    "__13/angled_extrusion"
                ),
                "impulse_ns": 0.30386159398198753,
                "separation_m": 8.073169738054276e-05,
            }
        ],
        finger_paths=FINGER_PATHS,
    )

    assert result["status"] == "FAIL"
    assert "FINGER_LIMIT_VIOLATION" in result["failure_codes"]
    assert (
        "ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION"
        in result["failure_codes"]
    )
    assert len(result["finger_environment_contacts"]) == 1


def test_runtime_frame_allows_bilateral_bottle_contact_inside_limits() -> None:
    bottle = "/World/ALOHA1Grasp20cmSession/Bottle500"
    contacts = [
        {
            "actor0_path": bottle,
            "actor1_path": FINGER_PATHS[finger],
            "collider0_path": f"{bottle}/Collisions/body",
            "collider1_path": f"{FINGER_PATHS[finger]}/collisions/finger",
            "impulse_ns": 0.01,
            "separation_m": -1.0e-5,
        }
        for finger in ("left_finger", "right_finger")
    ]

    result = evaluate_finger_runtime_frame(
        frame=454,
        phase="BILATERAL_CONTACT",
        targets=[0.048316874538855845, -0.048316874538855845],
        readback=[0.0491, -0.0495],
        source_limits=SOURCE_LIMITS,
        pair_overlap_volume_m3=0.0,
        contacts=contacts,
        finger_paths=FINGER_PATHS,
    )

    assert result["status"] == "PASS"
    assert result["failure_codes"] == []
    assert result["finger_environment_contacts"] == []


def test_runtime_frame_records_harmless_environment_contact_without_failure() -> None:
    result = evaluate_finger_runtime_frame(
        frame=100,
        phase="OPEN_PREGRASP",
        targets=[0.057, -0.057],
        readback=[0.0569, -0.0569],
        source_limits=SOURCE_LIMITS,
        pair_overlap_volume_m3=0.0,
        contacts=[
            {
                "actor0_path": FINGER_PATHS["left_finger"],
                "actor1_path": "/World/environment/worldBody/user_confirmed_table",
                "collider0_path": f"{FINGER_PATHS['left_finger']}/collisions/finger",
                "collider1_path": "/World/environment/worldBody/user_confirmed_table",
                "impulse_ns": 0.0,
                "separation_m": 0.001,
            }
        ],
        finger_paths=FINGER_PATHS,
    )

    assert result["status"] == "PASS"
    assert len(result["finger_environment_contacts"]) == 1


def test_runtime_frame_rejects_positive_finger_pair_overlap() -> None:
    result = evaluate_finger_runtime_frame(
        frame=1,
        phase="VALIDATE",
        targets=[0.021, -0.021],
        readback=[0.021, -0.021],
        source_limits=SOURCE_LIMITS,
        pair_overlap_volume_m3=1.0e-9,
        contacts=[],
        finger_paths=FINGER_PATHS,
    )

    assert result["status"] == "FAIL"
    assert result["failure_codes"] == ["FINGER_PAIR_OVERLAP"]
