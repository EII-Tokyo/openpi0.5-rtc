from pathlib import Path

from tools.aloha1_mapping.gripper_validation import build_gripper_validation_plan
from tools.aloha1_mapping.gripper_validation import canonicalize_contact_events
from tools.aloha1_mapping.gripper_validation import classify_gripper_trial
from tools.aloha1_mapping.gripper_validation import classify_repeat_determinism
from tools.aloha1_mapping.gripper_validation import summarize_contact_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_plan_uses_urdf_limits_and_explicit_mimic_relation() -> None:
    plan = build_gripper_validation_plan(PROJECT_ROOT)

    assert [robot["name"] for robot in plan["robots"]] == [
        "follower_left",
        "follower_right",
    ]
    for robot in plan["robots"]:
        assert robot["dof_order"][-2:] == ["left_finger", "right_finger"]
        assert robot["open_left_finger_m"] == 0.057
        assert robot["closed_left_finger_m"] == 0.021
        assert robot["mimic"] == {
            "target": "right_finger",
            "reference": "left_finger",
            "multiplier": -1.0,
            "offset": 0.0,
        }


def test_plan_marks_proxy_and_material_uncalibrated() -> None:
    plan = build_gripper_validation_plan(PROJECT_ROOT)

    assert plan["physics"]["solve_articulation_contact_last"] is True
    assert plan["physics"]["self_collision"] is False
    assert plan["physics"]["author_contact_rest_offsets"] is False
    assert plan["fingertip_material"]["status"] == "TEMPORARY_UNCALIBRATED"
    assert max(plan["fingertip_material"]["friction_scan"]) <= 0.7
    assert plan["bottle_proxy"]["diameter_m"] == 0.065
    assert plan["bottle_proxy"]["height_m"] == 0.210
    assert plan["bottle_proxy"]["mass_kg"] == 0.020
    assert plan["bottle_proxy"]["status"] == "PARTIAL_MEASURED_BODY_PROXY"
    assert plan["released_hold"]["surface_gripper_allowed"] is False
    assert plan["released_hold"]["fixed_constraint_allowed"] is False


def _passing_metrics() -> dict:
    return {
        "solve_articulation_contact_last_ok": True,
        "open_direction_ok": True,
        "close_direction_ok": True,
        "limits_ok": True,
        "readback_ok": True,
        "mimic_ok": True,
        "aperture_monotonic": True,
        "left_finger_contact": True,
        "right_finger_contact": True,
        "bilateral_contact_before_release": True,
        "impulses_finite": True,
        "persistent_penetration": False,
        "unexpected_gripper_collision": False,
        "released_without_constraint": True,
        "held_for_required_steps": True,
        "finite_state": True,
    }


def test_trial_is_partial_when_all_interface_checks_pass_but_calibration_is_missing() -> None:
    result = classify_gripper_trial(
        _passing_metrics(),
        hard_blockers=["measured fingertip friction", "bottle inertia"],
    )

    assert result["status"] == "PARTIAL"
    assert result["failed_checks"] == []
    assert result["passed_interface_gate"] is True


def test_any_required_failure_produces_fail() -> None:
    metrics = _passing_metrics()
    metrics["right_finger_contact"] = False

    result = classify_gripper_trial(metrics, hard_blockers=[])

    assert result["status"] == "FAIL"
    assert result["passed_interface_gate"] is False
    assert result["failed_checks"] == ["right_finger_contact"]


def test_contact_last_readback_is_a_required_gate() -> None:
    metrics = _passing_metrics()
    metrics["solve_articulation_contact_last_ok"] = False

    result = classify_gripper_trial(metrics, hard_blockers=[])

    assert result["status"] == "FAIL"
    assert result["failed_checks"] == ["solve_articulation_contact_last_ok"]


def _contact(
    frame: int,
    finger: str,
    *,
    separation: float = -0.0002,
    impulse: tuple[float, float, float] = (0.1, 0.0, 0.0),
) -> dict:
    return {
        "frame": frame,
        "type": "CONTACT_PERSISTS",
        "collider0": f"/Robot/{finger}_finger_link/collisions/mesh",
        "collider1": "/World/BottleProxy/Collision",
        "contacts": [
            {
                "position": [0.0, 0.0, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "impulse": list(impulse),
                "separation": separation,
                "material0": "/Robot/PhysicsMaterials/fingertip",
                "material1": "/World/Materials/bottle",
            }
        ],
    }


def test_contact_summary_requires_bilateral_finite_nonpersistent_contact() -> None:
    events = [_contact(10, "left"), _contact(10, "right")]

    result = summarize_contact_events(
        events,
        bottle_path_token="/BottleProxy/",
        penetration_limit_m=0.002,
        persistence_steps=5,
    )

    assert result["left_finger_contact"] is True
    assert result["right_finger_contact"] is True
    assert result["impulses_finite"] is True
    assert result["persistent_penetration"] is False
    assert result["minimum_separation_m"] == -0.0002
    assert result["maximum_penetration_depth_m"] == 0.0002
    assert result["unexpected_gripper_collision"] is False


def test_contact_summary_flags_persistent_penetration_and_internal_collision() -> None:
    events = [_contact(frame, "left", separation=-0.003) for frame in range(5)]
    events.append(
        {
            "frame": 5,
            "type": "CONTACT_FOUND",
            "collider0": "/Robot/left_finger_link/collisions/mesh",
            "collider1": "/Robot/gripper_bar_link/collisions/mesh",
            "contacts": [],
        }
    )

    result = summarize_contact_events(
        events,
        bottle_path_token="/BottleProxy/",
        penetration_limit_m=0.002,
        persistence_steps=5,
    )

    assert result["persistent_penetration"] is True
    assert result["unexpected_gripper_collision"] is True


def test_contact_events_are_canonicalized_without_losing_fields() -> None:
    first = _contact(11, "right")
    second = _contact(10, "left")
    first["contacts"].append(
        {
            "position": [1.0, 0.0, 0.0],
            "normal": [0.0, 1.0, 0.0],
            "impulse": [0.2, 0.0, 0.0],
            "separation": -0.0001,
            "material0": "/finger",
            "material1": "/bottle",
        }
    )

    forward = canonicalize_contact_events([first, second])
    reverse = canonicalize_contact_events([second, first])

    assert forward == reverse
    assert forward[0]["frame"] == 10
    assert len(forward[1]["contacts"]) == 2


def test_repeat_determinism_uses_exact_signatures() -> None:
    assert classify_repeat_determinism(None, "abc")["status"] == "PARTIAL"
    assert classify_repeat_determinism("abc", "abc") == {
        "status": "PASS",
        "deterministic": True,
        "previous_signature": "abc",
        "current_signature": "abc",
    }
    changed = classify_repeat_determinism("abc", "def")
    assert changed["status"] == "FAIL"
    assert changed["deterministic"] is False


def test_runtime_entrypoint_pins_required_isaac_5_1_contact_apis() -> None:
    source_path = PROJECT_ROOT / "tools/validate_aloha1_gripper.py"
    source = source_path.read_text(encoding="utf-8")

    assert "set_solve_articulation_contact_last(True)" in source
    assert "get_solve_articulation_contact_last()" in source
    assert "PhysxContactReportAPI.Apply" in source
    assert "subscribe_contact_report_events" in source
    assert "PhysicsSchemaTools.intToSdfPath" in source
    assert "SurfaceGripper" not in source
