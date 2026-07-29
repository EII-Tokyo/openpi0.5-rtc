from tools.aloha1_mapping import task7a_swept_collision as sweep
from tools.aloha1_mapping.task7a_swept_collision import ARM_JOINTS
from tools.aloha1_mapping.task7a_swept_collision import build_sweep_cases
from tools.aloha1_mapping.task7a_swept_collision import classify_contact_observation
from tools.aloha1_mapping.task7a_swept_collision import classify_contact_pair
from tools.aloha1_mapping.task7a_swept_collision import deterministic_case_signature
from tools.aloha1_mapping.task7a_swept_collision import summarize_sweep_cases


def _limits() -> dict[str, list[dict[str, float | str]]]:
    result = {}
    for robot in ("follower_left", "follower_right"):
        result[robot] = [
            {
                "name": name,
                "lower": -2.0 - index,
                "upper": 2.0 + index,
                "home": 0.0,
            }
            for index, name in enumerate(ARM_JOINTS)
        ]
    return result


def test_sweep_plan_has_six_dofs_two_directions_for_each_follower() -> None:
    cases = build_sweep_cases(_limits())

    assert len(cases) == 24
    assert {
        (case["robot"], case["joint"], case["direction"])
        for case in cases
    } == {
        (robot, joint, direction)
        for robot in ("follower_left", "follower_right")
        for joint in ARM_JOINTS
        for direction in ("negative", "positive")
    }
    assert all(case["lower"] < case["target"] < case["upper"] for case in cases)
    assert all(case["target"] != case["home"] for case in cases)


def test_contact_pair_classification_distinguishes_safety_boundaries() -> None:
    adjacent = {
        (
            "/World/follower_left/vx300s_left/base",
            "/World/follower_left/vx300s_left/upper_arm",
        )
    }

    assert classify_contact_pair(
        "/World/follower_left/vx300s_left/base",
        "/World/follower_left/vx300s_left/base",
        adjacent,
    )["classification"] == "SAME_RIGID_BODY"
    assert classify_contact_pair(
        "/World/follower_left/vx300s_left/upper_arm",
        "/World/follower_left/vx300s_left/base",
        adjacent,
    )["classification"] == "ADJACENT_BODY_CONTACT"
    assert classify_contact_pair(
        "/World/follower_left/vx300s_left/base",
        "/World/follower_left/vx300s_left/gripper",
        adjacent,
    )["classification"] == "NON_ADJACENT_SELF_CONTACT"
    assert classify_contact_pair(
        "/World/follower_left/vx300s_left/gripper",
        "/World/follower_right/vx300s_right/gripper",
        adjacent,
    )["classification"] == "CROSS_FOLLOWER_CONTACT"
    assert classify_contact_pair(
        "/World/follower_left/vx300s_left/gripper",
        "/World/environment/table",
        adjacent,
    )["classification"] == "ROBOT_ENVIRONMENT_CONTACT"


def test_user_confirmed_finger_table_contact_is_allowed_workspace_boundary() -> None:
    result = classify_contact_pair(
        (
            "/World/follower_left/vx300s_left/"
            "follower_left_left_finger_link"
        ),
        "/World/environment/worldBody/user_confirmed_table",
        set(),
    )
    generic = classify_contact_pair(
        "/World/follower_left/vx300s_left/upper_arm",
        "/World/environment/worldBody/user_confirmed_table",
        set(),
    )

    assert result["classification"] == (
        "USER_CONFIRMED_ALLOWED_FINGER_TABLE_CONTACT"
    )
    assert result["allowed"] is True
    assert result["policy_evidence"] == "USER_CONFIRMATION_2026_07_29"
    assert generic["classification"] == "ROBOT_ENVIRONMENT_CONTACT"
    assert generic["allowed"] is False


def test_allowed_table_contact_limits_workcell_reach_without_control_fail() -> None:
    partial = sweep.classify_sweep_case(
        direction_pass=True,
        target_reached=False,
        non_target_drift_pass=True,
        legal=True,
        finite=True,
        unexpected_contact_count=0,
        allowed_workspace_contact_count=2,
    )
    forbidden = sweep.classify_sweep_case(
        direction_pass=True,
        target_reached=False,
        non_target_drift_pass=True,
        legal=True,
        finite=True,
        unexpected_contact_count=1,
        allowed_workspace_contact_count=2,
    )
    unexplained = sweep.classify_sweep_case(
        direction_pass=True,
        target_reached=False,
        non_target_drift_pass=True,
        legal=True,
        finite=True,
        unexpected_contact_count=0,
        allowed_workspace_contact_count=0,
    )

    assert partial == {
        "status": "PASS",
        "motion_status": "CONTACT_LIMITED_WORKCELL_REACHABILITY",
        "control_direction_status": "PASS",
        "collision_policy_status": "PASS",
        "target_reachability_status": (
            "CONTACT_LIMITED_BY_ALLOWED_WORKCELL_CONTACT"
        ),
    }
    assert forbidden["status"] == "FAIL"
    assert forbidden["collision_policy_status"] == "FAIL"
    assert unexplained["status"] == "FAIL"
    assert unexplained["motion_status"] == "UNEXPLAINED_TARGET_SHORTFALL"


def test_positive_separation_zero_impulse_is_contact_envelope_only() -> None:
    envelope = classify_contact_observation(
        base_classification="ROBOT_ENVIRONMENT_CONTACT",
        base_allowed=False,
        minimum_separation_m=0.0098,
        maximum_impulse_norm_n_s=0.0,
    )
    penetration = classify_contact_observation(
        base_classification="ROBOT_ENVIRONMENT_CONTACT",
        base_allowed=False,
        minimum_separation_m=-0.00005,
        maximum_impulse_norm_n_s=0.0,
    )
    impulse = classify_contact_observation(
        base_classification="ROBOT_ENVIRONMENT_CONTACT",
        base_allowed=False,
        minimum_separation_m=0.001,
        maximum_impulse_norm_n_s=0.1,
    )

    assert envelope == {
        "classification": "CONTACT_ENVELOPE_ONLY",
        "geometric_classification": "ROBOT_ENVIRONMENT_CONTACT",
        "physical_contact": False,
        "allowed": True,
    }
    assert penetration["physical_contact"] is True
    assert penetration["allowed"] is False
    assert impulse["physical_contact"] is True
    assert impulse["allowed"] is False


def test_signature_ignores_event_order_but_not_contact_content() -> None:
    case = {
        "robot": "follower_left",
        "joint": "waist",
        "direction": "positive",
        "status": "PASS",
        "target": 1.0,
        "final_readback": 0.99,
        "maximum_non_target_drift": 0.001,
        "contact_pairs": [
            {
                "actor_pair": ["b", "a"],
                "collider_pair": ["d", "c"],
                "classification": "ADJACENT_BODY_CONTACT",
                "maximum_penetration_m": 0.0,
            },
            {
                "actor_pair": ["f", "e"],
                "collider_pair": ["h", "g"],
                "classification": "SAME_RIGID_BODY",
                "maximum_penetration_m": 0.0,
            },
        ],
    }
    reversed_case = {**case, "contact_pairs": list(reversed(case["contact_pairs"]))}
    changed_case = {
        **case,
        "contact_pairs": [
            {**case["contact_pairs"][0], "maximum_penetration_m": 0.002}
        ],
    }

    assert deterministic_case_signature(case) == (
        deterministic_case_signature(reversed_case)
    )
    assert deterministic_case_signature(case) != (
        deterministic_case_signature(changed_case)
    )


def test_summary_requires_24_cases_per_repeat_and_identical_signatures() -> None:
    plans = build_sweep_cases(_limits())
    cases = []
    for repeat in range(2):
        cases.extend(
            [
                {
                    **plan,
                    "repeat": repeat,
                    "status": "PASS",
                    "final_readback": plan["target"],
                    "maximum_non_target_drift": 0.0,
                    "contact_pairs": [],
                }
                for plan in plans
            ]
        )

    report = summarize_sweep_cases(cases, repeat_count=2)

    assert report["status"] == "PASS"
    assert report["case_count"] == 48
    assert report["case_count_per_repeat"] == [24, 24]
    assert report["coverage_status"] == "PASS"
    assert report["determinism"]["status"] == "PASS"

    cases[-1] = {**cases[-1], "status": "FAIL"}
    failed = summarize_sweep_cases(cases, repeat_count=2)
    assert failed["status"] == "FAIL"


def test_summary_preserves_partial_contact_limited_cases() -> None:
    plans = build_sweep_cases(_limits())
    cases = []
    for repeat in range(2):
        cases.extend(
            [
                {
                    **plan,
                    "repeat": repeat,
                    "status": "PASS",
                    "motion_status": (
                        "CONTACT_LIMITED_WORKCELL_REACHABILITY"
                        if plan["case_id"].endswith("shoulder:positive")
                        else "TARGET_REACHED"
                    ),
                    "final_readback": plan["target"],
                    "maximum_non_target_drift": 0.0,
                    "contact_pairs": [],
                }
                for plan in plans
            ]
        )

    report = summarize_sweep_cases(cases, repeat_count=2)

    assert report["status"] == "PASS"
    assert report["failed_case_count"] == 0
    assert report["partial_case_count"] == 0
    assert report["contact_limited_case_count"] == 4
