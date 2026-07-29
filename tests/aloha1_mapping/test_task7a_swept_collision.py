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
