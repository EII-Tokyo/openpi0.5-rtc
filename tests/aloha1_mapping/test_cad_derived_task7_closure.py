from tools.build_aloha1_cad_derived_task7_closure import build_report
from tools.build_aloha1_cad_derived_task7_closure import classify_task7


def test_runtime_pass_does_not_suppress_literal_rule_failures() -> None:
    result = classify_task7(
        runtime="PASS",
        visual="PARTIAL",
        velocity="PARTIAL",
        physics_rules="FAIL",
        robot_rules="FAIL",
        simready_rules="PASS",
    )

    assert result["runtime_grasp"] == "PASS"
    assert result["asset_promotion"] == "FAIL"
    assert result["task7"] == "PARTIAL"
    assert result["task8"] == "NOT_RUN"


def test_confirmed_visual_gate_removes_only_user_confirmation_blocker() -> None:
    report = build_report()

    assert "HARD_BLOCKER_EXACT_ATTEMPT7_VIDEO_USER_CONFIRMATION_NOT_RUN" not in report[
        "hard_blockers"
    ]
    assert "HARD_BLOCKER_BOTTLE_TENSOR_VELOCITY_SEMANTICS_INCONCLUSIVE" in report[
        "hard_blockers"
    ]
    assert "HARD_BLOCKER_LITERAL_PHYSICSRULES_FINDINGS_ON_DIAGNOSTIC_WORKCELL" in report[
        "hard_blockers"
    ]
    assert (
        "HARD_BLOCKER_LITERAL_ROBOTRULES_FINDINGS_ON_NON_ROBOT_PACKAGE_TARGET"
        in report["hard_blockers"]
    )
