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
