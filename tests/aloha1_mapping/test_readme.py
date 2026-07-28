from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_readme_separates_evidence_classes_and_gates_optimization() -> None:
    text = (
        PROJECT_ROOT / "README_ALOHA1_ISAACSIM_5_1.md"
    ).read_text(encoding="utf-8")

    for heading in (
        "### Confirmed directly from official/local source",
        "### Reused from existing project reports",
        "### Physical measurements",
        "### Engineering inferences and acceptance thresholds",
        "### Temporary, uncalibrated values",
        "## HARD_BLOCKER and measurement checklist",
    ):
        assert heading in text
    assert "Task 8 was not executed" in text
    assert "Task 8 optimization | **NOT_RUN**" in text
    assert "Supplier-CAD Task 5 / static bottle hold | **NOT_RUN**" in text
    assert "Supplier-CAD Task 7 validation | **NOT_RUN**" in text
    assert "Isaac Sim **5.1.0.0 / Kit 107.3.3**" in text


def test_readme_records_reproduction_and_machine_reports() -> None:
    text = (
        PROJECT_ROOT / "README_ALOHA1_ISAACSIM_5_1.md"
    ).read_text(encoding="utf-8")

    for required in (
        "bash tools/build_aloha1_urdf.sh",
        "tools/import_aloha1_to_usd.py",
        "tools/map_aloha1_public_cad_gripper.py",
        "tools/compare_aloha_viper_finger_tessellations.py",
        "tools/audit_aloha1_cad_finger_isaac_gate.py",
        "reports/aloha1_mapping/aloha_viper_gripper_screenshot_review.json",
        "reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json",
        "reports/aloha1_mapping/aloha_viper_finger_tessellation.json",
        "reports/aloha1_mapping/aloha_viper_cad_finger_isaac_stage_gate.json",
    ):
        assert required in text
