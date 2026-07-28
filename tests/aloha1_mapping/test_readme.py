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
    assert "BLOCKED / NOT RUN" in text
    assert "Isaac Sim **5.1.0.0 / Kit 107.3.3**" in text


def test_readme_records_reproduction_and_machine_reports() -> None:
    text = (
        PROJECT_ROOT / "README_ALOHA1_ISAACSIM_5_1.md"
    ).read_text(encoding="utf-8")

    for required in (
        "bash tools/build_aloha1_urdf.sh",
        "tools/import_aloha1_to_usd.py",
        "tools/validate_aloha1_gripper.py",
        "tools/validate_aloha1_asset.py",
        "reports/aloha1_mapping/gripper_validation.json",
        "reports/aloha1_mapping/validation_summary.json",
    ):
        assert required in text
