import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PRODUCERS = (
    "tools/aloha1_mapping/build_urdf.py",
    "tools/validate_aloha1_asset.py",
    "tools/validate_aloha1_gripper.py",
)
MACHINE_CSV_REPORTS = (
    "reports/aloha1_mapping/gripper_curves.csv",
    "reports/aloha1_mapping/joint_inventory.csv",
    "reports/aloha1_mapping/mesh_inventory.csv",
    "reports/aloha1_mapping/one_joint_curves.csv",
)


def test_csv_writers_explicitly_pin_lf_line_endings() -> None:
    missing = []
    for relative in CSV_PRODUCERS:
        tree = ast.parse(
            (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "DictWriter"
            ):
                continue
            line_ending = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "lineterminator"
                    and isinstance(keyword.value, ast.Constant)
                ),
                None,
            )
            if line_ending != "\n":
                missing.append(f"{relative}:{node.lineno}")

    assert missing == []


def test_machine_csv_reports_contain_no_carriage_returns() -> None:
    offenders = [
        relative
        for relative in MACHINE_CSV_REPORTS
        if b"\r" in (PROJECT_ROOT / relative).read_bytes()
    ]

    assert offenders == []
