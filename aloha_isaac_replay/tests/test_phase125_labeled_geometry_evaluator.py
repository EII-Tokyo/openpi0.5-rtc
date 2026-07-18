from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_phase125_labeled_geometry_evaluator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_phase125_labeled_geometry_evaluator", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_limit_zero_builds_all_available_phase120_command(tmp_path: Path) -> None:
    module = _load_module()

    command = module._build_phase120_command(
        reward=1,
        limit=0,
        date="2026-07-08",
        output_dir=tmp_path / "success",
    )

    assert "--reward" in command
    assert command[command.index("--reward") + 1] == "1"
    assert "--limit" in command
    assert command[command.index("--limit") + 1] == "100000"
    assert "--date" in command
    assert command[command.index("--date") + 1] == "2026-07-08"


def test_plan_contains_success_failure_and_comparison_steps(tmp_path: Path) -> None:
    module = _load_module()

    plan = module._build_plan(date="2026-07-08", success_limit=5, failure_limit=7, output_dir=tmp_path)

    assert [step["name"] for step in plan["steps"]] == ["success_cluster", "failure_cluster", "geometry_comparison"]
    assert str(tmp_path / "success_cluster") in plan["steps"][0]["command"]
    assert str(tmp_path / "failure_cluster") in plan["steps"][1]["command"]
    assert str(tmp_path / "geometry_comparison") in plan["steps"][2]["command"]
