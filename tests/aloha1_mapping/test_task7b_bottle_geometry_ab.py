from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tools.aloha1_mapping.task7b_bottle_geometry_ab import compare_geometry_groups
from tools.aloha1_mapping.task7b_bottle_geometry_ab import render_comparison_markdown
from tools.aloha1_mapping.task7b_bottle_geometry_ab import validate_single_geometry_variable

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_task7b_bottle_geometry_ab.yaml"
COMBINER = ROOT / "tools/validate_aloha1_task7b_bottle_geometry_ab.py"
RUNTIME = ROOT / "tools/validate_aloha_viper_cad_finger_task5_bottle.py"


@pytest.fixture
def task7b_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _summary(
    status: str,
    *,
    pass_count: int,
    signatures: int = 1,
) -> dict[str, object]:
    return {
        "status": status,
        "pass_count": pass_count,
        "trial_count": 20,
        "deterministic": signatures == 1,
        "unique_signature_count": signatures,
        "minimum_drop_m": 0.001,
        "maximum_drop_m": 0.004,
        "mean_drop_m": 0.002,
        "failure_modes": [] if status == "PASS" else ["continuous_slip"],
    }


def test_profiles_change_only_geometry_provider(
    task7b_config: dict[str, object],
) -> None:
    profiles = task7b_config["profiles"]
    result = validate_single_geometry_variable(
        profiles["procedural_cylinder"],
        profiles["project_bottle500"],
        allowed_differences=task7b_config["allowed_profile_differences"],
    )
    assert result["status"] == "PASS"
    assert result["unexpected_differences"] == []
    assert result["differences"] == sorted(
        task7b_config["allowed_profile_differences"]
    )


def test_simultaneous_friction_change_is_rejected(
    task7b_config: dict[str, object],
) -> None:
    profiles = task7b_config["profiles"]
    candidate = copy.deepcopy(profiles["project_bottle500"])
    candidate["physics"]["friction"] = 1.0
    result = validate_single_geometry_variable(
        profiles["procedural_cylinder"],
        candidate,
        allowed_differences=task7b_config["allowed_profile_differences"],
    )
    assert result["status"] == "FAIL"
    assert result["unexpected_differences"] == ["physics.friction"]


@pytest.mark.parametrize(
    ("baseline_status", "project_status", "expected_status", "conclusion"),
    [
        (
            "PASS",
            "PASS",
            "PASS",
            "PROJECT_BOTTLE_MATCHES_BASELINE",
        ),
        (
            "PASS",
            "FAIL",
            "FAIL",
            "PROJECT_BOTTLE_WORSENS_HOLD",
        ),
        (
            "FAIL",
            "PASS",
            "FAIL",
            "PROJECT_BOTTLE_IMPROVES_HOLD",
        ),
        ("FAIL", "FAIL", "FAIL", "INCONCLUSIVE"),
    ],
)
def test_group_comparison_uses_allowed_conclusions(
    baseline_status: str,
    project_status: str,
    expected_status: str,
    conclusion: str,
) -> None:
    result = compare_geometry_groups(
        _summary(
            baseline_status,
            pass_count=20 if baseline_status == "PASS" else 0,
        ),
        _summary(
            project_status,
            pass_count=20 if project_status == "PASS" else 0,
        ),
    )
    assert result["status"] == expected_status
    assert result["conclusion"] == conclusion
    assert result["task8"] == "NOT_RUN"
    assert result["acceptance_boundary"] == (
        "STATIC_FREE_BOTTLE_HOLD_ONLY_NOT_SUPPORT_TO_LIFT_PICKUP"
    )


def test_group_comparison_rejects_incomplete_or_nondeterministic_pass() -> None:
    nineteen = _summary("PASS", pass_count=19)
    nineteen["trial_count"] = 19
    result = compare_geometry_groups(nineteen, _summary("PASS", pass_count=20))
    assert result["status"] == "FAIL"
    assert result["conclusion"] == "INCONCLUSIVE"

    nondeterministic = _summary("PASS", pass_count=20, signatures=2)
    result = compare_geometry_groups(
        _summary("PASS", pass_count=20),
        nondeterministic,
    )
    assert result["status"] == "FAIL"
    assert result["conclusion"] == "INCONCLUSIVE"


def test_markdown_does_not_claim_pickup() -> None:
    report = compare_geometry_groups(
        _summary("PASS", pass_count=20),
        _summary("PASS", pass_count=20),
    )
    markdown = render_comparison_markdown(report)
    assert "PROJECT_BOTTLE_MATCHES_BASELINE" in markdown
    assert "STATIC_FREE_BOTTLE_HOLD_ONLY_NOT_SUPPORT_TO_LIFT_PICKUP" in markdown
    assert "Task 8" in markdown


def test_existing_task5_runtime_exposes_bottle_geometry_provider() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert "--bottle-profile" in source
    assert '"procedural_cylinder"' in source
    assert '"project_bottle500"' in source
    assert "AddReference" in source
    assert "UsdShade.Tokens.strongerThanDescendants" in source
    assert '"bottle_asset_readback"' in source
    assert '"bottle_mass_override_readback_kg"' in source
    assert 'default="procedural_cylinder"' in source


def test_combiner_writes_40_profiled_trials(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline_report.json"
    project_report = tmp_path / "project_report.json"
    baseline_trials = tmp_path / "baseline_trials.jsonl"
    project_trials = tmp_path / "project_trials.jsonl"
    output_json = tmp_path / "result.json"
    output_markdown = tmp_path / "result.md"
    output_trials = tmp_path / "trials.jsonl"

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    common_report = {
        "run_mode": "ACCEPTANCE",
        "summary": _summary("PASS", pass_count=20),
        "baseline_protection": {"protected_assets_immutable": True},
        "screenshots": {
            "status": "PASS",
            "captures": [
                {"capture_name": phase}
                for phase in (
                    "open",
                    "bilateral_contact",
                    "release",
                    "hold_end",
                )
            ],
        },
        "boundaries": {"task8": "NOT_RUN"},
    }
    for path, profile_name in (
        (baseline_report, "procedural_cylinder"),
        (project_report, "project_bottle500"),
    ):
        report = copy.deepcopy(common_report)
        report["bottle_profile"] = profile_name
        report["causal_profile"] = config["profiles"][profile_name]
        path.write_text(json.dumps(report), encoding="utf-8")

    baseline_trials.write_text(
        "\n".join(
            json.dumps({"trial_index": index, "status": "PASS"})
            for index in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    project_trials.write_text(
        "\n".join(
            json.dumps({"trial_index": index, "status": "PASS"})
            for index in range(20)
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(COMBINER),
            "--config",
            str(CONFIG),
            "--baseline-report",
            str(baseline_report),
            "--baseline-trials",
            str(baseline_trials),
            "--project-report",
            str(project_report),
            "--project-trials",
            str(project_trials),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            "--output-trials",
            str(output_trials),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["conclusion"] == "PROJECT_BOTTLE_MATCHES_BASELINE"
    combined = [
        json.loads(line)
        for line in output_trials.read_text(encoding="utf-8").splitlines()
    ]
    assert len(combined) == 40
    assert sum(
        item["bottle_profile"] == "procedural_cylinder"
        for item in combined
    ) == 20
    assert sum(
        item["bottle_profile"] == "project_bottle500"
        for item in combined
    ) == 20
