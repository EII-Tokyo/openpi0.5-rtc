from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.aloha1_mapping.gripper_collider_ab import ALLOWED_APPROXIMATIONS
from tools.aloha1_mapping.gripper_collider_ab import ALLOWED_CONTROL_MODES
from tools.aloha1_mapping.gripper_collider_ab import assert_profile_pair_is_frozen
from tools.aloha1_mapping.gripper_collider_ab import classify_decomposition_status
from tools.aloha1_mapping.gripper_collider_ab import classify_root_cause
from tools.aloha1_mapping.gripper_collider_ab import load_collision_profiles
from tools.aloha1_mapping.gripper_collider_ab import sha256_file
from tools.aloha1_mapping.gripper_collider_ab import summarize_ab_trials

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "configs/aloha1_gripper_collision_profiles.yaml"


def _trial(*, held: bool, bilateral: bool = True, persistent: bool = False) -> dict:
    return {
        "metrics": {
            "bilateral_contact_before_release": bilateral,
            "impulses_finite": True,
            "persistent_penetration": persistent,
            "unexpected_gripper_collision": False,
            "held_for_required_steps": held,
            "finite_state": True,
        },
        "runtime_s": 1.0,
        "deterministic_signature": "same" if held else "drop",
    }


def test_profile_freezes_every_non_experimental_variable() -> None:
    manifest = load_collision_profiles(PROFILE_PATH, PROJECT_ROOT)

    assert manifest["experiment"]["repeats_per_robot"] >= 20
    assert manifest["frozen"]["friction"] == 0.7
    assert manifest["frozen"]["restitution"] == 0.0
    assert manifest["frozen"]["bottle_mass_kg"] == 0.020
    assert manifest["frozen"]["bottle_diameter_m"] == 0.065
    assert manifest["frozen"]["physics_frequency_hz"] == 60
    assert manifest["frozen"]["solve_articulation_contact_last"] is True
    assert manifest["frozen"]["hold_interval_s"] == 2.0
    assert manifest["frozen"]["drop_gate_m"] == 0.010
    assert manifest["frozen"]["self_collision"] is False
    assert manifest["frozen"]["bottle_collision"] is True
    assert manifest["frozen"]["surface_gripper_allowed"] is False
    assert manifest["frozen"]["post_release_fixed_constraint_allowed"] is False
    assert set(manifest["profiles"]) == {"convex_hull", "convex_decomposition"}
    assert {"convexHull", "convexDecomposition"} == ALLOWED_APPROXIMATIONS
    assert {"current_mimic", "explicit_symmetric"} == ALLOWED_CONTROL_MODES


def test_only_approximation_may_differ_in_first_round() -> None:
    manifest = load_collision_profiles(PROFILE_PATH, PROJECT_ROOT)
    hull = manifest["profiles"]["convex_hull"]
    decomposition = manifest["profiles"]["convex_decomposition"]

    assert_profile_pair_is_frozen(hull, decomposition, allowed_differences={"approximation"})

    changed = dict(decomposition)
    changed["friction"] = 0.71
    with pytest.raises(ValueError, match="friction"):
        assert_profile_pair_is_frozen(hull, changed, allowed_differences={"approximation"})


def test_explicit_symmetric_control_is_diagnostic_only() -> None:
    manifest = load_collision_profiles(PROFILE_PATH, PROJECT_ROOT)
    control = manifest["control_modes"]["explicit_symmetric"]

    assert control["status"] == "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    assert control["right_target_expression"] == "-left_target"


def test_baseline_hash_manifest_matches_protected_files() -> None:
    manifest = load_collision_profiles(PROFILE_PATH, PROJECT_ROOT)

    for item in manifest["protected_baseline"]:
        path = PROJECT_ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha256_file(path) == item["sha256"], item["path"]


def test_ab_summary_preserves_original_hold_gate() -> None:
    summary = summarize_ab_trials(
        [_trial(held=True), _trial(held=False), _trial(held=True)],
        minimum_repeats=3,
    )

    assert summary["status"] == "FAIL"
    assert summary["hold_success_count"] == 2
    assert summary["hold_success_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["all_trials_pass_hold_gate"] is False
    assert summary["complete"] is True


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        (
            {
                "hull_current": [False, False],
                "decomposition_current": [True, True],
                "hull_explicit": [False, False],
                "decomposition_explicit": [True, True],
            },
            "collider_primary",
        ),
        (
            {
                "hull_current": [False, False],
                "decomposition_current": [False, False],
                "hull_explicit": [True, True],
                "decomposition_explicit": [True, True],
            },
            "mimic_primary",
        ),
        (
            {
                "hull_current": [False, False],
                "decomposition_current": [False, False],
                "hull_explicit": [False, False],
                "decomposition_explicit": [True, True],
            },
            "collider_and_mimic",
        ),
        (
            {
                "hull_current": [False, False],
                "decomposition_current": [False, False],
                "hull_explicit": [False, False],
                "decomposition_explicit": [False, False],
            },
            "neither_resolved",
        ),
    ],
)
def test_root_cause_classification(groups: dict[str, list[bool]], expected: str) -> None:
    assert classify_root_cause(groups, minimum_repeats=2)["classification"] == expected


def test_root_cause_is_inconclusive_with_incomplete_groups() -> None:
    groups = {
        "hull_current": [False],
        "decomposition_current": [True],
        "hull_explicit": [False],
        "decomposition_explicit": [True],
    }
    result = classify_root_cause(groups, minimum_repeats=2)

    assert result["classification"] == "inconclusive"
    assert result["status"] == "PARTIAL"


@pytest.mark.parametrize(
    ("hull", "decomposition", "expected"),
    [
        ([False, False], [True, True], "IMPROVES_HOLD"),
        ([True, False], [True, False], "NO_MEANINGFUL_EFFECT"),
        ([True, True], [False, False], "WORSENS_CONTACT"),
    ],
)
def test_decomposition_status(
    hull: list[bool],
    decomposition: list[bool],
    expected: str,
) -> None:
    assert (
        classify_decomposition_status(
            hull,
            decomposition,
            minimum_repeats=2,
        )["status"]
        == expected
    )


def test_decomposition_status_is_inconclusive_without_enough_repeats() -> None:
    result = classify_decomposition_status([False], [True], minimum_repeats=2)

    assert result["status"] == "INCONCLUSIVE"


def test_runtime_entrypoint_contains_required_isolated_controls() -> None:
    source_path = PROJECT_ROOT / "tools/validate_aloha1_gripper_collider_ab.py"
    source = source_path.read_text(encoding="utf-8")

    assert "set_solve_articulation_contact_last(True)" in source
    assert "PhysxContactReportAPI.Apply" in source
    assert "subscribe_contact_report_events" in source
    assert "create_new_stage()" in source
    assert "World.clear_instance()" in source
    assert "world.reset()" in source
    assert "right_target = -left_target" in source
    assert "SurfaceGripper" not in source


def test_generated_geometry_report_reads_actual_tokens_and_local_defaults() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/gripper_collider_comparison.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "PASS"
    assert report["local_api"]["urdf_importer"]["version"] == "2.4.30"
    defaults = report["local_api"]["convex_decomposition_api"]["defaults"]
    assert {name: item["value"] for name, item in defaults.items()} == {
        "errorPercentage": 10.0,
        "hullVertexLimit": 64,
        "maxConvexHulls": 32,
        "minThickness": pytest.approx(0.001),
        "shrinkWrap": False,
        "voxelResolution": 500000,
    }
    assert all(not item["authored"] for item in defaults.values())
    assert report["runtime_ab_evidence"]["CONVEX_DECOMPOSITION_STATUS"] == (
        "NO_MEANINGFUL_EFFECT"
    )
    for profile_name, expected in (
        ("convex_hull", "convexHull"),
        ("convex_decomposition", "convexDecomposition"),
    ):
        for asset in report["profiles"][profile_name]["assets"]:
            assert asset["layer"]["non_finger_collider_changes"] == []
            assert set(
                asset["layer"]["approximation_readback"].values()
            ) == {expected}
            for collider in asset["cooking"]["colliders"].values():
                assert collider["piece_count"] == (
                    1 if profile_name == "convex_hull" else 32
                )
                visualization = collider["visualization"]
                assert Path(
                    visualization["inner_gripping_surface_closeup"]
                ).is_file()


def test_generated_runtime_report_is_complete_and_physically_fails_hold() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/gripper_collider_ab_results.json"
        ).read_text(encoding="utf-8")
    )

    assert report["experiment_execution_status"] == "PASS"
    assert report["status"] == "FAIL"
    assert report["repeats_per_robot"] == 20
    assert report["CONVEX_DECOMPOSITION_STATUS"] == "NO_MEANINGFUL_EFFECT"
    assert report["root_cause_classification"]["classification"] == (
        "neither_resolved"
    )
    assert report["determinism"] == {
        "status": "PASS",
        "deterministic_within_every_robot_group": True,
    }
    assert set(report["groups"]) == {
        "hull_current",
        "decomposition_current",
        "hull_explicit",
        "decomposition_explicit",
    }
    trial_count = 0
    for group_name, group in report["groups"].items():
        assert group["combined"]["trial_count"] == 40
        assert group["combined"]["hold_success_count"] == 0
        assert group["combined"]["deterministic_per_robot"] is True
        assert group["diagnostic_metrics"]["bilateral_contact_trial_count"] == 40
        assert (
            group["diagnostic_metrics"]["persistent_penetration_trial_count"]
            == 0
        )
        assert (
            group["diagnostic_metrics"][
                "unexpected_internal_collision_trial_count"
            ]
            == 0
        )
        if group_name.endswith("explicit"):
            assert group["control_status"] == (
                "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
            )
        for robot in group["robots"].values():
            path = Path(robot["trial_file"])
            lines = path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 20
            first = json.loads(lines[0])
            assert first["fresh_reset"]["world_reset"] is True
            assert first["fresh_reset"]["resumed_from_contact_state"] is False
            assert first["metrics"]["actual_approximation_token_ok"] is True
            assert first["frozen"]["friction"] == 0.7
            assert first["frozen"]["bottle_mass_kg"] == 0.020
            assert first["frozen"]["physics_frequency_hz"] == 60
            assert first["released_hold"]["drop_gate_m"] == 0.010
            trial_count += len(lines)
    assert trial_count == 160


def test_readme_splits_gripper_gate_and_blocks_default_promotion() -> None:
    readme = (
        PROJECT_ROOT / "README_ALOHA1_ISAACSIM_5_1.md"
    ).read_text(encoding="utf-8")

    for gate in (
        "Finger motion direction",
        "Aperture monotonicity",
        "Mimic accuracy",
        "Collider geometry audit",
        "Bilateral contact establishment",
        "Contact normal quality",
        "Contact persistence",
        "Static bottle hold",
        "Determinism",
        "Performance",
    ):
        assert gate in readme
    assert "CONVEX_DECOMPOSITION_STATUS = NO_MEANINGFUL_EFFECT" in readme
    assert "default asset collider modified = `false`" in readme
    assert "Task 8 = `NOT_RUN`" in readme
