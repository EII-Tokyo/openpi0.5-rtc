from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_task7b2_support_to_lift.yaml"
MODULE = ROOT / "tools/aloha1_mapping/task7b2_support_to_lift.py"
RUNTIME = ROOT / "tools/validate_aloha1_task7b2_support_to_lift.py"


def _load_module() -> ModuleType:
    assert MODULE.is_file(), f"missing Task 7B.2 gate module: {MODULE}"
    spec = importlib.util.spec_from_file_location("task7b2_gate", MODULE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_metrics() -> dict[str, object]:
    return {
        "support_settle_pass": True,
        "support_contact_before_lift": True,
        "bilateral_contact_before_lift": True,
        "shoulder_delta_rad": -0.08,
        "expected_shoulder_delta_rad": -0.08,
        "non_target_arm_drift_within_gate": True,
        "bottle_left_support": True,
        "minimum_support_clearance_m": 0.006,
        "required_clearance_m": 0.005,
        "support_recontact_after_clear": False,
        "bilateral_contact_through_hold": True,
        "hold_drop_m": 0.002,
        "drop_gate_m": 0.010,
        "finite_state": True,
        "persistent_penetration": False,
        "forbidden_contact": False,
        "constraint_found": False,
        "surface_gripper_used": False,
        "parent_attachment_used": False,
    }


def test_config_freezes_stage_support_signal_and_task8() -> None:
    assert CONFIG.is_file(), f"missing Task 7B.2 config: {CONFIG}"
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    assert config["frozen_inputs"]["task7a_stage"]["sha256"] == (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
    )
    assert config["frozen_inputs"]["project_bottle_usd"]["sha256"] == (
        "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
    )
    assert config["support"]["prim_path"] == (
        "/World/environment/worldBody/user_confirmed_table"
    )
    assert config["robot"]["articulation_path"] == (
        "/World/follower_left/vx300s_left"
    )
    assert config["robot"]["left_finger_dof"] == "left_finger"
    assert config["robot"]["right_finger_dof"] == "right_finger"
    assert config["approach"] == {
        "source_case_id": "follower_left:shoulder:positive",
        "source_report_sha256": (
            "fb9340469c957c3f14ed4bc994209121f39cb738ccf21f749b4a6e36e78e4299"
        ),
        "source_curve_sha256": (
            "28187a5032f36fb1572b5b0aa671aec3f011963706f588de8ec80b8cf1c3d0be"
        ),
        "joint": "shoulder",
        "home_target_rad": -0.96,
        "sweep_target_rad": 1.1945033764839172,
        "sweep_steps": 180,
        "approach_frame": 98,
        "approach_target_rad": 0.2605069595575333,
        "trajectory": "cubic_smoothstep",
        "placement_xy_policy": "RUNTIME_APPROACH_APERTURE_MIDPOINT",
    }
    assert config["lift"] == {
        "joint": "shoulder",
        "start_target_rad": 0.2605069595575333,
        "lift_target_rad": 0.18050695955753326,
        "delta_rad": -0.08,
        "steps": 120,
        "trajectory": "cubic_smoothstep",
    }
    assert config["physics"]["friction"] == 0.7
    assert config["physics"]["mass_kg"] == 0.020
    assert config["physics"]["frequency_hz"] == 60
    assert config["physics"]["hold_steps"] == 120
    assert config["boundaries"]["task8"] == "NOT_RUN"


def test_placement_uses_table_top_bottle_bottom_and_aperture_xy() -> None:
    module = _load_module()
    translation = module.derive_supported_bottle_translation(
        table_bounds={
            "minimum": [-1.0, -1.0, -0.1],
            "maximum": [1.0, 1.0, 0.0],
        },
        bottle_bounds={
            "minimum": [-0.034, -0.034, -0.103],
            "maximum": [0.034, 0.034, 0.103],
        },
        aperture_midpoint=[0.2, -0.1, 0.3],
    )
    assert translation == pytest.approx([0.2, -0.1, 0.103])


@pytest.mark.parametrize(
    ("table_bounds", "bottle_bounds", "midpoint"),
    [
        (
            {"minimum": [0, 0, 0], "maximum": [0, 1, 1]},
            {"minimum": [-1, -1, -1], "maximum": [1, 1, 1]},
            [0, 0, 0],
        ),
        (
            {"minimum": [0, 0, 0], "maximum": [1, 1, 1]},
            {"minimum": [1, 1, 1], "maximum": [-1, -1, -1]},
            [0, 0, 0],
        ),
        (
            {"minimum": [0, 0, 0], "maximum": [1, 1, 1]},
            {"minimum": [-1, -1, -1], "maximum": [1, 1, 1]},
            [float("nan"), 0, 0],
        ),
    ],
)
def test_placement_rejects_invalid_geometry(
    table_bounds: dict[str, list[float]],
    bottle_bounds: dict[str, list[float]],
    midpoint: list[float],
) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="bounds|midpoint"):
        module.derive_supported_bottle_translation(
            table_bounds=table_bounds,
            bottle_bounds=bottle_bounds,
            aperture_midpoint=midpoint,
        )


def test_passing_pickup_requires_support_bilateral_lift_and_hold() -> None:
    module = _load_module()
    result = module.evaluate_pickup_trial(_passing_metrics())
    assert result["status"] == "PASS"
    assert result["failure_mode"] == "stable_support_to_lift_pickup"
    assert result["failed_checks"] == []


@pytest.mark.parametrize(
    ("mutation", "failure_mode"),
    [
        ({"support_settle_pass": False}, "support_settle_failed"),
        (
            {"bilateral_contact_before_lift": False},
            "bilateral_contact_not_established",
        ),
        ({"bottle_left_support": False}, "bottle_never_left_support"),
        (
            {"bilateral_contact_through_hold": False},
            "contact_lost_during_lift",
        ),
        (
            {"support_recontact_after_clear": True},
            "support_recontact_after_lift",
        ),
        ({"hold_drop_m": 0.02}, "continuous_slip_during_hold"),
        (
            {"persistent_penetration": True},
            "numerical_penetration_or_ejection",
        ),
        ({"forbidden_contact": True}, "forbidden_contact"),
    ],
)
def test_failure_classification_is_specific(
    mutation: dict[str, object],
    failure_mode: str,
) -> None:
    module = _load_module()
    metrics = _passing_metrics()
    metrics.update(mutation)
    result = module.evaluate_pickup_trial(metrics)
    assert result["status"] == "FAIL"
    assert result["failure_mode"] == failure_mode


def test_constraint_or_floating_hold_cannot_pass_as_pickup() -> None:
    module = _load_module()
    constrained = _passing_metrics()
    constrained["constraint_found"] = True
    assert module.evaluate_pickup_trial(constrained)["status"] == "FAIL"

    floating = _passing_metrics()
    floating["support_contact_before_lift"] = False
    result = module.evaluate_pickup_trial(floating)
    assert result["status"] == "FAIL"
    assert result["failure_mode"] == "support_settle_failed"


def _passing_trial(index: int, signature: str = "same") -> dict[str, object]:
    return {
        "trial_index": index,
        "status": "PASS",
        "fresh_world_reset": True,
        "deterministic_signature": signature,
        "metrics": _passing_metrics(),
        "failure_mode": "stable_support_to_lift_pickup",
    }


def test_group_pass_requires_twenty_fresh_deterministic_trials() -> None:
    module = _load_module()
    trials = [_passing_trial(index) for index in range(20)]
    summary = module.summarize_pickup_trials(trials, required_repeats=20)
    assert summary["status"] == "PASS"
    assert summary["pass_count"] == 20
    assert summary["trial_count"] == 20
    assert summary["deterministic"] is True
    assert summary["unique_signature_count"] == 1

    assert module.summarize_pickup_trials(
        trials[:19],
        required_repeats=20,
    )["status"] == "FAIL"
    trials[-1]["deterministic_signature"] = "different"
    assert module.summarize_pickup_trials(
        trials,
        required_repeats=20,
    )["status"] == "FAIL"


def test_markdown_preserves_task_boundaries() -> None:
    module = _load_module()
    report = {
        "status": "PASS",
        "conclusion": "SUPPORT_TO_LIFT_PICKUP_VERIFIED",
        "summary": module.summarize_pickup_trials(
            [_passing_trial(index) for index in range(20)],
            required_repeats=20,
        ),
        "boundaries": {
            "task7b_static_hold": "PASS",
            "asset_promotion": "PARTIAL",
            "task8": "NOT_RUN",
        },
    }
    markdown = module.render_pickup_markdown(report)
    assert "SUPPORT_TO_LIFT_PICKUP_VERIFIED" in markdown
    assert "Task 7B static hold: `PASS`" in markdown
    assert "Asset promotion: `PARTIAL`" in markdown
    assert "Task 8: `NOT_RUN`" in markdown
    json.dumps(report, allow_nan=False)


def test_runtime_source_contract_isolated_dynamic_pickup() -> None:
    assert RUNTIME.is_file(), f"missing Task 7B.2 runtime: {RUNTIME}"
    source = RUNTIME.read_text(encoding="utf-8")
    required = [
        "open_stage",
        "set_solve_articulation_contact_last(True)",
        '"/Bottle500"',
        "user_confirmed_table",
        "derive_supported_bottle_translation",
        "GetKinematicEnabledAttr().Set(False)",
        "APPROACH_FRAME",
        "SWEEP_STEPS",
        "LIFT_DELTA",
        "support_settle",
        "bilateral_contact_on_support",
        "support_clear",
        "hold_end",
        "subscribe_contact_report_events",
    ]
    for token in required:
        assert token in source

    forbidden = [
        "SurfaceGripper",
        "CreateFixedJoint",
        "parent_attachment_used = True",
        "source_layer.Save",
    ]
    for token in forbidden:
        assert token not in source
