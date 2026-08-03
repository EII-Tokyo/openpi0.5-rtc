import math
from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_correspondence import HOME_ARM
from tools.aloha1_mapping.home_sleep_correspondence import SLEEP_ARM
from tools.aloha1_mapping.home_sleep_correspondence import build_home_sleep_samples
from tools.aloha1_mapping.home_sleep_correspondence import command_index_for_physics_frame
from tools.aloha1_mapping.home_sleep_correspondence import command_signature
from tools.aloha1_mapping.home_sleep_correspondence import compare_aligned_joint_rows
from tools.aloha1_mapping.home_sleep_correspondence import count_follower_articulation_roots
from tools.aloha1_mapping.home_sleep_correspondence import digital_runtime_signature
from tools.aloha1_mapping.home_sleep_correspondence import evaluate_interbotix_group_limit_gate
from tools.aloha1_mapping.home_sleep_correspondence import validate_digital_preflight
from tools.aloha1_mapping.home_sleep_correspondence import values_within_float32_limits
from tools.audit_aloha1_sleep_limit_correspondence import _inspect_python_semantics
from tools.audit_aloha1_sleep_limit_correspondence import _isaac_sleep_saturation
from tools.audit_aloha1_sleep_limit_correspondence import _xacro_limits
from tools.audit_aloha1_sleep_limit_correspondence import build_root_cause_report
from tools.build_aloha1_home_sleep_command_manifest import build_manifest
from tools.build_aloha1_home_sleep_digital_report import build_digital_report
from tools.review_aloha1_home_sleep_digital_evidence import build_visual_review

ROOT = Path(__file__).resolve().parents[2]


def test_home_sleep_samples_freeze_three_cycles_and_end_at_home() -> None:
    samples = build_home_sleep_samples(
        home=HOME_ARM,
        sleep=SLEEP_ARM,
        command_hz=50,
        move_seconds=5,
        hold_seconds=1,
        cycles=3,
    )

    assert ARM_JOINT_ORDER == (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    )
    assert len(samples) == 1850
    assert samples[0].segment == "initial_home_hold"
    assert samples[-1].segment == "cycle_03_home_hold"
    assert samples[0].q_rad == pytest.approx(HOME_ARM)
    assert samples[-1].q_rad == pytest.approx(HOME_ARM)
    assert {len(sample.q_rad) for sample in samples} == {6}
    assert [sample.index for sample in samples] == list(range(1850))
    assert [sample.time_ns for sample in samples] == [index * 20_000_000 for index in range(1850)]


def test_home_sleep_segment_lengths_and_endpoints_are_exact() -> None:
    samples = build_home_sleep_samples(
        home=HOME_ARM,
        sleep=SLEEP_ARM,
        command_hz=50,
        move_seconds=5,
        hold_seconds=1,
        cycles=3,
    )
    by_segment: dict[str, list[object]] = {}
    for sample in samples:
        by_segment.setdefault(sample.segment, []).append(sample)

    assert len(by_segment["initial_home_hold"]) == 50
    for cycle in range(1, 4):
        prefix = f"cycle_{cycle:02d}"
        outbound = by_segment[f"{prefix}_home_to_sleep"]
        sleep_hold = by_segment[f"{prefix}_sleep_hold"]
        inbound = by_segment[f"{prefix}_sleep_to_home"]
        home_hold = by_segment[f"{prefix}_home_hold"]
        assert len(outbound) == 250
        assert len(sleep_hold) == 50
        assert len(inbound) == 250
        assert len(home_hold) == 50
        assert outbound[0].q_rad == pytest.approx(HOME_ARM)
        assert outbound[-1].q_rad == pytest.approx(SLEEP_ARM)
        assert sleep_hold[0].q_rad == pytest.approx(SLEEP_ARM)
        assert sleep_hold[-1].q_rad == pytest.approx(SLEEP_ARM)
        assert inbound[0].q_rad == pytest.approx(SLEEP_ARM)
        assert inbound[-1].q_rad == pytest.approx(HOME_ARM)
        assert home_hold[0].q_rad == pytest.approx(HOME_ARM)
        assert home_hold[-1].q_rad == pytest.approx(HOME_ARM)


def test_home_sleep_rejects_nonfinite_or_non_arm_vectors() -> None:
    with pytest.raises(ValueError, match="six finite arm joints"):
        build_home_sleep_samples(home=[0.0] * 7, sleep=SLEEP_ARM)
    with pytest.raises(ValueError, match="six finite arm joints"):
        build_home_sleep_samples(home=[0.0] * 5 + [math.nan], sleep=SLEEP_ARM)


def test_rational_scheduler_maps_sixty_hz_physics_to_fifty_hz_commands() -> None:
    assert command_index_for_physics_frame(0, physics_hz=60, command_hz=50, sample_count=1850) == 0
    assert command_index_for_physics_frame(6, physics_hz=60, command_hz=50, sample_count=1850) == 5
    assert command_index_for_physics_frame(60, physics_hz=60, command_hz=50, sample_count=1850) == 50
    assert command_index_for_physics_frame(999999, physics_hz=60, command_hz=50, sample_count=1850) == 1849


def test_command_signature_is_deterministic_and_changes_with_samples() -> None:
    first = build_home_sleep_samples(home=HOME_ARM, sleep=SLEEP_ARM)
    second = build_home_sleep_samples(home=HOME_ARM, sleep=SLEEP_ARM)
    changed = build_home_sleep_samples(
        home=HOME_ARM,
        sleep=(0.0, -2.04, 1.7, 0.0, -2.0, 0.0),
    )

    assert command_signature(first) == command_signature(second)
    assert command_signature(first) != command_signature(changed)


def test_home_sleep_manifest_freezes_official_sources_and_exclusions() -> None:
    config = yaml.safe_load((ROOT / "configs/aloha1_home_sleep_correspondence.yaml").read_text())

    manifest, source_audit = build_manifest(config, project_root=ROOT)

    assert source_audit["status"] == "PASS"
    assert source_audit["product"] == "aloha_vx300s"
    assert source_audit["home"]["value_rad"] == pytest.approx(HOME_ARM)
    assert source_audit["sleep"]["value_rad"] == pytest.approx(SLEEP_ARM)
    assert source_audit["command_dt_s"] == 0.02
    assert source_audit["moving_time_s"] == 5.0
    assert manifest["robot"] == "follower_left"
    assert manifest["joint_order"] == list(ARM_JOINT_ORDER)
    assert manifest["command_rate_hz"] == 50
    assert manifest["physics_rate_hz"] == 60
    assert manifest["sample_count"] == 1850
    assert len(manifest["samples"]) == 1850
    assert manifest["stationary_scope"] == {
        "follower_right": True,
        "follower_left_gripper": True,
        "follower_right_gripper": True,
    }
    assert manifest["real_execution_authorized"] is False
    assert manifest["candidate_promoted"] is False
    assert len(manifest["command_signature"]) == 64
    assert len(manifest["manifest_signature"]) == 64


def test_home_sleep_manifest_is_deterministic() -> None:
    config = yaml.safe_load((ROOT / "configs/aloha1_home_sleep_correspondence.yaml").read_text())

    first, _ = build_manifest(config, project_root=ROOT)
    second, _ = build_manifest(config, project_root=ROOT)

    assert first == second


def test_digital_preflight_fails_closed_on_any_frozen_contract_drift() -> None:
    contract = {
        "runtime_versions_match": True,
        "stage_hash_match": True,
        "manifest_hash_match": True,
        "default_prim": "/World",
        "root_prim_valid": True,
        "required_prims_valid": True,
        "articulation_count": 2,
        "dof_order_match": True,
        "finger_limit_hash_match": True,
        "home_finite_and_legal": True,
        "first_frame_arm_stable": True,
        "stationary_scope_declared": True,
        "source_hashes_immutable": True,
        "final_default_asset_modified": False,
    }
    passed = validate_digital_preflight(contract)
    assert passed["status"] == "PASS"
    assert passed["failed_gates"] == []

    for key in (
        "runtime_versions_match",
        "stage_hash_match",
        "manifest_hash_match",
        "root_prim_valid",
        "required_prims_valid",
        "dof_order_match",
        "finger_limit_hash_match",
        "home_finite_and_legal",
        "first_frame_arm_stable",
        "stationary_scope_declared",
        "source_hashes_immutable",
    ):
        drifted = dict(contract)
        drifted[key] = False
        result = validate_digital_preflight(drifted)
        assert result["status"] == "FAIL"
        assert key in result["failed_gates"]


def test_digital_runtime_signature_excludes_process_local_fields() -> None:
    run_a = {
        "runtime_pid": 100,
        "wall_time_s": 12.0,
        "rows": [
            {
                "physics_frame": 0,
                "command_index": 0,
                "target": [0.0, -0.96],
                "readback": [0.0, -0.959999],
            }
        ],
    }
    run_b = {
        **run_a,
        "runtime_pid": 200,
        "wall_time_s": 99.0,
    }
    assert digital_runtime_signature(run_a) == digital_runtime_signature(run_b)
    changed = {**run_b, "rows": [{**run_b["rows"][0], "command_index": 1}]}
    assert digital_runtime_signature(run_a) != digital_runtime_signature(changed)


def test_follower_articulation_count_excludes_environment_schema_root() -> None:
    roots = [
        "/World/follower_left/vx300s_left/root_joint",
        "/World/follower_right/vx300s_right/root_joint",
        "/World/environment/worldBody",
    ]
    assert count_follower_articulation_roots(roots) == roots[:2]


def test_limit_check_allows_only_float32_roundoff_not_physical_violation() -> None:
    assert values_within_float32_limits(
        [1.6057032346725464],
        [-1.7627824544906616],
        [1.6057027578353882],
    )
    assert not values_within_float32_limits(
        [1.6059],
        [-1.7627824544906616],
        [1.6057027578353882],
    )


def _visual_capture(mode: str, *, status: str = "PENDING_VISUAL_MODEL_REVIEW") -> dict:
    return {
        "status": status,
        "stage": {"sha256_before": "stage", "sha256_after": "stage"},
        "manifest": {"sha256": "manifest", "command_signature": "command"},
        "videos": {
            mode: {
                "absolute_path": f"/{mode}.mp4",
                "sha256": mode * 8,
                "frame_count": 558,
                "fps": 15,
                "duration_s": 37.2,
            }
        },
        "screenshots": [
            {
                "label": label,
                "mode": mode,
                "raw_absolute_path": f"/{label}_{mode}_raw.png",
                "raw_sha256": label * 8,
                "annotated_absolute_path": f"/{label}_{mode}_annotated.png",
                "annotated_sha256": (label + mode) * 4,
            }
            for label in (
                "before_limit_exceedance",
                "first_limit_exceedance",
                "first_sleep_hold_end",
                "final_home_recovery",
            )
        ],
    }


def test_visual_review_rejects_old_overlay_and_accepts_red_retake() -> None:
    original = _visual_capture("normal")
    original["videos"]["collision_overlay"] = {
        **original["videos"]["normal"],
        "absolute_path": "/old_collision.mp4",
        "sha256": "old" * 16,
    }
    original["screenshots"] += _visual_capture("collision_overlay")["screenshots"]
    retake = _visual_capture("collision_overlay")

    report = build_visual_review(
        original,
        retake,
        normal_review_status="PASS",
        collision_retake_review_status="PASS",
        rejected_collision_reason="REJECTED_COLLIDER_OVERLAY_NOT_DISTINCT",
    )

    assert report["status"] == "PASS_FAILURE_EVIDENCE"
    assert report["visual_review_is_auxiliary"] is True
    assert report["retained_videos"]["normal"]["absolute_path"] == "/normal.mp4"
    assert report["retained_videos"]["collision_overlay"]["absolute_path"] == "/collision_overlay.mp4"
    assert len(report["retained_screenshots"]) == 8
    assert report["rejected_attempts"][0]["reason"] == ("REJECTED_COLLIDER_OVERLAY_NOT_DISTINCT")


def _digital_run(status: str, signature: str) -> dict:
    return {
        "status": status,
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "stage": {"sha256_before": "stage", "sha256_after": "stage"},
        "manifest": {
            "sha256_before": "manifest",
            "sha256_after": "manifest",
            "command_signature": "command",
        },
        "source_or_final_asset_modified": False,
        "real_execution_authorized": False,
        "summary": {
            "status": status,
            "normalized_numeric_signature": signature,
            "gates": {
                "three_cycles_complete": True,
                "directions": True,
                "endpoints": status == "PASS",
                "legal_limits": status == "PASS",
                "final_home": True,
                "follower_right_stationary": True,
                "grippers_stationary": True,
                "finite_readback": True,
                "no_impulse_carrying_contact": True,
            },
            "endpoint_results": [],
            "contact": {"impulse_carrying_point_count": 0},
        },
        "preflight": {
            "status": "PASS",
            "limits": {
                "follower_left": [
                    [-3.14, -1.85, -1.76, -3.14, -1.868, -3.14],
                    [3.14, 1.257, 1.606, 3.14, 2.234, 3.14],
                ]
            },
        },
    }


def test_interbotix_group_gate_rejects_the_whole_first_illegal_sample() -> None:
    samples = build_home_sleep_samples(
        home=HOME_ARM,
        sleep=SLEEP_ARM,
        command_hz=50,
        move_seconds=5,
        hold_seconds=1,
        cycles=1,
    )
    outbound = [sample for sample in samples if sample.segment == "cycle_01_home_to_sleep"]

    result = evaluate_interbotix_group_limit_gate(
        outbound,
        lower_rad=[-math.pi, math.radians(-106), math.radians(-101), -math.pi, math.radians(-107), -math.pi],
        upper_rad=[math.pi, math.radians(72), math.radians(92), math.pi, math.radians(128), math.pi],
        moving_time_s=2.0,
        velocity_limits_rad_s=[math.pi] * 6,
    )

    assert result["first_rejected_segment_sample"] == 204
    assert result["accepted_sample_count"] == 204
    assert result["first_rejected_joint_names"] == ["shoulder"]
    assert result["last_published_q_rad"] == pytest.approx(
        [0.0, -1.8486345381526104, 1.6002409638554216, 0.0, -1.6859437751004016, 0.0]
    )
    assert result["first_rejected_q_rad"] == pytest.approx(
        [0.0, -1.853012048192771, 1.602409638554217, 0.0, -1.6927710843373494, 0.0]
    )
    assert result["command_semantics"] == "REJECT_WHOLE_GROUP_SAMPLE"


def test_digital_report_marks_visual_motion_pass_but_signal_gate_partial() -> None:
    run_1 = _digital_run("FAIL", "same")
    run_2 = _digital_run("FAIL", "same")
    visual = {
        "status": "PASS_FAILURE_EVIDENCE",
        "retained_videos": {},
        "retained_screenshots": [],
    }
    manifest = {
        "sleep_rad": [0.0, -2.05, 1.7, 0.0, -2.0, 0.0],
        "joint_order": list(ARM_JOINT_ORDER),
        "command_signature": "command",
        "real_execution_authorized": False,
    }

    report = build_digital_report(run_1, run_2, visual, manifest)

    assert report["status"] == "PARTIAL"
    assert report["classification"] == ("VISUAL_TRAJECTORY_PASS_SIGNAL_SEMANTICS_MISMATCH")
    assert report["layer_status"]["visual_trajectory"] == "PASS"
    assert report["layer_status"]["exact_sleep_endpoint"] == "FAIL"
    assert report["layer_status"]["real_api_signal_correspondence"] == "PARTIAL"
    assert report["numeric_repeatability"] == "PASS"
    assert report["real_execution_authorized"] is False
    assert [item["joint_name"] for item in report["limit_conflicts"]] == [
        "shoulder",
        "elbow",
        "wrist_angle",
    ]


def test_digital_report_requires_two_passing_fresh_runs_for_pass() -> None:
    run_1 = _digital_run("PASS", "same")
    run_2 = _digital_run("PASS", "same")
    manifest = {
        "sleep_rad": [0.0, -1.5, 1.5, 0.0, -1.5, 0.0],
        "joint_order": list(ARM_JOINT_ORDER),
        "command_signature": "command",
        "real_execution_authorized": False,
    }
    visual = {
        "status": "PASS_FAILURE_EVIDENCE",
        "retained_videos": {},
        "retained_screenshots": [],
    }

    report = build_digital_report(run_1, run_2, visual, manifest)

    assert report["status"] == "PASS"
    assert report["classification"] == "DIGITAL_HOME_SLEEP_VERIFIED"


def test_sleep_limit_root_cause_separates_video_from_signal_semantics() -> None:
    report = build_root_cause_report(
        official_sleep_rad=[0.0, -2.05, 1.7, 0.0, -2.0, 0.0],
        previous_sleep_rad=[0.0, -1.8, 1.55, 0.0, -1.57, 0.0],
        lower_rad=[-math.pi, math.radians(-106), math.radians(-101), -math.pi, math.radians(-107), -math.pi],
        upper_rad=[math.pi, math.radians(72), math.radians(92), math.pi, math.radians(128), math.pi],
        gate_result={
            "command_semantics": "REJECT_WHOLE_GROUP_SAMPLE",
            "first_rejected_segment_sample": 204,
            "accepted_sample_count": 204,
            "rejected_sample_count": 46,
            "first_rejected_joint_names": ["shoulder"],
            "last_published_q_rad": [0.0, -1.8486345381526104, 1.6002409638554216, 0.0, -1.6859437751004016, 0.0],
            "first_rejected_q_rad": [0.0, -1.853012048192771, 1.602409638554217, 0.0, -1.6927710843373494, 0.0],
        },
        source_facts={
            "aloha_sleep_uses_set_joint_positions": True,
            "set_joint_positions_checks_whole_group": True,
            "set_joint_positions_return_value_ignored": True,
            "generic_go_to_sleep_pose_bypasses_python_limit_check": True,
            "xs_sdk_group_callback_adds_no_urdf_limit_check": True,
        },
        runtime_facts={"isaac_sleep_saturation": {"status": "VERIFIED_INDIVIDUAL_DOF_LIMIT_SATURATION"}},
        source_records=[],
        branch_history=[],
    )

    assert report["status"] == "VERIFIED_ROOT_CAUSE"
    assert report["classification"] == ("OFFICIAL_ROS2_ALOHA_SLEEP_CONFIGURATION_OUTSIDE_ITS_OWN_URDF_LIMITS")
    assert report["video_interpretation"] == "PASS_TRAJECTORY_VISUAL"
    assert report["signal_correspondence_status"] == "PARTIAL"
    assert report["real_execution_status"] == "NOT_RUN_UNAUTHORIZED"
    assert [item["joint_name"] for item in report["limit_conflicts"]] == ["shoulder", "elbow", "wrist_angle"]
    assert report["previous_sleep_within_limits"] is True


def test_sleep_limit_audit_reads_radians_and_pi_offset_xacro_limits() -> None:
    path = (
        ROOT / ".codex/artifacts/20260802-aloha1-official-model-first/sources/"
        "interbotix_manipulators_b66d5b905725351dd71d3251a06cd3f4c777940f/"
        "aloha_vx300s.urdf.xacro"
    )

    lower, upper = _xacro_limits(path)

    assert lower == pytest.approx(
        [
            -math.pi + 0.00001,
            math.radians(-106),
            math.radians(-101),
            -math.pi + 0.00001,
            math.radians(-107),
            -math.pi + 0.00001,
        ]
    )
    assert upper == pytest.approx(
        [math.pi - 0.00001, math.radians(72), math.radians(92), math.pi - 0.00001, math.radians(128), math.pi - 0.00001]
    )


def test_sleep_limit_audit_finds_top_level_aloha_and_class_toolbox_methods() -> None:
    robot_utils = (
        ROOT / ".codex/artifacts/20260802-aloha1-official-model-first/sources/"
        "official_repo_probe/interbotix_aloha_main/aloha/robot_utils.py"
    )
    arm_module = (
        ROOT / ".codex/artifacts/20260803-aloha-home-sleep-root-cause/toolboxes_probe/"
        "interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/"
        "xs_robot/arm.py"
    )

    facts = _inspect_python_semantics(robot_utils, arm_module)

    assert facts == {
        "aloha_sleep_reads_group_sleep_positions": True,
        "aloha_sleep_uses_set_joint_positions": True,
        "set_joint_positions_return_value_ignored": True,
        "set_joint_positions_checks_whole_group": True,
        "generic_go_to_sleep_pose_bypasses_python_limit_check": True,
    }


def test_sleep_limit_audit_derives_individual_physx_saturation_from_telemetry(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "segment,target_arm_q,left_q\n"
        'cycle_01_sleep_hold,"[0,-2.05,1.7,0,-2.0,0]",'
        '"[0,-1.850049,1.605703,0,-1.867502,0]\n"',
        encoding="utf-8",
    )

    result = _isaac_sleep_saturation(
        telemetry,
        lower_rad=[-3.14, -1.850049, -1.76278, -3.14, -1.867502, -3.14],
        upper_rad=[3.14, 1.256637, 1.605703, 3.14, 2.234021, 3.14],
    )

    assert result["status"] == "VERIFIED_INDIVIDUAL_DOF_LIMIT_SATURATION"
    assert [item["joint_name"] for item in result["saturated_joints"]] == ["shoulder", "elbow", "wrist_angle"]


def test_real_digital_comparison_preserves_signed_joint_error() -> None:
    digital = [
        {"command_index": 0, "q": [0.0, -1.0]},
        {"command_index": 1, "q": [0.2, -1.2]},
        {"command_index": 2, "q": [0.4, -1.4]},
    ]
    real = [
        {"command_index": 0, "q": [0.1, -1.0]},
        {"command_index": 1, "q": [0.3, -1.1]},
        {"command_index": 2, "q": [0.5, -1.2]},
    ]

    result = compare_aligned_joint_rows(digital, real, joint_names=("a", "b"))

    assert result["matched_command_count"] == 3
    assert result["per_joint"][0]["signed_mean_error_rad"] == pytest.approx(0.1)
    assert result["per_joint"][1]["signed_mean_error_rad"] == pytest.approx(0.1)
    assert result["per_joint"][1]["maximum_abs_error_rad"] == pytest.approx(0.2)
