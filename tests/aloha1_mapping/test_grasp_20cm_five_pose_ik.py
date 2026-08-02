from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.aloha1_mapping.grasp_20cm_five_pose_ik import apply_frozen_bottle_transform
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import canonical_five_pose_signature
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import compose_initial_command
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import derive_sample_geometry
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import gripper_approach_axis_world
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import initial_tool_orientation_gate
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import line_yaw_distance_deg
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import minimum_pairwise_ee_distance_m
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import minimum_pairwise_line_yaw_separation_deg
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import place_bottle_center_and_yaw
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import sample_bottle_center_yaw_candidates
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import sample_initial_arm_joint_candidates
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import select_diverse_records
from tools.aloha1_mapping.grasp_20cm_five_pose_ik import solve_oriented_initial_arm_pose
from tools.plan_aloha1_grasp_20cm_five_pose_ik import classify_preflight_contacts
from tools.plan_aloha1_grasp_20cm_five_pose_ik import freeze_preflight_records
from tools.plan_aloha1_grasp_20cm_five_pose_ik import is_excluded_runtime_failure_candidate
from tools.plan_aloha1_grasp_20cm_five_pose_ik import preserve_accepted_preflight_records
from tools.plan_aloha1_grasp_20cm_five_pose_ik import replacement_slot
from tools.run_aloha1_grasp_20cm_five_pose_ik import _read_run_evidence
from tools.run_aloha1_grasp_20cm_five_pose_ik import build_five_pose_summary
from tools.run_aloha1_grasp_20cm_five_pose_ik import resume_verified_runtime_records
from tools.run_aloha1_grasp_20cm_five_pose_ik import reuse_accepted_runtime_records

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_grasp_20cm_five_pose_ik.yaml"
GUI_SCRIPT = ROOT / "tools/run_aloha1_grasp_20cm_gui.py"
RUNNER_SCRIPT = ROOT / "tools/run_aloha1_grasp_20cm_five_pose_ik.py"


def test_replacement_slot_preserves_noncontiguous_user_accepted_samples() -> None:
    selected = [
        {"sample_id": "sample_01"},
        {"sample_id": "sample_03"},
        {"sample_id": "sample_04"},
    ]
    assert replacement_slot(selected, ["sample_02", "sample_05"]) == (
        "sample_02",
        1,
    )
    selected.append({"sample_id": "sample_02"})
    assert replacement_slot(selected, ["sample_02", "sample_05"]) == (
        "sample_05",
        4,
    )
    selected.append({"sample_id": "sample_05"})
    assert replacement_slot(selected, ["sample_02", "sample_05"]) is None


def _horizontal_world_from_object() -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return result


def _candidate_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample_{index + 1:02d}",
            "preflight_status": "PASS",
            "bottle_line_yaw_deg": yaw,
            "initial_ee_position_world_m": [0.06 * index, 0.0, 0.30],
        }
        for index, yaw in enumerate((4.0, 34.0, 64.0, 94.0, 124.0))
    ]


def _five_runtime_pass_records() -> list[dict[str, object]]:
    records = []
    for index in range(5):
        signature = f"signature-{index}"
        initialization_signature = f"initialization-{index}"
        records.append(
            {
                "sample_id": f"sample_{index + 1:02d}",
                "primary": {
                    "process_id": 100 + index,
                    "exit_code": 0,
                    "machine_status": "PASS",
                    "evidence_status": "PASS",
                    "deterministic_signature": signature,
                    "initialization_contract_status": "PASS",
                    "initialization_signature": initialization_signature,
                    "finger_safety_status": "PASS",
                    "finger_safety_violation_count": 0,
                    "video_count": 2,
                },
                "collider_repeat": {
                    "process_id": 200 + index,
                    "exit_code": 0,
                    "machine_status": "PASS",
                    "evidence_status": "PASS",
                    "deterministic_signature": signature,
                    "initialization_contract_status": "PASS",
                    "initialization_signature": initialization_signature,
                    "finger_safety_status": "PASS",
                    "finger_safety_violation_count": 0,
                    "collision_record_count": 24,
                },
                "visual_review_status": "NOT_REVIEWED",
            }
        )
    return records


def test_five_pose_config_freezes_joint_sampling_and_diversity() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == 2
    assert config["sampling"]["seed"] == 2026073102
    assert config["sampling"]["formal_sample_count"] == 5
    assert config["sampling"]["candidate_count"] == 256
    assert config["sampling"]["bottle_line_yaw_domain_deg"] == [0.0, 180.0]
    assert config["gates"]["minimum_bottle_line_yaw_separation_deg"] == 25.0
    assert config["gates"]["minimum_initial_ee_separation_m"] == 0.050
    assert config["gates"]["initial_arm_readback_tolerance_rad"] == 0.020
    assert config["gates"]["first_frame_jump_tolerance_rad"] == 0.020
    assert config["gates"]["arm_phase_readback_tolerance_rad"] == 0.020
    assert (
        config["frozen_inputs"]["task7a_structure_validation"]["sha256"]
        == "668a1c83e14d28de50c3fa18c773c6f60ec2feb2263eac985546c2bb7e52048a"
    )
    assert config["formal_structure"]["sample_01"]["bottle_center_world_x_m"] == 0.0
    assert config["formal_structure"]["sample_01"]["bottle_center_y_sign"] == "positive"
    assert config["formal_structure"]["sample_04"]["bottle_center_world_x_m"] == 0.0
    assert config["formal_structure"]["sample_04"]["bottle_center_y_sign"] == "negative"
    assert config["runtime"]["allow_runtime_resampling"] is False
    assert config["runtime"]["required_primary_videos"] == 5
    assert config["boundaries"]["task8"] == "NOT_RUN"


def test_five_pose_config_uses_official_acceleration_limited_lula_path() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    trajectory = config["arm_trajectory"]

    assert trajectory["mode"] == "LULA_CSPACE_ACCELERATION_LIMITED"
    assert trajectory["velocity_limits_rad_s"] == pytest.approx([np.pi] * 6)
    assert trajectory["acceleration_limits_rad_s2"] == [5.0] * 6
    assert trajectory["jerk_limit_status"] == ("NOT_SET_NO_EXACT_MODEL_OFFICIAL_VALUE")
    assert trajectory["classification"] == ("DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING")
    assert trajectory["source"]["local_path"].endswith("vx300s_joint_limits.yaml")
    assert len(trajectory["source"]["sha256"]) == 64
    gui_source = GUI_SCRIPT.read_text(encoding="utf-8")
    runner_source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    assert '"--arm-trajectory-mode"' in gui_source
    assert '"--arm-acceleration-limits-rad-s2"' in gui_source
    assert '"--arm-trajectory-mode"' in runner_source
    assert '"--arm-acceleration-limits-rad-s2"' in runner_source


def test_bottle_transform_places_cad_center_on_vertical_centerline() -> None:
    nominal = _horizontal_world_from_object()
    nominal[:3, 3] = [0.2, -0.1, 0.03]

    result = place_bottle_center_and_yaw(
        nominal_world_from_object=nominal,
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.08],
        yaw_delta_rad=np.deg2rad(47.0),
    )

    center = result[:3, :3] @ np.array([0.0, 0.0, 0.103]) + result[:3, 3]
    assert center[:2] == pytest.approx([0.0, 0.08], abs=1e-12)
    assert center[2] == pytest.approx((nominal[:3, :3] @ np.array([0.0, 0.0, 0.103]) + nominal[:3, 3])[2])
    assert np.linalg.det(result[:3, :3]) == pytest.approx(1.0)


def test_rotated_ab_and_grasp_transform_follow_object_yaw() -> None:
    world_from_object = place_bottle_center_and_yaw(
        nominal_world_from_object=_horizontal_world_from_object(),
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.08],
        yaw_delta_rad=np.deg2rad(82.0),
    )

    result = derive_sample_geometry(
        world_from_object=world_from_object,
        a_local_m=[0.0, 0.0, 0.0],
        b_local_m=[0.0, 0.0, 0.206],
        object_from_gripper=np.eye(4),
    )

    assert result["line_yaw_deg"] == pytest.approx(82.0)
    assert result["axis_to_world_z_deg"] == pytest.approx(90.0)
    assert result["world_from_gripper"] == pytest.approx(world_from_object)


def test_line_yaw_distance_is_modulo_180_degrees() -> None:
    assert line_yaw_distance_deg(5.0, 175.0) == pytest.approx(10.0)
    assert line_yaw_distance_deg(15.0, 47.0) == pytest.approx(32.0)


def test_five_selected_samples_meet_yaw_and_ee_distance_gates() -> None:
    selected = select_diverse_records(
        records=_candidate_records(),
        count=5,
        minimum_line_yaw_separation_deg=25.0,
        minimum_ee_separation_m=0.050,
    )

    assert len(selected) == 5
    assert minimum_pairwise_line_yaw_separation_deg(selected) >= 25.0
    assert minimum_pairwise_ee_distance_m(selected) >= 0.050


def test_joint_candidate_sampling_is_fixed_seed_and_within_limits() -> None:
    lower = np.array([-1.0, -0.8, -1.2, -1.5, -1.0, -2.0])
    upper = np.array([1.0, 0.9, 1.3, 1.5, 1.1, 2.0])

    first = sample_initial_arm_joint_candidates(
        lower_limits=lower,
        upper_limits=upper,
        seed=2026073102,
        count=256,
    )
    second = sample_initial_arm_joint_candidates(
        lower_limits=lower,
        upper_limits=upper,
        seed=2026073102,
        count=256,
    )

    assert np.array_equal(first, second)
    assert np.all(first >= lower)
    assert np.all(first <= upper)


def test_initial_tool_orientation_gate_uses_gripper_local_positive_x() -> None:
    root_half = float(np.sqrt(0.5))
    downward_wxyz = [root_half, 0.0, root_half, 0.0]

    assert gripper_approach_axis_world(downward_wxyz) == pytest.approx([0.0, 0.0, -1.0])
    result = initial_tool_orientation_gate(
        downward_wxyz,
        maximum_angle_to_world_down_deg=23.241131059202324,
    )

    assert result["status"] == "PASS"
    assert result["approach_axis_local"] == [1.0, 0.0, 0.0]
    assert result["approach_axis_world"] == pytest.approx([0.0, 0.0, -1.0])
    assert result["angle_to_world_down_deg"] == pytest.approx(0.0)


def test_initial_tool_orientation_gate_rejects_upward_approach() -> None:
    result = initial_tool_orientation_gate(
        [0.0, 0.0, 1.0, 0.0],
        maximum_angle_to_world_down_deg=23.241131059202324,
    )

    assert result["status"] == "FAIL"
    assert result["angle_to_world_down_deg"] == pytest.approx(90.0)


class _FakeKinematicsSolver:
    def __init__(self) -> None:
        self.inverse_call: dict[str, object] | None = None

    def compute_inverse_kinematics(
        self,
        frame_name: str,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        **kwargs: object,
    ) -> tuple[np.ndarray, bool]:
        self.inverse_call = {
            "frame_name": frame_name,
            "target_position": target_position.copy(),
            "target_orientation": target_orientation.copy(),
            **kwargs,
        }
        return np.asarray([0.1, -0.2, 0.3, 0.0, 0.2, -0.1]), True

    def compute_forward_kinematics(
        self,
        frame_name: str,
        joint_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert frame_name == "ee_gripper_link"
        assert joint_positions.shape == (6,)
        return (
            np.asarray([0.10, -0.05, 0.40]),
            np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
        )


def test_oriented_initial_pose_is_solved_in_task_space_and_read_back() -> None:
    solver = _FakeKinematicsSolver()
    result = solve_oriented_initial_arm_pose(
        kinematics_solver=solver,
        frame_name="ee_gripper_link",
        target_position_world_m=[0.10, -0.05, 0.40],
        target_orientation_world_wxyz=[
            float(np.sqrt(0.5)),
            0.0,
            float(np.sqrt(0.5)),
            0.0,
        ],
        warm_start_arm_q_rad=[0.0] * 6,
        lower_limits_rad=[-1.0] * 6,
        upper_limits_rad=[1.0] * 6,
        position_tolerance_m=0.001,
        orientation_tolerance_rad=0.005,
        maximum_approach_angle_to_world_down_deg=23.241131059202324,
    )

    assert result["status"] == "PASS"
    assert result["initial_arm_q_rad"] == pytest.approx([0.1, -0.2, 0.3, 0.0, 0.2, -0.1])
    assert result["fk_position_error_m"] == pytest.approx(0.0)
    assert result["fk_orientation_error_rad"] == pytest.approx(0.0)
    assert result["orientation_gate"]["status"] == "PASS"
    assert solver.inverse_call is not None
    assert solver.inverse_call["warm_start"] == pytest.approx([0.0] * 6)


def test_preserved_successes_are_explicit_orientation_exceptions() -> None:
    records = [
        {
            "sample_id": f"sample_{index:02d}",
            "preflight_status": "PASS",
            "initial_arm_q_rad": [float(index)] * 6,
            **(
                {"initial_orientation_policy": ("TASK_SPACE_VALIDATED_T_O_G_ORIENTATION_THEN_LULA_IK")}
                if index == 2
                else {}
            ),
        }
        for index in range(1, 4)
    ]

    preserved = preserve_accepted_preflight_records(
        records,
        sample_ids=["sample_01", "sample_02"],
    )

    assert [record["sample_id"] for record in preserved] == [
        "sample_01",
        "sample_02",
    ]
    assert preserved[0]["initial_orientation_policy"] == ("USER_ACCEPTED_LEGACY_INITIAL_ORIENTATION_EXCEPTION")
    assert preserved[1]["initial_orientation_policy"] == ("TASK_SPACE_VALIDATED_T_O_G_ORIENTATION_THEN_LULA_IK")
    assert records[0].get("initial_orientation_policy") is None


def test_runtime_reuse_keeps_only_verified_accepted_successes() -> None:
    source = {"samples": _five_runtime_pass_records()}
    source["samples"][1]["initial_orientation_policy"] = "TASK_SPACE_VALIDATED_T_O_G_ORIENTATION_THEN_LULA_IK"

    reused = reuse_accepted_runtime_records(
        source,
        sample_ids=["sample_01", "sample_02"],
    )

    assert [record["sample_id"] for record in reused] == [
        "sample_01",
        "sample_02",
    ]
    assert all(record["execution_policy"] == "REUSED_USER_ACCEPTED_SUCCESS_NO_RERECORDING" for record in reused)
    assert reused[0]["initial_orientation_policy"] == ("USER_ACCEPTED_LEGACY_INITIAL_ORIENTATION_EXCEPTION")
    assert reused[1]["initial_orientation_policy"] == ("TASK_SPACE_VALIDATED_T_O_G_ORIENTATION_THEN_LULA_IK")
    assert source["samples"][0].get("execution_policy") is None


def test_interrupted_runtime_resume_keeps_only_complete_machine_successes() -> None:
    source = {"samples": _five_runtime_pass_records()[:3]}

    resumed = resume_verified_runtime_records(source)

    assert [record["sample_id"] for record in resumed] == [
        "sample_01",
        "sample_02",
        "sample_03",
    ]
    assert all(
        record["execution_policy"]
        == "RESUMED_INTERRUPTED_MACHINE_SUCCESS_NO_RERECORDING"
        for record in resumed
    )
    assert source["samples"][0].get("execution_policy") is None


def test_interrupted_runtime_resume_rejects_incomplete_collision_evidence() -> None:
    source = {"samples": _five_runtime_pass_records()[:3]}
    source["samples"][2]["collider_repeat"]["collision_record_count"] = 23

    with pytest.raises(ValueError, match="not a complete success: sample_03"):
        resume_verified_runtime_records(source)


def test_interrupted_runtime_resume_rejects_missing_initialization_contract() -> None:
    source = {"samples": _five_runtime_pass_records()[:3]}
    source["samples"][1]["primary"].pop("initialization_signature")

    with pytest.raises(ValueError, match="initialization"):
        resume_verified_runtime_records(source)


def test_failed_runtime_candidate_is_explicitly_excluded_by_sample() -> None:
    exclusions = {"sample_05": [108]}

    assert is_excluded_runtime_failure_candidate(
        exclusions,
        sample_id="sample_05",
        candidate_index=108,
    )
    assert not is_excluded_runtime_failure_candidate(
        exclusions,
        sample_id="sample_05",
        candidate_index=109,
    )
    assert not is_excluded_runtime_failure_candidate(
        exclusions,
        sample_id="sample_04",
        candidate_index=108,
    )


def test_gui_supports_machine_only_screening_without_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import run_aloha1_grasp_20cm_gui

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_aloha1_grasp_20cm_gui.py",
            "--skip-video-capture",
        ],
    )

    args = run_aloha1_grasp_20cm_gui._parse_args()  # noqa: SLF001

    assert args.skip_video_capture is True


def test_gui_supports_sparse_collision_evidence_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import run_aloha1_grasp_20cm_gui

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_aloha1_grasp_20cm_gui.py",
            "--collision-evidence-only",
        ],
    )

    args = run_aloha1_grasp_20cm_gui._parse_args()  # noqa: SLF001

    assert args.collision_evidence_only is True


def test_missing_candidate_manifest_is_reported_without_hashing_it(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "aloha1_grasp_20cm_runtime.json"
    telemetry = tmp_path / "aloha1_grasp_20cm_telemetry.jsonl"
    runtime.write_text(
        '{"status":"PASS","runtime":{"initial_pose":{}},'
        '"stage":{},"metrics":{},"boundaries":{}}\n',
        encoding="utf-8",
    )
    telemetry.write_text("{}\n", encoding="utf-8")

    result = _read_run_evidence(
        artifact_root=tmp_path,
        process={"exit_code": 0, "timed_out": False},
        collision_repeat=False,
        selected={
            "world_from_object": np.eye(4).tolist(),
            "initial_arm_q_rad": [0.0] * 6,
        },
        stage_sha256="0" * 64,
        readback_tolerance_rad=0.02,
        first_frame_jump_tolerance_rad=0.02,
        hold_frames=12,
    )

    assert result["evidence_status"] == "FAIL"
    assert result["candidate_manifest_sha256"] is None
    assert any(
        error.startswith("missing:")
        for error in result["evidence_errors"]
    )


@pytest.mark.parametrize(
    ("formal_sample_index", "x_relation", "y_sign"),
    [
        (0, "zero", "positive"),
        (1, "negative", "any"),
        (3, "zero", "negative"),
    ],
)
def test_bottle_candidate_sampling_obeys_formal_spatial_structure(
    formal_sample_index: int,
    x_relation: str,
    y_sign: str,
) -> None:
    records = sample_bottle_center_yaw_candidates(
        center_xy_bounds={"minimum": [-0.30, -0.20], "maximum": [0.10, 0.25]},
        yaw_domain_deg=[0.0, 180.0],
        seed=2026073102,
        count=8,
        formal_sample_index=formal_sample_index,
    )
    repeat = sample_bottle_center_yaw_candidates(
        center_xy_bounds={"minimum": [-0.30, -0.20], "maximum": [0.10, 0.25]},
        yaw_domain_deg=[0.0, 180.0],
        seed=2026073102,
        count=8,
        formal_sample_index=formal_sample_index,
    )

    assert records == repeat
    assert all(0.0 <= record["bottle_line_yaw_deg"] < 180.0 for record in records)
    if x_relation == "zero":
        assert all(record["bottle_center_xy_m"][0] == 0.0 for record in records)
    else:
        assert all(record["bottle_center_xy_m"][0] < 0.0 for record in records)
    if y_sign == "positive":
        assert all(record["bottle_center_xy_m"][1] > 0.0 for record in records)
    elif y_sign == "negative":
        assert all(record["bottle_center_xy_m"][1] < 0.0 for record in records)


def test_apply_frozen_transform_preserves_t_o_g_and_input_profile() -> None:
    nominal = _horizontal_world_from_object()
    nominal[:3, 3] = [-0.10, -0.15, 0.034]
    object_from_gripper = np.eye(4)
    object_from_gripper[:3, 3] = [0.0, 0.0, 0.069]
    original_grasp = nominal @ object_from_gripper
    profile = {
        "kinematics": {
            "placement": {
                "placement_matrix": nominal.tolist(),
                "bottle_axis": {
                    "a_world_m": (nominal @ [0.0, 0.0, 0.0, 1.0])[:3].tolist(),
                    "b_world_m": (nominal @ [0.0, 0.0, 0.206, 1.0])[:3].tolist(),
                    "grasp_point_world_m": original_grasp[:3, 3].tolist(),
                },
                "target_poses": {
                    "object_from_gripper": object_from_gripper.tolist(),
                    "grasp_ee_position_world_m": original_grasp[:3, 3].tolist(),
                    "pregrasp_ee_position_world_m": (original_grasp[:3, 3] + [0.0, 0.0, 0.08]).tolist(),
                    "lift_ee_position_world_m": (original_grasp[:3, 3] + [0.0, 0.0, 0.21]).tolist(),
                    "orientation_world_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            }
        }
    }
    original_matrix = np.asarray(profile["kinematics"]["placement"]["placement_matrix"]).copy()
    frozen = place_bottle_center_and_yaw(
        nominal_world_from_object=nominal,
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.10],
        yaw_delta_rad=np.deg2rad(35.0),
    )

    result = apply_frozen_bottle_transform(
        profile,
        world_from_object=frozen,
    )

    placement = result["kinematics"]["placement"]
    expected_world_from_gripper = frozen @ object_from_gripper
    assert placement["placement_matrix"] == pytest.approx(frozen)
    assert placement["target_poses"]["object_from_gripper"] == pytest.approx(object_from_gripper)
    assert placement["target_poses"]["grasp_ee_position_world_m"] == pytest.approx(expected_world_from_gripper[:3, 3])
    assert (
        np.asarray(placement["target_poses"]["pregrasp_ee_position_world_m"]) - expected_world_from_gripper[:3, 3]
    ) == pytest.approx([0.0, 0.0, 0.08])
    assert np.asarray(profile["kinematics"]["placement"]["placement_matrix"]) == pytest.approx(original_matrix)


def test_canonical_signature_is_deterministic_and_pose_sensitive() -> None:
    records = [
        {
            "sample_id": "sample_01",
            "candidate_index": 7,
            "bottle_geometric_center_world_m": [0.0, 0.08, 0.034],
            "bottle_line_yaw_deg": 15.0,
            "world_from_object": np.eye(4).tolist(),
            "initial_arm_q_rad": [0.0, -0.9, 1.1, 0.0, -0.3, 0.0],
            "initial_ee_position_world_m": [-0.2, 0.0, 0.3],
        }
    ]

    first = canonical_five_pose_signature(records)
    second = canonical_five_pose_signature(records)
    changed = [dict(records[0], bottle_line_yaw_deg=16.0)]

    assert first == second
    assert len(first) == 64
    assert first != canonical_five_pose_signature(changed)


def test_initial_command_replaces_only_explicit_six_arm_dofs() -> None:
    baseline = np.arange(9, dtype=float)
    sampled_arm = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

    result = compose_initial_command(
        baseline,
        sampled_arm,
        arm_dof_indices=[0, 1, 2, 3, 4, 5],
    )

    assert result[:6] == pytest.approx(sampled_arm)
    assert result[6:] == pytest.approx(baseline[6:])


def test_initial_command_rejects_duplicate_or_out_of_range_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        compose_initial_command(
            np.zeros(9),
            np.zeros(6),
            arm_dof_indices=[0, 1, 2, 3, 4, 4],
        )
    with pytest.raises(ValueError, match="out of range"):
        compose_initial_command(
            np.zeros(9),
            np.zeros(6),
            arm_dof_indices=[0, 1, 2, 3, 4, 9],
        )


def test_selector_does_not_replace_runtime_failures() -> None:
    records = [
        {
            "sample_id": f"sample_{index + 1:02d}",
            "preflight_status": "PASS",
            "runtime_status": "PASS",
        }
        for index in range(5)
    ]
    records[1]["runtime_status"] = "FAIL"

    selected = freeze_preflight_records(records, required=5)

    assert [item["sample_id"] for item in selected] == [
        "sample_01",
        "sample_02",
        "sample_03",
        "sample_04",
        "sample_05",
    ]


def test_centerline_record_binds_geometric_center_not_prim_translation() -> None:
    records = [
        {
            "sample_id": "sample_01",
            "preflight_status": "PASS",
            "bottle_geometric_center_world_m": [0.0, 0.08, 0.034],
            "world_from_object": [
                [0.0, 0.0, 1.0, -0.103],
                [0.0, 1.0, 0.0, 0.08],
                [-1.0, 0.0, 0.0, 0.034],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    ]

    selected = freeze_preflight_records(records, required=1)

    assert selected[0]["bottle_geometric_center_world_m"][0] == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert selected[0]["world_from_object"][0][3] != pytest.approx(0.0)


def test_preflight_contact_policy_allows_only_confirmed_finger_table_pair() -> None:
    allowed = classify_preflight_contacts(
        [
            {
                "actor0_path": ("/World/follower_left/vx300s_left/follower_left_left_finger_link"),
                "actor1_path": ("/World/environment/worldBody/user_confirmed_table"),
                "separation_m": -0.0002,
                "impulse_ns": 0.001,
            }
        ]
    )
    blocked = classify_preflight_contacts(
        [
            {
                "actor0_path": ("/World/follower_left/vx300s_left/follower_left_shoulder_link"),
                "actor1_path": ("/World/environment/worldBody/user_confirmed_table"),
                "separation_m": -0.0002,
                "impulse_ns": 0.001,
            }
        ]
    )

    assert allowed["status"] == "PASS"
    assert allowed["allowed_physical_contact_count"] == 1
    assert blocked["status"] == "FAIL"
    assert blocked["forbidden_physical_contact_count"] == 1


def test_five_pose_summary_requires_all_five_machine_passes() -> None:
    summary = build_five_pose_summary(_five_runtime_pass_records())

    assert summary["machine_status"] == "PASS"
    assert summary["machine_pass_count"] == 5
    assert summary["primary_video_count"] == 5
    assert summary["fresh_process_count"] == 10
    assert summary["failed_sample_ids"] == []


def test_one_runtime_failure_cannot_be_hidden_by_visual_pass() -> None:
    records = _five_runtime_pass_records()
    records[3]["primary"]["machine_status"] = "FAIL"
    records[3]["visual_review_status"] = "PASS"

    summary = build_five_pose_summary(records)

    assert summary["machine_status"] == "FAIL"
    assert summary["status"] == "FAIL"
    assert summary["failed_sample_ids"] == ["sample_04"]


def test_five_pose_summary_rejects_runtime_finger_violation() -> None:
    records = _five_runtime_pass_records()
    records[1]["primary"]["finger_safety_status"] = "FAIL"
    records[1]["primary"]["finger_safety_violation_count"] = 1

    summary = build_five_pose_summary(records)

    assert summary["machine_status"] == "FAIL"
    assert summary["per_sample_gates"][1]["machine_gates"][
        "primary_finger_safety_pass"
    ] is False


def test_video_failure_does_not_erase_physics_machine_pass() -> None:
    records = _five_runtime_pass_records()
    records[2]["primary"]["evidence_status"] = "FAIL"
    records[2]["primary"]["video_count"] = 0

    summary = build_five_pose_summary(records)

    assert summary["machine_status"] == "PASS"
    assert summary["machine_pass_count"] == 5
    assert summary["status"] == "FAIL"
    assert summary["failed_sample_ids"] == []
    assert summary["evidence_failed_sample_ids"] == ["sample_03"]


def test_missing_video_candidate_is_reported_without_erasing_machine_pass(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "primary"
    artifact_root.mkdir()
    stage_hash = "a" * 64
    world_from_object = np.eye(4, dtype=np.float64)
    initial_arm_q = np.asarray(
        [-1.0, -0.1, 0.8, 2.2, 2.0, -1.2],
        dtype=np.float64,
    )
    runtime = {
        "status": "PASS",
        "reason": "stable_20cm_hold",
        "deterministic_signature": "b" * 64,
        "stage": {
            "sha256_before": stage_hash,
            "sha256_after": stage_hash,
        },
        "bottle_random_position": {
            "pose_mode": "FROZEN_CENTER_AND_YAW_TRANSFORM",
            "world_from_object": world_from_object.tolist(),
        },
        "runtime": {
            "initial_pose": {
                "initial_arm_q_target_rad": initial_arm_q.tolist(),
                "initial_pose_hold_frames_required": 60,
                "initial_pose_hold_frames_observed": 60,
                "initial_arm_max_readback_error_rad": 0.001,
                "first_frame_jump_rad": 0.001,
            },
            "ik": {"status": "PASS"},
        },
        "metrics": {
            "dynamic_during_formal_phases": True,
            "finite_state": True,
        },
        "boundaries": {
            "surface_gripper": False,
            "fixed_joint": False,
            "parent_attachment": False,
            "task8": "NOT_RUN",
        },
    }
    (artifact_root / "aloha1_grasp_20cm_runtime.json").write_text(
        __import__("json").dumps(runtime),
        encoding="utf-8",
    )
    (artifact_root / "aloha1_grasp_20cm_telemetry.jsonl").write_text(
        '{"frame": 1, "phase": "VALIDATE"}\n',
        encoding="utf-8",
    )

    result = _read_run_evidence(
        artifact_root=artifact_root,
        process={"exit_code": -15, "timed_out": True},
        collision_repeat=False,
        selected={
            "world_from_object": world_from_object.tolist(),
            "initial_arm_q_rad": initial_arm_q.tolist(),
        },
        stage_sha256=stage_hash,
        readback_tolerance_rad=0.02,
        first_frame_jump_tolerance_rad=0.02,
        hold_frames=60,
    )

    candidate_path = (
        artifact_root
        / "video_attempt_001/video/candidate_manifest.json"
    ).resolve()
    assert result["machine_status"] == "PASS"
    assert result["evidence_status"] == "FAIL"
    assert result["candidate_manifest_sha256"] is None
    assert f"missing:{candidate_path}" in result["evidence_errors"]
