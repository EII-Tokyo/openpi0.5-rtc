from __future__ import annotations

import dataclasses
from typing import Any


CONTROL_INTERFACE_GATE = "level_1_control_interface"
FIXED_POSE_TASK_GATE = "level_2_fixed_pose_minimal_grasp"
RANDOMIZED_TRUE_STATE_GATE = "level_3_randomized_pose_true_state_task"
CAMERA_PERCEPTION_GATE = "level_4_camera_perception_task"
LOADED_GRIPPER_CALIBRATION_SUBGATE = "loaded_gripper_qpos_to_contact_surface_calibration"


@dataclasses.dataclass(frozen=True)
class ReadinessGate:
    level: int
    name: str
    status: str
    evidence: str
    next_action: str
    acceptance: str


def build_rl_readiness_report(
    *,
    drive_gate_pass: bool,
    drive_gate_evidence: str,
    loaded_gripper_calibration_pass: bool = False,
    loaded_gripper_calibration_evidence: str = "not evaluated; qpos is not yet calibrated to loaded finger-pad gap",
    fixed_pose_minimal_task_pass: bool = False,
    fixed_pose_minimal_task_evidence: str = "not evaluated by drive-target replay smoke",
    randomized_true_state_task_pass: bool = False,
    randomized_true_state_task_evidence: str = "not evaluated by fixed replay or fixed-pose smoke",
    camera_perception_task_pass: bool = False,
    camera_perception_task_evidence: str = "not evaluated before privileged-state task is stable",
    reset_gate_pass: bool = False,
    causality_gate_pass: bool = False,
    contact_reward_gate_pass: bool = False,
    observation_gate_pass: bool = False,
) -> dict[str, Any]:
    """Return a conservative four-level ALOHA RL-readiness report.

    Fixed-state replay only proves that control targets, DOF order, signs,
    limits, and deterministic reset are plausible under one initial condition.
    It does not prove that a policy can pick up a bottle from an unknown pose.

    The first useful RL milestone is level 3: randomized bottle poses with
    privileged simulator truth in the observation.  Camera-based policies are a
    later level-4 milestone.
    """

    if not drive_gate_pass:
        control_status = "FAIL"
        control_next_action = "fix_dof_mapping_limits_reset_and_tracking"
    elif not loaded_gripper_calibration_pass:
        control_status = "PARTIAL"
        control_next_action = "calibrate_loaded_gripper_qpos_to_contact_surface_gap"
    else:
        control_status = "PASS"
        control_next_action = "build_fixed_pose_approach_close_lift_task"

    gates = [
        ReadinessGate(
            level=1,
            name=CONTROL_INTERFACE_GATE,
            status=control_status,
            evidence=(
                f"drive: {drive_gate_evidence}; "
                f"loaded gripper calibration: {loaded_gripper_calibration_evidence}"
            ),
            next_action=control_next_action,
            acceptance=(
                "same initial state, correct DOF order/signs/limits, stable gripper open-close, "
                "stable reset, repeatable motion for the same action input, and a loaded gripper qpos-to-contact "
                "surface calibration for soft bottle grasping"
            ),
        ),
        ReadinessGate(
            level=2,
            name=FIXED_POSE_TASK_GATE,
            status="PASS" if fixed_pose_minimal_task_pass else "NOT_EVALUATED",
            evidence=fixed_pose_minimal_task_evidence,
            next_action=(
                "add_small_random_bottle_pose_reset_and_privileged_state_observation"
                if fixed_pose_minimal_task_pass
                else "prove scripted_or_ik_fixed_bottle_grasp_and_lift"
            ),
            acceptance=(
                "with one fixed bottle pose, a deterministic controller can approach the bottle, close the gripper "
                "on the bottle body, lift it, and satisfy target-contact gates without non-target collision hacks"
            ),
        ),
        ReadinessGate(
            level=3,
            name=RANDOMIZED_TRUE_STATE_GATE,
            status="PASS" if randomized_true_state_task_pass else "NOT_EVALUATED",
            evidence=randomized_true_state_task_evidence,
            next_action=(
                "introduce_camera_or_keypoint_observation_curriculum"
                if randomized_true_state_task_pass
                else "prove_reset_randomization_true_state_observation_reward_and_termination"
            ),
            acceptance=(
                "on reset, bottle pose varies over a small range; observation includes q, qdot, gripper state, "
                "bottle pose relative to the robot base, and target pose; policy performance is measured across "
                "held-out random seeds"
            ),
        ),
        ReadinessGate(
            level=4,
            name=CAMERA_PERCEPTION_GATE,
            status="PASS" if camera_perception_task_pass else "NOT_EVALUATED",
            evidence=camera_perception_task_evidence,
            next_action=(
                "start_camera_policy_or_asymmetric_actor_critic_training"
                if camera_perception_task_pass
                else "keep_privileged_state_task_as_control_and_physics_baseline_before_camera_training"
            ),
            acceptance=(
                "replace privileged object pose with RGB/RGB-D/keypoints/pose estimates, or use asymmetric "
                "training where the actor sees images and the critic may see simulator truth"
            ),
        ),
    ]
    privileged_state_ready = all(gate.status == "PASS" for gate in gates[:3])
    camera_ready = all(gate.status == "PASS" for gate in gates)
    if camera_ready:
        status = "READY_FOR_CAMERA_BASED_RL_TRAINING"
    elif privileged_state_ready:
        status = "READY_FOR_PRIVILEGED_STATE_RL_TRAINING_NOT_CAMERA"
    else:
        status = "NOT_READY_FOR_RL_TRAINING"
    return {
        "overall_rl_training_ready": bool(camera_ready),
        "privileged_state_rl_training_ready": bool(privileged_state_ready),
        "camera_based_rl_training_ready": bool(camera_ready),
        "status": status,
        "replay_scope": (
            "Fixed-initial-state replay is a calibration gate. It does not prove grasping from unknown bottle poses."
        ),
        "legacy_subgates": {
            "reset_gate_pass": bool(reset_gate_pass),
            "causality_gate_pass": bool(causality_gate_pass),
            "contact_reward_gate_pass": bool(contact_reward_gate_pass),
            "observation_gate_pass": bool(observation_gate_pass),
            "note": (
                "These legacy booleans are diagnostic only. The four readiness levels above are the canonical gates."
            ),
        },
        "control_interface_subgates": {
            "drive_gate_pass": bool(drive_gate_pass),
            LOADED_GRIPPER_CALIBRATION_SUBGATE: {
                "pass": bool(loaded_gripper_calibration_pass),
                "evidence": loaded_gripper_calibration_evidence,
                "note": (
                    "For soft bottle tasks, observed gripper qpos is not accepted as a direct finger-pad gap "
                    "measurement until loaded calibration or spacer evidence exists."
                ),
            },
        },
        "gates": [dataclasses.asdict(gate) for gate in gates],
    }
