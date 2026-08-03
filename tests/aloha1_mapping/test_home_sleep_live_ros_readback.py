from pathlib import Path

from tools.aloha1_mapping.home_sleep_live_runtime_report import build_live_runtime_report
from tools.build_aloha1_home_sleep_live_readback_report import _joint_state_summary
from tools.inspect_aloha1_home_sleep_ros1_readback import build_dry_run_report
from tools.inspect_aloha1_home_sleep_ros1_readback import classify_live_snapshot

ROOT = Path(__file__).resolve().parents[2]
INSPECTOR = ROOT / "tools/inspect_aloha1_home_sleep_ros1_readback.py"


def _snapshot() -> dict[str, object]:
    return {
        "nodes": [
            "/puppet_left/robot_state_publisher",
            "/puppet_left/xs_sdk",
            "/rosout",
        ],
        "joint_names": [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
            "left_finger",
            "right_finger",
        ],
        "joint_positions": [0.0] * 9,
        "joint_velocities": [0.0] * 9,
        "joint_efforts": [0.0] * 9,
        "joint_states_type": "sensor_msgs/JointState",
        "arm_command_type": "interbotix_xs_msgs/JointGroupCommand",
        "gripper_command_type": "interbotix_xs_msgs/JointSingleCommand",
        "arm_command_publishers": [],
        "gripper_command_publishers": [],
        "arm_command_subscribers": ["/puppet_left/xs_sdk"],
        "gripper_command_subscribers": ["/puppet_left/xs_sdk"],
        "load_configs": False,
        "mode_configs": "/root/interbotix_ws/src/aloha/config/puppet_modes_left.yaml",
        "motor_configs": "/root/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_control/config/vx300s.yaml",
    }


def test_readback_passes_only_exact_left_runtime_contract() -> None:
    report = classify_live_snapshot(_snapshot())

    assert report["status"] == "PASS_PUPPET_LEFT_READ_ONLY_RUNTIME"
    assert report["arm_joint_order"] == [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    assert report["commands_published_by_inspector"] == 0
    assert report["forbidden_driver_nodes"] == []


def test_right_driver_or_command_publisher_fails_contract() -> None:
    snapshot = _snapshot()
    snapshot["nodes"] = [*snapshot["nodes"], "/puppet_right/xs_sdk"]
    snapshot["arm_command_publishers"] = ["/unexpected_controller"]

    report = classify_live_snapshot(snapshot)

    assert report["status"] == "FAIL_PUPPET_LEFT_RUNTIME_SCOPE"
    assert report["checks"]["no_forbidden_driver_nodes"] is False
    assert report["checks"]["no_arm_command_publishers"] is False


def test_inspector_defaults_to_no_ros_access_and_has_no_publisher_call() -> None:
    report = build_dry_run_report()
    source = INSPECTOR.read_text(encoding="utf-8")

    assert report["status"] == "NOT_RUN_EXPLICIT_READ_ONLY_ROS_FLAG_REQUIRED"
    assert report["ros_connected"] is False
    assert report["commands_published_by_inspector"] == 0
    assert "rospy.Publisher(" not in source
    assert "rosservice" not in source


def test_aggregate_keeps_motion_and_workspace_gates_blocked() -> None:
    ros_report = classify_live_snapshot(_snapshot())
    camera = {
        "status": "PASS_CAM_HIGH_SINGLE_CAMERA_RUNTIME",
        "frames_captured": 600,
        "frames_published": 600,
        "hardware_resets": 0,
        "robot_command_publishers": 0,
    }
    aggregate = build_live_runtime_report(
        ros_report=ros_report,
        camera_pre=camera,
        camera_post=camera,
        driver_log="operating mode for the 'arm' group was changed to position.\n"
        "operating mode for the 'gripper' joint was changed to linear_position.\n"
        "Interbotix 'xs_sdk' node is up!\n",
        deployment_hashes={"launch": "a" * 64},
        artifact_paths={"camera_pre": "/tmp/pre.jpg"},
        driver_running=True,
        joint_state_samples={
            "rows": 20,
            "max_position_span_rad": 0.001534,
            "max_abs_reported_velocity": 0.0,
        },
    )

    assert aggregate["status"] == "PASS_READ_ONLY_RUNTIME_MOTION_NOT_RUN"
    assert aggregate["REAL_MOTION"] == "NOT_RUN_AUTHORIZATION_REQUIRED"
    assert aggregate["WORKSPACE_CLEAR_FOR_MOTION"] == "FAIL_CLUTTERED_TABLE"
    assert aggregate["driver"]["left_driver_running"] is True
    assert aggregate["commands_published"] == 0


def test_joint_state_summary_reads_rostopic_csv_columns(tmp_path: Path) -> None:
    source = tmp_path / "joint_states.csv"
    source.write_text(
        "%time,field.position0,field.position1,field.velocity0\n"
        "1,0.0,-1.0,0.0\n"
        "2,0.0015,-1.0002,0.02\n",
        encoding="utf-8",
    )

    summary = _joint_state_summary(source)

    assert summary["rows"] == 2
    assert summary["max_position_span_rad"] == 0.0015
    assert summary["max_abs_reported_velocity"] == 0.02
